#!/usr/bin/env python3
"""GPU feature extraction for the UTMOSv2 serving path (FEATURE_DEVICE=cuda).

Replicates the exact serving feature stack of `SSLLMultiSpecExtDataset.__getitem__`
(3 s SSL crop + num_frames x len(specs) mel-spectrograms with inner mixup + resize)
with the DSP running as a few batched kernels on a torch device instead of
spec-by-spec on CPU:

  waveform crops [reps*frames(*specs), L]  --one small pinned H2D-->
  torchaudio STFT (power=2, center, constant-pad)  ->  librosa mel filter matmul
  ->  power_to_db(ref=max, top_db=80)  ->  (x+norm)/norm  ->  inner mixup
  ->  3-channel  ->  torchvision Resize(512, 512)  ->  x2 [reps, F*S, 3, 512, 512]

Numerical parity with the CPU worker path:
  * identical mel filter banks (librosa, slaney) and torchaudio STFT settings,
    just moved to the device; the mel matmul forces TF32 OFF so it is IEEE fp32;
  * identical log/normalize constants and the same `cfg.transform["valid"]` Resize;
  * np.random is consumed in EXACTLY the dataset's order (ssl crop; per frame:
    y1 crop; per spec: y2 crop + beta lambda), so a seeded run yields the same
    crops/mixup as the CPU path and features match to float tolerance
    (benchmark_features.py relies on this).

Perf notes: crop/RNG stays on cheap CPU numpy; the H2D upload is the raw crops
(~1 MB/rep) instead of the ~25 MB/rep of finished spectrogram images the CPU
path must ship, staged through a reusable pinned buffer with async copies.
All GPU work is enqueued on the CALLER's current stream — the server wraps calls
in its featurizer stream and orders the model forward after it with events.
"""
from __future__ import annotations

import librosa
import numpy as np
import torch
import torchaudio

from utmosv2.dataset._utils import extend_audio, get_dataset_map, select_random_start


class GpuFeaturizer:
    def __init__(self, cfg, device: str | torch.device = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.ssl_len = int(cfg.dataset.ssl.duration * cfg.sr)
        self.spec_len = int(cfg.dataset.spec_frames.frame_sec * cfg.sr)
        self.frames = int(cfg.dataset.spec_frames.num_frames)
        self.mixup = bool(cfg.dataset.spec_frames.mixup_inner)
        self.alpha = float(cfg.dataset.spec_frames.mixup_alpha)
        self.extend = cfg.dataset.spec_frames.extend
        self.specs = list(cfg.dataset.specs)
        self.resize = cfg.transform["valid"]          # the same Resize((512,512)) the CPU path applies
        self.dataset_map = get_dataset_map(cfg)
        self._pin: torch.Tensor | None = None          # reusable pinned staging buffer
        self._pin_ev: torch.cuda.Event | None = None

        self.mel_fbs: list[torch.Tensor] = []
        self.stfts: list[torchaudio.transforms.Spectrogram] = []
        for sc in self.specs:
            assert sc.mode == "melspec", f"GPU featurizer only supports melspec specs (got {sc.mode})"
            fb = librosa.filters.mel(
                sr=cfg.sr, n_fft=sc.n_fft, n_mels=sc.n_mels,
                fmin=0.0, fmax=None, htk=False, norm="slaney",  # librosa 0.9 default (matches CPU path)
            )
            self.mel_fbs.append(torch.from_numpy(fb).float().to(self.device))
            self.stfts.append(
                torchaudio.transforms.Spectrogram(
                    n_fft=sc.n_fft, hop_length=sc.hop_length, win_length=sc.win_length,
                    power=2.0, center=True, pad_mode="constant",
                ).to(self.device)
            )

    def _upload(self, arrs: list[np.ndarray]) -> list[torch.Tensor]:
        """Pack numpy arrays into one pinned buffer, issue async H2D on the current stream."""
        if self.device.type != "cuda":
            return [torch.from_numpy(a) for a in arrs]
        total = sum(a.size for a in arrs)
        if self._pin is None or self._pin.numel() < total:
            self._pin = torch.empty(total, dtype=torch.float32).pin_memory()
            self._pin_ev = None
        if self._pin_ev is not None:
            self._pin_ev.synchronize()                 # previous copies out of this buffer drained
        pin_np = self._pin.numpy()
        outs, o = [], 0
        for a in arrs:
            n = a.size
            pin_np[o:o + n] = a.ravel()
            outs.append(self._pin[o:o + n].view(a.shape).to(self.device, non_blocking=True))
            o += n
        self._pin_ev = torch.cuda.Event()
        self._pin_ev.record()
        return outs

    def _mel_db(self, y: torch.Tensor, i: int) -> torch.Tensor:
        """Crops [B, L] -> normalized log-mel [B, n_mels, T]; matches _make_melspec_fast."""
        sc = self.specs[i]
        power = self.stfts[i](y)                       # [B, n_fft//2+1, T]
        mel = torch.matmul(self.mel_fbs[i], power)     # [B, n_mels, T]
        lg = 10.0 * mel.clamp(min=1e-10).log10()
        lg = lg - lg.amax(dim=(-2, -1), keepdim=True)  # power_to_db ref=max, per spectrogram
        lg = lg.clamp(min=-80.0)                       # top_db=80
        norm = getattr(sc, "norm", None)
        if norm is not None:
            lg = (lg + norm) / norm
        return lg

    @torch.inference_mode()
    def featurize(
        self, audio: np.ndarray, dataset_name: str = "sarulab", reps: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Silence-removed mono 16 kHz waveform ->
        (x1 [R, ssl_len], x2 [R, F*S, 3, 512, 512], d [R, n_domains]) on self.device."""
        R, F, S = max(1, reps), self.frames, len(self.specs)

        y_ssl = extend_audio(audio, self.ssl_len, method="tile")
        y_spec = extend_audio(audio, self.spec_len, method=self.extend)

        x1 = np.empty((R, self.ssl_len), dtype=np.float32)
        y1s = np.empty((R * F, self.spec_len), dtype=np.float32)
        y2s = np.empty((R * F * S, self.spec_len), dtype=np.float32) if self.mixup else None
        lmds = np.empty((R * F * S,), dtype=np.float32) if self.mixup else None
        # np.random consumed in the same order as SSLLMultiSpecExtDataset.__getitem__
        for r in range(R):
            x1[r] = select_random_start(y_ssl, self.ssl_len)
            for f in range(F):
                y1s[r * F + f] = select_random_start(y_spec, self.spec_len)
                if self.mixup:
                    base = (r * F + f) * S
                    for s in range(S):
                        y2s[base + s] = select_random_start(y_spec, self.spec_len)
                        lmds[base + s] = np.random.beta(self.alpha, self.alpha)

        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # mel matmul in IEEE fp32 (match CPU exactly)
        try:
            if self.mixup:
                t1, t2, tl, tx1 = self._upload([y1s, y2s, lmds, x1])
                t2 = t2.view(R * F, S, -1)
                tl = tl.view(R * F, S, 1, 1)
            else:
                t1, tx1 = self._upload([y1s, x1])
            per_spec = []
            for i in range(S):
                s1 = self._mel_db(t1, i)                        # [R*F, M, T]
                if self.mixup:
                    s2 = self._mel_db(t2[:, i], i)
                    s1 = tl[:, i] * s1 + (1.0 - tl[:, i]) * s2  # mixup of the two NORMALIZED specs
                img = s1.unsqueeze(1).expand(-1, 3, -1, -1)     # [R*F, 3, M, T]
                per_spec.append(self.resize(img))               # [R*F, 3, 512, 512]
            x2 = torch.stack(per_spec, dim=1)                   # [R*F, S, 3, 512, 512]
            x2 = x2.reshape(R, F * S, *x2.shape[2:])            # frame-major, spec-minor = CPU order
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

        d = torch.zeros((R, len(self.dataset_map)), dtype=torch.float32, device=self.device)
        d[:, self.dataset_map[dataset_name]] = 1.0
        return tx1, x2, d
