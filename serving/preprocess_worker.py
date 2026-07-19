#!/usr/bin/env python3
"""Preprocessing worker for the UTMOSv2 serving process pool.

decode (wav/mp3/flac/ogg/...) -> mono 16 kHz -> the exact UTMOSv2 feature stack:

  * SSL branch:   raw 3 s waveform  ->  x1  [ssl_len]              (float32)
  * spec branch:  num_frames x len(specs) mel-spectrograms, each 3x512x512
                  (the four multi-resolution mels the fusion model consumes)
                  ->  x2  [num_frames * n_specs, 3, 512, 512]      (float32)
  * domain:       one-hot data-domain embedding  ->  d  [n_domains] (float32)

All of this is Python/numpy/librosa/torchaudio-on-CPU and **holds the GIL**, so
running it in *threads* gives no real parallelism and caps the whole server while
the GPU starves.  Running it across *processes* bypasses the GIL and lets
preprocessing outrun the GPU -> the GPU saturates.  This module is imported by the
ProcessPool workers ONLY; it never touches CUDA (CUDA_VISIBLE_DEVICES is blanked in
init()), so workers stay CPU-only and never load the GPU model.

Because UTMOSv2 draws random crops (and mixup) per pass, a single request can be
preprocessed `reps` times (test-time augmentation); the caller averages the model
outputs over the reps.  Each rep is an independent stochastic view, so we stack
them and return one array per branch.
"""
from __future__ import annotations

import importlib
import io
import os
from types import SimpleNamespace

import numpy as np

_CFG = None      # UTMOSv2 config (SimpleNamespace)
_DATASET = None  # a reusable dataset instance (mel filters / stft transforms cached once)
_FE_SR = 16000


def init(config: str = "fusion_stage3", predict_dataset: str = "sarulab") -> None:
    """ProcessPool initializer: runs in each WORKER process only.

    Blank the GPU, then build the UTMOSv2 config + a single dataset instance whose
    mel filter banks and torchaudio STFT transforms are pre-cached (the expensive
    part of dataset construction).  Per request we only swap the in-memory audio.
    """
    global _CFG, _DATASET
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # workers must never grab the GPU
    os.environ.setdefault("OMP_NUM_THREADS", "1")      # spectrograms are single-threaded; parallelism is across procs
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    torch.set_num_threads(1)

    from utmosv2._settings import configure_execution
    from utmosv2.dataset._schema import InMemoryData
    from utmosv2.utils import get_dataset

    _mod = importlib.import_module(f"utmosv2.config.{config}")
    cfg = SimpleNamespace(**{k: v for k, v in _mod.__dict__.items() if not k.startswith("__")})
    configure_execution(cfg)
    # UTMOSv2's inference path removes silent sections before feature extraction.
    cfg.dataset.remove_silent_section = True
    _CFG = cfg

    # Build the dataset once with a dummy 1 s clip so the mel/STFT tables are cached.
    dummy = InMemoryData(data=np.zeros(_FE_SR, dtype=np.float32), dataset_name=predict_dataset)
    _DATASET = get_dataset(cfg, dummy, cfg.phase)


def _decode(raw: bytes) -> np.ndarray:
    """bytes -> mono float32 @ 16 kHz. Tries libsndfile first, falls back to librosa."""
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        import librosa

        audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        audio = audio.astype("float32")
    if sr != _FE_SR:
        try:
            import soxr

            audio = soxr.resample(audio, sr, _FE_SR).astype("float32")
        except Exception:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=_FE_SR).astype("float32")
    return np.ascontiguousarray(audio, dtype="float32")


def preprocess(raw: bytes, dataset_name: str = "sarulab", reps: int = 1):
    """bytes -> (x1[reps, ssl_len], x2[reps, F, 3, 512, 512], d[reps, n_domains]), all float32.

    Runs in a worker process (no GIL contention with the GPU thread).  `reps` random
    augmentation views are stacked; the server averages the model outputs over them.
    """
    global _CFG, _DATASET
    if _DATASET is None:
        init()

    from utmosv2.dataset._schema import InMemoryData

    audio = _decode(raw)
    # Guard: a clip that is empty (or all-silence, which the dataset's silence
    # removal would strip to empty) would divide-by-zero in extend_audio. Fall back
    # to a short low tone so the request returns a (meaningless) score, not a 500.
    if audio.size < _FE_SR // 10:
        t = np.arange(_FE_SR, dtype="float32") / _FE_SR
        audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")

    # Swap the in-memory audio + domain on the cached dataset (and its sub-datasets).
    data = InMemoryData(data=audio, dataset_name=dataset_name)
    _DATASET.data = data
    _DATASET.ssl.data = data
    _DATASET.multi_spec.data = data

    x1s, x2s, ds = [], [], []
    for _ in range(max(1, reps)):
        try:
            x1, x2, d, _target = _DATASET[0]  # (ssl wav, specs, domain one-hot, mos)
        except ZeroDivisionError:
            # Clip was all-silence -> silence removal emptied it. Fall back to a tone
            # so the request returns a score instead of a 500.
            t = np.arange(_FE_SR, dtype="float32") / _FE_SR
            tone = InMemoryData(data=(0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32"),
                                dataset_name=dataset_name)
            _DATASET.data = _DATASET.ssl.data = _DATASET.multi_spec.data = tone
            x1, x2, d, _target = _DATASET[0]
        x1s.append(x1.numpy())
        x2s.append(x2.numpy())
        ds.append(d.numpy())

    return (
        np.stack(x1s).astype("float32"),
        np.stack(x2s).astype("float32"),
        np.stack(ds).astype("float32"),
    )
