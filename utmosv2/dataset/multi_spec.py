from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
import torch
import torchaudio

from utmosv2.dataset._base import BaseDataset, DataDomainMixin
from utmosv2.dataset._utils import (
    extend_audio,
    select_random_start,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from utmosv2._settings._config import Config
    from utmosv2.dataset._schema import DatasetItem, InMemoryData


class MultiSpecDataset(BaseDataset):
    """
    Dataset class for mel-spectrogram feature extractor. This class is responsible for
    loading audio data, generating multiple spectrograms for each sample, and
    applying the necessary transformations.

    Args:
        cfg (SimpleNamespace): The configuration object containing dataset and model settings.
        data (list[DatasetSchema] | pd.DataFrame): The dataset containing file paths and labels.
        phase (str): The phase of the dataset, either "train" or any other phase (e.g., "valid").
        transform (str, dict[Callable[[torch.Tensor], torch.Tensor]] | None): Transformation function to apply to spectrograms.
    """

    def __init__(
        self,
        cfg: Config,
        data: pd.DataFrame | list[DatasetItem] | InMemoryData,
        phase: str,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        super().__init__(cfg, data, phase, transform)
        # Pre-create mel transforms once so filter banks are computed only at init,
        # not on every __getitem__ call.
        self._mel_transforms: dict[int, torchaudio.transforms.MelSpectrogram] = {
            i: torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sr,
                n_fft=spec_cfg.n_fft,
                hop_length=spec_cfg.hop_length,
                n_mels=spec_cfg.n_mels,
                win_length=spec_cfg.win_length,
                power=2.0,
            )
            for i, spec_cfg in enumerate(cfg.dataset.specs)
            if spec_cfg.mode == "melspec"
        }

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get the spectrogram and target MOS for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: The spectrogram (torch.Tensor) and target MOS (torch.Tensor) for the sample.
        """
        y, target = self._get_audio_and_mos(idx)
        specs = []
        length = int(self.cfg.dataset.spec_frames.frame_sec * self.cfg.sr)
        y = extend_audio(y, length, method=self.cfg.dataset.spec_frames.extend)
        for _ in range(self.cfg.dataset.spec_frames.num_frames):
            y1 = select_random_start(y, length)
            y1_t = torch.from_numpy(y1).unsqueeze(0)  # [1, T]
            for i, spec_cfg in enumerate(self.cfg.dataset.specs):
                mel_t = self._mel_transforms.get(i)
                if mel_t is not None:
                    spec = _make_melspec_torch(y1_t, mel_t, spec_cfg)
                else:
                    spec = _make_spctrogram(self.cfg, spec_cfg, y1)
                if self.cfg.dataset.spec_frames.mixup_inner and self.phase == "train":
                    y2 = select_random_start(y, length)
                    y2_t = torch.from_numpy(y2).unsqueeze(0)
                    if mel_t is not None:
                        spec2 = _make_melspec_torch(y2_t, mel_t, spec_cfg)
                    else:
                        spec2 = _make_spctrogram(self.cfg, spec_cfg, y2)
                    lmd = np.random.beta(
                        self.cfg.dataset.spec_frames.mixup_alpha,
                        self.cfg.dataset.spec_frames.mixup_alpha,
                    )
                    spec = lmd * spec + (1 - lmd) * spec2
                spec = np.stack([spec, spec, spec], axis=0)
                spec_tensor = torch.tensor(spec, dtype=torch.float32)
                phase = "train" if self.phase == "train" else "valid"
                assert self.transform is not None, "Transform must be provided."
                spec_tensor = self.transform[phase](spec_tensor)
                specs.append(spec_tensor)
        spec_tensor = torch.stack(specs).float()

        return spec_tensor, target


class MultiSpecExtDataset(MultiSpecDataset, DataDomainMixin):
    """
    Dataset class for mel-spectrogram feature extractor with data-domain embedding.

    Args:
        cfg (SimpleNamespace | ModuleType):
            The configuration object containing dataset and model settings.
        data (pd.DataFrame | list[DatasetSchema]):
            The dataset containing file paths and labels.
        phase (str):
            The phase of the dataset, either "train" or any other phase (e.g., "valid").
        transform (dict[str, Callable[[torch.Tensor], torch.Tensor]] | None):
            Transformation function to apply to spectrograms.
    """

    def __init__(
        self,
        cfg: Config,
        data: pd.DataFrame | list[DatasetItem] | InMemoryData,
        phase: str,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        MultiSpecDataset.__init__(self, cfg, data, phase, transform)
        DataDomainMixin.__init__(self, cfg)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get the spectrogram, data-domain embedding, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the generated spectrogram (torch.Tensor), data-domain embedding (torch.Tensor),
            and target MOS (torch.Tensor).
        """
        spec, target = super().__getitem__(idx)
        dt = self._get_data_domain_embedding(idx)

        return spec, dt, target


def _make_melspec_torch(
    y_tensor: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    spec_cfg: object,
) -> np.ndarray:
    """Compute mel spectrogram using a pre-created torchaudio transform.

    Equivalent to librosa.feature.melspectrogram + librosa.power_to_db(ref=np.max).
    """
    with torch.no_grad():
        power_spec = mel_transform(y_tensor).squeeze(0)  # [n_mels, T]
    # power_to_db with ref=max: 10 * log10(spec / max(spec))
    log_spec = 10.0 * power_spec.clamp(min=1e-10).log10()
    log_spec -= log_spec.max()
    spec = log_spec.numpy()
    if getattr(spec_cfg, "norm", None) is not None:
        spec = (spec + spec_cfg.norm) / spec_cfg.norm
    return spec


def _make_spctrogram(cfg: Config, spec_cfg: Config, y: np.ndarray) -> np.ndarray:
    return {
        "melspec": _make_melspec,
        "stft": _make_stft,
    }[spec_cfg.mode](cfg, spec_cfg, y)


def _make_melspec(cfg: Config, spec_cfg: Config, y: np.ndarray) -> np.ndarray:
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=spec_cfg.n_fft,
        hop_length=spec_cfg.hop_length,
        n_mels=spec_cfg.n_mels,
        win_length=spec_cfg.win_length,
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    if spec_cfg.norm is not None:
        spec = (spec + spec_cfg.norm) / spec_cfg.norm
    return spec


def _make_stft(cfg: Config, spec_cfg: Config, y: np.ndarray) -> np.ndarray:
    spec = librosa.stft(y=y, n_fft=spec_cfg.n_fft, hop_length=spec_cfg.hop_length)
    spec = np.abs(spec)
    spec = librosa.amplitude_to_db(spec)
    return spec
