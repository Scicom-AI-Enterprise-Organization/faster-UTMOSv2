from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from utmosv2.dataset import MultiSpecDataset, SSLExtDataset
from utmosv2.dataset._base import BaseDataset
from utmosv2.dataset._utils import extend_audio, select_random_start
from utmosv2.dataset.multi_spec import _make_melspec_torch, _make_spctrogram

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from utmosv2._settings._config import Config
    from utmosv2.dataset._schema import DatasetItem, InMemoryData


class SSLLMultiSpecExtDataset(BaseDataset):
    """
    Dataset class that combines both SSL (Self-Supervised Learning) and Multi-Spectrogram datasets.
    This dataset uses both SSLExtDataset and MultiSpecDataset to provide different representations
    of the same audio sample.

    Args:
        cfg (SimpleNamespace | ModuleType):
            The configuration object containing dataset and model settings.
        data (pd.DataFrame | list[DatasetSchema]):
            The dataset containing file paths and MOS labels.
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
        super().__init__(cfg, data, phase, transform)
        self.ssl = SSLExtDataset(cfg, data, phase)
        self.multi_spec = MultiSpecDataset(cfg, data, phase, transform)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get data for SSL feature extractor, mel-spectrogram feature extractor, data-domain embedding, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: data for SSL feature extractor (torch.Tensor), data for mel-spectrogram feature extractor (torch.Tensor),
            data-domain id (torch.Tensor), and target MOS (torch.Tensor).
        """
        # Load audio ONCE and share between SSL and spectrogram pipelines.
        y, target = self._get_audio_and_mos(idx)

        # SSL processing
        ssl_length = int(self.cfg.dataset.ssl.duration * self.cfg.sr)
        y_ssl = extend_audio(y, ssl_length, method="tile")
        y_ssl = select_random_start(y_ssl, ssl_length)
        x1 = torch.from_numpy(y_ssl)

        # Domain embedding (via SSLExtDataset's DataDomainMixin)
        d = self.ssl._get_data_domain_embedding(idx)

        # MultiSpec processing (inlined from MultiSpecDataset.__getitem__)
        specs = []
        spec_length = int(self.cfg.dataset.spec_frames.frame_sec * self.cfg.sr)
        y_spec = extend_audio(y, spec_length, method=self.cfg.dataset.spec_frames.extend)
        for _ in range(self.cfg.dataset.spec_frames.num_frames):
            y1 = select_random_start(y_spec, spec_length)
            y1_t = torch.from_numpy(y1).unsqueeze(0)
            for i, spec_cfg in enumerate(self.cfg.dataset.specs):
                mel_t = self.multi_spec._mel_transforms.get(i)
                if mel_t is not None:
                    spec = _make_melspec_torch(y1_t, mel_t, spec_cfg)
                else:
                    spec = _make_spctrogram(self.cfg, spec_cfg, y1)
                if self.cfg.dataset.spec_frames.mixup_inner and self.phase == "train":
                    y2 = select_random_start(y_spec, spec_length)
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
        x2 = torch.stack(specs).float()

        return x1, x2, d, target
