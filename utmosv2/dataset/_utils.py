from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
import torchaudio

if TYPE_CHECKING:
    from utmosv2._settings._config import Config


def load_audio(cfg: Config, file: Path) -> np.ndarray:
    if Path(file).suffix == ".npy":
        return np.load(file)
    waveform, sr = torchaudio.load(file)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != cfg.sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=cfg.sr)
    return waveform.squeeze(0).numpy()


def extend_audio(y: np.ndarray, length: int, method: str) -> np.ndarray:
    if y.shape[0] > length:
        return y
    elif method == "tile":
        n = length // y.shape[0] + 1
        y = np.tile(y, n)
        return y
    else:
        raise NotImplementedError


def select_random_start(y: np.ndarray, length: int) -> np.ndarray:
    start = np.random.randint(0, y.shape[0] - length)
    return y[start : start + length]


def get_dataset_map(cfg: Config) -> dict[str, int]:
    if cfg.data_config:
        with open(cfg.data_config, "r") as f:
            datasets = json.load(f)
        return {d["name"]: i for i, d in enumerate(datasets["data"])}
    else:
        return {
            "bvcc": 0,
            "sarulab": 1,
            "blizzard2008": 2,
            "blizzard2009": 3,
            "blizzard2010-EH1": 4,
            "blizzard2010-EH2": 5,
            "blizzard2010-ES1": 6,
            "blizzard2010-ES3": 7,
            "blizzard2011": 8,
            "somos": 9,
        }


def get_dataset_num(cfg: Config) -> int:
    return len(get_dataset_map(cfg))
