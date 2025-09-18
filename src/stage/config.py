from pathlib import Path
import os
from typing import Optional
import os

ENTITY: str = ""
PROJECT: str = ""

ROOT = Path(__file__).parent.parent.parent
LOCAL_WEIGHTS_DIR: Path = ROOT / "weights"
LOCAL_CKP_DIR: Path = ROOT / "checkpoints"
AUDIO_DIR: Path = ROOT / "audio"
EVAL_DIR: Path = ROOT / "eval"
EXP_DIR: Path = ROOT / "experiments"
CKP_DIR = LOCAL_CKP_DIR


class ConfigurationError(ValueError):
    ...


def first_existing(*paths: Path | str) -> Optional[Path]:
    for path in paths:
        if Path(path).exists():
            return Path(path)


def moises_path() -> Path:
    path = first_existing(Path.home() / "dev/dataset/moisesdb/moisesdb_v0.1",
                          Path.home() / "dev/datasets/moisesdb/moisesdb_v0.1",
                          Path.home() / "lag-data/lag-moisesdb",
                          ROOT / "datasets/moisesdb")
    if path is None:
        raise RuntimeError("I can't find moisesdb")
    return path


def mus_path() -> Path:
    path = first_existing(
        Path.home() / "dev/dataset/moisesdb/musdb",
        Path.home() / "datasets/moisesdb/musdb",
        Path.home() / "lag-data/musdb",
        ROOT / "datasets/musdb",
    )
    if path is None:
        raise RuntimeError("I can't find musdb")
    return path


def mixdata_path() -> Path:
    path = first_existing(
        Path.home() / "dev/datasets/moisesdb",
        Path.home() / "datasets/moisesdb",
        Path.home() / "lag-data",
        Path.home() / "datasets/lag-data",
    )
    if path is None:
        raise RuntimeError("I can't find the mixed dataset path")
    return path


def weights_dir() -> Path:
    return LOCAL_WEIGHTS_DIR


def output_dir() -> Path:
    return CKP_DIR


def shutdown():
    print(f"Shutting down myself 💀")
    os.system("sudo shutdown now")


import torch

torch.set_float32_matmul_precision("medium")
