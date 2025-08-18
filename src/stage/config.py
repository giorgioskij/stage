from pathlib import Path
import os
from typing import Optional
# import boto3
import os

# TODO: remove everything that's unnecessary to release

# ENTITY: str = "ballants-thesis"  # "latent-accompaniment-generation"
# PROJECT: str = "musicgen-conditioning"  # "accompaniment-generation"
ENTITY: str = "latent-accompaniment-generation"
PROJECT: str = "accompaniment-generation"

ROOT = Path(__file__).parent.parent.parent
LOCAL_WEIGHTS_DIR: Path = ROOT / "weights"
AUDIO_DIR: Path = ROOT / "audio"
EVAL_DIR: Path = ROOT / "eval"
EXP_DIR: Path = ROOT / "experiments"

EXT_CKP_DIR: Path = Path("/run/media/tkol/BIGSHAQ/lamsi/checkpoints")
LOCAL_CKP_DIR: Path = ROOT / "checkpoints"
CKP_DIR: Path = EXT_CKP_DIR if EXT_CKP_DIR.exists() else LOCAL_CKP_DIR


class ConfigurationError(ValueError):
    ...


def first_existing(*paths: Path | str) -> Optional[Path]:
    for path in paths:
        if Path(path).exists():
            return Path(path)


def running_on_sagemaker():
    return "SM_MODEL_DIR" in os.environ


def running_locally():
    return not running_on_sagemaker()


def moises_path() -> Path:
    if running_on_sagemaker():
        path = Path(os.environ["SM_CHANNEL_DATA"])
        return path
    path = first_existing(Path.home() / "dev/dataset/moisesdb/moisesdb_v0.1",
                          Path.home() / "dev/datasets/moisesdb/moisesdb_v0.1",
                          Path.home() / "lag-data/lag-moisesdb",
                          ROOT / "datasets/moisesdb")
    if path is None:
        raise RuntimeError("I can't find moisesdb")
    return path


def mus_path() -> Path:
    if running_on_sagemaker():
        path = Path(os.environ["SM_CHANNEL_DATA"])
        return path

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
    if running_on_sagemaker():
        path = Path(os.environ["SM_CHANNEL_DATA"])
        return path

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
    if running_on_sagemaker():
        return Path(os.environ["SM_CHANNEL_WEIGHTS"])
    return LOCAL_WEIGHTS_DIR


def output_dir() -> Path:
    if running_on_sagemaker():
        return Path(os.environ["SM_MODEL_DIR"])
    return CKP_DIR


# def stop_instance(instance_id):
#     ec2 = boto3.client('ec2')
#     ec2.stop_instances(InstanceIds=[instance_id])
#     print(f"Stopped instance: {instance_id}")


def shutdown():
    print(f"Shutting down myself 💀")
    os.system("sudo shutdown now")


import torch

torch.set_float32_matmul_precision("medium")
