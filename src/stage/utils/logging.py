from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
import stage.hyperparameters
import wandb.util
# import boto3


def to_loggable(x):
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {to_loggable(k): to_loggable(v) for k, v in x.items()}
    if isinstance(x, stage.hyperparameters.Loggable):
        return x.to_dict()
    return x


def get_or_create_run_id(basepath: Path) -> str:
    run_id_file = basepath / "wandb_run_id.txt"

    if run_id_file.exists():
        run_id = run_id_file.read_text().strip()
    else:
        basepath.mkdir(parents=True, exist_ok=True)
        run_id = wandb.util.generate_id()
        run_id_file.write_text(run_id)

    return run_id


# def upload_to_s3(filepath: Path, bucket: str, name: str):
#     s3 = boto3.client("s3")
#     try:
#         s3.upload_file(filepath, bucket, name)
#         return True
#     except Exception as e:
#         print(f"Failed to load {filepath} to S3 with error: {e}")
#         return False
