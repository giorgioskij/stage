import lightning as L
from pathlib import Path
from typing import Set, Optional
from stage.config import ConfigurationError
from stage.data.stem import Stem
from stage.data.stemmed_dataset import StemmedDataset
from stage.data.dataset_mixed import MixDataset
from stage import hyperparameters as hp
import torch.utils.data
from torch import Tensor
import random


class StemmedDatamodule(L.LightningDataModule):

    def __init__(self, params: hp.StemmedDatasetParams):
        super().__init__()
        # self.stems: Set[Stem] = params.stems
        # self.single_stem: bool = params.single_stem
        # self.root_dir: Path = Path(params.root_dir)
        # self.clip_length_in_seconds: int = params.clip_length_in_seconds
        # self.sample_rate: int = params.sample_rate
        # self.batch_size_train: int = params.batch_size_train
        # self.batch_size_test: int = params.batch_size_test
        # self.num_workers: int = params.num_workers
        # self.speed_transform_p: float = params.speed_transform_p
        # self.pitch_transform_p: float = params.pitch_transform_p
        # self.n_samples_per_epoch: int = params.n_samples_per_epoch
        # self.target_stem: Stem = params.target_stem
        # self.add_click: bool = params.add_click
        # self.sync_chunks: bool = params.sync_chunks
        # self.bpm_in_caption: bool = params.bpm_in_caption
        self.params = params

        if isinstance(params, hp.MixDatasetParams):
            self.dataset_class = MixDataset
        else:
            self.dataset_class = StemmedDataset

        self.setup(None)

        self.lengths = {
            "train": len(self.train_dataloader()),
            "valid": len(self.val_dataloader())
        }

    def _collate_fn(self, batch):
        if self.params.min_context_seconds > self.params.clip_length_in_seconds:
            raise ConfigurationError(
                "Context has to be smaller than clip length")
        if (self.params.min_context_seconds ==
                self.params.clip_length_in_seconds):
            inputs = {
                k:
                    torch.stack([s[k] for s in batch]) if isinstance(
                        batch[0][k], Tensor) else [s[k] for s in batch]
                for k in batch[0].keys()
                # if k != "name"
            }
        else:
            inputs = {
                k:
                    torch.stack([s[k] for s in batch]) if
                    (isinstance(batch[0][k], Tensor) and
                     k != "context") else [s[k] for s in batch]
                for k in batch[0].keys()
                # if k != "name"
            }

        # inputs = {
        #     "target": torch.stack([s["target"] for s in batch]),
        #     "context": torch.stack([s["context"] for s in batch]),
        #     "description": [s["description"] for s in batch],
        #     "style": torch.stack([s["style"] for s in batch])
        # }
        return inputs

    def setup(self, stage: Optional[str]):
        self.train_dataset = self.dataset_class(
            Path(self.params.root_dir),
            self.params.stems,
            train=True,
            target_stem=self.params.target_stem,
            single_stem=self.params.single_stem,
            min_context_seconds=self.params.min_context_seconds,
            use_style_conditioning=self.params.use_style_conditioning,
            use_beat_conditioning=self.params.use_beat_conditioning,
            add_click=self.params.add_click,
            sync_chunks=self.params.sync_chunks,
            bpm_in_caption=self.params.bpm_in_caption,
            sample_rate=self.params.sample_rate,
            type_of_context=self.params.type_of_context,
            chunk_size_samples=self.params.clip_length_in_seconds *
            self.params.sample_rate,
            speed_transform_p=self.params.speed_transform_p,
            pitch_transform_p=self.params.pitch_transform_p,
            n_samples_per_epoch=self.params.n_samples_per_epoch,
            stereo=False,
            max_genres_in_description=3,
            max_moods_in_description=3,
        )

        self.val_dataset = self.dataset_class(
            Path(self.params.root_dir),
            self.params.stems,
            train=False,
            target_stem=self.params.target_stem,
            single_stem=self.params.single_stem,
            min_context_seconds=self.params.min_context_seconds,
            use_style_conditioning=self.params.use_style_conditioning,
            use_beat_conditioning=self.params.use_beat_conditioning,
            add_click=self.params.add_click,
            sync_chunks=self.params.sync_chunks,
            bpm_in_caption=self.params.bpm_in_caption,
            sample_rate=self.params.sample_rate,
            type_of_context=self.params.type_of_context,
            chunk_size_samples=self.params.clip_length_in_seconds *
            self.params.sample_rate,
            speed_transform_p=self.params.speed_transform_p,
            pitch_transform_p=self.params.pitch_transform_p,
            n_samples_per_epoch=None,
            stereo=False,
            max_genres_in_description=3,
            max_moods_in_description=3,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.params.batch_size_train,
            shuffle=True,
            num_workers=self.params.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            worker_init_fn=lambda id: random.seed(id),
            # prefetch_factor=1,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            self.params.batch_size_test,
            shuffle=False,
            num_workers=self.params.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            worker_init_fn=lambda id: random.seed(id),
            # prefetch_factor=1,
        )


# def main():
if __name__ == "__main__":

    from tqdm import tqdm
    from stage import config as cfg

    # root_dir = cfg.mixdata_path()
    root_dir = cfg.moises_path()
    stems = {
        Stem.DRUMS, Stem.GUITAR, Stem.BASS, Stem.PIANO, Stem.KEYBOARD,
        Stem.STRINGS, Stem.OTHER
    }
    dataset_params = hp.MixDatasetParams(
        root_dir=root_dir,
        stems=stems,
        single_stem=True,
        min_context_seconds=5,
        use_style_conditioning=True,
        use_beat_conditioning=True,
        target_stem=Stem.DRUMS,
        add_click=False,
        sync_chunks=False,
        bpm_in_caption=False,
        batch_size_train=4,
        batch_size_test=4,
        num_workers=8,
        clip_length_in_seconds=10,
        sample_rate=32_000,
        speed_transform_p=1,
        pitch_transform_p=0.5,
        n_samples_per_epoch=2000,
    )
    d = dataset_params.instantiate()

    vd = d.val_dataloader()
    vbatch = next(iter(vd))

    td = d.train_dataloader()
    tbatch = next(iter(td))
