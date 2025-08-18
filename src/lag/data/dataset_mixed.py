import torch
import torch.utils.data
from pathlib import Path

from lag.data.stemmed_dataset import StemmedDataset
from lag import config as cfg


class MixDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir: Path, *args, **kwargs):

        for dirname in [
                "moisesdb_v0.1", "lag-moisesdb", "lag_moisesdb", "moisesdb",
                "moises"
        ]:
            if (root_dir / dirname).exists():
                self.root_dir_moises: Path = root_dir / dirname
                break
        else:
            raise FileNotFoundError(
                f"Couldn't find subdirectory for moisesdb under {root_dir}")

        self.root_dir_mus: Path = root_dir / "musdb"

        self.moisesdb_dataset: StemmedDataset = StemmedDataset(
            self.root_dir_moises,
            *args,
            **kwargs,
        )
        self.musdb_dataset: StemmedDataset = StemmedDataset(
            self.root_dir_mus,
            *args,
            **kwargs,
        )

        # self.n_samples = len(self.moisesdb_dataset) + len(self.musdb_dataset)
        self.n_samples_moises = len(self.moisesdb_dataset)
        self.n_samples_mus = len(self.musdb_dataset)

    def __len__(self):
        return self.n_samples_moises + self.n_samples_mus

    def __getitem__(self, x):
        if x < self.n_samples_moises:
            return self.moisesdb_dataset[x]
        else:
            return self.musdb_dataset[x - self.n_samples_moises]


if __name__ == "__main__":
    from lag.data.stem import Stem

    moises_root = cfg.moises_path()
    mus_root = cfg.mus_path()

    stems = {
        Stem.DRUMS, Stem.GUITAR, Stem.BASS, Stem.PIANO, Stem.KEYBOARD,
        Stem.STRINGS
    }

    d = MixDataset(
        cfg.mixdata_path(),
        stems,
        target_stem=Stem.DRUMS,
        single_stem=True,
        add_click=False,
        bpm_in_caption=False,
        sync_chunks=False,
        train=True,
        sample_rate=32_000,
        chunk_size_samples=32_000 * 10,
        speed_transform_p=1,
        pitch_transform_p=1,
        stereo=False,
        n_samples_per_epoch=2000,
    )

    sample = next(iter(d))
