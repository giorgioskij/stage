from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from beat_this.inference import File2Beats, Audio2Beats
from lag import config as cfg
from lag.conditioning.beat_embedder import Beat


def extract_beats(root_dir: Path):
    from tqdm import tqdm

    subdirs: List[Path] = sorted([p for p in root_dir.iterdir() if p.is_dir()],
                                 key=lambda x: x.name)

    beatthis_model = File2Beats(checkpoint_path="final0",
                                device="cuda",
                                dbn=False)

    logfile = cfg.ROOT / "beattrack.log"
    openfile = logfile.open(mode="a")
    openfile.write(f"Starting preprocessing. "
                   f"It's {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    errors_path: List[Path] = []
    errors_beats: List[Path] = []
    for song in tqdm(subdirs):

        mix_path = song / "mixed.wav"
        if not mix_path.exists():
            print(f"song {mix_path} doesn't exist")
            errors_path.append(song)
            openfile.write(f"song {song} doesn't have a mix\n")

        output_file = song / "beatthis.npz"
        try:
            beats, downbeats = beatthis_model(mix_path)
            np.savez(output_file, beats=beats, downbeats=downbeats)
        except:
            print(f"Error tracking beats of song {song.name}")
            errors_beats.append(song)
            openfile.write(f"error tracking beats of song {song}\n")

    openfile.write("\n\n---- Job Complete! ----\n")
    openfile.write(f"It's {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    openfile.write(f"{len(errors_path)} errors in loading songs:\n")
    for song in errors_path:
        openfile.write(f"{song}\n")

    openfile.write(f"\n\n{len(errors_beats)} errors in tracking beats:\n")
    for song in errors_beats:
        openfile.write(f"{song}\n")

    openfile.close()


def beat_from_wav(audio: Tensor, sr: int) -> Beat:

    beatthis_model = Audio2Beats("final0", device="cuda")

    beats, downbeats = beatthis_model(audio.cpu().squeeze().numpy(), sr)

    return Beat(torch.Tensor(beats), torch.Tensor(downbeats), audio.shape[-1])


if __name__ == "__main__":
    # extract_beats(cfg.moises_path())
    extract_beats(cfg.mus_path())
