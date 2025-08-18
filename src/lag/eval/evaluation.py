import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from lag import config as cfg
from lag.data.stem import Stem
from lag.data.stemmed_dataset import StemmedDataset
from lag.models.lightning_musicgen import LightningMusicgen
from lag.utils import audio


def generate_evaluation_dataset(dataset_path: Path = None, type_of_context: str = "beats", stem: str = "drums"):
    assert type_of_context in ["stems", "beats", "stems and beats"]
    assert stem in ["drums", "bass"]

    match stem:  # todo add more
        case "drums":
            stem_to_evaluate: Stem = Stem.DRUMS
        case "bass":
            stem_to_evaluate: Stem = Stem.BASS

    L.seed_everything(42)

    ds_name = dataset_path.name if dataset_path else "bpm"

    if type_of_context == "stems":

        stems = {
            Stem.DRUMS, Stem.GUITAR, Stem.BASS, Stem.PIANO, Stem.KEYBOARD,
            Stem.STRINGS
        }
        dataset = StemmedDataset(
            dataset_path,
            stems,
            target_stem=stem_to_evaluate,
            single_stem=True,
            min_context_seconds=10,
            use_style_conditioning=False,
            use_beat_conditioning=True,
            add_click=False,
            bpm_in_caption=False,
            sync_chunks=False,
            train=False,
            sample_rate=32_000,
            chunk_size_samples=32_000 * 10,
            type_of_context=type_of_context,
            speed_transform_p=0,
            pitch_transform_p=0,
            stereo=False,
            n_samples_per_epoch=None,
        )

        i = 0
        n_epochs: int = 4
        for epoch_idx in range(n_epochs):
            for sample in tqdm(dataset, total=len(dataset), desc="Retrieving samples..."):

                description = sample["description"]
                target = sample["target"]
                exp_beats = sample["beat_seconds"]
                context = sample["context"]

                sample_name = f"{i:03d}"
                save_path = cfg.EVAL_DIR / type_of_context / ds_name / stem / sample_name
                os.makedirs(save_path, exist_ok=True)

                if target is not None:
                    target_filename: Path = save_path / "target.wav"
                    audio.save_audio(target, target_filename)
                if description is not None:
                    desc_filename: Path = save_path / f"description.txt"
                    desc_filename.write_text(description)
                if context is not None:
                    context_filename: Path = save_path / f"context.wav"
                    audio.save_audio(context, context_filename)
                if exp_beats is not None:
                    exp_beats_filename: Path = save_path / f"gt_beats.npz"
                    np.savez(exp_beats_filename, beats=exp_beats)
                i += 1


def generate_with_stage(model_ckp: str, type_of_context: str, stem: str, eval_dataset: str):
    assert type_of_context in ["stems", "beats", "stems and beats"]

    input_path: Path = cfg.EVAL_DIR / type_of_context / eval_dataset / stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningMusicgen.load_from_checkpoint(cfg.LOCAL_CKP_DIR / f"{model_ckp}.ckpt")
    model = model.to(device)
    model.eval()

    sample_list = os.listdir(input_path)
    num_samples = len(sample_list)

    for sample in tqdm(sample_list, total=num_samples, desc="Generating songs"):
        generation_path: Path = cfg.EVAL_DIR / model_ckp / type_of_context / eval_dataset / stem / sample
        os.makedirs(generation_path, exist_ok=True)

        sample_path: Path = input_path / sample
        context_path: Path = sample_path / "context.wav"
        description_path: Path = sample_path / "description.txt"

        context, sample_rate = torchaudio.load(context_path)
        context = audio.to_mono(context)
        context = context.unsqueeze(0).to(device)

        with open(description_path, "r", encoding="utf-8") as file:
            description = file.read()

        style = None
        beat = None

        with torch.autocast(device_type="cuda"):
            gen_audio = model.generate(n_samples=1,
                                       gen_seconds=10,
                                       prompt=None,
                                       context=context,
                                       style=style,
                                       beat=[beat],
                                       description=[description],
                                       prog_bar=False)

        gen_filename: Path = generation_path / f"gen.wav"
        audio.save_audio(gen_audio[0], gen_filename)


if __name__ == '__main__':
    '''generate_evaluation_dataset(cfg.moises_path(), type_of_context="stems", stem="drums")
    generate_evaluation_dataset(cfg.moises_path(), type_of_context="stems", stem="bass")
    generate_evaluation_dataset(cfg.mus_path(), type_of_context="stems", stem="drums")
    generate_evaluation_dataset(cfg.mus_path(), type_of_context="stems", stem="bass")'''

    '''generate_with_stage(model_ckp="leblanc_drums/epoch=39-step=50000", type_of_context="stems", stem="drums",
                        eval_dataset="moisesdb")'''
    generate_with_stage(model_ckp="leblanc_drums/epoch=39-step=50000", type_of_context="stems", stem="drums",
                        eval_dataset="musdb")
    generate_with_stage(model_ckp="galio_drums", type_of_context="stems", stem="drums",
                        eval_dataset="moisesdb")
    generate_with_stage(model_ckp="galio_drums", type_of_context="stems", stem="drums",
                        eval_dataset="musdb")
