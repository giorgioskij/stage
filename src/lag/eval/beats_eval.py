import json
import os
import random
from itertools import cycle
from pathlib import Path

import librosa
import lightning as L
import mir_eval
import numpy as np
import pandas as pd
import torch
from beat_this.inference import File2Beats
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from lag import config as cfg
from lag.cocola import constants as cocola_constants
from lag.cocola.contrastive_model import CoCola
from lag.cocola.feature_extraction import CoColaFeatureExtractor
from lag.config import LOCAL_CKP_DIR
from lag.data.stem import Stem
from lag.data.stemmed_dataset import StemmedDataset
from lag.models.lightning_musicgen import LightningMusicgen
from lag.utils import audio


def evaluate_beats(ground_truth, predicted, tolerance=0.07):
    """
        Compare predicted beats with ground truth using a tolerance window.

        Args:
            predicted (np.array): Predicted beats (in seconds).
            ground_truth (np.array): Ground truth beats (in seconds).
            tolerance (float): Tolerance window in seconds.

        Returns:
        dict:
            "Precision",
            "Recall",
            "F-measure",
            "MAE": "Mean Absolute Error"
            "EMD",
            ...

    """

    if len(predicted) == 0 or len(ground_truth) == 0:
        return {
            "Precision (Predicted)": 0,
            "Recall (Ground Truth)": 0,
            "F1-Score": 0,
            "F1-score mir_eval": 0,
            "MAE": float('inf'),
            "EMD": float('inf'),
            "eheh": float('inf'),
            "eheh_bps": float('inf')
        }

    # --- Scenario 1: No Alignment ---
    # Count true positives (TP): Predicted beats that are within tolerance of a ground truth beat
    '''tp = sum(np.any(np.abs(predicted[:, None] - ground_truth) <= tolerance, axis=1))
    fp = len(predicted) - tp  # Beats in predicted that are unmatched (False Positives)
    fn = len(ground_truth) - tp  # Beats in ground truth that are unmatched (False Negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0'''

    tp_pred = sum(np.any(np.abs(predicted[:, None] - ground_truth) <= tolerance, axis=1))
    fp_pred = len(predicted) - tp_pred  # False Positives in predicted
    fn_gt = len(ground_truth) - tp_pred  # False Negatives in ground truth

    # True Positives (TP) for ground truth beats
    tp_gt = sum(np.any(np.abs(ground_truth[:, None] - predicted) <= tolerance, axis=1))
    fp_gt = len(ground_truth) - tp_gt  # False Positives in ground truth
    fn_pred = len(predicted) - tp_gt  # False Negatives in predicted

    # Precision, Recall, and F1-Score for both sides
    precision_pred = tp_pred / (tp_pred + fp_pred) if (tp_pred + fp_pred) > 0 else 0
    recall_gt = tp_gt / (tp_gt + fn_gt) if (tp_gt + fn_gt) > 0 else 0
    f1_pred = 2 * precision_pred * recall_gt / (precision_pred + recall_gt) if (precision_pred + recall_gt) > 0 else 0

    f1_mir_eval = mir_eval.beat.f_measure(ground_truth, predicted, tolerance)

    # Mean Absolute Error (MAE): Each predicted beat is matched to its closest ground-truth beat
    if len(predicted) > len(ground_truth):
        mae = np.mean(np.abs(np.subtract.outer(predicted, ground_truth).min(axis=1)))
    else:
        mae = np.mean(np.abs(np.subtract.outer(ground_truth, predicted).min(axis=1)))

    # Earth Mover's Distance (EMD) - only meaningful if sequences have similar distributions
    emd = wasserstein_distance(ground_truth, predicted)

    # --- Inter-Beat Interval (IBI) Comparison ---
    ibi_gt = np.diff(ground_truth)
    ibi_pred = np.diff(predicted)

    if len(ibi_pred) > 0 and len(ibi_gt) > 0:
        delta_gt = np.mean(ibi_gt)
        delta_pred = np.mean(ibi_pred)

        diff1 = np.abs(delta_gt - delta_pred)
        diff2 = np.abs(delta_gt - 2 * delta_pred)
        diff3 = np.abs(2 * delta_gt - delta_pred)
        eheh = min(diff1, diff2, diff3)

        bps_gt = np.mean(60 / ibi_gt)
        bps_pred = np.mean(60 / ibi_pred)
        diff1 = np.abs(bps_gt - bps_pred)
        diff2 = np.abs(bps_gt - 2 * bps_pred)
        diff3 = np.abs(2 * bps_gt - bps_pred)
        eheh_bps = min(diff1, diff2, diff3)
    else:
        eheh = float('inf')
        eheh_bps = float('inf')

    results = {
        "Precision (Predicted)": precision_pred,
        "Recall (Ground Truth)": recall_gt,
        "F1-Score": f1_pred,
        "F1-score mir_eval": f1_mir_eval,
        "MAE": mae,
        "EMD": emd,
        "eheh": eheh,
        "eheh_bps": eheh_bps
    }

    return results


def compute_cocola_score(gen_audio, context):
    cocola = CoCola(
        embedding_mode=cocola_constants.EmbeddingMode.BOTH)
    cocola.load_state_dict(
        torch.load(cfg.weights_dir() / "cocola-weights.pt",
                   weights_only=True))

    cocola.eval()
    cocola.set_embedding_mode(cocola_constants.EmbeddingMode.BOTH)
    feature_extractor = CoColaFeatureExtractor()
    gen_features = feature_extractor(gen_audio.cpu())

    assert len(gen_audio) == len(context)
    context_features = feature_extractor(context.cpu())
    context_score = cocola.score(gen_features, context_features)
    return context_score


def generate_beat_times(bpm, total_time=10.0, max_offset=0.3):
    """
    Generate a NumPy array of beat times (in seconds) up to 'total_time',
    based on the given BPM (beats per minute). A random offset (0 to max_offset)
    is added at the start.
    """
    # Calculate the interval in seconds between beats
    beat_interval = 60.0 / bpm

    # Random offset at the beginning
    offset = random.uniform(0, max_offset)

    # Generate all beat times until we reach or exceed total_time
    beat_times = []
    t = offset
    while t <= total_time:
        beat_times.append(t)
        t += beat_interval

    # Convert list to a NumPy array
    return np.array(beat_times)


def get_description(bpm, bpm_descriptions):
    """Selects a description based on the BPM value."""
    if bpm < 120:
        index_range = bpm_descriptions["100-119 BPM"]
    elif bpm < 140:
        index_range = bpm_descriptions["120-139 BPM"]
    elif bpm < 160:
        index_range = bpm_descriptions["140-159 BPM"]
    elif bpm < 180:
        index_range = bpm_descriptions["160-179 BPM"]
    else:
        index_range = bpm_descriptions["180-200 BPM"]

    return random.choice(index_range)


def generate_evaluation_dataset(model_ckp: str,
                                type_of_context: str,
                                eval_dataset: Path = None):
    assert type_of_context in ["stems", "beats"]  # , "stems and beats"]

    for inst in ["drum", "bass", "guitar"]:  # todo add more
        if inst in model_ckp:
            instrument = inst

    device = torch.device("cuda")
    model = LightningMusicgen.load_from_checkpoint(LOCAL_CKP_DIR / f"{model_ckp}.ckpt")
    model = model.to(device)
    model.eval()
    L.seed_everything(42)

    ds = eval_dataset.name if eval_dataset else "bpm"
    save_path = cfg.EVAL_DIR / model_ckp / type_of_context / ds
    os.makedirs(save_path, exist_ok=True)

    if type_of_context == "stems":
        stems = {
            Stem.DRUMS, Stem.GUITAR, Stem.BASS, Stem.PIANO, Stem.KEYBOARD,
            Stem.STRINGS
        }
        dataset = StemmedDataset(
            eval_dataset,
            stems,
            target_stem=Stem.DRUMS,
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

                exp_beats = sample["beat_seconds"]
                context = sample["context"]
                context = context.unsqueeze(0).to(
                    device) if context is not None else None

                style = None
                beat = sample["beat"]

                with torch.autocast(device_type="cuda"):
                    gen_audio = model.generate(n_samples=1,
                                               gen_seconds=10,
                                               prompt=None,
                                               context=context,
                                               style=style,
                                               beat=[beat],
                                               description=[description],
                                               prog_bar=False)

                gen_filename: Path = save_path / f"demo{i}_gen.wav"
                audio.save_audio(gen_audio[0], gen_filename)

                if description is not None:
                    desc_filename: Path = save_path / f"demo{i}_description.txt"
                    desc_filename.write_text(description)
                if context is not None:
                    context_filename: Path = save_path / f"demo{i}_context.wav"
                    mix_filename: Path = save_path / f"demo{i}_mix.wav"
                    audio.save_audio(context[0], context_filename)
                    mix = torch.nn.functional.pad(
                        context[0],
                        (0, gen_audio[0].shape[-1] - context[0].shape[-1]),
                        value=0) + gen_audio[0]
                    audio.save_audio(mix, mix_filename)
                if exp_beats is not None:
                    exp_beats_filename: Path = save_path / f"demo{i}_gt_beats.npz"
                    np.savez(exp_beats_filename, beats=exp_beats)
                i += 1

    else:  # type_of_context == "beats"
        if eval_dataset:
            stems = {
                Stem.DRUMS, Stem.GUITAR, Stem.BASS, Stem.PIANO, Stem.KEYBOARD,
                Stem.STRINGS
            }
            dataset = StemmedDataset(
                eval_dataset,
                stems,
                target_stem=Stem.DRUMS,
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

                    exp_beats = sample["beat_seconds"]
                    context = sample["context"]
                    context = context.unsqueeze(0).to(
                        device) if context is not None else None

                    style = None
                    beat = sample["beat"]

                    with torch.autocast(device_type="cuda"):
                        gen_audio = model.generate(n_samples=1,
                                                   gen_seconds=10,
                                                   prompt=None,
                                                   context=context,
                                                   style=style,
                                                   beat=[beat],
                                                   description=[description],
                                                   prog_bar=False)

                    gen_filename: Path = save_path / f"demo{i}_gen.wav"
                    audio.save_audio(gen_audio[0], gen_filename)

                    if description is not None:
                        desc_filename: Path = save_path / f"demo{i}_description.txt"
                        desc_filename.write_text(description)
                    if context is not None:
                        context_filename: Path = save_path / f"demo{i}_context.wav"
                        mix_filename: Path = save_path / f"demo{i}_mix.wav"
                        audio.save_audio(context[0], context_filename)
                        mix = torch.nn.functional.pad(
                            context[0],
                            (0, gen_audio[0].shape[-1] - context[0].shape[-1]),
                            value=0) + gen_audio[0]
                        audio.save_audio(mix, mix_filename)
                    if exp_beats is not None:
                        exp_beats_filename: Path = save_path / f"demo{i}_gt_beats.npz"
                        np.savez(exp_beats_filename, beats=exp_beats)
                    i += 1

        else:
            with open("/home/ballants/PycharmProjects/lag-modular/src/lag/eval/musicgen_bpm_descriptions.json",
                      "r") as json_file:
                bpm_descriptions = json.load(json_file)

            i = 0
            for t in range(2):
                for bpm in tqdm(range(100, 180),
                                total=len(range(100, 180)),
                                desc="Generating songs"):
                    description = get_description(bpm, bpm_descriptions)
                    # add instrument
                    # description = description + f". Instrument: {instrument}"
                    exp_beats = generate_beat_times(bpm=bpm)
                    exp_downbeats = exp_beats[::4]
                    clicks = librosa.clicks(times=exp_beats,
                                            sr=32_000,
                                            click_freq=1000,
                                            length=32_000 * 10)
                    downbeat_clicks = librosa.clicks(times=exp_downbeats,
                                                     sr=32_000,
                                                     click_freq=1000,
                                                     length=32_000 * 10)
                    audio_clicks = clicks + downbeat_clicks
                    audio_clicks = np.clip(audio_clicks, -1.0, 1.0)
                    if audio_clicks.ndim > 1:
                        audio_clicks = audio_clicks.mean(axis=0)
                    context = torch.tensor(audio_clicks)
                    context = context.unsqueeze(0).unsqueeze(0).to(device)

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

                    gen_filename: Path = save_path / f"demo{i}_gen.wav"
                    audio.save_audio(gen_audio[0], gen_filename)

                    if description is not None:
                        desc_filename: Path = save_path / f"demo{i}_description.txt"
                        desc_filename.write_text(description)
                    if context is not None:
                        context_filename: Path = save_path / f"demo{i}_context.wav"
                        mix_filename: Path = save_path / f"demo{i}_mix.wav"
                        audio.save_audio(context[0], context_filename)
                        mix = torch.nn.functional.pad(
                            context[0],
                            (0, gen_audio[0].shape[-1] - context[0].shape[-1]),
                            value=0) + gen_audio[0]
                        audio.save_audio(mix, mix_filename)
                    if exp_beats is not None:
                        exp_beats_filename: Path = save_path / f"demo{i}_gt_beats.npz"
                        np.savez(exp_beats_filename, beats=exp_beats)
                    i += 1


def evaluate_dataset(evalds_dir: str,
                     type_of_context: str,
                     eval_dataset: Path = None):
    assert type_of_context in ["stems", "beats"]

    eval_path = cfg.EVAL_DIR / evalds_dir / type_of_context
    if eval_dataset is not None:
        eval_path = eval_path / eval_dataset.name
    else:
        eval_path = eval_path / "bpm"

    if type_of_context == "beats":
        total_demos = len(list(eval_path.iterdir()))
        assert total_demos % 5 == 0
        total_demos = int(total_demos / 5)

        beatthis_model = File2Beats(checkpoint_path="final0",
                                    device="cuda",
                                    dbn=False)

        results = []
        for i in range(total_demos):
            audio_path = eval_path / f"demo{i}_gen.wav"
            real_beats, real_downbeats = beatthis_model(audio_path)
            gt_beats_path = eval_path / f"demo{i}_gt_beats.npz"
            loaded_beats = np.load(gt_beats_path)
            gt_beats = loaded_beats["beats"]
            res = evaluate_beats(gt_beats, real_beats, tolerance=0.07)
            results.append(res)

        df_results = pd.DataFrame(results)
        aggregated_results = {
            "Mean": df_results.mean().to_dict(),
            "Std Dev": df_results.std().to_dict()
        }
        with open(eval_path / 'final_results.json', 'w') as f:
            json.dump(aggregated_results, f, indent=4)
    else:
        total_demos = len(list(eval_path.iterdir()))
        assert total_demos % 5 == 0
        total_demos = int(total_demos / 5)

        pass


if __name__ == '__main__':
    # generate_evaluation_dataset(model_ckp="leblanc_drums/epoch=39-step=50000", type_of_context="beats", eval_dataset=None)
    # evaluate_dataset(evalds_dir="leblanc_drums/epoch=39-step=50000", type_of_context="beats", eval_dataset=None)

    generate_evaluation_dataset(model_ckp="leblanc_drums/epoch=39-step=50000", type_of_context="beats",
                                eval_dataset=cfg.mus_path())
    evaluate_dataset(evalds_dir="leblanc_drums/epoch=39-step=50000", type_of_context="beats",
                     eval_dataset=cfg.mus_path())

    '''generate_evaluation_dataset(model_ckp="leblanc_drums/epoch=39-step=50000", type_of_context="stems",
                                eval_dataset=cfg.mus_path(), n_samples=100)

    generate_evaluation_dataset(model_ckp="leblanc_drums/epoch=39-step=50000", type_of_context="stems",
                                eval_dataset=cfg.moises_path(), n_samples=100)'''
