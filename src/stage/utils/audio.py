from functools import partial
import random
import time
import numba
from torch import Tensor
from pathlib import Path
import torch
import torchaudio
from typing import Dict, List, Optional, Sequence, Tuple, Union
import pylibrb
import concurrent.futures
import numpy as np

from stage.data.stem import Stem


def pad_stack(tensors: List[Tensor],
              pad_value: int | float,
              padding_dim: int = -1,
              pad_start: bool = False,
              stack_dim: int = 0):
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) > 1:
        raise ValueError("Input tensors have different types")
    ndims = {t.ndim for t in tensors}
    if len(ndims) > 1:
        raise ValueError("Input tensors have different number of dimensions")
    if len(tensors) == 0:
        raise ValueError("Input list cannot be empty")
    ndim = len(tensors[0].shape)
    shapes = [t.shape for t in tensors]
    for i in range(1, len(shapes)):
        shape_i = list(tensors[i].shape)
        shape_ii = list(tensors[i - 1].shape)
        shape_i.pop(padding_dim)
        shape_ii.pop(padding_dim)
        if shape_i != shape_ii:
            raise ValueError(
                f"Shape of tensors at indices ({i-1}, {i}) don't match")

    max_len = max(t.shape[padding_dim] for t in tensors)
    if padding_dim < 0:
        padding_dim = ndim + padding_dim

    def get_pad_code(n):
        pre = [0] * (2 * (ndim - padding_dim - 1))
        mid = [n, 0] if pad_start else [0, n]
        post = [0] * (2 * (padding_dim))
        pad_code = pre + mid + post
        return pad_code

    padded_tensors = [
        torch.nn.functional.pad(t,
                                get_pad_code(max_len - t.shape[padding_dim]),
                                value=pad_value) for t in tensors
    ]
    return torch.stack(padded_tensors, dim=stack_dim)


def to_stereo(audio: Tensor) -> Tensor:
    if audio.dim() == 1:
        return audio.repeat(2, 1)
    if audio.dim() >= 2 and audio.shape[-2] == 2:
        return audio
    return torch.cat((audio, audio), dim=-2)


def to_mono(audio: Tensor) -> Tensor:
    if audio.ndim == 1:
        return audio
    if audio.shape[-2] == 1:
        return audio
    return audio.mean(dim=-2, keepdim=True)


def save_audio(audio: Tensor, path: Path, sample_rate: int = 32_000):
    audio = audio.float()
    if audio.shape[-1] == 1:
        audio.squeeze(-1)
    l = audio.shape[-1]
    audio = audio.reshape(-1, l)
    # audio = mono_to_stereo(audio.detach().cpu())
    audio = to_stereo(audio.detach().cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() != 2:
        print(f'{audio.shape=}')

    torchaudio.save(str(path), audio, sample_rate=sample_rate)  # type: ignore


def load_audio(path: Path,
               sample_rate: int = 32_000,
               stereo: bool = False) -> Tensor:
    audio, orig_sr = torchaudio.load(str(path))  # type: ignore
    audio = torchaudio.functional.resample(audio, orig_sr, sample_rate)
    if not stereo:
        audio = to_mono(audio)
    return audio.reshape(1, 1, -1)  # type: ignore


def load_audio_chunk(audio_path: Path, start_offset: int, num_frames: int,
                     stereo: bool) -> Tensor:

    # info = torchaudio.info(str(audio_path))
    # length = info.num_frames
    # file_sample_rate = info.sample_rate

    # assert file_sample_rate == sample_rate
    try:
        wav, sr = torchaudio.load(str(audio_path),
                                  frame_offset=start_offset,
                                  num_frames=num_frames,
                                  backend="soundfile")
    except:
        wav = torch.zeros(2, num_frames)

    # if start_offset + num_frames >= length:
    #     wav = torch.zeros(2, num_frames)

    # wav = torchaudio.functional.resample(wav, sr, sample_rate)

    if wav.shape[-1] < num_frames:
        wav = torch.nn.functional.pad(wav,
                                      pad=(0, num_frames - wav.shape[-1]),
                                      mode="constant",
                                      value=0)

    if not stereo:
        # wav: Tensor = stereo_to_mono(wav).reshape(1, -1)
        wav: Tensor = to_mono(wav).reshape(1, -1)

    return wav


def is_silent(audio: Tensor, threshold: float = 1e-2):
    return audio.max().item() < threshold


def create_click(shape: Sequence[int],
                 sr: int,
                 beats: Sequence[int],
                 click_freq: int = 440,
                 click_length: int = 200) -> Tensor:
    click_track: Tensor = torch.zeros(shape)
    sinewave: Tensor = create_sine_wave(click_freq, sr, click_length)
    for beat in beats:
        if beat >= click_track.shape[-1]:
            break
        for offset in range(200):
            idx = beat + offset
            if idx < click_track.shape[-1]:
                click_track[..., idx] = sinewave[offset]
    return click_track


def create_sine_wave(freq: float, sr: int, length: int) -> Tensor:
    cycle_len = int(sr // freq)
    cycle = torch.linspace(start=0, end=2 * torch.pi, steps=cycle_len)
    cycle = cycle.repeat(length // cycle_len + 1)
    cycle = cycle[:length]
    wave = cycle.sin()
    return wave


def stretch(audio: Tensor, sample_rate: int, speed_factor: float,
            pitch_factor: int) -> Tensor:
    stretcher = pylibrb.RubberBandStretcher(
        sample_rate=sample_rate,
        channels=1,
        options=pylibrb.Option.PROCESS_OFFLINE | pylibrb.Option.ENGINE_FASTER,
        initial_time_ratio=speed_factor,
        initial_pitch_scale=pow(2, pitch_factor / 12))
    stretcher.set_max_process_size(audio.shape[-1])
    audio_in = audio.reshape(1, -1).numpy()
    stretcher.study(audio_in, final=True)
    stretcher.process(audio_in, final=True)
    audio_out = torch.from_numpy(stretcher.retrieve_available()).reshape(
        1, -1).float()
    return audio_out


def stretch_with_timeout(audio: Tensor, sample_rate: int, speed_factor: float,
                         pitch_factor: int, timeout_seconds: float):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(stretch,
                                 audio,
                                 sample_rate=sample_rate,
                                 speed_factor=speed_factor,
                                 pitch_factor=pitch_factor)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            print("Timeout occurred in stretching audio.")
            return audio
        except Exception as e:
            print(f"Exception occurred in stretching audio: {e}.")
            return audio


def make_variable_frequency_sinewave(t_end: int,
                                     peak_indices: Tensor) -> Tensor:
    if not isinstance(peak_indices, Tensor):
        peak_indices = torch.tensor(peak_indices, dtype=torch.int32)

    device = peak_indices.device

    if peak_indices.shape[-1] < 2:
        return torch.zeros((t_end,), device=device)

    # Define the fine-grained time array
    # time = torch.linspace(t_start, t_end, 100_000)
    time = torch.arange(t_end, device=device)

    # Calculate frequencies for each interval
    intervals = torch.diff(peak_indices)  # Time intervals between beats
    frequencies = 1 / intervals  # Frequencies for each interval

    # Find segment indices for each time point
    segment_indices = torch.searchsorted(peak_indices, time, right=True) - 1
    segment_indices = torch.clamp(segment_indices, 0, len(frequencies) - 1)

    # Compute sinewave for all time points
    phase_shift = torch.pi / 2
    relative_time = time - peak_indices[segment_indices]
    wave = torch.sin(2 * torch.pi * frequencies[segment_indices] *
                     relative_time + phase_shift)

    # Extend before the first peak
    before_mask = time < peak_indices[0]
    freq_before = 1 / (peak_indices[1] - peak_indices[0])
    wave[before_mask] = torch.sin(2 * torch.pi * freq_before *
                                  (time[before_mask] - peak_indices[0]) +
                                  phase_shift)

    # Extend after the last peak
    after_mask = time >= peak_indices[-1]
    freq_after = 1 / (peak_indices[-1] - peak_indices[-2])
    wave[after_mask] = torch.sin(2 * torch.pi * freq_after *
                                 (time[after_mask] - peak_indices[-1]) +
                                 phase_shift)

    return wave


def normalize(audio: Tensor, new_min, new_max):
    if len(audio.shape) == 2 and audio.shape[0] == 2:
        audio = audio.mean(dim=-1)
    audio = to_mono(audio)
    # Calculate the min and max of the original array
    old_min = audio.min()
    old_max = audio.max()

    # Apply the normalization formula
    normalized_arr = (audio - old_min) / (old_max - old_min) * (
        new_max - new_min) + new_min
    return normalized_arr


def play(waveform: torch.Tensor, sr: int):
    import IPython.display
    import sounddevice as sd
    waveform = to_stereo(waveform)
    waveform_np = waveform.cpu().float().detach().numpy()
    if is_interactive():
        IPython.display.display(IPython.display.Audio(waveform_np, rate=sr))
    else:
        sd.play(waveform_np.T, sr)
        sd.wait()


def is_interactive():
    import sys
    return "ipykernel" in sys.modules


def inject_clicks(audio_tensor: Tensor, beat_positions: Tensor,
                  sample_rate: int):
    """
    Add short clicks at `beat_positions` (sample indices) in `audio_tensor`.
    If audio is multi-channel, we'll apply the same clicks to each channel.
    """
    import librosa
    beat_positions_seconds = beat_positions / sample_rate
    click_track: Tensor = torch.tensor(
        librosa.clicks(times=beat_positions_seconds.cpu().detach().numpy(),
                       hop_length=1,
                       length=audio_tensor.shape[-1],
                       sr=sample_rate))

    click_track = click_track.broadcast_to(audio_tensor.shape)
    return click_track + audio_tensor.detach().cpu()
