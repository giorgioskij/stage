from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
import torch
from torch import Tensor, norm
import matplotlib

from lag import config as cfg
from lag.utils.audio import normalize


def plot_waveforms(*waveforms: Tensor,
                   savepath: Optional[Path] = None,
                   **kwargs):
    if savepath is not None:
        b = matplotlib.get_backend()
        matplotlib.use("agg")

    t = torch.linspace(0, waveforms[0].shape[-1],
                       waveforms[0].shape[-1])  # 5 seconds of audio
    # audio_tensor = torch.sin(2 * np.pi * freq * t)  # Generate sinewave

    # Plot the waveform
    plt.style.use(cfg.ROOT / "tkol.mplstyle")
    colors = [c["color"] for c in list(plt.rcParams["axes.prop_cycle"])]

    plt.figure(**kwargs)
    for i, wave in enumerate(waveforms):
        wave = wave.squeeze()
        wave = normalize(wave, -1, 1)
        # If your audio is stereo (2 channels), you can average over channels, or just plot one
        if wave.ndimension() > 1:
            wave = wave.mean(dim=0)  # Take the mean over channels if stereo

        plt.plot(t, wave, color=colors[i])
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    for s in plt.gca().spines.values():
        s.set_visible((False))

    if savepath is not None:
        plt.savefig(savepath)
        matplotlib.use(b)
    else:
        plt.show()
