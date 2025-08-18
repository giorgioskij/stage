from collections import Counter
from lag import config, hyperparameters as hp
import torch
from torch import Tensor
from lightning.pytorch.utilities.seed import isolate_rng
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np

# plt.style.use("dark_background")
plt.style.use(config.ROOT / "tkol.mplstyle")
plt.tight_layout()

from lag.models.encodec import EncodecModel
from lag.utils import audio as au

encodec_params = hp.pretrained_encodec_meta_32khz_params
encodec: EncodecModel = EncodecModel.from_params(encodec_params).eval().cpu()

diff_indices = []
nonclose_indices = []
audio_lengths = []

audio_paths = [
    p for p in config.AUDIO_DIR.iterdir()
    if p.is_file and p.name.endswith(".wav")
]

# for audio_path in tqdm(audio_paths):
N_TRIES = 2
for idx in range(N_TRIES):
    audio: Tensor = torch.rand(1, 1,
                               torch.randint(100_000, 500_000, (1,)).item())
    audio_lengths.append(audio.shape[-1])
    # audio = au.load_audio(audio_path).reshape(1, 1, -1)

    audio_codes = encodec.encode(audio)

    codes_cut = audio_codes[..., :-100]

    with torch.inference_mode():
        with isolate_rng():
            decoded_full = encodec.decode(audio_codes)

        decoded_cut = encodec.decode(codes_cut)

    notclose = torch.abs(decoded_full[..., :decoded_cut.shape[-1]] -
                         decoded_cut) > 1e-4
    first_nonclose_index = notclose.long().argmax(dim=-1).item()
    nonclose_indices.append(decoded_cut.shape[-1] - first_nonclose_index)

    error = torch.abs(
        decoded_cut -
        decoded_full[..., :decoded_cut.shape[-1]]).detach().numpy().flatten()

    plt.subplots(1, 2, figsize=(20, 8))
    plt.suptitle(f"Random audio of {audio.shape[-1]} samples")
    plt.subplot(1, 2, 1)
    plt.gca().tick_params(axis="both", labelsize=16)
    plt.plot(torch.arange(len(error)).numpy(), error)
    plt.axvline(x=first_nonclose_index,
                color=list(plt.rcParams["axes.prop_cycle"])[1]["color"],
                linestyle='--',
                alpha=0.8)

    plt.subplot(1, 2, 2)
    plt.gca().tick_params(axis="both", labelsize=16)
    plt.plot(notclose.cumsum(dim=-1).squeeze())
    plt.show()

pearson_corr, _ = pearsonr(audio_lengths, nonclose_indices)
spearman_corr, _ = spearmanr(audio_lengths, nonclose_indices)
print(f'{pearson_corr=}')
print(f'{spearman_corr=}')

# Create scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(audio_lengths, nonclose_indices, color="blue", label="Data points")

# Annotate the plot
plt.title("Correlation Between Audio Length and Decoding Loss", fontsize=14)
plt.xlabel("Audio Length (seconds)", fontsize=12)
plt.ylabel("Different samples", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# Show the plot
plt.show()
