from typing import List, Sequence
from torch import Tensor
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass

from stage.utils.audio import load_audio, make_variable_frequency_sinewave
from stage.utils.plotting import plot_waveforms
from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.conditioning.embedder import Embedder, LinearProjectionEmbedder
from stage import config as cfg


@dataclass
class Beat:
    """
        beats / downbeats: 
            Tensor of Int or Long type, containing indices of samples where
            beats / downbeats occur
        seq_len: int representing the total length of the audio in samples
    """
    beats: Tensor
    downbeats: Tensor
    seq_len: int


class DirectSinusoidalBeatEmbedder(Embedder):

    def __init__(self, embedding_dim: int):
        super().__init__(input_dim=2, embedding_dim=embedding_dim)
        self.encodec_framerate: int = 50
        self.sample_rate: int = 32_000

    def forward(self,
                x: Sequence[Beat],
                duplicate_for_cfg: bool = False) -> EmbeddedCondition:
        """
            Returns: A tensor of shape [B, S, H], where B is batch, H is 
            embedding dim, and S is the length of the sequence
        """

        assert len(set([t.seq_len for t in x])) == 1

        emb_list: List[Tensor] = [None] * len(x)  # type: ignore

        ratio = self.encodec_framerate / self.sample_rate
        for i, track in enumerate(x):

            # convert indices from explicit to latent space
            beats = torch.floor(track.beats.float() * ratio).long()
            downbeats = torch.floor(track.downbeats.float() * ratio).long()
            seq_len = round(track.seq_len * ratio)

            # make a variable-frequency sinewave lined with the beats
            beat_emb = make_variable_frequency_sinewave(seq_len, beats)
            downbeat_emb = make_variable_frequency_sinewave(seq_len, downbeats)

            emb_list[i] = torch.stack((beat_emb, downbeat_emb), dim=-1)

        embeds: Tensor = torch.stack(emb_list)

        # create an embedding that's just the beat sinewave in every dimension
        embeds = torch.cat(
            (embeds[..., :1].repeat(1, 1, self.embedding_dim // 2),
             embeds[..., 1:].repeat(1, 1, self.embedding_dim // 2)),
            dim=-1)
        mask = torch.ones(embeds.shape[:-1],
                          device=embeds.device,
                          dtype=torch.bool)

        # concatenate an empty condition
        if duplicate_for_cfg:
            embeds = torch.cat((embeds, torch.zeros_like(embeds)), dim=0)
            mask = torch.cat((mask, torch.zeros_like(mask)), dim=0)

        return EmbeddedCondition(embeds, mask)

    def null_condition(self, batch_size: int) -> EmbeddedCondition:
        return EmbeddedCondition(
            torch.zeros(batch_size,
                        1,
                        self.embedding_dim,
                        dtype=torch.float32,
                        device=self.output_proj.weight.device),
            torch.zeros(batch_size,
                        1,
                        dtype=torch.bool,
                        device=self.output_proj.weight.device))


class SinusoidalBeatEmbedder(LinearProjectionEmbedder):

    def __init__(self, embedding_dim: int):
        super().__init__(input_dim=2, embedding_dim=embedding_dim)
        self.encodec_framerate: int = 50
        self.sample_rate: int = 32_000

    def forward(self,
                x: Sequence[Beat],
                duplicate_for_cfg: bool = False) -> EmbeddedCondition:
        """
            Returns: A tensor of shape [B, S, H], where B is batch, H is 
            embedding dim, and S is the length of the sequence
        """

        assert len(set([t.seq_len for t in x])) == 1

        emb_list: List[Tensor] = [None] * len(x)  # type: ignore

        for i, track in enumerate(x):

            beats = (track.beats / self.sample_rate *
                     self.encodec_framerate).round().int()
            downbeats = (track.downbeats / self.sample_rate *
                         self.encodec_framerate).round().int()

            seq_len = round(track.seq_len / self.sample_rate *
                            self.encodec_framerate)

            beat_emb = make_variable_frequency_sinewave(seq_len, beats)
            downbeat_emb = make_variable_frequency_sinewave(seq_len, downbeats)

            emb_list[i] = torch.stack((beat_emb, downbeat_emb), dim=-1)

        embeds: Tensor = torch.stack(emb_list)

        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        mask = torch.ones(embeds.shape[:-1],
                          device=embeds.device,
                          dtype=torch.bool)

        if duplicate_for_cfg:
            embeds = torch.cat((embeds, torch.zeros_like(embeds)), dim=0)
            mask = torch.cat((mask, torch.zeros_like(mask)), dim=0)

        return EmbeddedCondition(embeds, mask)

    def null_condition(self, batch_size: int) -> EmbeddedCondition:
        return EmbeddedCondition(
            torch.zeros(batch_size,
                        1,
                        self.embedding_dim,
                        dtype=torch.float32,
                        device=self.output_proj.weight.device),
            torch.zeros(batch_size,
                        1,
                        dtype=torch.bool,
                        device=self.output_proj.weight.device))


class SinusoidalBeatEmbedderMLP(Embedder):

    def __init__(self, embedding_dim: int):
        super().__init__(input_dim=2, embedding_dim=embedding_dim)
        # self.embedding_dim: int = embedding_dim
        self.encodec_framerate: int = 50
        self.sample_rate: int = 32_000

        self.output_proj: nn.Sequential = nn.Sequential(
            nn.Linear(self.input_dim, 512), nn.ReLU(),
            nn.Linear(512, self.embedding_dim))

    def forward(self,
                x: Sequence[Beat],
                duplicate_for_cfg: bool = False) -> EmbeddedCondition:
        """
            Returns: A tensor of shape [B, S, H], where B is batch, H is 
            embedding dim, and S is the length of the sequence
        """

        assert len(set([t.seq_len for t in x])) == 1

        emb_list: List[Tensor] = [None] * len(x)  # type: ignore

        for i, track in enumerate(x):

            beats = (track.beats / self.sample_rate *
                     self.encodec_framerate).round().int()
            downbeats = (track.downbeats / self.sample_rate *
                         self.encodec_framerate).round().int()

            seq_len = round(track.seq_len / self.sample_rate *
                            self.encodec_framerate)

            beat_emb = make_variable_frequency_sinewave(seq_len, beats)
            downbeat_emb = make_variable_frequency_sinewave(seq_len, downbeats)

            emb_list[i] = torch.stack((beat_emb, downbeat_emb), dim=-1)

        embeds: Tensor = torch.stack(emb_list)

        embeds = self.output_proj(embeds.to(self.output_proj[-1].weight))
        mask = torch.ones(embeds.shape[:-1],
                          device=embeds.device,
                          dtype=torch.bool)

        if duplicate_for_cfg:
            embeds = torch.cat((embeds, torch.zeros_like(embeds)), dim=0)
            mask = torch.cat((mask, torch.zeros_like(mask)), dim=0)

        return EmbeddedCondition(embeds, mask)

    def null_condition(self, batch_size: int) -> EmbeddedCondition:
        return EmbeddedCondition(
            torch.zeros(batch_size,
                        1,
                        self.embedding_dim,
                        dtype=torch.float32,
                        device=self.output_proj[-1].weight.device),
            torch.zeros(batch_size,
                        1,
                        dtype=torch.bool,
                        device=self.output_proj[-1].weight.device))


if __name__ == "__main__":
    song_path = cfg.moises_path(
    ) / "0d528a19-cb0f-4421-b250-444f9343e51c/mixed.wav"
    song = load_audio(song_path)
    n_samples = song.shape[-1]
    loaded = np.load(
        "/home/tkol/dev/datasets/moisesdb/moisesdb_v0.1/0d528a19-cb0f-4421-b250-444f9343e51c/beatthis.npz"
    )
    beats = torch.tensor([int(b * 32_000) for b in loaded["beats"]],
                         dtype=torch.int32)
    downbeats = torch.tensor([int(b * 32_000) for b in loaded["downbeats"]],
                             dtype=torch.int32)
    assert n_samples > beats.max()

    b = Beat(beats=beats, downbeats=downbeats, seq_len=n_samples)
    embedder = SinusoidalBeatEmbedder(1024)
    emb = embedder([b])

    directembedder = DirectSinusoidalBeatEmbedder(1024)
    directemb = directembedder([b])
    # from matplotlib import pyplot as plt
    # plt.style.use(cfg.ROOT / "tkol.mplstyle")
    # plt.figure(figsize=(100, 10))
    # plt.scatter(x=torch.arange(emb.shape[-1]) / 50 * 32_000, y=emb)
    # plt.show()

    # plot_waveforms(
    #     song,
    #     emb,
    #     # savepath=cfg.ROOT / "plots/beatontrack.png",
    #     figsize=(50, 3),
    #     dpi=100,
    # )
