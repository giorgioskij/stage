"""
    Embedders that encode audio data using the CLAP model.
"""

import laion_clap
from torch import Tensor
import torch
from torch import nn

from lag.conditioning.embedded_condition import EmbeddedCondition
from lag.conditioning.embedder import Embedder, LinearProjectionEmbedder
from lag import config as cfg


class LinearClapEmbedder(LinearProjectionEmbedder):

    def __init__(self, embedding_dim):
        super().__init__(input_dim=512, embedding_dim=embedding_dim)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False,
                                           amodel='HTSAT-base')
        self.clap.load_state_dict(
            torch.load(cfg.weights_dir() / "clap-weights.pt",
                       weights_only=True))
        for p in self.clap.parameters():
            p.requires_grad = False

    def null_condition(self, batch_size: int):
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

    def forward(self, x: Tensor, duplicate_for_cfg: bool = False):
        self.clap.eval()
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                embeds = self.clap.get_audio_embedding_from_data(
                    x.reshape(-1, x.shape[-1]),
                    use_tensor=True,
                )

        embeds = embeds.unsqueeze(1)
        embeds = self.output_proj(embeds)

        mask = torch.ones(embeds.shape[:-1],
                          dtype=torch.bool,
                          device=embeds.device)

        if duplicate_for_cfg:
            embeds = torch.cat((embeds, torch.zeros_like(embeds)), dim=0)
            mask = torch.cat((mask, torch.zeros_like(mask)), dim=0)

        return EmbeddedCondition(embeds, mask)


class SequenceClapEmbedder(Embedder):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.clap = laion_clap.CLAP_Module(enable_fusion=False,
                                           amodel='HTSAT-base')
        self.clap.load_state_dict(
            torch.load(cfg.weights_dir() / "clap-weights.pt",
                       weights_only=True))
        for p in self.clap.parameters():
            p.requires_grad = False
        self.prefix_length = 10

        # create MLP for conversion
        intermediate_size = self.embedding_dim * self.prefix_length // 2
        output_size = self.embedding_dim * self.prefix_length
        self.mlp = nn.Sequential(
            nn.Linear(512, intermediate_size, bias=True),
            nn.GELU(),
            nn.Linear(intermediate_size, output_size, bias=True),
        )

    def null_condition(self, batch_size: int):
        return EmbeddedCondition(
            torch.zeros(batch_size,
                        self.prefix_length,
                        self.embedding_dim,
                        dtype=torch.float32,
                        device=self.mlp[0].weight.device),
            torch.zeros(batch_size,
                        self.prefix_length,
                        dtype=torch.bool,
                        device=self.mlp[0].weight.device))

    def forward(self, audio: Tensor, duplicate_for_cfg: bool = False):
        self.clap.eval()
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                embeds = self.clap.get_audio_embedding_from_data(
                    audio.reshape(-1, audio.shape[-1]),
                    use_tensor=True,
                )

        embeds = self.mlp(embeds).view(-1, self.prefix_length,
                                       self.embedding_dim)
        mask = torch.ones(embeds.shape[:-1],
                          device=embeds.device,
                          dtype=torch.bool)

        if duplicate_for_cfg:
            embeds = torch.cat((embeds, torch.zeros_like(embeds)), dim=0)
            mask = torch.cat((mask, torch.zeros_like(mask)), dim=0)

        return EmbeddedCondition(embeds, mask)


if __name__ == "__main__":
    from lag.utils import audio as audio_utils
    from lag.utils.inspection import print_params
    device = "cpu"

    embedder = LinearClapEmbedder(1024).to(device)
    print_params(embedder, 1, False)
    audio = audio_utils.load_audio(cfg.AUDIO_DIR / "42cpu.wav").to(device)
    audio = torch.cat((audio, audio), dim=0)

    emb = embedder(audio)
    print(emb.data.shape)

    embcfg = embedder(audio, duplicate_for_cfg=True)
    print(embcfg)
