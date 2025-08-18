from abc import ABC, abstractmethod
from torch import nn, Tensor
import torch
import x_transformers as xt
from typing import Optional

from lag import config as cfg


class ResidualTokenEmbedding(nn.Module):

    def __init__(self,
                 dim: int,
                 num_tokens: int,
                 n_layers: int,
                 padding_token: Optional[int] = None):
        super().__init__()
        self.padding_token: Optional[int] = padding_token
        self.emb = nn.ModuleList([
            nn.Embedding(num_tokens, dim, padding_idx=self.padding_token)
            for _ in range(n_layers)
        ])

        for layer in self.emb:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        n_res_layers = x.shape[1]
        assert n_res_layers == len(self.emb)
        token_emb: Tensor = sum(  # type: ignore
            [self.emb[i](x[:, i]) for i in range(n_res_layers)])
        return token_emb


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device,
                        dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([],
                                   max_period,
                                   device=positions.device,
                                   dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor**(adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class ResidualSinusoidalEmbedding(nn.Module):

    def __init__(self, dim, theta=10000):
        super().__init__()
        assert dim % 2 == 0
        self.scale = 1
        self.theta = theta
        self.dim = dim

    def forward(self, x, pos=None, seq_start_pos=None):
        B, K, T = x.shape
        positions = (pos if pos is not None else torch.arange(
            T, device=x.device).view(1, -1, 1))
        pos_emb = create_sin_embedding(positions,
                                       self.dim,
                                       max_period=self.theta,
                                       dtype=x.dtype)
        return pos_emb * self.scale


class ResidualOutputProj(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, n_layers: int,
                 use_bias: bool):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, output_dim, use_bias) for _ in range(n_layers)
        ])

    def forward(self, x):
        return torch.stack([layer(x) for layer in self.linears], dim=1)


class TransformerDecoder(ABC, nn.Module):

    @abstractmethod
    def forward(x, mask, context, context_mask, prepend_data, prepend_mask,
                sum_data) -> Tensor:
        ...


class XTransformerDecoder(TransformerDecoder):

    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int,
        use_abs_pos_emb: bool,
        scaled_sinu_pos_emb: bool,
        dim: int,
        depth: int,
        heads: int,
        attn_dim_head: int,
        attn_flash: bool,
        ff_no_bias: bool,
        cross_attend: bool,
    ):

        self.decoder: xt.TransformerWrapper = xt.TransformerWrapper(
            num_tokens=self.input_card,
            max_seq_len=500,
            use_abs_pos_emb=True,
            scaled_sinu_pos_emb=True,
            attn_layers=xt.Decoder(
                dim=self.dim,
                depth=self.n_layers,
                heads=self.n_heads,
                attn_dim_head=64,
                attn_flash=True,
                ff_no_bias=True,
                cross_attend=self.cross_attend,
            ),
        )

        self.decoder.token_emb = ResidualTokenEmbedding(  # type: ignore
            self.dim,
            self.input_card,
            self.n_q,
            self.padding_token,
        )
        self.decoder.pos_emb = ResidualSinusoidalEmbedding(  # type: ignore
            dim=self.dim)
        self.decoder.to_logits = ResidualOutputProj(self.dim, self.card,
                                                    self.n_q, False)

        # TODO: this is horrendous, gotta find a fix
        if not self.cross_attend:
            self.nullwav_embeds = torch.load(cfg.weights_dir() /
                                             "nullwav_embeds.pt")[0]
        else:
            self.nullwav_embeds = None
