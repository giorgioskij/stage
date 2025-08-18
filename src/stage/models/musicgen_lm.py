from pathlib import Path
from torch import nn, Tensor
import torch
from typing import Optional, Tuple, List, Dict
import x_transformers as xt

from stage.conditioning.condition_type import ConditionType
from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.embedded_condition import EmbeddedCondition
from stage import hyperparameters as hp
from stage.utils.inspection import printshape
from stage import config as cfg


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


class DropoutModule(nn.Module):
    """Base module for all dropout modules."""

    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class ConditioningDropout(DropoutModule):

    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.training:
            return x, mask
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return x, mask
        return torch.zeros_like(x), torch.zeros_like(mask)


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """

    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: List) -> List[str | None]:
        if not self.training:
            return samples
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples
        return [None for _ in range(len(samples))]

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class MusicgenLm(nn.Module):

    def __init__(self, params: hp.LmParams):
        super().__init__()

        self.dim: int = params.dim
        self.n_layers: int = params.n_layers
        self.n_heads: int = params.n_heads
        self.card: int = params.card
        self.cross_attend: bool = params.cross_attend
        self.padding_token: Optional[int] = params.padding_token
        self.sep_token: Optional[int] = params.sep_token

        # if using an extra padding token, input card is card+1
        self.input_card: int = self.card + (
            1 if self.padding_token is not None and
            self.padding_token >= self.card else 0)

        if self.sep_token is not None:
            self.input_card += 1

        self.cfg_coef: float = 3.0
        self.n_q = 4

        # DECODER
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

        if params.weights is not None:

            ckpt_state_dict: Dict = torch.load(Path(params.weights),
                                               map_location=None,
                                               weights_only=True)
            for key in ("conditioner.output_proj.weight",
                        "conditioner.output_proj.bias"):
                if key in ckpt_state_dict:
                    ckpt_state_dict.pop(key)

            if self.sep_token is not None:
                for k in range(self.n_q):
                    param_name = f"decoder.token_emb.emb.{k}.weight"
                    param = ckpt_state_dict[param_name]
                    assert self.sep_token == param.shape[0]
                    sep_token_emb = torch.empty_like(param[:1])
                    nn.init.kaiming_normal_(sep_token_emb)
                    param = torch.cat((param, sep_token_emb))
                    ckpt_state_dict[param_name] = param

            self.load_state_dict(ckpt_state_dict, strict=True)

            # zero-out padding token in embedding
            # TODO: figure out how to handle doing or not this shit
            if self.cross_attend:
                if self.padding_token is not None:
                    with torch.no_grad():
                        for k in range(self.n_q):
                            self.decoder.token_emb.emb[  # type: ignore
                                k].weight[
                                    self.padding_token] = torch.zeros_like(
                                        self.decoder.token_emb.
                                        emb[k].  # type: ignore
                                        weight[0])

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        cross_attention_input: Optional[EmbeddedCondition] = None,
        prepend_embeds: Optional[EmbeddedCondition] = None,
        sum_embeds: Optional[EmbeddedCondition] = None,
    ) -> Tensor:

        # unpack cross_attention
        if cross_attention_input is not None:
            cross_attention_data = cross_attention_input.data
            cross_attention_mask = cross_attention_input.mask

            # for samples in the batch in which xatt-conditioning is all padding,
            # place dummy zero-valued conditioning vector, with mask set to True
            if cross_attention_mask is not None:
                masked_rows_coords = cross_attention_mask.sum(dim=-1) == 0
                cross_attention_data[masked_rows_coords] = torch.zeros_like(
                    cross_attention_data[masked_rows_coords])
                # cross_attention_mask[masked_rows_coords, ..., 0] = True
                cross_attention_mask[masked_rows_coords] = torch.ones_like(
                    cross_attention_mask[masked_rows_coords])
        else:
            if self.cross_attend:
                cross_attention_data = torch.zeros(x.shape[0],
                                                   1,
                                                   self.dim,
                                                   dtype=torch.float32,
                                                   device=x.device)
                cross_attention_mask = torch.ones(x.shape[0],
                                                  1,
                                                  dtype=torch.bool,
                                                  device=x.device)
            else:
                cross_attention_data = None
                cross_attention_mask = None

        # unpack prepend embeddings
        prepend_embeds_data, prepend_embeds_mask = None, None
        if prepend_embeds is not None:
            prepend_embeds_data = prepend_embeds.data
            prepend_embeds_mask = prepend_embeds.mask

            if self.nullwav_embeds is not None:
                raise RuntimeError(
                    "CAREFUL! You are using the musicgen-melody hack!")
                nwe = self.nullwav_embeds.repeat(prepend_embeds_data.shape[0],
                                                 1, 1).to(prepend_embeds_data)
                prepend_embeds_data = torch.cat((nwe, prepend_embeds_data),
                                                dim=1)
                if prepend_embeds_mask is not None:
                    nwm = torch.ones(nwe.shape[0],
                                     nwe.shape[1],
                                     device=nwe.device,
                                     dtype=torch.bool)
                    prepend_embeds_mask = torch.cat((nwm, prepend_embeds_mask),
                                                    dim=1)

        # unpack sum embeddings
        sum_embeds_data, sum_embeds_mask = None, None
        if sum_embeds is not None:
            sum_embeds_data = sum_embeds.data
            sum_embeds_mask = sum_embeds.mask
            if sum_embeds_mask is not None:
                sum_embeds_data[~sum_embeds_mask] = 0

        assert attention_mask.shape == torch.Size([x.shape[0], x.shape[-1]])

        # create manually positional embedding indices
        positions = torch.zeros_like(attention_mask, dtype=torch.int64)
        B, S = positions.shape
        first_valid_indices = torch.argmax(attention_mask.long(), dim=-1)
        sequence = torch.arange(S, device=positions.device).unsqueeze(0).expand(
            B, S)
        mask = torch.arange(S, device=positions.device).unsqueeze(
            0) >= first_valid_indices.unsqueeze(1)
        positions[mask] = (sequence - first_valid_indices.unsqueeze(1))[mask]
        positions = positions.unsqueeze(-1)

        # create mask
        mask = torch.ones((x.shape[0], x.shape[-1]),
                          dtype=torch.bool,
                          device=x.device)

        # positional embeddings are manually applied to prepend data
        # if not self.cross_attend and prepend_embeds_data is not None:
        if prepend_embeds_data is not None:
            positions += prepend_embeds_data.shape[-2]
            prepend_embeds_data = prepend_embeds_data + self.decoder.pos_emb(
                prepend_embeds_data[..., 0].unsqueeze(1))

        # forward through decoder
        logits = self.decoder(
            x,
            # mask=mask,
            mask=attention_mask,
            pos=positions,
            context=cross_attention_data,
            context_mask=cross_attention_mask,
            prepend_embeds=prepend_embeds_data,
            prepend_mask=prepend_embeds_mask,
            sum_embeds=sum_embeds_data,
        )
        # if something is prepended to the input, remove it from the logits
        if prepend_embeds_data is not None:
            logits = logits[:, :, prepend_embeds_data.shape[-2]:, :]

        return logits


if __name__ == "__main__":
    # params = hp.PretrainedSmallLmParams()

    # model = params.instantiate()

    # model.eval()
    res = ResidualSinusoidalEmbedding(1024)

    x = torch.rand(1, 1024, 3)
    emb = res(x)
    print(f'{emb.shape=}')
    print(f'{emb=}')
