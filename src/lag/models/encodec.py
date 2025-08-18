# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models.
Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

import logging
import contextlib
import os
import math
from pathlib import Path
import typing as tp
import numpy as np
import torch
from torch import nn
# from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

# lamsi imports
import lag.config as cfg
from lag.models.quantization.base import QuantizedResult
from lag.models import quantization as qt
from lag import hyperparameters as hp
# from modules import SEANetDecoder, SEANetEncoder

# audiocraft imports
with contextlib.redirect_stderr(open(os.devnull, "w")):
    from lag.models.modules import SEANetEncoder, SEANetDecoder

logger = logging.getLogger()


class EncodecModel(nn.Module):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """
    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0  # type: ignore
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: SEANetEncoder,
                 decoder: SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 sample_rate: int,
                 channels: int,
                 causal: bool = True,
                 renormalize: bool = False):
        super().__init__()
        self.encoder: SEANetEncoder = encoder
        self.decoder: SEANetDecoder = decoder
        self.quantizer: qt.ResidualVectorQuantizer = quantizer
        # self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        self.frame_rate: int = int(
            math.ceil(self.sample_rate /
                      np.prod(self.encoder.ratios)))  # type: ignore

        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def preprocess(
            self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward_with_sum_loss(
            self,
            x: torch.Tensor,
            sum_loss_multiplier: float = 1.) -> qt.QuantizedResult:
        """ Forward pass enforcing additivity in the latent space:
            Q(x1) + Q(x2) = Q(x1 + x2)
        """
        raise NotImplementedError()

        # if we are using sum_loss, batch size needs to be even
        if x.shape[0] % 2 != 0:
            raise ValueError("Batch size needs to be even to use sum loss. "
                             f"Received {x.shape}")
        # y = x1 + x2
        y = x.clone()
        half_bs: int = x.shape[0] // 2
        y = y[:half_bs] + y[half_bs:]

        # encode y: E(y)
        y, y_scale = self.preprocess(y)
        encoded_y = self.encoder(y)

        # encode x: E(x)
        length = x.shape[-1]
        x, scale = self.preprocess(x)
        encoded_x = self.encoder(x)

        # quantize x = x1, x2: get Q(x) = Q(x1), Q(x2)
        x_quantized: QuantizedResult = self.quantizer(encoded_x,
                                                      self.frame_rate)

        # quantize y = x1 + x2: get Q(y) = Q(x1 + x2)
        y_quantized = self.quantizer(encoded_y, self.frame_rate)  # type: ignore

        # sum_of_quantized_layers = Q(x1) + Q(x2)
        sum_of_quantized_layers = (
            x_quantized.quantized_layers[:half_bs] +  # type: ignore
            x_quantized.quantized_layers[half_bs:])  # type: ignore

        # quantization_of_sum = Q(x1 + x2) = Q(y)
        quantization_of_sum_layers = y_quantized.quantized_layers

        # compute sum_loss with L2.  prediction: Q(x1) + Q(x2) target: Q(x1+x2)
        sum_loss = nn.functional.mse_loss(sum_of_quantized_layers,
                                          quantization_of_sum_layers)
        sum_loss = sum_loss * sum_loss_multiplier
        x_quantized.sum_loss = sum_loss

        # decode Q(x) to get D(x) = D(x1), D(x2)
        decoded_x = self.decoder(x_quantized.x)

        # remove extra padding added by the encoder and decoder
        assert decoded_x.shape[-1] >= length, (decoded_x.shape[-1], length)
        decoded_x = decoded_x[..., :length]
        # put in x_quantized.x the decoded version to return it
        x_quantized.x = self.postprocess(decoded_x, scale)

        # compute informative losses (metrics)
        with torch.no_grad():

            # only take last layer (sum of all layers)
            sum_of_quantized = sum_of_quantized_layers[:, -1, ...]
            quantization_of_sum = quantization_of_sum_layers[:, -1, ...]

            # decode Q(y) to get D(y)
            decoded_y = self.decoder(y_quantized.x)
            assert decoded_y.shape[-1] >= length, (decoded_y.shape[-1], length)
            decoded_y = decoded_y[..., :length]

            # decode Q(x1) + Q(x2) to get D(Q(x1) + Q(x2))
            decoded_sum_of_quantized = self.decoder(sum_of_quantized)
            assert decoded_sum_of_quantized.shape[-1] >= length, (
                decoded_sum_of_quantized.shape[-1], length)
            decoded_sum_of_quantized = decoded_sum_of_quantized[..., :length]

            # sum_of_decoded_quantized = D(Q(x1)) + D(Q(x2))
            sum_of_decoded_quantized = decoded_x[:half_bs] + decoded_x[half_bs:]

            # variable names recap:
            # sum_of_quantized = Q(x1) + Q(x2)
            # quantization_of_sum = Q(x1 + x2)
            # decoded_sum_of_quantized = D(Q(x1) + Q(x2))
            # sum_of_decoded_quantized = D(Q(x1)) + D(Q(x2))

            # cosine similarity between Q(x1) + Q(x2) and Q(x1 + x2)
            cos_sim = torch.nn.functional.cosine_similarity(sum_of_quantized,
                                                            quantization_of_sum,
                                                            dim=2).mean()

            # recon-2
            # sisdr1:       prediction: D(Q(x1)) + D(Q(x2))     target: D(y)
            sisdr1 = scale_invariant_signal_distortion_ratio(
                sum_of_decoded_quantized, decoded_y).mean()

            # comp-1
            # sisdr2:       prediction: D(Q(x1) + Q(x2))        target: D(y)
            sisdr2 = scale_invariant_signal_distortion_ratio(
                decoded_sum_of_quantized, decoded_y).mean()

            # recon-1
            # sisdr3:       prediction: D(y)                    target: y
            sisdr3 = scale_invariant_signal_distortion_ratio(decoded_y,
                                                             y).mean()

            # comp-2
            # sisdr4:       prediction: D(Q(x1) + Q(x2))        target: y
            sisdr4 = scale_invariant_signal_distortion_ratio(
                decoded_sum_of_quantized, y).mean()

        metrics: tp.Dict[str, float] = {
            "quantizer/cos1": cos_sim.item(),
            "quantizer/recon2": sisdr1.item(),
            "quantizer/comp1": sisdr2.item(),
            "quantizer/recon1": sisdr3.item(),
            "quantizer/comp2": sisdr4.item(),
        }
        x_quantized.metrics = metrics

        return x_quantized

    def forward(
        self,
        x: torch.Tensor,
        sum_loss_amount: float = 0.,
    ) -> qt.QuantizedResult:

        assert x.dim() == 3
        if sum_loss_amount > 0:
            return self.forward_with_sum_loss(x, sum_loss_amount)

        length = x.shape[-1]
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        q_res: QuantizedResult = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]
        q_res.x = self.postprocess(out, scale)

        return q_res

    def quantize_embedding(self, emb: torch.Tensor) -> QuantizedResult:
        return self.quantizer(emb, self.frame_rate)

    def quantize(self, x: torch.Tensor) -> QuantizedResult:
        """ Pass x through encoder and quantizer and return QuantizedResult"""
        assert x.dim() == 3
        emb = self.encoder(x)
        quantized: QuantizedResult = self.quantizer(emb, self.frame_rate)
        return quantized

    def dequantize(self, x: torch.Tensor | QuantizedResult) -> torch.Tensor:
        if isinstance(x, QuantizedResult):
            x = x.x
        decoded: torch.Tensor = self.decoder(x)
        return decoded

    def encode(
        self,
        x: torch.Tensor,
        return_scales: bool = False
        # ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]] | torch.Tensor:
    ) -> torch.Tensor:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        # if return_scales:
        #     return codes, scale
        return codes

    def decode(self,
               codes: torch.Tensor,
               scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)

    @staticmethod
    def from_pretrained(name: str):
        if name == "facebook/encodec_32khz":
            model = torch.load(cfg.weights_dir() / "encodec_32khz.pt")
        else:
            raise NotImplementedError()

        return model

    @staticmethod
    def from_params(params: hp.EncodecParams):

        seanet_params = params.seanet_params
        qt_params = params.quantizer_params
        encoder: SEANetEncoder = SEANetEncoder(**seanet_params.__dict__)
        decoder: SEANetDecoder = SEANetDecoder(**seanet_params.__dict__)
        quantizer: qt.ResidualVectorQuantizer = qt.ResidualVectorQuantizer(
            **qt_params.__dict__)

        model = EncodecModel(encoder,
                             decoder,
                             quantizer,
                             params.sample_rate,
                             channels=1,
                             causal=params.seanet_params.causal,
                             renormalize=False)

        if params.weights is not None:
            weights = torch.load(Path(params.weights), weights_only=True)
            model.load_state_dict(weights)

        return model

    # @staticmethod
    # def _get_model(target_bandwidths: tp.List[float],
    #                sample_rate: int = 24_000,
    #                channels: int = 1,
    #                causal: bool = True,
    #                model_norm: str = 'weight_norm',
    #                audio_normalize: bool = False,
    #                segment: tp.Optional[float] = None,
    #                name: str = 'unset'):
    #     encoder = m.SEANetEncoder(channels=channels,
    #                               norm=model_norm,
    #                               causal=causal)
    #     decoder = m.SEANetDecoder(channels=channels,
    #                               norm=model_norm,
    #                               causal=causal)
    #     n_q = int(1000 * target_bandwidths[-1] //
    #               (math.ceil(sample_rate / encoder.hop_length) * 10))  # = 32
    #     quantizer = qt.ResidualVectorQuantizer(
    #         dimension=encoder.dimension,
    #         n_q=n_q,
    #         bins=1024,
    #     )
    #     model = EncodecModel(
    #         encoder,
    #         decoder,
    #         quantizer,
    #         # target_bandwidths,
    #         sample_rate,
    #         channels,
    #         causal=True,
    #         renormalize=audio_normalize,
    #         # segment=segment,
    #         # name=name,
    #     )
    #     return model

    # @staticmethod
    # def _get_pretrained(checkpoint_name: str,
    #                     repository: tp.Optional[Path] = None):
    #     if repository is not None:
    #         if not repository.is_dir():
    #             raise ValueError(f"{repository} must exist and be a directory.")
    #         file = repository / checkpoint_name
    #         checksum = file.stem.split('-')[1]
    #         _check_checksum(file, checksum)
    #         return torch.load(file)
    #     else:
    #         url = _get_checkpoint_url(cfg.ROOT_URL, checkpoint_name)
    #         return torch.hub.load_state_dict_from_url(
    #             url, map_location=cfg.device, check_hash=True)  # type:ignore

    # @staticmethod
    # def encodec_model_24khz(pretrained: bool = True,
    #                         repository: tp.Optional[Path] = None):
    #     """Return the pretrained causal 24khz model.
    #     """
    #     if repository:
    #         assert pretrained
    #     target_bandwidths = [1.5, 3., 6, 12., 24.]
    #     checkpoint_name = 'encodec_24khz-d7cc33bc.th'
    #     sample_rate = 24_000
    #     channels = 1
    #     model = EncodecModel._get_model(
    #         target_bandwidths,
    #         sample_rate,
    #         channels,
    #         causal=True,
    #         model_norm='weight_norm',
    #         audio_normalize=False,
    #         name='encodec_24khz' if pretrained else 'unset')
    #     if pretrained:
    #         state_dict = EncodecModel._get_pretrained(checkpoint_name,
    #                                                   repository)
    #         model.load_state_dict(state_dict)
    #     model.eval()
    #     return model
