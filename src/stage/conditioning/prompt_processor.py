"""
    Processing the prompt of the language model.
    With "prompt", we always refer to the sequence that is fed as input to 
    the transformer model.

    This sequence might contain conditioning data depending on the architecture
    of the model.

    Every prompt processor should handle at least:
    - Encoding of the input tokens with encodec
    - Interleaving the EnCodec tokens according to the proper pattern

    All classes should extend the abstract PromptProcessor class and implement
    their custom preprocessing strategy
"""
from torch import Tensor, nn
import torch
from typing import Callable, Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from stage.config import ConfigurationError
from stage.models.modules.codebooks_patterns import DelayedPatternProvider, Pattern
from stage.models.encodec import EncodecModel
from stage.utils.audio import pad_stack


class PromptProcessor(ABC, nn.Module):
    uses_sep_token: bool = False
    """
        Abstract base class for any prompt processor.
    """

    def __init__(self, encodec_model: EncodecModel, special_token: int,
                 keep_only_valid_steps: bool, **kwargs):
        super().__init__()

        self.__dict__["encodec_model"] = encodec_model
        # self.encodec_model: EncodecModel = encodec_model
        self.special_token: int = special_token
        self.keep_only_valid_steps = keep_only_valid_steps
        self.pattern_provider = DelayedPatternProvider(
            self.encodec_model.num_codebooks)

    def interleave(
            self, prompt: Tensor,
            keep_only_valid_steps: bool) -> Tuple[Tensor, Tensor, Pattern]:
        # interleave codes
        prompt = prompt.contiguous()
        pattern: Pattern = self.pattern_provider.get_pattern(prompt.shape[-1])
        prompt, indices, mask = pattern.build_pattern_sequence(
            prompt,
            self.special_token,
            keep_only_valid_steps=keep_only_valid_steps)

        return prompt, mask, pattern

    def deinterleave_logits(self, logits: Tensor,
                            pattern: Pattern) -> Tuple[Tensor, Tensor]:
        logits = logits.permute(0, 3, 1, 2)
        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits,
            float("nan"),
            keep_only_valid_steps=self.keep_only_valid_steps)
        logits = logits.permute(0, 2, 3, 1)
        logits_mask = logits_mask[None, :, :].expand(logits.shape[0], -1, -1)
        return logits, logits_mask

    def encode_and_pad_list(self, tensors: List[Tensor],
                            pad_value: int) -> Tensor:
        encoded = [self.encode(t.view(1, 1, -1)).squeeze(0) for t in tensors]

        max_len = max(t.shape[-1] for t in encoded)
        padded_tensors = [
            torch.nn.functional.pad(t, (max_len - t.shape[-1], 0),
                                    value=pad_value) for t in encoded
        ]
        stack = torch.stack(padded_tensors, dim=0)
        return stack

    def encode(self, x: Tensor) -> Tensor:
        if x.shape[-1] < 7:
            return torch.empty(x.shape[1],
                               self.encodec_model.num_codebooks,
                               0,
                               dtype=torch.long,
                               device=x.device)
        self.encodec_model.eval()
        with torch.no_grad():
            return self.encodec_model.encode(x)

    @abstractmethod
    def preprocess(
        self, batch: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor, Tensor, Callable[[Tensor], Tuple[Tensor,
                                                                Tensor]]]:
        ...


class ContextPromptProcessor(PromptProcessor):
    """
        Base class for a prompt processor that encodes target and
        context with encodec.
    """

    def __init__(self, encodec_model: EncodecModel, special_token: int,
                 keep_only_valid_steps: bool, context_dropout: float):
        super().__init__(encodec_model, special_token, keep_only_valid_steps)
        self.context_dropout: float = context_dropout

    def encode_target_and_context_sequential(
            self, target: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        target_codes = self.encode(target)
        context_codes = self.encode(context)
        return target_codes, context_codes

    def encode_target_and_context_parallel(
            self, target: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:

        batch_size = target.shape[0]
        assert context.shape[0] == batch_size

        # concatenate along batch target and context to encode in parallel
        encodec_input = torch.cat((target, context), dim=0)

        # encode with encodec
        encodec_output: Tensor = self.encode(encodec_input)

        target_codes: Tensor = encodec_output[:batch_size, ...]
        context_codes: Tensor = encodec_output[-batch_size:, ...]
        return target_codes, context_codes

    def encode_target_and_context(
            self, target: Tensor,
            context: Tensor | List[Tensor]) -> Tuple[Tensor, Tensor]:

        if isinstance(context, list):
            context_codes: Tensor = self.encode_and_pad_list(
                context, self.special_token)
            target_codes: Tensor = self.encode(target)
            return target_codes, context_codes

        assert isinstance(context, Tensor)

        target_len = target.shape[-1]
        context_len = context.shape[-1]

        if target_len == context_len:
            return self.encode_target_and_context_parallel(target, context)
        else:
            return self.encode_target_and_context_sequential(target, context)

    @abstractmethod
    def prepare_for_generation(
        self, prompt: Optional[Tensor], context: Optional[Tensor],
        gen_sequence: Tensor, use_cfg: bool,
        context_dropout_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, int, Callable[[Tensor], Tuple[Tensor, Tensor]]]:
        """
            Take prompt and context, prepare them, apply cfg, and insert them
            in dummy sequence full of -1 for generation. Return: 
            - gen_sequence
            - gen_mask 
            - start_offset 
            - closure to decode sequence once filled with gen tokens
        """


class DefaultPromptProcessor(PromptProcessor):
    """
        A prompt processor that only encodes the target with encodec, without
        conditioning on context.
    """

    def preprocess(self, batch: Dict[str, Any]):
        target = batch["target"]

        # encode target
        target_codes: Tensor = self.encode(target)

        # interleave
        prompt, _, pattern = self.interleave(target_codes,
                                             self.keep_only_valid_steps)

        def decode_logits_fn(logits: Tensor) -> Tuple[Tensor, Tensor]:
            return self.deinterleave_logits(logits, pattern)

        mask = (prompt != 2048)

        return prompt, mask, target_codes, decode_logits_fn

    def prepare_for_generation(
        self,
        prompt: Optional[Tensor],
        context: Optional[Tensor],
        gen_sequence: Tensor,
        use_cfg: bool,
        context_dropout_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, int, Callable[[Tensor], Tuple[Tensor, Tensor]]]:

        if context is not None or context_dropout_mask is not None:
            raise ConfigurationError(
                "This model does not support context conditioning in the prompt"
            )
        if prompt is None:
            prompt_codes = torch.empty(gen_sequence.shape[0],
                                       gen_sequence.shape[1],
                                       0,
                                       dtype=torch.long,
                                       device=gen_sequence.device)
        else:
            prompt_codes = self.encode(prompt)

        # TODO: make sure that duplicating for cfg is needed here
        if use_cfg:
            gen_sequence = torch.cat((gen_sequence, gen_sequence), dim=0)
            prompt_codes = torch.cat((prompt_codes, prompt_codes), dim=0)

        # insert encoded prompt into sequence
        gen_sequence[..., :prompt_codes.shape[-1]] = prompt_codes

        # interleave sequence (keep_only_valid_steps always false for inference)
        gen_sequence, gen_mask, pattern = self.interleave(
            gen_sequence, keep_only_valid_steps=False)

        # compute first offset of the sequence to generate
        start_offset: int = pattern.get_first_step_with_timesteps(  # type: ignore
            prompt_codes.shape[-1])

        mask: Tensor = (gen_sequence != 2048)

        def decode_sequence_closure(sequence) -> Tuple[Tensor, Tensor]:
            out_codes, _, out_mask = pattern.revert_pattern_sequence(
                sequence, special_token=-1, keep_only_valid_steps=False)

            return out_codes, out_mask

        return gen_sequence, mask, start_offset, decode_sequence_closure


class StraightContextPromptProcessor(ContextPromptProcessor):
    """
        Corresponding implementation to Bart and Azir models.
        
        Interleaves only the target. Concatenates straight context to 
        interleaved target.
    """

    def preprocess(self, batch: Dict[str, Any]):
        target = batch["target"]
        context = batch["context"]

        # encode target and context
        target_codes, context_codes = self.encode_target_and_context(
            target, context)

        # apply dropout to context
        if self.training and self.context_dropout > 0:
            drop = torch.rand(context_codes.shape[0]) < self.context_dropout
            context_codes[drop] = torch.full_like(context_codes[drop],
                                                  self.special_token)

        # interleave target only
        prompt, mask, pattern = self.interleave(target_codes,
                                                self.keep_only_valid_steps)

        # concatenate context and prompt
        prompt = torch.cat((context_codes, prompt), dim=-1)

        def decode_logits_fn(logits: Tensor) -> Tuple[Tensor, Tensor]:
            # remove context
            n_context_codes = context_codes.shape[-1]
            logits = logits[:, :, n_context_codes:, :]

            # deinterleave
            return self.deinterleave_logits(logits, pattern)

        raise NotImplementedError("mask implementation missing")

        return prompt, target_codes, decode_logits_fn

    def prepare_for_generation(
        self,
        prompt: Optional[Tensor],
        context: Optional[Tensor],
        gen_sequence: Tensor,
        use_cfg: bool,
        context_dropout_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, int, Callable[[Tensor], Tuple[Tensor, Tensor]]]:
        raise NotImplementedError("Missing context dropout mask implmentation")
        # dummy prompt and context
        prompt_codes = torch.empty(gen_sequence.shape[0],
                                   gen_sequence.shape[1],
                                   0,
                                   dtype=torch.long,
                                   device=gen_sequence.device)
        context_codes = torch.empty(gen_sequence.shape[0],
                                    gen_sequence.shape[1],
                                    0,
                                    dtype=torch.long,
                                    device=gen_sequence.device)

        # encode prompt and context
        if prompt is not None and context is not None:
            prompt_codes, context_codes = self.encode_target_and_context(
                prompt, context)
        elif prompt is not None:
            prompt_codes = self.encode(prompt)
        elif context is not None:
            context_codes = self.encode(context)

        # duplicate prompt, gen_sequence and context if using cfg
        if use_cfg:
            gen_sequence = torch.cat((gen_sequence, gen_sequence), dim=0)
            prompt_codes = torch.cat((prompt_codes, prompt_codes), dim=0)
            context_codes = torch.cat(
                (context_codes,
                 torch.full_like(context_codes, self.special_token)),
                dim=0)

        # insert encoded prompt into sequence
        gen_sequence[..., :prompt_codes.shape[-1]] = prompt_codes

        # interleave sequence
        gen_sequence, gen_mask, pattern = self.interleave(
            gen_sequence, keep_only_valid_steps=False)
        start_offset: int = pattern.get_first_step_with_timesteps(
            prompt_codes.shape[-1]) + context_codes.shape[-1]  # type: ignore

        # prepend non-interleaved context codes to the sequence
        gen_sequence = torch.cat((context_codes, gen_sequence), dim=-1)
        gen_mask = torch.cat((torch.zeros(context_codes.shape[1:],
                                          dtype=torch.bool,
                                          device=gen_mask.device), gen_mask),
                             dim=-1)

        def decode_sequence_closure(sequence):
            # remove context
            sequence = sequence[..., context_codes.shape[-1]:]
            # de-interleave
            sequence = sequence.contiguous()
            out_codes, _, out_mask = pattern.revert_pattern_sequence(
                sequence, special_token=-1, keep_only_valid_steps=False)
            return out_codes, out_mask

        raise NotImplementedError("mask implementation missing")

        return gen_sequence, gen_mask, start_offset, decode_sequence_closure


class InterleavedContextPromptProcessor(ContextPromptProcessor):
    uses_sep_token: bool = True
    """
        Concatenates context and target, separating them with a single timestep
        of a "stop tensor", made of n_q special tokens.

        The sequence (context, stop_token, target) is then interleaved.
    """

    def preprocess(self, batch: Dict[str, Any]):
        target = batch["target"]
        context = batch["context"]

        # encode target and context
        target_codes, context_codes = self.encode_target_and_context(
            target, context)

        # apply dropout to context
        if self.training and self.context_dropout > 0:
            drop = torch.rand(context_codes.shape[0]) < self.context_dropout
            context_codes[drop] = torch.full_like(context_codes[drop],
                                                  self.special_token)

        # concatenate context and target with stop token
        stop_tensor = torch.full(
            (target_codes.shape[0], target_codes.shape[1], 1),
            self.special_token + 1,
            dtype=target_codes.dtype,
            device=target_codes.device)
        prompt = torch.cat((context_codes, stop_tensor, target_codes), dim=-1)

        # interleave prompt
        prompt, prompt_mask, pattern = self.interleave(
            prompt, self.keep_only_valid_steps)

        # all tokens behind the stop tokens are conditioning
        cond_tokens_mask = torch.zeros_like(prompt, dtype=torch.bool)
        n_q = prompt.shape[1]
        context_len = context_codes.shape[-1]
        for i in range(n_q):
            cond_tokens_mask[:, i, range(context_len + i + 1)] = True
        special_tokens_mask = prompt == self.special_token
        invalid_cond_mask = special_tokens_mask * cond_tokens_mask
        final_mask = (~invalid_cond_mask) * prompt_mask.repeat(
            invalid_cond_mask.shape[0], 1, 1)

        def decode_logits_closure(logits) -> Tuple[Tensor, Tensor]:
            # remove interleaving pattern
            deinterleaved_logits, logits_mask = self.deinterleave_logits(
                logits, pattern)

            # remove context
            n_target_codes = target_codes.shape[-1]
            deinterleaved_logits = deinterleaved_logits[:, :,
                                                        -n_target_codes:, :]
            logits_mask = logits_mask[:, :, -n_target_codes:]

            return deinterleaved_logits, logits_mask

        return prompt, final_mask, target_codes, decode_logits_closure

    def prepare_for_generation(
        self, prompt: Optional[Tensor], context: Optional[Tensor],
        gen_sequence: Tensor, use_cfg: bool,
        context_dropout_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, int, Callable[[Tensor], Tuple[Tensor, Tensor]]]:
        """
            Take prompt and context, prepare them, apply cfg, and insert them
            in dummy sequence full of -1 for generation. Return: 
            - gen_sequence
            - gen_mask 
            - start_offset 
            - closure to decode sequence once filled with gen tokens
        """
        # encode prompt and context
        if prompt is not None and context is not None:
            prompt_codes, context_codes = self.encode_target_and_context(
                prompt, context)
            # context_mask = torch.ones_like(context_codes, dtype=torch.bool)
        else:
            if prompt is None:
                prompt_codes = torch.empty(gen_sequence.shape[0],
                                           gen_sequence.shape[1],
                                           0,
                                           dtype=torch.long,
                                           device=gen_sequence.device)
            else:
                prompt_codes = self.encode(prompt)
            if context is None:
                context_codes = torch.full(
                    (gen_sequence.shape[0], gen_sequence.shape[1], 0),
                    self.special_token,
                    dtype=torch.long,
                    device=gen_sequence.device)
                # context_mask = torch.zeros_like(context_codes, dtype=torch.bool)
            else:
                if isinstance(context, list):
                    context_codes = self.encode_and_pad_list(
                        context, self.special_token)
                else:
                    assert isinstance(context, Tensor)
                    context_codes = self.encode(context)
                # context_mask = torch.ones_like(context_codes, dtype=torch.bool)

            if context_dropout_mask is not None:
                if context_dropout_mask.shape != torch.Size(
                    (context_codes.shape[0],)):
                    raise ValueError(f"Context dropout mask should be of shape "
                                     f"[{context_codes.shape[0]} but it is "
                                     f"{context_dropout_mask.shape}]")
                for i in range(len(context_dropout_mask)):
                    if not context_dropout_mask[i]:
                        context_codes[i] = torch.full_like(
                            context_codes[i], self.special_token)
                # context_codes = torch.where(
                #     context_dropout_mask, context_codes,
                #     torch.full_like(context_codes, self.special_token))

        # duplicate prompt, gen_sequence and context if using cfg
        if use_cfg:
            gen_sequence = torch.cat((gen_sequence, gen_sequence), dim=0)
            prompt_codes = torch.cat((prompt_codes, prompt_codes), dim=0)
            context_codes = torch.cat(
                (context_codes,
                 torch.full_like(context_codes, self.special_token)),
                dim=0)
            # context_mask = torch.cat(
            #     (context_mask, torch.zeros_like(context_mask,
            #                                     dtype=torch.bool)),
            #     dim=0)

        # insert encoded prompt into sequence
        gen_sequence[..., :prompt_codes.shape[-1]] = prompt_codes

        # prepend context and stop token to gen_sequence
        stop_tensor = torch.ones(
            (gen_sequence.shape[0], gen_sequence.shape[1], 1),
            dtype=gen_sequence.dtype,
            device=gen_sequence.device) * self.special_token + 1
        # stop_mask = torch.ones_like(stop_tensor, dtype=torch.bool)
        # conditioning_mask = torch.cat(
        #     (context_mask, stop_mask,
        #      torch.ones_like(gen_sequence, dtype=torch.bool)),
        #     dim=-1)
        gen_sequence = torch.cat((context_codes, stop_tensor, gen_sequence),
                                 dim=-1)

        # interleave sequence (keep_only_valid_steps always false for inference)
        gen_sequence, gen_mask, pattern = self.interleave(
            gen_sequence, keep_only_valid_steps=False)

        # interleaved_conditioning_mask, mask_mask, _ = self.interleave(
        #     conditioning_mask, keep_only_valid_steps=False)

        # assert gen_sequence.shape == interleaved_conditioning_mask.shape

        # interleaved_conditioning_mask[~gen_mask] = 0
        # interleaved_conditioning_mask[~mask_mask] = 0

        # compute first offset of the sequence to generate
        start_offset: int = pattern.get_first_step_with_timesteps(  # type: ignore
            prompt_codes.shape[-1] + context_codes.shape[-1] + 1)

        # set gen_mask of stop token to false
        n_q = gen_sequence.shape[1]
        # gen_mask[range(n_q),
        #          range(start_offset - 1, start_offset - 1 + n_q)] = False

        # all tokens behind the stop token are conditioning
        cond_tokens_mask = torch.zeros_like(gen_sequence, dtype=torch.bool)
        for i in range(n_q):
            cond_tokens_mask[:, i, range(start_offset + i - 1)] = True
        special_tokens_mask = gen_sequence == self.special_token
        invalid_cond_mask = special_tokens_mask * cond_tokens_mask

        final_mask = (~invalid_cond_mask) * gen_mask.repeat(
            invalid_cond_mask.shape[0], 1, 1)

        def decode_sequence_closure(sequence):
            # deinterleave
            sequence = sequence.contiguous()
            out_codes, _, out_mask = pattern.revert_pattern_sequence(
                sequence, special_token=-1, keep_only_valid_steps=False)
            # remove context and stop token
            out_codes = out_codes[..., (context_codes.shape[-1] + 1):]
            return out_codes, out_mask

        backwards_compatible_mode = False
        if backwards_compatible_mode:
            return gen_sequence, gen_mask, start_offset, decode_sequence_closure

        return gen_sequence, final_mask, start_offset, decode_sequence_closure


def parallel_vs_sequential_test():

    from time import time
    from stage import hyperparameters as hp
    from stage.models.encodec import EncodecModel
    device = torch.device("cuda")

    encodec_params = hp.pretrained_encodec_meta_32khz_params
    encodec = EncodecModel.from_params(encodec_params).eval().to(device)

    processor = StraightContextPromptProcessor(encodec_model=encodec,
                                               special_token=2048,
                                               keep_only_valid_steps=False,
                                               context_dropout=0.9).eval()

    # quick benchmark for parallel vs sequential encoding performance
    batch_parallel = {
        "target": torch.rand(4, 1, 320_000).to(device),
        "context": torch.rand(4, 1, 320_000).to(device),
    }
    t0 = time()
    out1 = processor.preprocess(batch_parallel)
    t1 = time()
    print(f"parallel encoding: {round((t1 - t0) * 1000, 2)} ms")

    batch_sequential = {
        "target": torch.rand(4, 1, 320_000).to(device),
        "context": torch.rand(4, 1, 310_000).to(device),
    }
    t0 = time()
    out2 = processor.preprocess(batch_sequential)
    t1 = time()
    print(f"sequential encoding: {round((t1 - t0) * 1000, 2)} ms")

    batch_parallel = {
        "target": torch.rand(4, 1, 320_000).to(device),
        "context": torch.rand(4, 1, 320_000).to(device),
    }
    t0 = time()
    out3 = processor.preprocess(batch_parallel)
    t1 = time()
    print(f"parallel encoding: {round((t1 - t0) * 1000, 2)} ms")

    batch_sequential = {
        "target": torch.rand(4, 1, 320_000).to(device),
        "context": torch.rand(4, 1, 310_000).to(device),
    }
    t0 = time()
    out4 = processor.preprocess(batch_sequential)
    t1 = time()
    print(f"sequential encoding: {round((t1 - t0) * 1000, 2)} ms")

    batch_parallel = {
        "target": torch.rand(4, 1, 320_000).to(device),
        "context": torch.rand(4, 1, 320_000).to(device),
    }
    t0 = time()
    out5 = processor.preprocess(batch_parallel)
    t1 = time()
    print(f"parallel encoding: {round((t1 - t0) * 1000, 2)} ms")

    batch_sequential = {
        "target": torch.rand(4, 1, 320_000).to(device),
        "context": torch.rand(4, 1, 310_000).to(device),
    }
    t0 = time()
    out6 = processor.preprocess(batch_sequential)
    t1 = time()
    print(f"sequential encoding: {round((t1 - t0) * 1000, 2)} ms")


if __name__ == "__main__":
    from stage import hyperparameters as hp
    from stage.models.encodec import EncodecModel
    device = torch.device("cuda")

    encodec_params = hp.pretrained_encodec_meta_32khz_params
    encodec = EncodecModel.from_params(encodec_params).eval().to(device)

    processor = InterleavedContextPromptProcessor(encodec_model=encodec,
                                                  special_token=2048,
                                                  keep_only_valid_steps=False,
                                                  context_dropout=0.9).eval()
