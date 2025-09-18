import itertools
from pathlib import Path
import lightning as L
from typing import Dict, Any, List, Mapping, Type, Optional, Sequence
from torch import Tensor
import torch
from dataclasses import dataclass
from lightning.pytorch import utilities as lightning_utils
from tqdm import tqdm
import loralib as lora

from stage import hyperparameters as hp
from stage.conditioning.beat_embedder import Beat, SinusoidalBeatEmbedder
from stage.conditioning.clap_embedder import LinearClapEmbedder, SequenceClapEmbedder
from stage.conditioning.condition_dispatcher import ConditionDispatcher
from stage.conditioning.condition_provider import ConditionProvider
from stage.conditioning.condition_type import ConditionType
from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.conditioning.t5embedder import T5Embedder, T5EmbedderCPU, T5EmbedderGPU
from stage.models.encodec import EncodecModel
from stage.conditioning.prompt_processor import (
    DefaultPromptProcessor, InterleavedContextPromptProcessor, PromptProcessor,
    StraightContextPromptProcessor)
from stage.models.loss import compute_cross_entropy
from stage.models.musicgen_lm import MusicgenLm
from stage.utils.audio import load_audio, save_audio
from stage.utils.inspection import sanity_check
from stage.utils.sample import eval_decorator, sample_top_k
from stage import config as cfg


class LightningMusicgen(L.LightningModule):

    def __init__(self, params: hp.MusicgenParams):
        super().__init__()
        self.params: hp.MusicgenParams = params

        # instantiate EnCodec
        self.encodec_model: EncodecModel = EncodecModel.from_params(
            self.params.encodec_params)
        self.sample_rate: int = self.encodec_model.sample_rate
        self.n_q: int = self.encodec_model.num_codebooks
        self.special_token: int = 2048
        assert self.n_q == 4

        # freeze encodec
        for p in self.encodec_model.parameters():
            p.requires_grad = False

        # instantiate prompt processor
        self.prompt_processor: PromptProcessor = self.params.prompt_processor_params.model_class(
            self.encodec_model,
            self.special_token,
            keep_only_valid_steps=params.prompt_processor_params.
            keep_only_valid_steps,
            context_dropout=params.prompt_processor_params.context_dropout)

        # check consistency in prompt processor and lm params
        if self.prompt_processor.uses_sep_token:
            if self.params.lm_params.sep_token is None:
                raise cfg.ConfigurationError(
                    "This prompt processor requires an LM with "
                    "support for a separator token.")

        # instantiate lm
        self.lm: MusicgenLm = MusicgenLm(self.params.lm_params)

        # check consistency in conditioning parameters
        if (self.params.conditioning_params.embedder_types.keys()
                != self.params.conditioning_params.conditioning_methods.keys()):
            t1 = set(
                p.value
                for p in self.params.conditioning_params.embedder_types.keys())
            t2 = set(
                p.value for p in
                self.params.conditioning_params.conditioning_methods.keys())
            raise ValueError(
                "Embeddings produced by the condition provider don't match the "
                f"conditioning methods given in params. "
                f"processed conditions: {t1} "
                f"conditioning methods: {t2}")

        # instantiate condition provider (embedder)
        self.condition_provider = ConditionProvider(
            self.params.lm_params.dim,
            embedder_types=self.params.conditioning_params.embedder_types,
        )

        if self.params.lm_params.weights is not None:
            w = torch.load(Path(self.params.lm_params.weights),
                           map_location=None,
                           weights_only=True)
            newstatedict = {
                "condition_provider.embedders.description.output_proj.weight":
                    w['conditioner.output_proj.weight'],
                "condition_provider.embedders.description.output_proj.bias":
                    w['conditioner.output_proj.bias']
            }
            self.load_state_dict(newstatedict, strict=False)

        # instantiate condition dispatcher (contains fusers)
        self.condition_dispatcher: ConditionDispatcher = ConditionDispatcher(
            self.params.conditioning_params.conditioning_methods,
            self.params.lm_params.dim,
            self.params.conditioning_params.conditioning_dropout)

        # inject lora if needed
        if self.params.lora_params is not None:
            self._inject_lora()

        self.save_hyperparameters()

    def _inject_lora(self):
        assert self.params.lora_params is not None

        # freeze the decoder model
        # for p in self.lm.decoder.parameters():
        #     p.requires_grad = False

        layers: torch.nn.ModuleList = self.lm.decoder.attn_layers.layers
        # inject lora in every attention layer
        for att_idx in range(self.params.lm_params.n_layers):
            for sublayer_idx in range(2):
                layeridx = att_idx * 3 + sublayer_idx
                sublayer: torch.nn.ModuleList = layers[layeridx]  # type: ignore
                # for all layer types we want to swap
                for layername in self.params.lora_params.layers:
                    source_layer = sublayer[1].__getattr__(f"to_{layername}")
                    new_layer = lora.Linear(source_layer.in_features,
                                            source_layer.out_features,
                                            self.params.lora_params.r,
                                            self.params.lora_params.alpha,
                                            self.params.lora_params.dropout,
                                            bias=False)
                    with torch.no_grad():
                        new_layer.weight.data.copy_(source_layer.weight.data)
                    if layername == "q":
                        sublayer[1].to_q = new_layer
                    elif layername == "k":
                        sublayer[1].to_k = new_layer
                    elif layername == "v":
                        sublayer[1].to_v = new_layer
                    elif layername == "out":
                        sublayer[1].to_out = new_layer
                    else:
                        raise RuntimeError(f"unknown layer name {layername}")
                    # sublayer[1].__setattr__(f"to_{layername}", new_layer)

        lora.mark_only_lora_as_trainable(self)

    def configure_optimizers(self):  # type: ignore
        opt = torch.optim.AdamW(self.parameters(),
                                lr=1e-5,
                                betas=(0.9, 0.95),
                                weight_decay=0.001)
        return opt
        # n_warmup_steps: int = 1000

        # projections = (
        #     p for n, p in self.named_parameters() if "output_proj" in n)
        # embeddings = (p for n, p in self.named_parameters() if "token_emb" in n)
        # warmup_params = itertools.chain(projections, embeddings)
        # others = (p for n, p in self.named_parameters()
        #           if "output_proj" not in n and "token_emb" not in n)

        # multigroup_optim = torch.optim.AdamW(
        #     ({
        #         "params": warmup_params
        #     }, {
        #         "params": others
        #     }),
        #     lr=2e-5,
        #     betas=(0.9, 0.95),
        #     weight_decay=0.1,
        # )

        # lambda_encoder = lambda x: (1 + 1.5 * (1 - (x / n_warmup_steps))
        #                            ) if x < n_warmup_steps else 1.
        # lambda_decoder = lambda x: 0. if x < n_warmup_steps else 1.

        # multigroup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     multigroup_optim,
        #     lr_lambda=[lambda_encoder, lambda_decoder],
        # )

        # scheduler_config = {
        #     "scheduler": multigroup_scheduler,
        #     "interval": "step"
        # }

        # return {"optimizer": multigroup_optim, "lr_scheduler": scheduler_config}

    # def on_train_batch_end(self, outputs: Tensor | Mapping[str, Any] | None,
    #                        batch: Any, batch_idx: int) -> None:
    # def on_before_optimizer_step(self, optimizer):
    #     decoder_grads = lightning_utils.grad_norm(self.lm.decoder, 2)
    #     self.log_dict(decoder_grads)

    def training_step(self, batch, batch_idx) -> Tensor:
        self.train(True)
        loss = self.run_step(batch)
        self.log(
            "train/loss",
            loss,
            # prog_bar=True,
            batch_size=len(batch["target"]),
            # sync_dist=True,
            # on_step=True,
        )
        self.log("global_step", self.global_step, prog_bar=True, logger=False)
        if self._trainer is not None and self.lr_schedulers() is not None:
            self.log(
                "train/new_params_lr",
                self.lr_schedulers().get_last_lr()[0],  # type: ignore
                prog_bar=True)
            self.log(
                "train/old_params_lr",
                self.lr_schedulers().get_last_lr()[1],  # type: ignore
                prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        self.train(False)
        with torch.no_grad():
            loss = self.run_step(batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            batch_size=len(batch["target"]),
            sync_dist=True,
        )
        # sanity_check(self, interrupt=True)
        return loss

    def run_step(self, batch: Dict[str, Any]) -> Tensor:
        """
            Expects batch to be a dictionary like: 
            {
                "target": Tensor,
                "context": Tensor,     - optional
                "style": Tensor,       - optional
                "description": string, - optional
            }
        """

        # call prompt pre-processor
        (prompt, prompt_mask, target,
         decode_logits_fn) = self.prompt_processor.preprocess(batch)

        attention_mask = prompt_mask.sum(dim=-2) > 0

        # embed/encode conditioning data
        processed_conditions: Dict[ConditionType, EmbeddedCondition] = (
            self.condition_provider.process_conditions(batch))

        # dispatch eatch conditioning to the proper method, fusing if necessary
        method_to_cond: Dict[ConditioningMethod,
                             EmbeddedCondition] = self.condition_dispatcher(
                                 processed_conditions)

        # call language model
        logits = self.lm(
            x=prompt,
            attention_mask=attention_mask,
            cross_attention_input=method_to_cond.get(
                ConditioningMethod.CROSS_ATTENTION),
            prepend_embeds=method_to_cond.get(ConditioningMethod.INPUT_PREPEND),
            sum_embeds=method_to_cond.get(ConditioningMethod.INPUT_SUM),
        )

        # de-interleave logits and postprocess prompt
        logits, logits_mask = decode_logits_fn(logits)

        # compute cross entropy
        cross_entropy_loss, _ = compute_cross_entropy(logits, target,
                                                      logits_mask)

        return cross_entropy_loss

    def sample_next_token(
            self, current_sequence: Tensor, attention_mask: Tensor,
            method_to_cond: Dict[ConditioningMethod,
                                 EmbeddedCondition]) -> Tensor:

        if not current_sequence.isfinite().all():
            if current_sequence.isnan().any():
                print(f"Before forward pass some logits are nan")
            else:
                print(f"Before forward pass some logits are not finite")

        # call language model
        logits: Tensor = self.lm(
            x=current_sequence,
            attention_mask=attention_mask,
            cross_attention_input=method_to_cond.get(
                ConditioningMethod.CROSS_ATTENTION),
            prepend_embeds=method_to_cond.get(ConditioningMethod.INPUT_PREPEND),
            sum_embeds=method_to_cond.get(ConditioningMethod.INPUT_SUM),
        )

        if not logits.isfinite().all():
            if logits.isnan().any():
                print(f"After forward pass some logits are nan")
            else:
                print(f"After forward pass some logits are not finite")

        # classifier-free guidance
        cond_logits, uncond_logits = logits.split(
            current_sequence.shape[0] // 2,
            dim=0,
        )
        logits = uncond_logits + (cond_logits - uncond_logits) * 3.0

        # get logits for last token
        logits = logits.permute(0, 1, 3, 2)  # B, K, card, T
        logits = logits[..., -1]  # B, K, card,

        # apply softmax
        probs = torch.softmax(logits, dim=-1)

        # sample
        next_token = sample_top_k(probs, k=250)
        return next_token

    def predict_step(self, batch):
        batch["prog_bar"] = False
        return self.generate(**batch)

    # @torch.inference_mode()
    @eval_decorator
    @torch.no_grad()
    def generate(self,
                 n_samples: int,
                 gen_seconds: float | int,
                 prompt: Optional[Tensor],
                 context: Optional[Tensor | List[Tensor]],
                 style: Optional[Tensor],
                 beat: Optional[List[Beat]],
                 description: Optional[List[str]],
                 context_dropout_mask: Optional[Tensor] = None,
                 prog_bar: bool = False) -> Tensor:
        """Run autoregressive generation

        Args:
            n_samples (int): number of samples to generate (batch size). All other input parameters should match this.
            gen_seconds (float | int): total length of generation in seconds, including prompt if present.
            prompt (Optional[Tensor]): a piece of input to continue.
            context (Optional[Tensor  |  List[Tensor]]): a musical context to generate an accompaniment for.
            style (Optional[Tensor]): a piece of music to use as stylistic reference.
            beat (Optional[List[Beat]]): a beat object to follow
            description (Optional[List[str]]): a list of descriptions to use as conditioning
            context_dropout_mask (Optional[Tensor], optional): Defaults to None.
            prog_bar (bool, optional): whether to display a progress bar. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            Tensor: _description_
        """

        n_gen_frames = int(self.encodec_model.frame_rate * gen_seconds)

        # generate empty sequence
        gen_sequence = torch.full((n_samples, self.n_q, n_gen_frames),
                                  -1,
                                  dtype=torch.long,
                                  device=self.device)

        # pre-process prompt to feed to the lm
        (gen_sequence, gen_mask, start_offset,
         decode_sequence_fn) = self.prompt_processor.prepare_for_generation(
             prompt,
             context,
             gen_sequence,
             use_cfg=True,
             context_dropout_mask=context_dropout_mask)

        # attention mask: in timesteps in which ALL residual layers are invalid,
        # set attention mask to False.
        attention_mask = gen_mask.sum(dim=-2) > 0

        # from now on we only need the first part of the gen_mask, the second was cfg
        gen_mask = gen_mask[:gen_mask.shape[0] // 2]

        # embed/encode conditioning data
        conditions = {
            "description": description,
            # "context": context,
            "style": style,
            "beat": beat,
        }

        # check for compatibilty of conditions
        for c_name, c_value in conditions.items():
            if c_value is None:
                continue
            condtype: ConditionType = ConditionType(c_name)
            if condtype.value not in self.condition_provider.embedders:
                raise ValueError(
                    f"This version of the model does not support conditioning "
                    f"with {c_name}. You should pass None.")

        processed_conditions: Dict[ConditionType, EmbeddedCondition] = (
            self.condition_provider.process_conditions(conditions,
                                                       duplicate_for_cfg=True,
                                                       batch_size=n_samples))

        # dispatch eatch conditioning to the proper method, fusing if necessary
        method_to_cond: Dict[ConditioningMethod,
                             EmbeddedCondition] = self.condition_dispatcher(
                                 processed_conditions)
        # autoregression

        iterator = range(start_offset, gen_sequence.shape[-1])
        if prog_bar:
            iterator = tqdm(iterator, desc="generating autoregressively...")

        for offset in iterator:
            current_sequence = gen_sequence[..., :offset]
            current_mask = attention_mask[..., :offset]
            next_token = self.sample_next_token(current_sequence, current_mask,
                                                method_to_cond)
            valid_mask = gen_mask[
                ...,  # TODO: I can't figure out if this is correct or if it matters at all anyways
                offset:offset + 1].expand(n_samples, -1, -1)
            next_token[~valid_mask] = self.special_token
            gen_sequence[:n_samples, :, offset:offset + 1] = torch.where(
                gen_sequence[:n_samples, :, offset:offset + 1] == -1,
                next_token,
                gen_sequence[:n_samples, :, offset:offset + 1],
            )
            gen_sequence[n_samples:, :, offset:offset + 1] = torch.where(
                gen_sequence[n_samples:, :, offset:offset + 1] == -1,
                next_token,
                gen_sequence[n_samples:, :, offset:offset + 1],
            )

            if prog_bar:
                torch.cuda.synchronize()

        assert not (gen_sequence == -1).any()
        gen_sequence = gen_sequence[:gen_sequence.shape[0] // 2]
        # assert (gen_sequence == torch.where(
        #     gen_mask[None, ...].expand(n_samples, -1, -1),
        #     gen_sequence,
        #     self.special_token,
        # )).all()

        out_codes, out_mask = decode_sequence_fn(gen_sequence)

        self.encodec_model.eval()
        with torch.no_grad():
            out_audio = self.encodec_model.decode(out_codes)
        return out_audio

    @staticmethod
    def load_from_checkpoint_replacing_paths(ckp_path: Path | str):
        ckp_path = Path(ckp_path)

        def swap_parent(filepath: Path | str, new_parent: Path) -> Path | str:
            if isinstance(filepath, Path):
                typeout = Path
            elif isinstance(filepath, str):
                typeout = str
            else:
                raise RuntimeError("expected Path or str")

            filepath = Path(filepath)
            return typeout(new_parent / filepath.name)

        ckp = torch.load(ckp_path, map_location="cpu")
        params = ckp["hyper_parameters"]["params"]
        params.encodec_params.weights = swap_parent(
            params.encodec_params.weights, cfg.weights_dir())
        params.lm_params.weights = swap_parent(params.lm_params.weights,
                                               cfg.weights_dir())
        model = LightningMusicgen(params)
        model.load_state_dict(ckp["state_dict"])
        return model


if __name__ == "__main__":
    from time import time

    device = torch.device("cuda")

    # musicgen params
    model_params = hp.MusicgenParams(
        encodec_params=hp.pretrained_encodec_meta_32khz_params,
        prompt_processor_params=hp.PromptProcessorParams(
            model_class=InterleavedContextPromptProcessor,
            keep_only_valid_steps=True,
            context_dropout=0.5,
        ),
        conditioning_params=hp.ConditioningParams(
            embedder_types={
                ConditionType.DESCRIPTION: T5EmbedderGPU,
                ConditionType.BEAT: SinusoidalBeatEmbedder,
            },
            conditioning_methods={
                ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
                ConditionType.BEAT: ConditioningMethod.INPUT_PREPEND,
            },
            conditioning_dropout=0.5,
        ),
        lm_params=hp.PretrainedSmallLmParams(sep_token=2049),
    )

    model = model_params.instantiate().to(device)

    context = [
        torch.rand(1, 1, 200_000).to(device),
        # torch.rand(1, 1, 1234).to(device)
    ]

    # TEST TRAINING/VALIDATION STEP
    n_tries = 5
    for _ in range(n_tries):
        batch = {
            "target": torch.rand(1, 1, 320_000).to(device),
            "context": context,
            "beat": [
                Beat(beats=(torch.arange(18) * 16_000).long(),
                     downbeats=(torch.arange(0, 18, 4) * 16_000).long(),
                     seq_len=320_000)
            ],
            "description": [""],
        }
        t0 = time()
        loss = model.run_step(batch)
        torch.cuda.synchronize()
        t1 = time()
        print(f"training step in {t1 - t0} seconds")

    # print("MODEL COMPILED")
    # model = torch.compile(model, fullgraph=True,
    #                       backend="eager")  # type: ignore
    # n_tries = 5
    # for _ in range(n_tries):
    #     batch = {
    #         "target": torch.rand(2, 1, 1_000).to(device),
    #         "context": torch.rand(2, 1, 1_000).to(device) * 2,
    #         "style": torch.rand(2, 1, 1_000).to(device),
    #         "description": ["", ""],
    #     }
    #     t0 = time()
    #     loss = model.run_step(batch)
    #     t1 = time()
    #     print(f"training step in {t1 - t0} seconds")

    # TEST INFERENCE
    L.seed_everything(42)
    model.eval()

    # audio1 = load_audio(cfg.AUDIO_DIR / "42cpu.wav").to(device)
    # audio2 = load_audio(cfg.AUDIO_DIR / "42gpuT5cpu.wav").to(device)

    # PROMPT
    # prompt = audio1
    prompt = None

    # CONTEXT
    # context = torch.rand(2, 1, 320_000).to(device)
    # context = None
    # context = torch.cat((audio1, audio2), dim=0)
    # context = audio1.reshape(1, 1, -1)
    context = load_audio(cfg.EXP_DIR / "experiment_1" /
                         "context.wav").to(device).reshape(1, 1, -1)
    # context[0, ...] = 0

    # STYLE
    # style = torch.cat((audio1, audio2), dim=0)
    # style = torch.rand(2, 1, 320_000).to(device)
    style = None

    # DESCRIPTION
    description = [
        ""
        # "lo-fi chill beat with drums, keyboard and bass playing in a relaxed mood",
        # "lo-fi chill beat with drums, keyboard and bass playing in a relaxed mood"
    ]
    # description = None

    # with torch.autocast(device_type="cuda"):
    # t0 = time()
    # gen_audio = model.generate(
    #     n_samples=len(description),
    #     gen_seconds=10,
    #     prompt=prompt,
    #     context=context,
    #     style=style,
    #     description=description,
    #     prog_bar=True,
    # )
    # torch.cuda.synchronize()
    # t1 = time()
    # print(f"inference completed in {t1 - t0} seconds")

    # # for i in range(gen_audio.shape[0]):
    # save_audio(gen_audio, cfg.AUDIO_DIR / f"temp.wav")

    # args = {
    #     "n_samples": 1,
    #     "gen_seconds": 10,
    #     "prompt": prompt,
    #     "context": context,
    #     "style": style,
    #     "description": description,
    # }
    # i = iter((args,))

    # trainer = L.Trainer(precision="32", enable_progress_bar=False)

    # t0 = time()
    # gen_audio = trainer.predict(model, i)
    # torch.cuda.synchronize()
    # t1 = time()
    # print(f"predict completed in {t1 - t0} seconds")

    # save_audio(gen_audio[0], cfg.AUDIO_DIR / f"temp.wav")  # type: ignore
