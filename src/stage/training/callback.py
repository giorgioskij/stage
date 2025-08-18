from concurrent.futures import ThreadPoolExecutor
import pickle
import librosa
import lightning as L
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from typing import Dict, Any
from pathlib import Path
import random
import wandb
import torch
from lightning.pytorch.utilities import rank_zero_only

from stage.conditioning.beat_embedder import Beat
from stage.utils import audio
from stage import config as cfg
# from stage.cocola.contrastive_model import CoCola
# from stage.cocola.feature_extraction import CoColaFeatureExtractor
# from stage.cocola import constants as cocola_constants
# from stage.utils.logging import upload_to_s3

# Custom progress bar to refresh only twice per epoch
# class SlowProgressBar(TQDMProgressBar):

#     def init_train_tqdm(self):
#         bar = super().init_train_tqdm()
#         # Refresh rate: update the bar only twice per epoch
#         bar.refresh_rate = self.total_train_batches // 2
#         return bar


class SaveDemoOnValidationCallback(Callback):

    @rank_zero_only
    def __init__(self, save_path: Path, save_model: bool, n_demos: int):
        self.save_path: Path = save_path
        self.save_model: bool = save_model
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.n_demos: int = n_demos
        self.executor = ThreadPoolExecutor(max_workers=3)

    # @rank_zero_only
    # def on_validation_end(self, trainer: L.Trainer,
    #                       pl_module: L.LightningModule) -> None:
    #     if self.save_model:
    #         if not (self.save_path / "params.pkl").exists():
    #             with open(self.save_path / "params.pkl", "wb") as f:
    #                 pickle.dump(pl_module.params, f)

    #         checkpoint_path = self.save_path / f"step={trainer.global_step}.ckpt"
    #         trainer.save_checkpoint(checkpoint_path, weights_only=False)

    #         # upload checkpoint to S3
    #         run_name = self.save_path.name
    #         future = self.executor.submit(
    #             upload_to_s3,
    #             checkpoint_path,
    #             "lag-modular",
    #             f"checkpoints/{run_name}/{checkpoint_path.name}",
    #         )

    #         def log_upload_status(fut):
    #             try:
    #                 success = fut.result()
    #                 status = "successful" if success else "failed"
    #                 print(f"Checkpoint upload on S3 {status}")
    #             except Exception as e:
    #                 print(f"Error during upload of {checkpoint_path}: {e}")

    #         future.add_done_callback(log_upload_status)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: L.Trainer,
                                  pl_module: L.LightningModule) -> None:

        total_batches = trainer.num_val_batches[0]
        assert isinstance(total_batches, int)
        self.random_batch_idx = random.randint(0, total_batches - 1)

    def on_train_end(self, trainer, pl_module):
        print("Shutting down uploader")
        self.executor.shutdown(wait=True)
        print("Uploader shut down")

    @rank_zero_only
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: Any,
                                batch: Dict[str, Any],
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if batch_idx == self.random_batch_idx:
            self.generate_and_save_demo(trainer, pl_module, batch)

    def generate_and_save_demo(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Dict[str, Any],
    ):
        logger = trainer.logger.experiment if trainer.logger is not None else None  # type: ignore

        save_path: Path = self.save_path / "demos" / f"step={trainer.global_step}"
        save_path.mkdir(parents=True, exist_ok=True)

        # batch_size = min(len(batch["target"]), self.n_demos)
        batch_size = len(batch["target"])
        if self.n_demos < len(batch["target"]):
            chosen_samples = random.sample(list(range(batch_size)),
                                           self.n_demos)
        else:
            chosen_samples = list(range(batch_size))
        n_samples = len(chosen_samples)

        with isolate_rng(include_cuda=True):
            # extract conditioning data from batch
            target = batch.get("target")
            assert target is not None
            context = batch.get("context")
            context_dropout_mask = None
            if context is not None:
                if isinstance(context, list):
                    context = [
                        c for i, c in enumerate(context) if i in chosen_samples
                    ]
                else:
                    assert isinstance(context, Tensor)
                    context = context[chosen_samples]
                context_dropout_mask = torch.full((len(context),),
                                                  True,
                                                  dtype=torch.bool)
                # context_dropout_mask[:n_samples // 2] = False # todo uncomment me
            style = batch.get("style")
            if style is not None:
                style = style[chosen_samples]
            description = batch.get("description")
            if description is not None:
                description = [
                    v for i, v in enumerate(description) if i in chosen_samples
                ]
            beat = batch.get("beat")
            if beat is not None:
                beat = [v for i, v in enumerate(beat) if i in chosen_samples]

            # generate
            L.seed_everything(42)
            with torch.autocast(device_type="cuda"):
                gen_audio = pl_module.generate(
                    n_samples=n_samples,
                    gen_seconds=10,
                    prompt=None,
                    context=context,
                    context_dropout_mask=context_dropout_mask,
                    style=style,
                    beat=beat,
                    description=description,
                    prog_bar=cfg.running_locally())

            # compute cocola score between generated audio and context/style
            '''if style is not None or context is not None:
                cocola = CoCola(
                    embedding_mode=cocola_constants.EmbeddingMode.BOTH)
                cocola.load_state_dict(
                    torch.load(cfg.weights_dir() / "cocola-weights.pt",
                               weights_only=True))
                # cocola = CoCola.load_from_checkpoint(cfg.weights_dir() /
                #                                      "cocola.pt",
                #                                      map_location="cpu").eval()
                cocola.eval()
                cocola.set_embedding_mode(cocola_constants.EmbeddingMode.BOTH)
                feature_extractor = CoColaFeatureExtractor()
                gen_features = feature_extractor(gen_audio.cpu())
                if style is not None:
                    assert len(gen_audio) == len(style)
                    style_features = feature_extractor(style.cpu())
                    style_score = cocola.score(gen_features, style_features)
                if context is not None and isinstance(context, Tensor):
                    assert len(gen_audio) == len(context)
                    context_features = feature_extractor(context.cpu())
                    context_score = cocola.score(gen_features, context_features)'''

        columns = ([
            s for s in ("description", "context", "beat") if s in batch
        ] + ["generated"])
        if "context" in columns:
            columns.append("mix")
            # mixed_audio = gen_audio + context
        if "beat" in columns and "context" in columns:
            columns.append("context with beat")
        # if "style" in columns:
        #     columns.append("style cocola score")
        # if "context" in columns and isinstance(context, Tensor):
        #     columns.append("context cocola score")

        data = [[] for _ in range(n_samples)]

        # for idx in range(n_samples):
        for i, idx in enumerate(chosen_samples):
            gen_filename: Path = save_path / f"demo{i}_gen.wav"
            audio.save_audio(gen_audio[i], gen_filename)

            if description is not None:
                desc_filename: Path = save_path / f"demo{i}_description.txt"
                desc_filename.write_text(description[i])
            '''if style is not None:
                style_filename: Path = save_path / f"demo{i}_style.wav"
                audio.save_audio(style[i], style_filename)'''
            if context is not None and context_dropout_mask[i]:
                context_filename: Path = save_path / f"demo{i}_context.wav"
                mix_filename: Path = save_path / f"demo{i}_mix.wav"
                audio.save_audio(context[i], context_filename)
                mix = torch.nn.functional.pad(
                    context[i],
                    (0, gen_audio[i].shape[-1] - context[i].shape[-1]),
                    value=0) + gen_audio[i]
                audio.save_audio(mix, mix_filename)
            if "target" in batch:
                target_filename: Path = save_path / f"demo{i}_target.wav"
                audio.save_audio(target[idx], target_filename)

            if beat is not None:
                beat_filename: Path = save_path / f"demo{i}_beat.wav"
                b: Beat = beat[i]
                beats = (b.beats / pl_module.sample_rate).cpu().numpy()
                downbeats = (b.downbeats / pl_module.sample_rate).cpu().numpy()
                clicks = librosa.clicks(times=beats,
                                        sr=pl_module.sample_rate,
                                        click_freq=1000,
                                        length=gen_audio[i].shape[-1])
                high_clicks = librosa.clicks(times=downbeats,
                                             sr=pl_module.sample_rate,
                                             click_freq=2000,
                                             length=gen_audio[i].shape[-1])
                gen_with_beat = gen_audio[i].cpu() + (clicks *
                                                      0.5) + (high_clicks * 0.5)
                audio.save_audio(gen_with_beat, beat_filename)
                if context is not None and context_dropout_mask[i]:
                    context_with_beats_filename: Path = (
                        save_path / f"demo{i}_context_with_beats.wav")
                    context_with_beats = torch.nn.functional.pad(
                        context[i].cpu(),
                        (0, len(clicks) - context[i].shape[-1])) + (
                            clicks * 0.5) + (high_clicks * 0.5)
                    audio.save_audio(context_with_beats,
                                     context_with_beats_filename)

            if logger is not None:
                # if pl_module.logger is not None:
                row = []
                if description is not None:
                    row.append(description[i])
                '''if style is not None:
                    row.append(
                        wandb.Audio(str(style_filename),
                                    sample_rate=pl_module.sample_rate))'''
                if context is not None:
                    assert context_dropout_mask is not None
                    if context_dropout_mask[i]:
                        row.append(
                            wandb.Audio(str(context_filename),
                                        sample_rate=pl_module.sample_rate))
                    else:
                        row.append(None)

                if beat is not None:
                    row.append(
                        wandb.Audio(str(beat_filename),
                                    sample_rate=pl_module.sample_rate))

                row.append(
                    wandb.Audio(str(gen_filename),
                                sample_rate=pl_module.sample_rate))
                if context is not None:
                    assert context_dropout_mask is not None
                    if context_dropout_mask[i]:
                        row.append(
                            wandb.Audio(str(mix_filename),
                                        sample_rate=pl_module.sample_rate))
                    else:
                        row.append(None)
                if context is not None and beat is not None:
                    assert context_dropout_mask is not None
                    if context_dropout_mask[i]:
                        row.append(
                            wandb.Audio(str(context_with_beats_filename),
                                        sample_rate=pl_module.sample_rate))
                    else:
                        row.append(None)
                # if style is not None:
                #     row.append(style_score[i])
                # if context is not None and isinstance(context, Tensor):
                #     row.append(context_score[i])

                data[i] = row

        # if pl_module.logger is not None:
        if logger is not None:
            table = wandb.Table(columns=columns, data=data)
            logger.log({f"demos/step:{trainer.global_step}": table})
            # if style is not None:
            #     logger.log({"valid/style_cocola": style_score.mean()})
            # if context is not None and isinstance(context, Tensor):
            #     logger.log({"valid/context_cocola": context_score.mean()})
            # logger.log_table(  # type: ignore
            #     key=f"demos_step{trainer.global_step}",
            #     columns=columns,
            #     data=data)


class PrintLossesCallback(L.Callback):

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Compute and print the average training loss for the epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is None:
            return
        print(f"Epoch {trainer.current_epoch +1} / "
              f"train loss: {train_loss:.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute and print the average validation loss for the epoch
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        print(f"Epoch {trainer.current_epoch + 1} / "
              f"validation loss: {val_loss:.4f}")
