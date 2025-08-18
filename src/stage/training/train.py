from pathlib import Path
from time import sleep
import traceback
from typing import List, Optional
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import OnExceptionCheckpoint
import wandb
from torch import Tensor
import torch

from stage import hyperparameters as hp
from stage.conditioning.clap_embedder import LinearClapEmbedder
from stage.conditioning.condition_type import ConditionType
from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.prompt_processor import InterleavedContextPromptProcessor, StraightContextPromptProcessor
from stage.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU
from stage import config as cfg
from stage.data.stem import Stem
from stage.models.lightning_musicgen import LightningMusicgen
from stage.training.callback import PrintLossesCallback, SaveDemoOnValidationCallback
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from stage.utils.inspection import print_params
from stage.utils.logging import get_or_create_run_id


def train(model_params: hp.ModelParams,
          dataset_params: hp.DatasetParams,
          max_steps: Optional[int] = None,
          max_time: Optional[str] = None,
          max_epochs: Optional[int] = None,
          log: Optional[bool] = None,
          accumulate_grad_batches: int = 1,
          validate_every_n_steps: Optional[int] = None,
          distributed_strategy: Optional[str] = None,
          seed: Optional[int] = None,
          run_name: Optional[str] = None,
          kill_on_end: bool = False,
          n_demos_per_epoch: int = 6,
          devices: int = -1):

    if log is None:
        log = run_name is not None
    if log and run_name is None:
        raise ValueError("Need a run name to log on wandb")

    # init model and datamodule
    full_run_params = {
        "model_params": model_params.to_dict(),
        "data_params": dataset_params.to_dict()
    }
    model = model_params.instantiate()
    datamodule = dataset_params.instantiate()
    n_train_batches, n_valid_batches = (datamodule.lengths["train"],
                                        datamodule.lengths["valid"])

    # setup callbacks and logger
    logger: WandbLogger | bool = False
    resume_from_checkpoint: Optional[Path] = None
    if log:
        assert run_name is not None
        output_dir = cfg.output_dir() / run_name
        if output_dir.exists():
            print(
                f"Output directory for a run named {run_name} exists. Resuming training..."
            )
            resume_from_checkpoint = output_dir / "last.ckpt"
        wandb_run_id = get_or_create_run_id(output_dir)
        logger = WandbLogger(
            entity=cfg.ENTITY,
            project=cfg.PROJECT,
            name=run_name,
            id=wandb_run_id,
            resume="allow",
            config=full_run_params,
            # settings=wandb.Settings(start_method="fork"),
        )
    callbacks: List[L.Callback] = []
    if not cfg.running_locally():
        # printlossescallback = PrintLossesCallback()
        # callbacks.append(printlossescallback)
        progbar: L.Callback = TQDMProgressBar(refresh_rate=n_train_batches // 2)
        callbacks.append(progbar)
    if run_name is not None:
        output_dir = cfg.output_dir() / run_name
        savedemocallback = SaveDemoOnValidationCallback(
            output_dir, save_model=False, n_demos=n_demos_per_epoch)
        interruptcallback = OnExceptionCheckpoint(output_dir,
                                                  filename="interrupted")
        modelcheckpoint = ModelCheckpoint(dirpath=output_dir,
                                          save_last=True,
                                          every_n_epochs=1,
                                          save_top_k=-1)
        callbacks += [
            interruptcallback,
            savedemocallback,
            modelcheckpoint,
        ]

    # init trainer
    trainer = L.Trainer(
        enable_model_summary=True,
        accelerator="auto",
        max_steps=max_steps or -1,
        max_epochs=max_epochs,
        max_time=max_time,
        devices=devices,
        strategy=distributed_strategy or "auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_algorithm="value",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=validate_every_n_steps or 1.0,
        # check_val_every_n_epoch=None,
        limit_train_batches=n_train_batches,
        limit_val_batches=n_valid_batches,
        num_sanity_val_steps=-1,
        enable_progress_bar=True,
    )

    # set seed
    if seed is not None:
        L.seed_everything(seed)

    # run training
    if kill_on_end:
        try:
            trainer.fit(model, datamodule=datamodule)
            print("Training is finished. Killing myself in five minutes.")
            try:
                wandb.finish()
                sleep(300)
                cfg.shutdown()
            except KeyboardInterrupt:
                print("You saved me! I'll never forget that.")
                return
            except Exception:
                cfg.shutdown()

        except KeyboardInterrupt:
            print("Received keyboard interrupt. Stopping training "
                  "without shutting down...")
            wandb.finish()

        except Exception as e:
            (cfg.output_dir() / "exception.txt").write_text(
                f"Exception: {str(e)}\n\n "
                f"Stacktrace: {traceback.format_exc()}\n")
            print(f"training broke with exception {e}")
            print(f"Killing myself in five minutes")
            try:
                wandb.finish()
                sleep(300)
                cfg.shutdown()
            except KeyboardInterrupt:
                print("You saved me! I'll never forget that.")
                return
            except Exception:
                cfg.shutdown()

    else:
        trainer.fit(model, datamodule=datamodule)
        if run_name is not None:
            wandb.finish()

    return


if __name__ == "__main__":
    from time import time

    encodec_params = hp.pretrained_encodec_meta_32khz_params
    lm_params = hp.FioraSmallLmParams()
    prompt_processor_params = hp.PromptProcessorParams(
        keep_only_valid_steps=True,
        model_class=InterleavedContextPromptProcessor,
        context_dropout=0.5)
    conditioning_params = hp.ConditioningParams(
        embedder_types={
            ConditionType.DESCRIPTION: T5EmbedderGPU,
            ConditionType.STYLE: LinearClapEmbedder
        },
        conditioning_methods={
            ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
            ConditionType.STYLE: ConditioningMethod.INPUT_SUM,
        },
        conditioning_dropout=0.5)

    model_params: hp.MusicgenParams = hp.MusicgenParams(
        encodec_params=encodec_params,
        lm_params=lm_params,
        prompt_processor_params=prompt_processor_params,
        conditioning_params=conditioning_params)

    dataset_params = hp.MixDatasetParams(clip_length_in_seconds=10,
                                         sample_rate=32_000,
                                         root_dir=cfg.mixdata_path(),
                                         single_stem=True,
                                         target_stem=Stem.DRUMS,
                                         min_context_seconds=5,
                                         use_style_conditioning=True,
                                         use_beat_conditioning=False,
                                         type_of_context="stems",
                                         add_click=False,
                                         sync_chunks=False,
                                         bpm_in_caption=False,
                                         batch_size_train=2,
                                         batch_size_test=12,
                                         num_workers=8,
                                         speed_transform_p=0.5,
                                         pitch_transform_p=0.5,
                                         n_samples_per_epoch=2000)

    device = "cuda"
    model: LightningMusicgen = model_params.instantiate().to(device)
    datamodule = dataset_params.instantiate()

    # validation step test
    # model.eval()
    # vd = iter(datamodule.val_dataloader())
    # for i in range(2):
    #     batch = next(vd)
    #     batch = {
    #         k: v.to(device) if isinstance(v, Tensor) else v
    #         for k, v in batch.items()
    #     }
    #     t0 = time()
    #     with torch.autocast(device_type="cuda"):
    #         val_loss = model.validation_step(batch, i)
    #     t1 = time()
    #     print(f"val step in {t1 - t0} seconds")

    # training step test
    # td = iter(datamodule.train_dataloader())
    # model.train()
    # for i in range(10):
    #     batch = next(td)
    #     batch = {
    #         k: v.to(device) if isinstance(v, Tensor) else v
    #         for k, v in batch.items()
    #     }
    #     t0 = time()
    #     with torch.autocast(device_type="cuda"):
    #         train_loss = model.training_step(batch, i)
    #     t1 = time()
    #     print(f"training step in {t1 - t0} seconds")

    trainer = L.Trainer(accelerator="auto",
                        precision="16-mixed",
                        enable_model_summary=True,
                        logger=None,
                        enable_checkpointing=False,
                        num_sanity_val_steps=2,
                        limit_train_batches=50,
                        limit_val_batches=2,
                        max_epochs=10)
    trainer.fit(model, datamodule=datamodule)
