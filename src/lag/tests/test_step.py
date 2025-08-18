import torch
import lightning as L

from lag import config as cfg
from lag import hyperparameters as hp
from lag.conditioning.beat_embedder import Beat, SinusoidalBeatEmbedder
from lag.conditioning.clap_embedder import LinearClapEmbedder
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import InterleavedContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderGPU
from lag.data.stem import Stem
from lag.models.lightning_musicgen import LightningMusicgen
from lag.training.callback import SaveDemoOnValidationCallback
from lag.utils.inspection import print_params

lm_params = hp.PretrainedSmallLmParams(sep_token=2049)

conditioning_params = hp.ConditioningParams(
    embedder_types={
        ConditionType.DESCRIPTION: T5EmbedderGPU,
        ConditionType.STYLE: LinearClapEmbedder,
        ConditionType.BEAT: SinusoidalBeatEmbedder,
    },
    conditioning_methods={
        ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
        ConditionType.STYLE: ConditioningMethod.INPUT_SUM,
        ConditionType.BEAT: ConditioningMethod.CROSS_ATTENTION,
    },
    conditioning_dropout=0.5)

prompt_params = hp.PromptProcessorParams(
    keep_only_valid_steps=True,
    model_class=InterleavedContextPromptProcessor,
    context_dropout=0.5)

encodec_params = hp.pretrained_encodec_meta_32khz_params

model_params = hp.MusicgenParams(encodec_params=encodec_params,
                                 prompt_processor_params=prompt_params,
                                 conditioning_params=conditioning_params,
                                 lm_params=lm_params)

model: LightningMusicgen = model_params.instantiate().cpu()
# print_params(model, 3, False)

dataset_params = hp.StemmedDatasetParams(clip_length_in_seconds=10,
                                         sample_rate=32_000,
                                         root_dir=cfg.moises_path(),
                                         single_stem=True,
                                         target_stem=Stem.DRUMS,
                                         min_context_seconds=5,
                                         use_style_conditioning=True,
                                         use_beat_conditioning=True,
                                         add_click=False,
                                         sync_chunks=False,
                                         bpm_in_caption=False,
                                         batch_size_train=2,
                                         batch_size_test=2,
                                         num_workers=11,
                                         speed_transform_p=0.5,
                                         pitch_transform_p=0.5,
                                         n_samples_per_epoch=2000)

datamodule = dataset_params.instantiate()
train_dataloader = datamodule.train_dataloader()
batch = next(iter(train_dataloader))

# print(batch)
L.seed_everything(43)
# model.run_step(batch)
# model.generate(n_samples=2,
#                gen_seconds=10,
#                prompt=None,
#                context=batch["context"],
#                style=batch["style"],
#                beat=batch["beat"],
#                description=batch["description"],, device=device
#                prog_bar=True)

trainer = L.Trainer(max_epochs=1,
                    num_sanity_val_steps=-1,
                    callbacks=[
                        SaveDemoOnValidationCallback(cfg.CKP_DIR / "temp",
                                                     save_model=False,
                                                     n_demos=4)
                    ])
trainer.fit(model, datamodule)
