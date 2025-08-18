"""
Kaisa is ONLY beat conditioning and description (with context), as simple as possible
"""
import torch
import lightning as L

from lag import hyperparameters as hp
from lag.conditioning.beat_embedder import Beat, DirectSinusoidalBeatEmbedder, SinusoidalBeatEmbedder
from lag.conditioning.clap_embedder import LinearClapEmbedder
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import DefaultPromptProcessor, InterleavedContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderGPU
from lag.data.stem import Stem
from lag.training.train import train
from lag import config as cfg

RUN_NAME = "try"
STRATEGY = None

def launch_train():
    lm_params = hp.PretrainedSmallLmParams(sep_token=2049)

    conditioning_params = hp.ConditioningParams(
        embedder_types={
            ConditionType.DESCRIPTION: T5EmbedderGPU,
        },
        conditioning_methods={
            ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
        },
        conditioning_dropout=0.5)

    prompt_params = hp.PromptProcessorParams(
        keep_only_valid_steps=True,
        model_class=InterleavedContextPromptProcessor,
        context_dropout=0.1)

    encodec_params = hp.pretrained_encodec_meta_32khz_params

    model_params = hp.MusicgenParams(encodec_params=encodec_params,
                                     prompt_processor_params=prompt_params,
                                     conditioning_params=conditioning_params,
                                     lm_params=lm_params)
    batch_size_train = 2
    max_steps = 100_000
    max_time = "00:24:00:00"

    # 1 epoch ~= 30 mins of training
    # batch_size * n_gpus * 60 seconds * 30 minutes / 2 datasets
    # 4 * 4 * 60 * 30 / 2 = 14_400
    n_samples_per_epoch = 10

    accumulate_grad_batches = 4

    dataset_params = hp.StemmedDatasetParams(
        clip_length_in_seconds=10,
        sample_rate=32_000,
        root_dir=cfg.moises_path(),
        single_stem=True,
        target_stem=Stem.DRUMS,
        min_context_seconds=5,
        use_style_conditioning=True,
        use_beat_conditioning=True,
        type_of_context="stems or beats",
        add_click=False,
        sync_chunks=False,
        bpm_in_caption=False,
        batch_size_train=batch_size_train,
        batch_size_test=12,
        num_workers=11,
        speed_transform_p=0.5,
        pitch_transform_p=0.5,
        n_samples_per_epoch=n_samples_per_epoch)

    train(
        model_params=model_params,
        dataset_params=dataset_params,
        max_time=max_time,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        run_name=RUN_NAME,
        distributed_strategy=STRATEGY,
        log=False,
        kill_on_end=False,
    )


if __name__ == "__main__":
    launch_train()
