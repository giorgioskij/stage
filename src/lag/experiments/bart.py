from lag import hyperparameters as hp
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import StraightContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU
from lag.training.train import train
from lag import config as cfg
from lag.data.stem import Stem

RUN_NAME = "bart-modular-drums-mixdata"

if __name__ == "__main__":
    lm_params = hp.PretrainedSmallLmParams()
    conditioning_params = hp.ConditioningParams(
        embedder_types={ConditionType.DESCRIPTION: T5EmbedderCPU},
        conditioning_methods={
            ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION
        })
    prompt_params = hp.PromptProcessorParams(
        model_class=StraightContextPromptProcessor,
        keep_only_valid_steps=True,
        context_dropout=0.5)
    encodec_params = hp.pretrained_encodec_meta_32khz_params

    bart_params = hp.MusicgenParams(encodec_params=encodec_params,
                                    prompt_processor_params=prompt_params,
                                    conditioning_params=conditioning_params,
                                    lm_params=lm_params)

    dataset_params = hp.MixDatasetParams(clip_length_in_seconds=10,
                                         sample_rate=32_000,
                                         root_dir=cfg.mixdata_path(),
                                         single_stem=True,
                                         target_stem=Stem.DRUMS,
                                         add_click=False,
                                         sync_chunks=False,
                                         bpm_in_caption=False,
                                         batch_size_train=8,
                                         batch_size_test=12,
                                         num_workers=4,
                                         speed_transform_p=0.5,
                                         pitch_transform_p=0.5,
                                         n_samples_per_epoch=2000)
    train(
        model_params=bart_params,
        dataset_params=dataset_params,
        n_steps=2000,
        validate_every_n_steps=2000 * 2 // 8,
        run_name=RUN_NAME,
        # run_name=None,
        log=False,
    )
