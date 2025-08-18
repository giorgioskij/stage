from lag import hyperparameters as hp
from lag.conditioning.clap_embedder import LinearClapEmbedder
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import InterleavedContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderCPU
from lag.training.train import train
from lag import config as cfg
from lag.data.stem import Stem

RUN_NAME = None
STRATEGY = None

if __name__ == "__main__":
    lm_params = hp.PretrainedLargeLmParams()

    conditioning_params = hp.ConditioningParams(
        embedder_types={
            ConditionType.DESCRIPTION: T5EmbedderCPU,
            ConditionType.STYLE: LinearClapEmbedder,
        },
        conditioning_methods={
            ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
            ConditionType.STYLE: ConditioningMethod.INPUT_SUM,
        },
        conditioning_dropout=0.5)

    prompt_params = hp.PromptProcessorParams(
        model_class=InterleavedContextPromptProcessor,
        keep_only_valid_steps=True,
        context_dropout=0.5)

    encodec_params = hp.pretrained_encodec_meta_32khz_params

    lora_params: hp.LoraParams = hp.LoraParams(r=16,
                                               alpha=32,
                                               dropout=0.1,
                                               layers=["q"])

    model_params = hp.MusicgenParams(encodec_params=encodec_params,
                                     prompt_processor_params=prompt_params,
                                     conditioning_params=conditioning_params,
                                     lm_params=lm_params,
                                     lora_params=lora_params)

    # model = model_params.instantiate()

    # from torchinfo import summary
    # import torch
    # print(
    #     summary(model.lm,
    #             input_data=torch.randint(2048, (2, 4, 500),
    #                                      device=model.device)))

    batch_size_train = 1
    batch_size_test = 1
    max_steps = 50_000
    max_time = "00:12:00:00"

    # 1 epoch ~= 30 mins of training
    # batch_size * n_gpus * 60 seconds * 30 minutes / 2 datasets
    # 4 * 4 * 60 * 30 / 2 = 14_400
    n_samples_per_epoch = 10_000

    accumulate_grad_batches = 1
    dataset_params = hp.MixDatasetParams(
        clip_length_in_seconds=10,
        sample_rate=32_000,
        root_dir=cfg.mixdata_path(),
        single_stem=True,
        target_stem=Stem.DRUMS,
        use_style_conditioning=True,
        add_click=False,
        sync_chunks=False,
        bpm_in_caption=False,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        num_workers=11,
        speed_transform_p=0.5,
        pitch_transform_p=0.5,
        n_samples_per_epoch=n_samples_per_epoch)

    train(
        model_params=model_params,
        dataset_params=dataset_params,
        max_time=max_time,
        max_steps=max_steps // accumulate_grad_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        run_name=RUN_NAME,
        log=False,
        distributed_strategy=STRATEGY,
        kill_on_end=False,
    )
