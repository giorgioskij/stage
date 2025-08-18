from lag import hyperparameters as hp
from lag.conditioning.clap_embedder import LinearClapEmbedder, SequenceClapEmbedder
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import InterleavedContextPromptProcessor, StraightContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU
from lag.training.train import train
from lag import config as cfg
from lag.data.stem import Stem

RUN_NAME = "camille-drums-ec2"
STRATEGY = "ddp_find_unused_parameters_true"


def launch_train():
    if __name__ == "__main__":
        lm_params = hp.PretrainedSmallLmParams()

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

        model_params = hp.MusicgenParams(
            encodec_params=encodec_params,
            prompt_processor_params=prompt_params,
            conditioning_params=conditioning_params,
            lm_params=lm_params)

        batch_size_train = 2
        max_steps = 50_000
        max_time = "00:12:00:00"

        # 1 epoch ~= 30 mins of training
        # batch_size * n_gpus * 60 seconds * 30 minutes / 2 datasets
        # 4 * 4 * 60 * 30 / 2 = 14_400
        n_samples_per_epoch = 14_400

        accumulate_grad_batches = 1
        dataset_params = hp.MixDatasetParams(
            clip_length_in_seconds=10,
            sample_rate=32_000,
            root_dir=cfg.mixdata_path(),
            single_stem=True,
            target_stem=Stem.DRUMS,
            min_context_seconds=10,
            use_style_conditioning=True,
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
            max_steps=max_steps // accumulate_grad_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            run_name=RUN_NAME,
            log=True,
            distributed_strategy=STRATEGY,
            kill_on_end=True,
        )


if __name__ == "__main__":
    launch_train()
    # import wandb
    # try:
    #     launch_train()
    #     print("Training is finished")
    # except Exception as e:
    #     (cfg.output_dir() / "exception.txt").write_text(str(e))
    #     print(f"training broke with exception {e}")
    # finally:
    #     print(f"Shutting down myself 💀")
    #     wandb.finish()
    #     cfg.stop_instance("i-0e1706d9ddb9845aa")

    # if cfg.running_locally():
    #     from pathlib import Path
    #     from lag.utils.aws import launch_aws_job
    #     launch_aws_job(
    #         RUN_NAME,
    #         "ml.g5.12xlarge",
    #         16,
    #         str(Path(__file__).relative_to(Path(__file__).parents[2])),
    #         input_mode="File",
    #     )
    # else:
    #     launch_train()
