import torch
import lightning as L
import os

from lag import config as cfg
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import DefaultPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderCPU
from lag.models.lightning_musicgen import LightningMusicgen
from lag.utils.audio import load_audio, save_audio
from lag import hyperparameters as hp

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)

if __name__ == "__main__":

    device = torch.device("cpu")

    BATCH_SIZE = 1

    params = hp.MusicgenParams(
        encodec_params=hp.pretrained_encodec_meta_32khz_params,
        prompt_processor_params=hp.PromptProcessorParams(
            keep_only_valid_steps=True, model_class=DefaultPromptProcessor),
        conditioning_params=hp.ConditioningParams(
            embedder_types={ConditionType.DESCRIPTION: T5EmbedderCPU},
            conditioning_methods={
                ConditionType.DESCRIPTION: ConditioningMethod.INPUT_PREPEND
            },
            conditioning_dropout=0.5),
        lm_params=hp.PretrainedMelodyLmParams())

    model = params.instantiate()
    model = model.to(device)
    model.eval()
    L.seed_everything(42)

    # context = torch.rand(BATCH_SIZE, 1, 320_000).cuda()
    # context = None
    # context = load_audio(cfg.EXP_DIR / "experiment_sample1-oldmother" /
    #                      "context.wav").to(device).reshape(1, 1, -1)
    context = None

    # style = torch.rand(BATCH_SIZE, 1, 160_000).cuda()
    style = None

    description = [
        # "lo-fi chill beat with drums, keyboard and bass playing in a relaxed mood"
        "happy rock",
    ]

    # with torch.autocast(device_type="cuda", dtype=torch.float16):
    out = model.generate(BATCH_SIZE,
                         gen_seconds=10,
                         prompt=None,
                         beat=None,
                         context=context,
                         style=style,
                         description=description,
                         prog_bar=True)

    save_audio(out[0], cfg.AUDIO_DIR / "melody" / "rock42_cpu.wav")
