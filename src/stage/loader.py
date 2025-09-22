from pathlib import Path
from tabnanny import check
from typing import Optional
import torch
from safetensors import torch as sft

from stage.conditioning.condition_type import ConditionType
from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.prompt_processor import InterleavedContextPromptProcessor
from stage.conditioning.t5embedder import T5EmbedderGPU
from stage.models.lightning_musicgen import LightningMusicgen
from stage import hyperparameters as hp


def load_model(checkpoint_path: Path,
               device: Optional[str] = None) -> LightningMusicgen:

    stage_params = hp.MusicgenParams(
        encodec_params=hp.pretrained_encodec_meta_32khz_params,
        prompt_processor_params=hp.PromptProcessorParams(
            keep_only_valid_steps=True,
            model_class=InterleavedContextPromptProcessor,
            context_dropout=0.1),
        conditioning_params=hp.ConditioningParams(
            embedder_types={
                ConditionType.DESCRIPTION: T5EmbedderGPU,
            },
            conditioning_methods={
                ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
            },
            conditioning_dropout=0.5),
        lm_params=hp.PretrainedSmallLmParams(sep_token=2049))

    model: LightningMusicgen = stage_params.instantiate()
    # state_dict = torch.load(checkpoint_path)
    # model.load_state_dict(state_dict)
    sft.load_model(model, checkpoint_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    return model
