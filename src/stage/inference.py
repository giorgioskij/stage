import torch
import lightning as L

from stage import config as cfg
from stage.conditioning.condition_type import ConditionType
from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.prompt_processor import InterleavedContextPromptProcessor
from stage.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU
from stage.data.stem import Stem
from stage.models.lightning_musicgen import LightningMusicgen
from stage import hyperparameters as hp
from stage.utils.audio import load_audio, save_audio
"""
REQUIREMENTS:

weights/
 - encodec_32khz.pt
 - lm-small-weights.pt in

checkpoints/
 - stage_drums_statedict.pt
"""

#%% Load model
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

#%% Load checkpoint, run inference

# load context audio, description
SAMPLE = "sample4"
INSTRUMENT = Stem.DRUMS
SEED = 42

# load state dict
match INSTRUMENT:
    case Stem.DRUMS:
        state_dict = torch.load(cfg.CKP_DIR /
                                "epoch=23-step=30000-statedict.pt")
    case Stem.BASS:
        state_dict = torch.load(cfg.CKP_DIR /
                                "epoch=45-step=57500-statedict.pt")

model.load_state_dict(state_dict)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

desc = (cfg.AUDIO_DIR / SAMPLE / "desc.txt").read_text().strip()
wav = load_audio(cfg.AUDIO_DIR / SAMPLE / "context.wav").to(model.device)

# generate
L.seed_everything(SEED)
out = model.generate(n_samples=1,
                     gen_seconds=10,
                     prompt=None,
                     context=wav,
                     style=None,
                     beat=None,
                     description=[desc],
                     prog_bar=True)

# save output and mix
save_audio(
    out,
    cfg.AUDIO_DIR / "gen" / f"{SAMPLE}_{INSTRUMENT.name.lower()}_{SEED}_2.wav")
padded_wav = torch.nn.functional.pad(wav,
                                     tuple((0, out.shape[-1] - wav.shape[-1])),
                                     value=0)
mix = out + padded_wav
save_audio(
    mix, cfg.AUDIO_DIR / "gen" /
    f"{SAMPLE}_{INSTRUMENT.name.lower()}_{SEED}_mix_2.wav")
