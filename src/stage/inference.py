import torch
import lightning as L

from stage import config as cfg
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
model_params = hp.stage_params
model: LightningMusicgen = model_params.instantiate()

#%% Load checkpoint, run inference

# load context audio, description
SAMPLE = "sample4"
INSTRUMENT = Stem.BASS
SEED = 777

# load state dict
match INSTRUMENT:
    case Stem.DRUMS:
        state_dict = torch.load(cfg.CKP_DIR /
                                "epoch=39-step=50000-statedict.pt")
    case Stem.BASS:
        state_dict = torch.load(cfg.CKP_DIR /
                                "epoch=45-step=57500-statedict.pt")

model.load_state_dict(state_dict)
model = model.cuda().eval()

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
    cfg.AUDIO_DIR / "gen" / f"{SAMPLE}_{INSTRUMENT.name.lower()}_{SEED}.wav")
padded_wav = torch.nn.functional.pad(wav,
                                     tuple((0, out.shape[-1] - wav.shape[-1])),
                                     value=0)
mix = out + padded_wav
save_audio(
    mix, cfg.AUDIO_DIR / "gen" /
    f"{SAMPLE}_{INSTRUMENT.name.lower()}_{SEED}_mix.wav")
