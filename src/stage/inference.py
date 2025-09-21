import torch
import lightning as L

from stage import config as cfg
from stage.loader import load_model
from stage.utils.audio import load_audio, save_audio
"""
REQUIREMENTS:

weights/
 - encodec_32khz.pt
 - lm-small-weights.pt in

checkpoints/
 - stage-drums-ckp1.pt
 - stage-bass-ckp1.pt
"""

#%% Load model
INSTRUMENT = "bass"
checkpoint_path = cfg.CKP_DIR / f"stage-{INSTRUMENT}-ckp1.pt"
model = load_model(checkpoint_path)

#%% Load conditioning, generate and save

# load context audio, description
SAMPLE = "sample1"
SEED = 42

# load description if present
desc_path = cfg.AUDIO_DIR / INSTRUMENT / f"{SAMPLE}-desc.txt"
desc = desc_path.read_text().strip() if desc_path.exists() else None

# load audio context
wav = load_audio(cfg.AUDIO_DIR / INSTRUMENT / f"{SAMPLE}.wav").to(model.device)

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
save_audio(out, cfg.AUDIO_DIR / "gen" / f"{SAMPLE}_{INSTRUMENT}_{SEED}.wav")
padded_wav = torch.nn.functional.pad(wav,
                                     tuple((0, out.shape[-1] - wav.shape[-1])),
                                     value=0)
mix = out + padded_wav
save_audio(mix, cfg.AUDIO_DIR / "gen" / f"{SAMPLE}_{INSTRUMENT}_{SEED}_mix.wav")
