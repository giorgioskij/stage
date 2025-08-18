import torch
import lightning as L
import os

from lag import config as cfg
from lag.conditioning.beat_embedder import Beat
from lag.models.lightning_musicgen import LightningMusicgen
from lag.preprocessing.extract_beats import beat_from_wav
from lag.utils.audio import create_click, load_audio, save_audio

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)

if __name__ == "__main__":

    device = torch.device("cuda")

    BATCH_SIZE = 1

    model = LightningMusicgen.load_from_checkpoint_replacing_paths(
        cfg.CKP_DIR / "irelia-drums" / "epoch=18-step=5947.ckpt")

    model = model.to(device)
    model.eval()
    L.seed_everything(42)

    # ---------------- CONTEXT ----------------
    # context = torch.rand(BATCH_SIZE, 1, 320_000).cuda()
    context = None
    # context = load_audio(cfg.EXP_DIR / "experiment_sample2-venus" /
    #                      "context.wav").to(device).reshape(1, 1, -1)

    # ---------------- STYLE ----------------
    # style = torch.rand(BATCH_SIZE, 1, 160_000).cuda()
    style = None

    # ---------------- BEAT ----------------
    # beatlist = None
    audio_for_beats = load_audio(cfg.EXP_DIR / "experiment_sample2-venus" /
                                 "context.wav").to(device).reshape(1, -1)
    beat: Beat = beat_from_wav(audio_for_beats, 32_000)
    click_track = create_click(audio_for_beats.shape, 32_000,
                               (beat.beats * 32_000).long())
    save_audio(click_track, cfg.AUDIO_DIR / "temp" / f"click_track.wav")
    beat_frames = Beat(beat.beats * 32_000, beat.downbeats * 32_000,
                       beat.seq_len)
    beatlist = [beat_frames]

    # ---------------- DESCRIPTION ----------------
    description = ["A lo-fi chill beat with a relaxed mood"]

    with torch.autocast(device_type="cuda"):
        out = model.generate(BATCH_SIZE,
                             gen_seconds=10,
                             prompt=None,
                             context=context,
                             style=style,
                             beat=beatlist,
                             description=description,
                             prog_bar=True)

    save_audio(out[0], cfg.AUDIO_DIR / "temp" / "gen.wav")

    if context is not None:
        save_audio(out[0] + context,
                   cfg.AUDIO_DIR / "temp" / "gen_with_context.wav")

    if beatlist is not None:
        save_audio(out[0].cpu() + click_track * 0.5,
                   cfg.AUDIO_DIR / "temp" / "gen_with_beat.wav")
