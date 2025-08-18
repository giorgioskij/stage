import torch
import lightning as L

from lag import hyperparameters as hp
from lag.conditioning.clap_embedder import LinearClapEmbedder
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import InterleavedContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU
from lag.models.lightning_musicgen import LightningMusicgen
from lag import config as cfg
from lag.utils.audio import save_audio

if __name__ == "__main__":
    encodec_params: hp.EncodecParams = hp.pretrained_encodec_meta_32khz_params
    lm_params: hp.PretrainedSmallLmParams = hp.PretrainedSmallLmParams()

    # conditioning parameters
    conditioning_params = hp.ConditioningParams(
        embedder_types={
            ConditionType.DESCRIPTION: T5EmbedderGPU,
            ConditionType.STYLE: LinearClapEmbedder
        },
        conditioning_methods={
            ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
            ConditionType.STYLE: ConditioningMethod.INPUT_SUM,
        },
        conditioning_dropout=0.5,
    )

    # prompt processing parameters
    prompt_processor_params = hp.PromptProcessorParams(
        model_class=InterleavedContextPromptProcessor,
        keep_only_valid_steps=True,
        context_dropout=0.5)

    # musicgen params
    musicgen_params = hp.MusicgenParams(
        encodec_params=encodec_params,
        prompt_processor_params=prompt_processor_params,
        conditioning_params=conditioning_params,
        lm_params=lm_params,
    )

    # model = musicgen_params.instantiate()
    # ckpt = torch.load(cfg.CKP_DIR / "camille-drums-last.ckpt")
    # model.load_state_dict(ckpt["state_dict"], strict=False)
    # model = LightningMusicgen.load_from_checkpoint(
    #     cfg.CKP_DIR / "camille-drums-last.ckpt",
    #     params=musicgen_params).eval().to("cuda")
    model = LightningMusicgen.load_from_checkpoint_replacing_paths(
        cfg.CKP_DIR / "camille-drums-last-t5gpu.ckpt")

    model = model.to("cuda")
    model.eval()
    L.seed_everything(42)
    out = model.generate(1,
                         gen_seconds=10,
                         prompt=None,
                         context=None,
                         style=None,
                         description=["jazzy groove with relaxed mood"],
                         prog_bar=True)

    save_audio(out[0], cfg.AUDIO_DIR / "temp_gpu_new.wav")

    ##### CONVERT #####
    # newname = "camille-drums-last-t5gpu.ckpt"
    # basename = newname[:-11] + newname[-5:]

    # print(f"converting {basename} into {newname}")

    # model = musicgen_params.instantiate().cpu()
    # ckpt_base = torch.load(cfg.CKP_DIR / basename, map_location="cpu")
    # model.load_state_dict(ckpt_base["state_dict"], strict=False)

    # ckpt_base["hyper_parameters"]["params"] = model.params
    # ckpt_base["state_dict"] = model.state_dict()
    # torch.save(ckpt_base, cfg.CKP_DIR / newname)
