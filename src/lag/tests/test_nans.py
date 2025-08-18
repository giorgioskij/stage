from pathlib import Path
import torch
from torch import Tensor
import lightning as L
from time import time
import os

from tqdm import tqdm

from lag import config as cfg
from lag.conditioning.clap_embedder import LinearClapEmbedder
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import InterleavedContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderCPU
from lag.models.lightning_musicgen import LightningMusicgen
from lag import hyperparameters as hp
from lag.utils.inspection import print_params

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
torch.use_deterministic_algorithms(True)

if __name__ == "__main__":

    # DEVICE: torch.device = torch.device(
    #     "cuda" if torch.cuda.is_available() else "cpu")
    DEVICE: torch.device = torch.device("cuda")
    BATCH_SIZE = 2

    # musicgen params
    params = hp.MusicgenParams(
        encodec_params=hp.pretrained_encodec_meta_32khz_params,
        prompt_processor_params=hp.PromptProcessorParams(
            model_class=InterleavedContextPromptProcessor,
            keep_only_valid_steps=True,
            context_dropout=0.5),
        conditioning_params=hp.ConditioningParams(
            embedder_types={
                ConditionType.DESCRIPTION: T5EmbedderCPU,
                ConditionType.STYLE: LinearClapEmbedder
            },
            conditioning_methods={
                ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
                ConditionType.STYLE: ConditioningMethod.INPUT_SUM,
            },
            conditioning_dropout=0.5,
        ),
        lm_params=hp.LmParams(dim=1024, n_layers=24, n_heads=16),
    )
    model_nice: LightningMusicgen = params.instantiate()
    ckp_path: Path = cfg.CKP_DIR / "camille-bass-last.ckpt"
    model_nice.load_from_checkpoint_replacing_paths(ckp_path)
    model_nice = model_nice.to(DEVICE)

    L.seed_everything(42)
    model_nice.eval()
    # context = torch.rand(1, 1, 320_000)
    # context = context.repeat(BATCH_SIZE, 1, 1).to(DEVICE)
    context = torch.rand(2, 1, 320_000).to(DEVICE)
    # style = None
    # style = torch.rand(1, 1, 320_000)
    # style = style.repeat(BATCH_SIZE, 1, 1).to(DEVICE)
    style = torch.rand(2, 1, 320_000).to(DEVICE)
    desc = [
        "lo-fi chill beat with drums, keyboard and bass playing in a relaxed mood",
        "lo-fi chill beat with drums, keyboard and bass playing in a relaxed mood",
    ]

    try:
        with torch.autocast(device_type="cuda"):
            gen_audio: Tensor = model_nice.generate(n_samples=BATCH_SIZE,
                                                    gen_seconds=1,
                                                    prompt=None,
                                                    context=context,
                                                    style=style,
                                                    description=desc,
                                                    prog_bar=True)
        print("model nice works")
    except:
        print("model nice broke")

    model_bad = LightningMusicgen.load_from_checkpoint_replacing_paths(ckp_path)
    model_bad = model_bad.to(DEVICE)

    for seed in tqdm(range(1)):
        L.seed_everything(seed)
        model_bad.eval()
        # context = torch.rand(1, 1, 320_000)
        # context = context.repeat(BATCH_SIZE, 1, 1).to(DEVICE)
        context = torch.rand(2, 1, 320_000).to(DEVICE)
        # style = None
        # style = torch.rand(1, 1, 320_000)
        # style = style.repeat(BATCH_SIZE, 1, 1).to(DEVICE)
        style = torch.rand(2, 1, 320_000).to(DEVICE)
        desc = [
            "lo-fi chill beat with drums, keyboard and bass playing in a relaxed mood",
            "lo-fi chill beat with drums, keyboard and bass playing in a relaxed mood",
        ]

        # with torch.autocast(device_type="cuda"):
        try:
            with torch.autocast(device_type="cuda"):
                gen_audio: Tensor = model_bad.generate(n_samples=BATCH_SIZE,
                                                       gen_seconds=0.2,
                                                       prompt=None,
                                                       context=context,
                                                       style=style,
                                                       description=desc,
                                                       prog_bar=False)
        except Exception as e:
            print(f"model breaks on seed {seed}")
            print(e)

    # print(f'{(model_nice.params == model_bad.params)=}')

    # for (nice_name, nice_p), (bad_name,
    #                           bad_p) in zip(model_nice.named_parameters(),
    #                                         model_bad.named_parameters()):
    #     if not nice_name == bad_name:
    #         print(f"wtf? i got parameters {nice_name} and {bad_name}")
    #     if not nice_p.shape == bad_p.shape:
    #         print(f"wtf? parameter {nice_name} differs in shape")
    #     if not nice_p.equal(bad_p):
    #         print(f"model nice and model bad parameters differ in {nice_name}")
    #         break
    # else:
    #     print("models have same parameters")

    # print("MODEL NICE")
    # print(print_params(model_nice, 2, False))

    # print("MODEL BAD")
    # print(print_params(model_bad, 2, False))
