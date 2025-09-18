from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Type, Set
from pydoc import locate
from frozendict import frozendict
from copy import deepcopy
from pathlib import Path

from stage import config as cfg
from stage.conditioning.condition_type import ConditionType
from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning import ConcreteEmbedder
from stage.conditioning.prompt_processor import InterleavedContextPromptProcessor
from stage.conditioning.t5embedder import T5EmbedderGPU
from stage.data.stem import Stem
from stage.utils.logging import to_loggable

import typing
if typing.TYPE_CHECKING:
    from stage.conditioning.prompt_processor import PromptProcessor


class Loggable:

    def to_dict(self) -> Dict:
        return to_loggable(self.__dict__)  # type: ignore


# generic parameters for a model that can be instantiated
# from a hyperparameter class
@dataclass(unsafe_hash=True, kw_only=True)
class ModelParams(Loggable):
    model_class: str

    # def to_dict(self) -> Dict:
    #     return to_loggable(self.__dict__)  # type: ignore

    def instantiate(self):
        klass = locate(self.model_class)
        return klass(self)  # type: ignore


# --- ENCODEC ---
@dataclass(unsafe_hash=True)
class QuantizerParams:
    dimension: int
    n_q: int
    bins: int

    q_dropout: bool = False
    decay: float = 0.99
    kmeans_init: bool = True
    kmeans_iters: int = 50
    threshold_ema_dead_code: int = 2
    orthogonal_reg_weight: float = 0.0
    orthogonal_reg_active_codes_only: bool = False
    orthogonal_reg_max_codes: int | None = None


@dataclass(unsafe_hash=True)
class SeaNetParams:
    dimension: int
    n_filters: int
    ratios: Tuple[int, int, int, int]
    causal: bool
    true_skip: bool

    channels: int = 1
    n_residual_layers: int = 1
    activation: str = "ELU"
    activation_params: frozendict = field(default_factory=frozendict)
    norm: str = "weight_norm"
    norm_params: frozendict = field(default_factory=frozendict)
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    pad_mode: str = "reflect"
    compress: int = 2
    lstm: int = 2


@dataclass(unsafe_hash=True, kw_only=True)
class EncodecParams(ModelParams):
    sample_rate: int
    seanet_params: SeaNetParams
    quantizer_params: QuantizerParams
    sum_loss_mulitiplier: int
    weights: Optional[str] = None

    model_class: str = "stage.models.lightning_encodec.LightningEncodec"

    def to_dict(self) -> Dict:
        d = deepcopy(self.__dict__)
        for key, value in self.seanet_params.__dict__.items():
            d["seanet_" + key] = value
        for key, value in self.quantizer_params.__dict__.items():
            d["qt_" + key] = value
        del d["quantizer_params"]
        del d["seanet_params"]
        return d


# --- CONDITIONING ---
@dataclass(unsafe_hash=True)
class PromptProcessorParams(Loggable):
    keep_only_valid_steps: bool
    model_class: Type["PromptProcessor"]
    context_dropout: Optional[float] = None


@dataclass(unsafe_hash=True)
class ConditioningParams(Loggable):
    embedder_types: Dict[ConditionType, Type[ConcreteEmbedder]]
    conditioning_methods: Dict[ConditionType, ConditioningMethod]
    conditioning_dropout: float


# --- LM ---
@dataclass(unsafe_hash=True, kw_only=True)
class LmParams(ModelParams):
    dim: int
    n_layers: int
    n_heads: int
    card: int = 2048
    padding_token: Optional[int] = 2048
    sep_token: Optional[int] = None
    cross_attend: bool = True
    weights: Optional[str] = None
    model_class: str = "stage.models.musicgen_lm.MusicgenLm"


@dataclass(unsafe_hash=True, kw_only=True)
class PretrainedSmallLmParams(LmParams):
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    weights: Optional[str] = str(cfg.weights_dir() / "lm-small-weights.pt")


@dataclass(unsafe_hash=True, kw_only=True)
class PretrainedLargeLmParams(LmParams):
    dim: int = 2048
    n_layers: int = 48
    n_heads: int = 32
    weights: Optional[str] = str(cfg.weights_dir() / "lm-large-weights.pt")


@dataclass(unsafe_hash=True, kw_only=True)
class PretrainedMelodyLmParams(LmParams):
    dim: int = 1536
    n_layers: int = 48
    n_heads: int = 24
    cross_attend: bool = False
    weights: Optional[str] = str(cfg.weights_dir() / "lm-melody-weights.pt")


@dataclass(unsafe_hash=True, kw_only=True)
class FioraSmallLmParams(PretrainedSmallLmParams):
    sep_token: Optional[int] = 2049


# --- LORA ---
@dataclass(unsafe_hash=True)
class LoraParams:
    r: int
    alpha: int
    dropout: float
    layers: List[str]


# --- MUSICGEN ---
@dataclass(unsafe_hash=True, kw_only=True)
class MusicgenParams(ModelParams):
    encodec_params: EncodecParams
    prompt_processor_params: PromptProcessorParams
    conditioning_params: ConditioningParams
    lm_params: LmParams
    lora_params: Optional[LoraParams] = None
    model_class: str = "stage.models.lightning_musicgen.LightningMusicgen"


# ------ DATA -------


@dataclass(unsafe_hash=True, kw_only=True)
class DatasetParams:
    datamodule_class: str
    clip_length_in_seconds: int
    sample_rate: int

    def to_dict(self) -> Dict:
        return self.__dict__

    def instantiate(self):
        klass = locate(self.datamodule_class)
        return klass(self)  # type: ignore


@dataclass(kw_only=True)
class StemmedDatasetParams(DatasetParams):
    root_dir: Path
    stems: Set[Stem] = field(
        default_factory=lambda: {
            Stem.DRUMS,
            Stem.BASS,
            Stem.GUITAR,
            Stem.KEYBOARD,
            Stem.PIANO,
            Stem.STRINGS,
            Stem.OTHER,
        })
    single_stem: bool
    target_stem: Stem
    min_context_seconds: int
    use_style_conditioning: bool
    use_beat_conditioning: bool
    type_of_context: str
    add_click: bool
    sync_chunks: bool
    bpm_in_caption: bool
    batch_size_train: int
    batch_size_test: int
    num_workers: int
    clip_length_in_seconds: int
    sample_rate: int
    speed_transform_p: float
    pitch_transform_p: float
    n_samples_per_epoch: int
    datamodule_class: str = "stage.data.stemmed_datamodule.StemmedDatamodule"


@dataclass(kw_only=True)
class MixDatasetParams(StemmedDatasetParams):
    ...


# --------- CONFIGURATIONS ---------
pretrained_encodec_meta_32khz_params: EncodecParams = EncodecParams(
    sample_rate=32_000,
    seanet_params=SeaNetParams(128, 64, (8, 5, 4, 4), False, True),
    quantizer_params=QuantizerParams(128, 4, 2048),
    sum_loss_mulitiplier=0,
    weights=str(cfg.weights_dir() / "encodec_32khz.pt"),
)

stage_params = MusicgenParams(
    encodec_params=pretrained_encodec_meta_32khz_params,
    prompt_processor_params=PromptProcessorParams(
        keep_only_valid_steps=True,
        model_class=InterleavedContextPromptProcessor,
        context_dropout=0.1),
    conditioning_params=ConditioningParams(
        embedder_types={
            ConditionType.DESCRIPTION: T5EmbedderGPU,
        },
        conditioning_methods={
            ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
        },
        conditioning_dropout=0.5),
    lm_params=PretrainedSmallLmParams(sep_token=2049))
