from lag import hyperparameters as hp
from lag.conditioning.clap_embedder import LinearClapEmbedder
from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.prompt_processor import InterleavedContextPromptProcessor
from lag.conditioning.t5embedder import T5EmbedderCPU

ppp = hp.PromptProcessorParams(model_class=InterleavedContextPromptProcessor,
                               keep_only_valid_steps=True,
                               context_dropout=0.5)

params = hp.MusicgenParams(
    encodec_params=hp.pretrained_encodec_meta_32khz_params,
    lm_params=hp.PretrainedSmallLmParams(),
    conditioning_params=hp.ConditioningParams(
        embedder_types={
            ConditionType.DESCRIPTION: T5EmbedderCPU,
            ConditionType.CONTEXT: LinearClapEmbedder,
        },
        conditioning_methods={
            ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
            ConditionType.CONTEXT: ConditioningMethod.CROSS_ATTENTION,
        },
        conditioning_dropout=0.5),
    prompt_processor_params=ppp,
)

model = params.instantiate()
