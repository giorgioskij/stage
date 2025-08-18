from typing import Union
from stage.conditioning.beat_embedder import DirectSinusoidalBeatEmbedder, SinusoidalBeatEmbedder
from stage.conditioning.clap_embedder import LinearClapEmbedder, SequenceClapEmbedder
from stage.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU

ConcreteEmbedder = Union[T5EmbedderCPU, T5EmbedderGPU, LinearClapEmbedder,
                         SequenceClapEmbedder, SinusoidalBeatEmbedder,
                         DirectSinusoidalBeatEmbedder]
