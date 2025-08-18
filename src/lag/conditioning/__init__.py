from typing import Union
from lag.conditioning.beat_embedder import DirectSinusoidalBeatEmbedder, SinusoidalBeatEmbedder
from lag.conditioning.clap_embedder import LinearClapEmbedder, SequenceClapEmbedder
from lag.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU

ConcreteEmbedder = Union[T5EmbedderCPU, T5EmbedderGPU, LinearClapEmbedder,
                         SequenceClapEmbedder, SinusoidalBeatEmbedder,
                         DirectSinusoidalBeatEmbedder]
