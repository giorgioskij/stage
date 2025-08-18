"""
    Abstract base classes for conditioning embedders.
    An embedder takes some conditioning data and embeds/encodes it into vectors
    that are then fed to the LM according to the condition strategy specified 
    in the config.
    
    Embedding vectors should have the same latent dimensionality of the 
    transformer, so every embedder takes as input to the constructor a 
    parameter `embedding_dim`, that should be the same as the hidden dim of 
    the transformer.
"""

from torch import nn
from abc import ABC, abstractmethod

from stage.conditioning.embedded_condition import EmbeddedCondition


class Embedder(ABC, nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.embedding_dim: int = embedding_dim

    @abstractmethod
    def forward(self, x, duplicate_for_cfg: bool) -> EmbeddedCondition:
        ...

    @abstractmethod
    def null_condition(self, batch_size: int) -> EmbeddedCondition:
        ...


class LinearProjectionEmbedder(ABC, nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.embedding_dim: int = embedding_dim
        self.output_proj: nn.Linear = nn.Linear(self.input_dim,
                                                self.embedding_dim)

    @abstractmethod
    def forward(self, x, duplicate_for_cfg: bool) -> EmbeddedCondition:
        ...

    @abstractmethod
    def null_condition(self, batch_size: int) -> EmbeddedCondition:
        ...
