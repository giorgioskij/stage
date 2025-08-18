from typing import Dict, Optional, Type, Any, Union
from torch import Tensor, nn
import torch

from stage.conditioning.condition_type import ConditionType
from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.conditioning.t5embedder import T5EmbedderCPU
from stage.conditioning import ConcreteEmbedder


class ConditionProvider(nn.Module):

    def __init__(self, embedding_dim: int,
                 embedder_types: Dict[ConditionType, Type[ConcreteEmbedder]]):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.embedders: nn.ModuleDict = nn.ModuleDict()

        # instantiate embedders for all condition types
        for condition, embedder_type in embedder_types.items():
            self.embedders[condition.value] = embedder_type(
                embedding_dim=self.embedding_dim)

    # def duplicate_conditions_for_cfg(self, conditions: Dict[str, Any]):
    #     for cond_name, cond_data in conditions.items():
    #         if isinstance(cond_data, list) and isinstance(cond_data[0], str):
    #             conditions[cond_name] += [""] * len(cond_data)
    #         elif isinstance(cond_data, Tensor):
    #             conditions[cond_name] = torch.cat(
    #                 (cond_data, torch.zeros_like(cond_data)), dim=0)
    #         else:
    #             raise RuntimeError(
    #                 f"I don't know what an emtpy condition for type: {type}")

    # embed condition into tensors with the corresponding embedders
    def process_conditions(
        self,
        conditions: Dict[str, Any],
        duplicate_for_cfg: bool = False,
        batch_size: Optional[int] = None,
    ) -> Dict[ConditionType, EmbeddedCondition]:

        processed_conditions: Dict[ConditionType, EmbeddedCondition] = {}

        # for each condition in the dict, try to find the corresponding embedder
        # for cond_name, cond_data in conditions.items():
        #     condtype: ConditionType = ConditionType(cond_name)
        #     if condtype.value not in self.embedders:
        #         raise RuntimeError(f"I don't have an embedder for a condition "
        #                            f"named {cond_name}")
        #     embedded: EmbeddedCondition = self.embedders[condtype.value](
        #         cond_data, duplicate_for_cfg=duplicate_for_cfg)
        #     processed_conditions[condtype] = embedded

        # for each of my embedders, find the corresponding condition in the dict
        for cond_name, embedder in self.embedders.items():
            condtype: ConditionType = ConditionType(cond_name)
            if (condtype.value in conditions and
                    conditions[condtype.value] is not None):
                embedded: EmbeddedCondition = embedder(
                    conditions[condtype.value],
                    duplicate_for_cfg=duplicate_for_cfg)
                processed_conditions[condtype] = embedded
            else:
                if batch_size is None:
                    raise RuntimeError(
                        f"Condition {cond_name} was not provided in batch. "
                        f"I need the batch size to generate a null condition")
                embedded: EmbeddedCondition = embedder.null_condition(
                    batch_size + (batch_size * duplicate_for_cfg))
                processed_conditions[condtype] = embedded

        return processed_conditions


if __name__ == "__main__":
    import torch

    conditions = {
        "description": "The quick brown fox jumps over the lazy dog.",
        # "context": torch.rand(4, 32_000),
        # "style": torch.rand(4, 32_000),
    }

    provider = ConditionProvider(embedding_dim=2048,
                                 embedder_types={
                                     ConditionType.DESCRIPTION: T5EmbedderCPU,
                                 })

    with torch.no_grad():
        processed_conditions = provider.process_conditions(conditions)
