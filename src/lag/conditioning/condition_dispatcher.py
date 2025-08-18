from typing import Callable, List, Sequence, Dict, Type
from torch import nn, Tensor
import torch

from lag.conditioning.condition_type import ConditionType
from lag.conditioning.conditioning_method import ConditioningMethod
from lag.conditioning.embedded_condition import EmbeddedCondition


class ConditionDispatcher(nn.Module):

    def __init__(self, cond_to_method: Dict[ConditionType, ConditioningMethod],
                 embedding_dim: int, condition_dropout: float):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.condition_dropout: float = condition_dropout

        # what fusing method to use for each conditioning type
        self.cond_to_method: Dict[ConditionType,
                                  ConditioningMethod] = cond_to_method

        # what condition types is each method applied to
        self.method_to_conds: Dict[ConditioningMethod, List[ConditionType]] = {}
        for c, m in self.cond_to_method.items():
            self.method_to_conds[m] = self.method_to_conds.get(m, []) + [c]

        # instantiate fuser classes
        self.method_to_fuser: nn.ModuleDict = nn.ModuleDict({
            m.value:
                METHOD_TO_FUSER_TYPE[m](len(self.method_to_conds[m]),
                                        self.embedding_dim)
            for m in self.method_to_conds.keys()
        })

    def dropout(self, x: EmbeddedCondition):
        if self.training and self.condition_dropout > 0:
            drop = torch.rand(x.data.shape[0]) < self.condition_dropout
            # x.data[drop] = torch.zeros_like(x.data[drop])  #.detach()
            x.data[drop] = x.data[drop] * 0
            if x.mask is not None:
                x.mask[drop] = torch.zeros_like(x.mask[drop])  #.detach()
        return x

    def forward(
        self, embedded_conditions: Dict[ConditionType, EmbeddedCondition]
    ) -> Dict[ConditioningMethod, EmbeddedCondition]:
        assert embedded_conditions.keys() == self.cond_to_method.keys()

        # figure out where does each conditioning go, between:
        #  - summed to the input of the lm (after embedding)
        #  - prepended to the input of the lm (after embedding)
        #  - in cross-attention of the lm
        method_to_embeddings: Dict[ConditioningMethod,
                                   List[EmbeddedCondition]] = {}

        # for condition_type, emb in embedded_conditions.items():
        for condition_type in sorted(embedded_conditions.keys()):
            emb = embedded_conditions[condition_type]

            # apply dropout to each condition
            emb = self.dropout(emb)

            method: ConditioningMethod = self.cond_to_method[condition_type]
            method_to_embeddings[method] = method_to_embeddings.get(method,
                                                                    []) + [emb]

        # has to return a dict that associates one embedded condition to each
        # conditioning method
        method_to_fused_embedding: Dict[ConditioningMethod,
                                        EmbeddedCondition] = {}

        # if multiple conditions are dispatched to the same method, we have to
        # fuse them
        for m, emb_list in method_to_embeddings.items():
            fused_embedding: EmbeddedCondition = self.method_to_fuser[m.value](
                emb_list)
            method_to_fused_embedding[m] = fused_embedding

        return method_to_fused_embedding


class CrossAttentionFuser(nn.Module):

    def __init__(self, n_conditions: int, embedding_dim: int):
        super().__init__()
        self.n_conditions: int = n_conditions
        self.embedding_dim: int = embedding_dim

        # use concatenation with segment embedding to merge conditions
        if self.n_conditions > 1:
            self.segment_embedding = nn.Embedding(self.n_conditions - 1,
                                                  self.embedding_dim)

    def forward(self, conds: Sequence[EmbeddedCondition]) -> EmbeddedCondition:
        if len(conds) == 0:
            raise RuntimeError("Received a list of 0 length. There is a bug.")
        if len(conds) == 1:
            return conds[0]

        # apply segment embedding
        if len(conds) == 2:
            assert self.n_conditions == 2
            assert conds[0].mask is not None and conds[1].mask is not None
            cond1_emb = conds[1].data
            cond1_ids = torch.zeros(cond1_emb.shape[0],
                                    cond1_emb.shape[1],
                                    dtype=torch.long,
                                    device=conds[1].data.device)
            cond1_segembed = self.segment_embedding(cond1_ids)
            cond1_emb = cond1_emb + cond1_segembed
            total_cond = torch.cat((conds[0].data, cond1_emb), dim=1)
            total_mask = torch.cat((conds[0].mask, conds[1].mask), dim=1)
            return EmbeddedCondition(total_cond, total_mask)

        # multiple conditions in cross-atteniton are not implemented
        raise NotImplementedError()


class PrependFuser(nn.Module):

    def __init__(self, n_conditions: int, embedding_dim: int):
        super().__init__()
        self.n_conditions: int = n_conditions
        self.embedding_dim: int = embedding_dim

    def forward(self, conds: Sequence[EmbeddedCondition]):
        if len(conds) == 0:
            raise RuntimeError("Received a list of 0 length. There is a bug.")
        if len(conds) == 1:
            return conds[0]

        # multiple conditions prepended to input are not implemented
        raise NotImplementedError()


class SumFuser(nn.Module):

    def __init__(self, n_conditions: int, embedding_dim: int):
        super().__init__()
        self.n_conditions: int = n_conditions
        self.embedding_dim: int = embedding_dim

    def forward(self, conds: Sequence[EmbeddedCondition]):
        if len(conds) == 0:
            raise RuntimeError("Received a list of 0 length. There is a bug.")
        if len(conds) == 1:
            return conds[0]

        # multiple conditions summed to input are not implemented
        raise NotImplementedError()


METHOD_TO_FUSER_TYPE: Dict[ConditioningMethod, Type[nn.Module]] = {
    ConditioningMethod.CROSS_ATTENTION: CrossAttentionFuser,
    ConditioningMethod.INPUT_PREPEND: PrependFuser,
    ConditioningMethod.INPUT_SUM: SumFuser,
}
