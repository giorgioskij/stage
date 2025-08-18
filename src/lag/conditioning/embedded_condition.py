from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass(repr=False)
class EmbeddedCondition:
    data: Tensor
    mask: Optional[Tensor]

    def __repr__(self) -> str:
        return (f"data: {list(self.data.shape)}; mask: "
                f"{list(self.mask.shape) if self.mask is not None else 'None'}")
