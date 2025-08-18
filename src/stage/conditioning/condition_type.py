from enum import Enum
from functools import total_ordering


class ConditionType(Enum):
    DESCRIPTION = "description"
    CONTEXT = "context"
    STYLE = "style"
    BEAT = "beat"

    def __lt__(self, other):

        order = ["DESCRIPTION", "BEAT", "CONTEXT", "STYLE"]
        if not isinstance(other, ConditionType):
            raise NotImplementedError()
        return order.index(self.name) < order.index(other.name)


if __name__ == "__main__":
    c = ConditionType("description")
    print(f'{c=}')
    print(f'{c.value=}')

    c2 = ConditionType("beat")

    print(c < c2)
