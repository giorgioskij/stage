from enum import Enum


class ConditioningMethod(Enum):

    INPUT_PREPEND = "input_prepend"
    INPUT_SUM = "input_sum"
    CROSS_ATTENTION = "cross_attention"


if __name__ == "__main__":
    c = ConditioningMethod.INPUT_PREPEND
    print(f'{c=}')
    print(f'{c.value=}')
