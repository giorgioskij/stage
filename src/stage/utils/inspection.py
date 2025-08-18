from typing import Optional
from torch import Tensor, nn

MAXROWLEN = 50
N1 = 17
N2 = MAXROWLEN - N1 - 1


def print_params(model: nn.Module, depth: int = 0, summary: bool = False):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    name = model._get_name()
    s_header = f"{'MODEL:':<{N1}} {name:>{N2}}"
    if not summary:
        s_header += f"   DEVICE   GRAD   NAN?    SHAPE"
    print(s_header)

    print(f"{'TOTAL PARAMS:':<{N1}} {total_params:>{N2},}")
    print(f"{'TRAINABLE PARAMS:':<{N1}} {trainable_params:>{N2},}")
    print()

    _print_params(model, depth, summary, tabs=0)


def _print_params(model: nn.Module,
                  depth: int = 0,
                  summary: bool = False,
                  tabs=0):

    if depth == 0:
        if not summary:
            for name, p in model.named_parameters():
                if len(name) > MAXROWLEN:
                    name = f"...{name[-37:]}"
                print(f"{name:>{MAXROWLEN}}   {str(p.device):>6}   "
                      f"{'Grad' if p.requires_grad else '----'}   "
                      f"{'NaNs!' if p.isnan().any() else 'clean'}   "
                      f"{tuple(p.data.shape)}")
        return

    for childname, childmodule in model.named_children():
        if len(list(childmodule.parameters())) == 0:
            continue
        total_count = sum(p.numel() for p in childmodule.parameters())
        trainable_count = sum(
            p.numel() for p in childmodule.parameters() if p.requires_grad)
        print("  " * tabs + "-" * (len(childname) + 1))
        print("  " * tabs + f"{childname.upper()}:  {trainable_count:,} "
              f"trainable of {total_count:,}")
        print("  " * tabs + "-" * (len(childname) + 1))
        _print_params(childmodule, depth - 1, summary=summary, tabs=tabs + 1)
    return


def printshape(*x: Optional[Tensor]) -> None:
    for item in x:
        print(f"{list(item.shape) if item is not None else 'None'}")
    return


def sanity_check(model: nn.Module, interrupt: bool = True):
    clean: bool = True
    if not all(p.isfinite().all() for p in model.parameters()):
        print("Model has some inf or NaN parameters")
        clean = False

    if not all(p.grad.isfinite().all()
               for p in model.parameters()
               if p.requires_grad and p.grad is not None):
        print("Model has some inf or NaN gradients")
        clean = False

    if interrupt and not clean:
        raise RuntimeError("Model sanity check failed")

    return clean
