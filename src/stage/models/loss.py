import torch
from torch import Tensor
from typing import Tuple, List
from torch.nn import functional as F


def compute_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute cross entropy between multi-codebook targets and model's logits.
    The cross entropy is computed per codebook to provide codebook-level cross entropy.
    Valid timesteps for each of the codebook are pulled from the mask, where invalid
    timesteps are set to 0.

    Args:
        logits (torch.Tensor): Model's logits of shape [B, K, T, card].
        targets (torch.Tensor): Target codes, of shape [B, K, T].
        mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
    Returns:
        ce (torch.Tensor): Cross entropy averaged over the codebooks
        ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
    """
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    ce = torch.zeros([], device=targets.device)
    ce_per_codebook: List[Tensor] = []
    for k in range(K):
        logits_k = (logits[:, k, ...].contiguous().view(-1, logits.size(-1))
                   )  # [B x T, card]
        targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]

        # if the codebook is masked out, the loss is 0
        if mask_k.sum() == 0:
            q_ce = torch.tensor(0.0, device=targets.device)
        else:
            q_ce = F.cross_entropy(ce_logits, ce_targets)

        ce += q_ce
        ce_per_codebook.append(q_ce.detach())
    # average cross entropy across codebooks
    ce = ce / K
    return ce, ce_per_codebook
