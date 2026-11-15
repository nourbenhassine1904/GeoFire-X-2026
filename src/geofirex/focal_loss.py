"""Isolated binary Focal loss on logits; unused by the production Studio."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalWithLogits(nn.Module):
    """Sum over valid pixels / valid count, including fire-free images.

    Positive/negative weights are 10/1, equivalent to alpha=10/11 scaled by
    11. No second alpha factor, Dice term, probability clamp or batch balancing.
    gamma=0 is weighted BCE. An entirely invalid batch has zero loss/gradient.
    Invalid padding is removed BEFORE arithmetic, including NaN padding.
    """
    def __init__(self, gamma=2., positive_weight=10., negative_weight=1.):
        super().__init__()
        if (not all(math.isfinite(v) for v in (gamma, positive_weight, negative_weight))
                or gamma < 0 or min(positive_weight, negative_weight) <= 0):
            raise ValueError("Finite gamma>=0 and positive class weights required")
        self.gamma = float(gamma)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)

    def forward(self, logits, target, valid=None):
        if logits.shape != target.shape or not logits.is_floating_point():
            raise ValueError("Matching logits/target shapes and floating logits required")
        if valid is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            if valid.shape != logits.shape or not torch.all((valid == 0) | (valid == 1)):
                raise ValueError("Matching binary validity mask required")
            mask = valid.bool()
        z, y = logits[mask], target[mask]
        if not torch.isfinite(z).all() or not torch.all((y == 0) | (y == 1)):
            raise ValueError("Finite valid logits and binary valid targets required")
        # Preserve float64 for numerical gradient tests; promote low precision.
        if z.dtype in (torch.float16, torch.bfloat16):
            z = z.float()
        if z.numel() == 0:
            return z.sum()
        positive = y.bool()
        signed = torch.where(positive, z, -z)
        weight = torch.where(positive, self.positive_weight, self.negative_weight)
        # sigmoid(-signed) avoids cancellation in 1-sigmoid(signed).
        return (weight * torch.sigmoid(-signed).pow(self.gamma)
                * F.softplus(-signed)).mean()
