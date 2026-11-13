"""Small, genuine 2D U-Net. Forward returns logits; inference applies sigmoid."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        # GroupNorm works with small batches and has no running train/val statistics.
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(1, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(1, outgoing), nn.ReLU(inplace=True),
        )


class LightUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 8, depth: int = 3):
        super().__init__()
        if min(in_channels, base_channels, depth) < 1 or depth > 5:
            raise ValueError("Positive channels and depth in [1,5] required")
        self.depth = depth
        widths = [base_channels * 2**i for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths[:-1]:
            self.encoders.append(ConvBlock(previous, width))
            previous = width
        self.bottleneck = ConvBlock(widths[-2], widths[-1])
        self.decoders = nn.ModuleList(
            ConvBlock(widths[i + 1] + widths[i], widths[i])
            for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(widths[0], 1, 1)
        nn.init.constant_(self.head.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or min(x.shape[-2:]) < 2**self.depth:
            raise ValueError("Expected [B,C,H,W] with H,W >= 2**depth")
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.head(x)


class SegmentationLoss(nn.Module):
    """BCEWithLogits + per-image soft Dice, with optional padding mask.

    dice_weight=0 is plain BCE. No class weights are inferred from held-out data.
    """
    def __init__(self, dice_weight: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        if not 0 <= dice_weight <= 1 or pos_weight <= 0:
            raise ValueError("dice_weight in [0,1], pos_weight > 0 required")
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits, target, valid=None):
        if logits.shape != target.shape:
            raise ValueError("logits and target shapes differ")
        valid = torch.ones_like(target) if valid is None else valid
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        bce = (bce * valid).sum() / valid.sum().clamp_min(1)
        p = logits.sigmoid() * valid
        t = target * valid
        axes = (1, 2, 3)
        dice = (2 * (p * t).sum(axes) + 1e-6) / (p.sum(axes) + t.sum(axes) + 1e-6)
        # Ignore entirely padded images, rather than declaring them perfect.
        keep = valid.sum(axes) > 0
        dice_loss = (1 - dice[keep]).mean() if keep.any() else logits.sum() * 0
        return (1 - self.dice_weight) * bce + self.dice_weight * dice_loss
