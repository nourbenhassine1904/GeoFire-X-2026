"""Versioned nearest-neighbor decoders for deterministic CUDA training.

Historical bilinear models and checkpoints keep their original implementation.
Nearest interpolation avoids the nondeterministic CUDA bilinear backward.
"""
import torch
from torch.nn import functional as F

from geofirex.benchmark_models import ForecastModel, StackedUNet, ConvLSTM
from geofirex.unet import LightUNet


class NearestLightUNet(LightUNet):
    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 2**self.depth:
            raise ValueError("Expected [B,C,H,W] with H,W >= 2**depth")
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat((skip, x), dim=1))
        return self.head(x)


class NearestStackedUNet(StackedUNet):
    def __init__(self, channels, context, base_channels=8, depth=3):
        ForecastModel.__init__(self, channels, context)
        self.network = NearestLightUNet(channels*context, base_channels, depth)


class NearestConvLSTM(ConvLSTM):
    def forward(self, x):
        self.validate(x)
        hidden = cell = None
        for step in x.unbind(1):
            encoded = self.encoder(step)
            if hidden is None:
                hidden = torch.zeros_like(encoded)
                cell = torch.zeros_like(encoded)
            i, f, o, g = self.gates(torch.cat((encoded, hidden), dim=1)).chunk(4, dim=1)
            cell = f.sigmoid()*cell + i.sigmoid()*g.tanh()
            hidden = o.sigmoid()*cell.tanh()
        return F.interpolate(self.head(hidden), size=x.shape[-2:], mode="nearest")
