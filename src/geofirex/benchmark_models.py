"""Untrained benchmark prototypes sharing [B,T,C,H,W] -> [B,1,H,W] logits.

B0 overrides predict_proba for exact binary Persistence. B1-B3 are trainable.
No performance or computational feasibility claim follows from a forward test.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from geofirex.unet import LightUNet


class ForecastModel(nn.Module):
    def __init__(self, channels, context):
        super().__init__()
        if channels < 1 or context < 1:
            raise ValueError("Positive channel and context counts required")
        self.channels, self.context = channels, context

    def validate(self, x):
        if x.ndim != 5 or x.shape[1:3] != (self.context,self.channels):
            raise ValueError("Expected [B,T,C,H,W] matching the model configuration")

    def predict_proba(self, x):
        return self(x).sigmoid()


class Persistence(ForecastModel):
    def __init__(self, channels, context, fire_channel=7):
        super().__init__(channels,context)
        if not 0 <= fire_channel < channels:
            raise ValueError("Invalid fire channel")
        self.fire_channel=fire_channel

    def predict_proba(self,x):
        self.validate(x)
        return x[:,-1,self.fire_channel:self.fire_channel+1]

    def forward(self,x):
        return torch.logit(self.predict_proba(x).clamp(1e-6,1-1e-6))


class ConvLogistic(ForecastModel):
    def __init__(self,channels,context):
        super().__init__(channels,context)
        self.classifier=nn.Conv2d(channels*context,1,3,padding=1)

    def forward(self,x):
        self.validate(x)
        return self.classifier(x.flatten(1,2))


class StackedUNet(ForecastModel):
    def __init__(self,channels,context,base_channels=8,depth=3):
        super().__init__(channels,context)
        self.network=LightUNet(channels*context,base_channels,depth)

    def forward(self,x):
        self.validate(x)
        return self.network(x.flatten(1,2))


class SpatialCNN(ForecastModel):
    """Versioned three-layer CNN. Legacy single-layer B1 remains loadable."""
    def __init__(self, channels, context, width=16):
        super().__init__(channels, context)
        self.network = nn.Sequential(nn.Conv2d(channels*context, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, 1, 1))

    def forward(self, x):
        self.validate(x)
        return self.network(x.flatten(1, 2))


class ConvLSTM(ForecastModel):
    """Spatial encoder -> ConvLSTM gates -> upsampled one-channel decoder."""
    def __init__(self,channels,context,hidden_channels=8):
        super().__init__(channels,context)
        self.hidden=hidden_channels
        self.encoder=nn.Sequential(nn.Conv2d(channels,hidden_channels,3,stride=2,padding=1),nn.ReLU())
        self.gates=nn.Conv2d(2*hidden_channels,4*hidden_channels,3,padding=1)
        self.head=nn.Conv2d(hidden_channels,1,3,padding=1)

    def forward(self,x):
        self.validate(x)
        hidden=cell=None
        for step in x.unbind(1):
            encoded=self.encoder(step)
            if hidden is None:
                hidden=torch.zeros_like(encoded); cell=torch.zeros_like(encoded)
            i,f,o,g=self.gates(torch.cat((encoded,hidden),dim=1)).chunk(4,dim=1)
            cell=f.sigmoid()*cell+i.sigmoid()*g.tanh()
            hidden=o.sigmoid()*cell.tanh()
        return F.interpolate(self.head(hidden),size=x.shape[-2:],mode="bilinear",align_corners=False)


def make_model(name,channels=8,context=1,**kwargs):
    if name == "B1":
        architecture = kwargs.pop("architecture", "legacy_conv_v1")
        if architecture == "spatial_cnn_v2":
            return SpatialCNN(channels, context, **kwargs)
        if architecture != "legacy_conv_v1":
            raise ValueError("Unknown B1 architecture version")
    factories={"B0":Persistence,"B1":ConvLogistic,"B2":StackedUNet,"B3":ConvLSTM}
    if name == "B4":
        raise NotImplementedError("Temporal U-Net/UTAE deferred until resource pilot and architecture review")
    if name not in factories:
        raise ValueError(f"Unknown model: {name}")
    return factories[name](channels,context,**kwargs)
