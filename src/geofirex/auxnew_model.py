"""One untrained C8 auxiliary-task candidate; primary head is the sole forecast."""
import torch
from torch import nn
from torch.nn import functional as F
from geofirex.propagation_experiment import PropagationUNet
from geofirex.unet import SegmentationLoss


class AuxiliaryUNet(PropagationUNet):
    """Reuse pinned C8 decoder and initialization, add only nine auxiliary weights."""
    def forward(self,x):
        return self.heads(x)[0]


class AuxiliaryLoss(nn.Module):
    def __init__(self,coefficient=.25):
        super().__init__()
        if coefficient not in (0.,.25):raise ValueError('Only fixed .25 or CPU parity control 0')
        self.coefficient=coefficient
        self.primary=SegmentationLoss(.5,10.)

    def forward(self,main,aux,target,valid,observed):
        if not (main.shape==aux.shape==target.shape==valid.shape==observed.shape):
            raise ValueError('Exact head/target/mask shapes required')
        if not all(torch.isin(a,torch.tensor([0.,1.],device=a.device)).all() for a in (target,valid,observed)):
            raise ValueError('Binary targets, validity and observed-J fire required')
        primary=self.primary(main,target,valid)
        if self.coefficient==0:return primary,primary,aux.sum()*0
        mask=valid*(observed==0)
        auxiliary=F.binary_cross_entropy_with_logits(aux,target,pos_weight=self.primary.pos_weight,reduction='none')
        auxiliary=(auxiliary*mask).sum()/mask.sum().clamp_min(1)
        return primary+self.coefficient*auxiliary,primary,auxiliary
