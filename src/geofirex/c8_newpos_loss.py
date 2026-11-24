"""Single scientific change: multiply new positive BCE terms by four."""
import torch
from torch.nn import functional as F
from geofirex.unet import SegmentationLoss


class NewPositiveLoss(SegmentationLoss):
    def __init__(self, new_positive_weight=4., dice_weight=.5, pos_weight=10.):
        super().__init__(dice_weight, pos_weight)
        if not torch.isfinite(torch.tensor(new_positive_weight)) or new_positive_weight < 1:
            raise ValueError('new_positive_weight must be finite and >=1')
        self.new_positive_weight = float(new_positive_weight)

    def forward(self, logits, target, valid, observed_J):
        if not logits.shape == target.shape == valid.shape == observed_J.shape:
            raise ValueError('Logits, target, valid and explicit J mask must match')
        if any(not torch.isin(a,torch.tensor([0,1],device=a.device)).all() for a in (target,valid,observed_J)):
            raise ValueError('Binary target/valid/observed_J required')
        base = super().forward(logits,target,valid)
        if self.new_positive_weight == 1.:
            return base  # bitwise value/gradient equivalence, including empty images
        bce = F.binary_cross_entropy_with_logits(logits,target,pos_weight=self.pos_weight,reduction='none')
        extra = (bce*valid*target*(observed_J==0)).sum()/valid.sum().clamp_min(1)
        return base+(1-self.dice_weight)*(self.new_positive_weight-1)*extra
