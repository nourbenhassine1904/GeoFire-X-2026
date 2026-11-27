"""One-factor C8 sampling ablation: replace easy negatives by fire-context negatives.

Every target-positive tile stays exactly once. No probability, checkpoint, new
feature or validation label is used to choose training exposures.
"""
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Sampler
from geofirex.fast_compact import LocalitySampler
from geofirex.protocol_v2 import make_v2_loaders


class PropHNSampler(Sampler):
    def __init__(self,dataset,seed=42,block_size=256):
        if dataset.role!='train':raise ValueError('Sampling is train-only')
        self.dataset=dataset;self.seed=seed;self.block_size=block_size;self.epoch=0
        r=dataset.records
        if not len(r) or not np.isin(r['category'],[0,1,2]).all():raise ValueError('Certified tile categories required')
        self.easy=np.flatnonzero(r['category']==0)
        self.positive=np.flatnonzero(r['category']==1)
        self.hard=np.flatnonzero(r['category']==2)
        if not len(self.hard) or not len(self.positive):raise ValueError('Both positive and hard-negative strata required')
        # Reserve one easy tile for each archive to preserve all training events.
        _,first=np.unique(r[self.easy]['archive'],return_index=True)
        reserve=self.easy[first]
        self.removable=np.setdiff1d(self.easy,reserve,assume_unique=True)
        if len(self.removable)<len(self.hard):raise ValueError('Insufficient removable easy negatives')

    def __len__(self):return len(self.dataset)

    def order(self):
        if self.epoch not in range(4):raise ValueError('Only four registered epochs')
        rng=np.random.default_rng(self.seed+self.epoch)
        dropped=rng.choice(self.removable,size=len(self.hard),replace=False)
        easy=np.setdiff1d(self.easy,dropped,assume_unique=True)
        selected=np.concatenate((easy,self.positive,self.hard,self.hard))
        # Same locality ordering primitive, including block size and epoch seed.
        base=LocalitySampler(self.dataset,self.seed,self.block_size);base.epoch=self.epoch
        inverse=np.empty(len(self),dtype=np.int64);inverse[base.order()]=np.arange(len(self))
        return selected[np.argsort(inverse[selected],kind='stable')]

    def __iter__(self):return iter(self.order().tolist())

    def receipt(self):
        order=self.order();r=self.dataset.records;counts=np.bincount(order,minlength=len(self))
        if (not np.all(counts[self.positive]==1) or not np.all(counts[self.hard]==2)
                or not np.isin(counts[self.easy],[0,1]).all() or len(order)!=len(self)):
            raise ValueError('Sampling exposure contract differs')
        actual,expected=set(r[order]['archive'].tolist()),set(r['archive'].tolist())
        if actual!=expected:raise ValueError('A training event was omitted')
        return dict(epoch=self.epoch,draws=len(order),unique_tiles=int((counts>0).sum()),
            repeated_draws=int((counts-1).clip(min=0).sum()),omitted_easy_tiles=int((counts==0).sum()),
            positive_draws=int(counts[self.positive].sum()),hard_negative_draws=int(counts[self.hard].sum()),
            easy_negative_draws=int(counts[self.easy].sum()),events=len(actual),
            order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            positive_and_new_positive_pixel_exposures='Identical to C8: every target-positive tile exactly once',
            no_validation_or_model_scores_used=True)


def make_prop_loaders(corpus,index,*,mounts=None,batch_size=16,workers=0):
    train,val=make_v2_loaders(corpus,index,mounts=mounts,batch_size=batch_size,workers=workers)
    sampler=PropHNSampler(train.dataset)
    options=dict(batch_size=batch_size,num_workers=workers,drop_last=False,pin_memory=False,
                 prefetch_factor=2 if workers else None)
    return DataLoader(train.dataset,sampler=sampler,**options),val
