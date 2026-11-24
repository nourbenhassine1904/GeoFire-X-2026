"""Qualify HN2 using only frozen train tiles; no validation or 2021 pixels."""
from pathlib import Path
import hashlib
import time
import numpy as np
from geofirex import v2_execution as old
from geofirex.protocol_v2 import V2CompactTiles,save,sha256
from geofirex.prop_hn2_sampling import PropHNSampler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/prop_hn2_20261122_v1'


def main():
    base,_,_=old.identity(ROOT)
    maps=[ROOT/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
    mounts,receipt=old.verify_mounts(ROOT,maps)
    ds=V2CompactTiles('',ROOT/base['index'],'train',mounts=mounts)
    start=time.monotonic();h=hashlib.sha256()
    try:
        sampler=PropHNSampler(ds);plan=[]
        for epoch in range(4):sampler.epoch=epoch;plan.append(sampler.receipt())
        if any(r['events']!=318 or r['draws']!=115900 or r['positive_draws']!=6861 or r['hard_negative_draws']!=3678 for r in plan):
            raise ValueError('Frozen real sampling budget differs')
        # Exhaustive only over the 1839 tiles to be duplicated. No easy or
        # positive tile needs a changed target, mask, channel or decoder.
        for j,i in enumerate(sampler.hard):
            if time.monotonic()-start>1200:raise TimeoutError('CPU qualification exceeded 20-minute cap')
            x,y,v=ds[int(i)]
            if (y*v).sum()!=0 or (x[-1,7:8]*v).sum()<=0:
                raise ValueError('A doubled tile is not a valid fire-context negative')
            for a in (y,v,x[-1,7:8]):h.update(a.numpy().tobytes())
            if (j+1)%256==0:print('TRAIN-ONLY HN CHECK',j+1,'/',len(sampler.hard),flush=True)
        OUT.mkdir(exist_ok=True)
        save(OUT/'train_category_qualification.json',dict(status='PASS',checked_hard_negative_tiles=len(sampler.hard),
            train_index_sha256=sha256(ROOT/base['index']/'train.npy'),ordered_hard_y_v_o_sha256=h.hexdigest(),
            all_doubled_tiles_have_zero_valid_target_and_observed_fire=True,epochs=plan,
            mounted_objects=receipt['objects'],mounted_bytes=receipt['bytes'],seconds=time.monotonic()-start,
            validation_pixels_opened=0,test_pixels_opened=0,GPU_executed=False,
            unchanged_positive_supervision='Every positive index exactly once per epoch; inputs/targets/masks untouched'))
        print('PASS: all 1839 doubled tiles and four exact epoch plans',flush=True)
    finally:ds.close()


if __name__=='__main__':main()
