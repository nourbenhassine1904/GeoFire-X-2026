"""Targeted artifact/original comparisons, including edge tiles and both roles."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from torch.utils.data import DataLoader
from geofirex.mission_streaming import CompactTiles,load,read_tile,padded_patch,save,peak_ram

def verify(corpus,prepared,root,out,reference=None):
    corpus=Path(corpus);_,_,_,n=load(prepared,root);checked=[]
    for role in ('train','val'):
        ds=CompactTiles(corpus,role);b=next(iter(DataLoader(ds,batch_size=4,num_workers=0)));assert b[0].shape[1:]==(3,8,64,64)
        indices={0,len(ds)-1,len(ds)//2}
        for category in ('positive','hard_negative','negative'):
            candidates=[i for i,s in enumerate(ds.samples) if s['category']==category]
            if candidates:indices.add(candidates[0])
        for i in sorted(indices):
            s=ds.samples[i];x,y,v=ds[i];raw_x=[]
            for d in s['context_dates']:
                raw,_=read_tile(Path(root)/s['event_id']/f'{d}.tif',s['row'],s['col'],64);raw_x.append(padded_patch(n.transform(raw),0,0,64))
            assert np.array_equal(x.numpy(),np.stack(raw_x))
            raw,valid=read_tile(Path(root)/s['event_id']/f"{s['target_date']}.tif",s['row'],s['col'],64)
            target=padded_patch(np.isfinite(raw[22:23]).astype('float32'),0,0,64)
            for fn in s['domain']:
                day=Path(fn.rsplit('!',1)[-1]).name[:10]
                _,mask=read_tile(Path(root)/s['event_id']/f'{day}.tif',s['row'],s['col'],64);valid*=mask
            assert np.array_equal(y.numpy(),target) and np.array_equal(v.numpy(),padded_patch(valid,0,0,64))
            checked.append(dict(role=role,index=i,event_id=s['event_id'],target_date=s['target_date'],category=s['category']))
    archive_members=0
    if reference:
        original=json.loads((Path(reference)/'shards.json').read_text());hashes={(s['event_id'],s['date'],s['row'],s['col']):s['sha256'] for s in original}
        records=json.loads((corpus/'shards.json').read_text());archive=None;prior=None
        for s in records:
            name,member=s['path'].split('!',1)
            if name!=prior:
                if archive:archive.close()
                archive=zipfile.ZipFile(corpus/name);prior=name
            value=hashlib.sha256(archive.read(member)).hexdigest()
            assert value==s['sha256']==hashes[(s['event_id'],s['date'],s['row'],s['col'])]
            archive_members+=1
        if archive:archive.close()
    result=dict(status='PASS',original_samples=checked,archive_members_identical_to_reference=archive_members,train_and_validation_loader=True,peak_rss_bytes_including_metadata_load=peak_ram(),comparison='float32 arrays and masks exactly equal; no tolerance')
    save(out,result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--prepared',default='reports/full_corpus_v1/prepared_spark_v1');p.add_argument('--root',default='D:/GeoFire-X/WildfireSpreadTS/rasters');p.add_argument('--out',required=True);p.add_argument('--reference');a=p.parse_args();verify(a.corpus,a.prepared,a.root,a.out,a.reference)
