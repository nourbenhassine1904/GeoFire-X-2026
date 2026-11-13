"""CPU-only certificate against the pinned private publication objects."""
import hashlib
import json
from pathlib import Path
import time
import tempfile
import zipfile
import numpy as np
import torch
from compact_publication import restore
from geofirex.fast_compact import FastCompactTiles, diagnostic_indices
from geofirex.mission_streaming import CompactTiles, save, peak_ram


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(expected, workspace):
    start=time.perf_counter();torch.set_num_threads(2)
    workspace=Path(workspace)
    result=dict(status='CPU_CERTIFICATION_STARTED',gpu_used=False)
    try:
        maps=list(Path('/kaggle/input').rglob('mount-map.json'))
        actual={json.loads(p.read_text())['identity']:p for p in maps}
        assert len(maps)==len(actual)==len(expected)==11
        assert set(actual)=={p['identity'] for p in expected}
        partitions=[]
        for part in expected:
            mount=actual[part['identity']].parent
            assert {p.name for p in mount.iterdir() if p.is_file()}==set(part['files'])
            for name,row in part['files'].items():
                p=mount/name
                assert p.stat().st_size==row['bytes'],name
                assert sha(p)==row['sha256'],name
            partitions.append(dict(handle=part['handle'],identity=part['identity'],
                mounted_path=str(mount),files=len(part['files']),all_sizes_sha256_match=True))
            print('CPU SHA256 PASS',part['handle'],flush=True)
        # Keep dataset links out of /kaggle/working so kernel outputs contain
        # certificates rather than another materialization of the 9.85 GB corpus.
        temporary=Path(tempfile.mkdtemp(prefix='geofirex-compact-',dir='/tmp'))
        corpus=restore([p.parent for p in actual.values()],temporary/'corpus',verify=False)
        index=corpus/'index'
        meta=json.loads((index/'index.json').read_text())
        assert len(meta['archives'])==len(set(meta['archives']))==451
        assert len(set(meta['events'].values()))==451
        assert sha(corpus/'samples.json')==meta['source_samples_sha256']
        assert sha(corpus/'normalization.json')==meta['normalization_sha256']
        sets={};counts={};samples=[];positive=0
        for role,n_events,n_tiles,years in [('train',250,93603,{'2018','2019'}),('val',201,78771,{'2020'})]:
            ds=FastCompactTiles(corpus,index,role,cache_mb=16,zip_handles=2)
            codes,first=np.unique(ds.records['archive'],return_index=True)
            sets[role]={ds.event(int(i)) for i in first}
            assert len(ds)==n_tiles and len(codes)==len(sets[role])==n_events
            assert {e[:4] for e in sets[role]}<=years
            for i in first:
                x,y,v=ds[int(i)]
                assert x.shape==(3,8,64,64) and y.shape==(1,64,64) and v.shape==(1,64,64)
                assert torch.isfinite(x).all() and torch.isfinite(y).all()
                assert ((y==0)|(y==1)).all() and ((v==0)|(v==1)).all()
            # Check every archive central directory without expanding archive data.
            for code in codes:
                with zipfile.ZipFile(ds.paths[int(code)]) as z:
                    assert z.namelist() and all(n.endswith('.npz') for n in z.namelist())
            if role=='val':
                diagnostic=diagnostic_indices(ds,12)
                positive=sum(int((ds[i][1]*ds[i][2]).sum()) for i in diagnostic)
                assert positive>0
            # Compatibility with original CompactTiles, using exact real tensors.
            old=CompactTiles(corpus,role)
            assert len(old)==len(ds)
            for i in sorted(set([0,len(ds)//2,len(ds)-1])):
                assert all(torch.equal(a,b) for a,b in zip(old[i],ds[i]))
                samples.append(dict(role=role,index=i,event=ds.event(i)))
            del old
            counts[role]=dict(events=len(sets[role]),tiles=len(ds),real_samples_checked=len(first))
            ds.close()
        assert not sets['train']&sets['val']
        assert len(sets['train']|sets['val'])==451
        assert not any(e.startswith('2021/') for e in sets['train']|sets['val'])
        normalization=json.loads((corpus/'normalization.json').read_text())
        fit_events={'/'.join(p.replace('\\','/').split('/')[-3:-1]) for p in normalization['fit_paths']}
        assert fit_events and fit_events<=sets['train']
        report=json.loads((corpus/'report.json').read_text())
        assert report['test_pixels_opened']==0
        result.update(status='PASS',partitions=partitions,roles=counts,
            objects_verified=sum(len(p['files']) for p in expected),archives_opened=451,
            real_pytorch_event_samples=451,legacy_fast_tensor_comparisons=samples,
            validation_diagnostic_positive_pixels=positive,
            train_val_disjoint=True,test_2021_absent=True,T=3,target='J+1',seed=42,
            normalization_sha256=meta['normalization_sha256'],normalizer_refitted=False,
            normalizer_train_only=True,normalizer_fit_events=len(fit_events),
            temporal_metadata_hash_verified=True,archives_rebuilt=False,
            paths_reconstructed_by_verified_links=True)
    except Exception as e:
        result.update(status='FAIL',error_type=type(e).__name__,error=str(e)[:200])
    result.update(seconds=time.perf_counter()-start,peak_parent_rss_bytes=peak_ram())
    save(workspace/'certificate.json',result)
    print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise RuntimeError('CPU compact certification failed')
    return result
