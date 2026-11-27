"""Targeted ADATA inventory, cost pilot and CPU qualification. No GPU/download."""
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
from geofirex import v2_execution as old
from geofirex.protocol_v2 import V2CompactTiles,sha256,save,sample_keys
from geofirex.fire13_features import Fire13Tiles,fit_moments

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fire13_20261122_v1'
RAW=Path('D:/GeoFire-X/WildfireSpreadTS/rasters')


def inventory():
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.windows import Window
    from geofirex.v2_c36_overlay import require_exact_geometry
    OUT.mkdir(exist_ok=False)
    base,lock,_=old.identity(ROOT)
    # Freeze existing scientific deliverables, including the untouched fallback.
    files=dict(lock['files'])
    for folder in ('src','scripts','configs','notebooks','splits'):
        files.update({p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT/folder).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and 'fire13' not in p.name.lower()})
    for folder in ('reports/prop_hn2_20261122_v1','reports/newpos2_20261122_v1'):
        files.update({p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT/folder).rglob('*') if p.is_file()})
    save(OUT/'preservation_before.json',dict(files=files,scientific_identities=77))
    rows=list(csv.DictReader((ROOT/'reports/c36_cpu_20261121_v1/geometry_headers.csv').open(encoding='utf8')))
    obs={r['relative_path']:r for r in pq.read_table(ROOT/'reports/full_corpus_v1/prepared_v1/observations.parquet').to_pylist()}
    geometry={};samples=[];total=0
    for row in rows:
        rel=Path(row['relative_path']);r=obs[row['relative_path']]
        if row['role'] not in ('train','val') or rel.parts[0]=='2021' or row['status']!='PASS':raise ValueError('Only certified development paths')
        p=(RAW/rel).resolve()
        if not p.is_relative_to(RAW.resolve()) or p.stat().st_size!=r['bytes']:raise ValueError('Raw path/size changed')
        total+=p.stat().st_size
        g=dict(height=r['height'],width=r['width'],crs=r['crs'],transform=r['transform'])
        event=r['event_id']
        if event in geometry:require_exact_geometry(g,geometry[event]['header'])
        else:geometry[event]=dict(shape=[r['height'],r['width']],header=g)
        if row['role'] not in [s['role'] for s in samples]:
            with rasterio.open(p) as src:
                require_exact_geometry(g,dict(height=src.height,width=src.width,crs=str(src.crs),transform=list(src.transform)))
                a=src.read(window=Window(0,0,min(64,src.width),min(64,src.height)))
                if src.count!=23:raise ValueError('23 raw bands required')
                samples.append(dict(role=row['role'],path=str(p),sha256=sha256(p),count=src.count,
                    descriptions=list(src.descriptions),tags=src.tags(),nodata=src.nodata,
                    scales=list(src.scales),offsets=list(src.offsets),units=list(src.units),
                    sampled_pixels_per_band=a.shape[1]*a.shape[2],
                    finite=[int(np.isfinite(v).sum()) for v in a],nan=[int(np.isnan(v).sum()) for v in a],
                    note='Two top-left windows only; not global missingness estimates'))
    if len(rows)!=6913 or len(geometry)!=398:raise ValueError('Frozen input/event coverage differs')
    maps=[ROOT/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
    mounts,mount_receipt=old.verify_mounts(ROOT,maps)
    save(OUT/'scene_geometry.json',geometry)
    save(OUT/'inventory.json',dict(status='PASS',TIFFs_stat_verified=len(rows),raw_bytes=total,
        existing_header_certificate_sha256=sha256(ROOT/'reports/c36_cpu_20261121_v1/geometry_headers.json'),
        raw_header_audit_repeated=False,all_development_geometry_metadata_checked=True,geometry_events=len(geometry),
        compact=mount_receipt,raw_samples=samples,test_pixels_opened=0,downloads=0,copied_corpus_bytes=0))
    matrix=list(csv.DictReader((ROOT/'reports/scientific_improvements_v1/feature_availability_matrix.csv').open(encoding='utf8')))
    save(OUT/'raw_band_decisions.json',dict(bands=matrix,external_additions_certified=[],
        reason='No new reviewed asset/version/acquisition/composite/issue/publication/availability dossier',
        existing_C8_raw_indices=[12,13,14,22],
        fire13_admission='Derived from accepted C8 observations only; inherited operational latency assumption, not a new real-time certification'))
    return base,mounts,geometry


def pilot(base,mounts,geometry):
    from geofirex.benchmark_models import StackedUNet
    import torch
    torch.set_num_threads(2)
    ds=Fire13Tiles(V2CompactTiles('',ROOT/base['index'],'train',mounts=mounts),geometry,
        dict(mean=[0.,0.],std=[1.,1.]))
    try:
        # Consecutive LocalitySampler presentations, not a score-selected subset.
        from geofirex.fast_compact import LocalitySampler
        indices=LocalitySampler(ds,seed=42).order()[:256]
        start=time.monotonic()
        for i in indices:sample=ds[int(i)]
        seconds=time.monotonic()-start
        model=StackedUNet(13,3);x,y,v=[t.unsqueeze(0) for t in sample]
        before=time.monotonic();logits=model(x)
        loss=__import__('geofirex.unet',fromlist=['SegmentationLoss']).SegmentationLoss(.5,10)(logits,y,v)
        loss.backward();cpu_step=time.monotonic()-before
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Finite CPU model/gradients required')
        if sum(p.numel() for p in model.parameters())!=125057:raise ValueError('Mechanical input-layer parameter change differs')
        max_scene=max(np.prod(g['shape']) for g in geometry.values())
        receipt=dict(status='PASS',pilot_tiles=256,loader_seconds=seconds,CPU_single_image_step_seconds=cpu_step,
            loader_seconds_linear_four_epochs_estimate=seconds/256*115900*4,
            estimate_is_not_measured_full_fit=True,parameters=125057,baseline_parameters=123977,
            input_channels_per_day=13,stacked_channels=39,first_conv_added_parameters=1080,
            descriptor_cache_bytes=ds.budget,base_cache_bytes=ds.base.budget,
            max_scene_pixels=int(max_scene),max_daily_scene_maps_bytes=int(max_scene*11),
            float32_input_batch_bytes=16*3*13*64*64*4,baseline_input_batch_bytes=16*3*8*64*64*4,
            additional_corpus_storage_bytes=0,overlay_required=False,scipy_available=True,
            GPU_measured=False,test_pixels_opened=0)
        if seconds>120:raise RuntimeError('Pilot too slow for immediate full CPU preparation')
        save(OUT/'cost_pilot.json',receipt)
    finally:ds.close()


def qualify(base,mounts,geometry):
    import torch
    torch.set_num_threads(2)
    ds=Fire13Tiles(V2CompactTiles('',ROOT/base['index'],'train',mounts=mounts),geometry)
    try:stats=fit_moments(ds)
    finally:ds.close()
    stats.update(train_index_sha256=sha256(ROOT/base['index']/'train.npy'),
        geometry_sha256=sha256(OUT/'scene_geometry.json'),base_normalization_sha256=base['normalization_sha256'])
    save(OUT/'derived_normalization.json',stats)
    result={};started=time.monotonic()
    from geofirex.fast_compact import LocalitySampler
    for role in ('train','val'):
        ds=Fire13Tiles(V2CompactTiles('',ROOT/base['index'],role,mounts=mounts),geometry,stats)
        domain={};support=0;pixels=0;stage=time.monotonic()
        try:
            keys=sample_keys(ds)
            # Match locality access while digesting canonical keys, not order.
            for j,i in enumerate(LocalitySampler(ds).order()):
                if time.monotonic()-started>2400:raise TimeoutError('CPU exhaustive parity cap exhausted')
                i=int(i);bx,by,bv=ds.base[i];cx,cy,cv=ds[i]
                if not (np.array_equal(bx.numpy(),cx[:,:8].numpy()) and torch_equal(by,cy) and torch_equal(bv,cv)):
                    raise ValueError('C8 target/mask/shared-feature parity differs')
                if cx.shape!=(3,13,64,64) or not np.isfinite(cx.numpy()).all():raise ValueError('FIRE13 schema differs')
                h=hashlib.sha256(json.dumps([keys[i],[64,64]]).encode())
                for a in (by,bv,bx[-1,7:8]):h.update(a.numpy().astype(bool).tobytes())
                domain[keys[i]]=h.hexdigest();support+=int(cx[-1,12].sum());pixels+=4096
                if (j+1)%8192==0:print('EXHAUSTIVE PARITY',role,j+1,'/',len(ds),flush=True)
            digest=hashlib.sha256(json.dumps(sorted(domain.items())).encode()).hexdigest()
            if role=='val' and digest!='efc995be596d7d520bac472ceb1b5122fe2111fe9d5a53c5edbec02851c1accc':
                raise ValueError('Exact official validation domain differs')
            result[role]=dict(tiles=len(ds),events=len({k[0] for k in keys}),evaluation_domain_sha256=digest,
                derived_supported_pixels=support,padded_pixels=pixels,seconds=time.monotonic()-stage)
        finally:ds.close()
    save(OUT/'parity_full.json',dict(status='PASS',roles=result,seconds=time.monotonic()-started,
        source='all frozen train and validation keys; no balancing; exact y/valid/observed_J/C8 equality',test_pixels_opened=0))


def torch_equal(a,b):return bool(__import__('torch').equal(a,b))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['inventory','qualify'])
    stage=p.parse_args().stage
    if stage=='inventory':
        b,m,g=inventory();pilot(b,m,g)
    else:
        b,_,_=old.identity(ROOT)
        maps=[ROOT/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
        m,_=old.verify_mounts(ROOT,maps);g=old.read(OUT/'scene_geometry.json');qualify(b,m,g)
