"""Private CPU certification of one frozen lot 002..008; no network or training."""
import datetime as dt
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time


def run(reference, observations, normalization, split, helpers, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    deadline=started+1800
    import psutil
    proc=psutil.Process()
    stop=threading.Event()
    resources=dict(peak_rss_bytes=0,minimum_available_bytes=psutil.virtual_memory().available)
    def guard():
        rss=proc.memory_info().rss
        available=psutil.virtual_memory().available
        resources['peak_rss_bytes']=max(resources['peak_rss_bytes'],rss)
        resources['minimum_available_bytes']=min(resources['minimum_available_bytes'],available)
        if time.monotonic()>deadline: raise TimeoutError('1800 second certification deadline')
        if rss>3*1024**3 or available<2*1024**3: raise MemoryError('CPU certification reserve/limit reached')
    def watch():
        while not stop.wait(1):
            try: guard()
            except Exception as error:
                (output/'resource-stop.json').write_text(json.dumps(dict(error=str(error),**resources)))
                os._exit(75)
    watcher=threading.Thread(target=watch,daemon=True)
    watcher.start()
    try:
        guard()
        if reference['number'] not in range(2,9) or reference['role'] not in ('train','val'):
            raise ValueError('Only frozen lots 002..008 allowed; lot 001 is closed')
        if os.environ.get('KAGGLE_ACCELERATOR_TYPE','').lower() not in ('','none','cpu') or list(Path('/dev').glob('nvidia[0-9]*')):
            raise RuntimeError('CPU-only session required')
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        role=reference['role']
        permitted=(2018,2019) if role=='train' else (2020,)
        if any(split.get(e)!=role or int(e.split('/')[0]) not in permitted for e in reference['events']):
            raise ValueError('Event role/year mismatch')
        for f in reference['files']:
            if dt.date.fromisoformat(PurePosixPath(f['path']).stem).year not in permitted:
                raise ValueError('Observation outside lot role years: '+f['path'])
        inventories=list(Path('/kaggle/input').rglob('lot-inventory.json'))
        if len(inventories)!=1: raise ValueError('Exactly one data lot must be attached')
        root=inventories[0].parent
        attached=[p for p in Path('/kaggle/input').rglob('*') if p.is_file()]
        if len(attached)!=reference['observations']+1 or any('2021' in p.parts or p.stem.startswith('2021-') for p in attached):
            raise ValueError('Unexpected attached files or held-out 2021 data')
        canonical=output/'canonical.json'
        canonical.write_text(json.dumps(reference))
        proof=helpers['certify_lot'](canonical,root,output/'hashes',pinned_identity=reference['identity'],seconds=1700,inspect_geotiff=False)
        guard()
        import numpy as np
        import rasterio
        import torch
        from rasterio.windows import Window
        from torch.utils.data import DataLoader
        from geofirex.unet_data import Normalizer,validate_layout
        from geofirex.benchmark import aligned_windows
        from geofirex.runner_data import FoldData,TemporalStore,TrainTiles
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        rasters=root/'rasters'
        resolver=FoldData.__new__(FoldData)
        resolver.config={'corpus_scope':'train_validation'}
        resolver.root=rasters.resolve()
        resolver.roots=[rasters.resolve()]
        expected={f['path'] for f in reference['files']}
        if {r['relative_path'] for r in observations}!=expected: raise ValueError('QC observation coverage mismatch')
        for name in expected:
            if resolver.resolve_path(name)!=(rasters/name).resolve(): raise ValueError('Loader path resolution mismatch')
        samples=[]
        loader_reads=[]
        with rasterio.Env(GDAL_CACHEMAX=64*1024**2,GDAL_NUM_THREADS='1'):
            for i in (0,len(reference['files'])//2,-1):
                guard()
                name=reference['files'][i]['path']
                with rasterio.open(rasters/name) as src:
                    validate_layout(src)
                    a=src.read(window=Window(0,0,min(64,src.width),min(64,src.height)))
                    if src.crs is None or a.shape[0]!=23 or np.isinf(a).any(): raise ValueError('GeoTIFF layout/read invalid')
                    samples.append(dict(path=name,bands=src.count,width=src.width,height=src.height,crs=str(src.crs),tile_shape=list(a.shape)))
            relocated=[{**r,'relative_path':str(resolver.resolve_path(r['relative_path']))} for r in observations]
            groups,rejected=aligned_windows(relocated,split,role,contexts=(1,3,5))
            norm=Normalizer(**normalization['normalization'])
            for context in (1,3,5):
                guard()
                if not groups[context]: raise ValueError('No valid temporal window')
                window=groups[context][0]
                maximum=groups[5][0]
                geometry=None
                for p in (*maximum.context_paths,maximum.target_path):
                    with rasterio.open(p) as src:
                        validate_layout(src)
                        current=(src.shape,str(src.crs),tuple(src.transform))
                        if geometry is not None and current!=geometry: raise ValueError('Temporal geometry mismatch')
                        geometry=current
                store=TemporalStore([window],norm,domain_windows=[maximum],cache_bytes=0)
                dataset=TrainTiles(store,size=64,samples=1,seed=42)
                x,y,valid=next(iter(DataLoader(dataset,batch_size=1,num_workers=0,pin_memory=False)))
                if list(x.shape)!=[1,context,8,64,64] or list(y.shape)!=[1,1,64,64]: raise ValueError('Loader shape mismatch')
                if any(t.device.type!='cpu' or not torch.isfinite(t).all() for t in (x,y,valid)): raise ValueError('Loader device/finite mismatch')
                loader_reads.append(dict(context_days=context,X=list(x.shape),Y=list(y.shape),device=str(x.device),finite=True))
        proof.update(status='LOT_REMOTE_CPU_CERTIFIED',number=reference['number'],anomalies=[],gpu_used=False,training=False,
            rasterio_reads=samples,loader_reads=loader_reads,aligned_windows={str(k):len(v) for k,v in groups.items()},
            window_rejections=rejected,test_events=0,test_2021_files=0,data_root=str(rasters),windows_drive_dependency=False,
            normalizer_reused_without_fitting=True,resources=dict(**resources,elapsed_seconds=time.monotonic()-started))
        (output/'certificate.json').write_text(json.dumps(proof,indent=2))
        (output/'verified.jsonl').write_bytes((output/'hashes/verified.jsonl').read_bytes())
        (output/'progress.json').write_text(json.dumps(dict(status='COMPLETE',files_certified=proof['files'],bytes_certified=proof['bytes'])))
        print(json.dumps(proof,indent=2),flush=True)
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(status='FAILED_STOP',error_type=type(error).__name__,error=str(error))))
        raise
    finally:
        stop.set()
        watcher.join(timeout=2)
