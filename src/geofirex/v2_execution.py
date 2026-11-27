"""Isolated B2 C8 V2 execution and recovery. Default CLI is CPU preflight.

No annual runner, test reader, Studio integration, threshold search or auto-fit.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
import numpy as np
from geofirex.protocol_v2 import sha256, save, make_v2_loaders, sample_keys

GIB = 1024**3
REPORT = Path('reports/protocol_v2_execution')

@contextmanager
def evaluation_deadline(seconds):
    """Kaggle/Linux alarm also bounds the CPU external merges, after inference."""
    import signal
    def expired(signum, frame):
        raise TimeoutError('Evaluation budget exhausted; use final checkpoint recovery')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def verify_manifest(root, manifest):
    root = Path(root).resolve()
    for name, digest in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError('Frozen input differs: '+name)

def identity(root):
    root = Path(root)
    lock = read(root / REPORT / 'execution_manifest.json')
    verify_manifest(root, lock['files'])
    c = read(root / 'configs/protocol_v2/baseline_c8.json')
    verify_manifest(root,c['source_sha256'])
    index = root / c['index']; meta = read(index / 'index.json')
    records = np.load(index / 'train.npy', mmap_mode='r', allow_pickle=False)
    updates = math.ceil(len(records)/c['batch_size'])
    if (meta['cutoff'] != '2020-08-23' or len(records) != c['train_tiles'] or
            updates != c['updates_per_epoch'] or updates*c['epochs'] != c['total_updates']):
        raise ValueError('Frozen split/update contract differs')
    return c, lock, dict(train_tiles=len(records), updates_per_epoch=updates,
                         total_updates=updates*c['epochs'])

def disk_check(path, required):
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError(f'Insufficient space at {path}: {free} < {required}')
    return free

def resource_check(policy, work, scratch):
    total,available = host_memory()
    if total < policy['min_host_ram_bytes'] or available < policy['min_available_ram_bytes']:
        raise RuntimeError('Insufficient total/available host RAM')
    return dict(host_ram_total=total, host_ram_available=available,
                working_free=disk_check(work, policy['required_free_bytes']),
                scratch_free=disk_check(scratch, policy['required_free_bytes']))

def host_memory():
    """No package installation needed in the offline notebook."""
    if os.name == 'nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
        value=MemoryStatus(); value.length=ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError('Cannot inspect host RAM')
        return value.total,value.available
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value,*_=line.split(); values[key.rstrip(':')]=int(value)*1024
    return values['MemTotal'],values['MemAvailable']

def host_rss():
    # Kaggle/Linux: process plus direct CPU DataLoader workers, no dependency.
    pids=[os.getpid()]
    children=Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children')
    if children.exists(): pids += [int(p) for p in children.read_text().split()]
    rss=0
    for pid in pids:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'): rss+=int(line.split()[1])*1024
        except FileNotFoundError:
            pass
    return rss

def verify_mounts(root, mapping_paths):
    expected = read(Path(root)/REPORT/'partitions.json')
    if len(mapping_paths) != 11:
        raise ValueError('Exactly eleven mount maps required')
    seen = set(); objects = 0; total = 0; mounts = []
    for supplied in mapping_paths:
        supplied = Path(supplied).resolve(); m = read(supplied)
        item = expected.get(m['identity'])
        if item is None or m['identity'] in seen or sha256(supplied) != item['map_sha256']:
            raise ValueError('Missing, duplicate, unknown or changed partition')
        canonical = json.dumps([{k:f[k] for k in ('path','alias','bytes','sha256')} for f in m['files']],
                               sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(canonical).hexdigest() != m['identity']:
            raise ValueError('Invalid partition identity')
        seen.add(m['identity']); mounts.append(str(supplied.parent))
        for row in m['files']:
            alias = row['alias']
            if Path(alias).name != alias or '\\' in alias or ':' in alias or '2021' in Path(row['path']).parts:
                raise ValueError('Unsafe or test object')
            if (supplied.parent/alias).stat().st_size != row['bytes']:
                raise ValueError('Published object size differs')
            objects += 1; total += row['bytes']
    return mounts, dict(partitions=sorted(seen), objects=objects, bytes=total,
                        current_check='pinned manifest identities and every object size',
                        current_full_archive_hashes_verified=False,
                        prior_full_hash_certificate='historical Kaggle PASS; not a new byte attestation')

def preflight(root, maps=None, work=None, scratch=None, gpu=False):
    import torch
    c, lock, updates = identity(root)
    policy = read(Path(root)/REPORT/'budget.json')
    result = dict(status='CPU_IDENTITY_PASS', **updates, torch=str(torch.__version__),
                  numpy=np.__version__, cuda_build=torch.version.cuda,
                  python=sys.version, gpu_checked=False, training_executed=False, test_pixels_opened=0)
    if maps is not None:
        mounts, receipt = verify_mounts(root, maps)
        train, val = make_v2_loaders('', Path(root)/c['index'], mounts=mounts, workers=0)
        result['mount_receipt'] = receipt
        result['loader_lengths'] = [len(train.dataset), len(val.dataset)]
        train.dataset.close(); val.dataset.close()
    if work is not None:
        result['resources'] = resource_check(policy, work, scratch)
    if gpu:
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1 or
                torch.cuda.get_device_name(0) != 'Tesla T4'):
            raise ValueError('Exactly one visible Tesla T4 required')
        if str(torch.__version__) != policy['torch'] or np.__version__ != policy['numpy'] or torch.version.cuda != policy['cuda_build']:
            raise ValueError('Unqualified PyTorch/NumPy/CUDA version; do not install/update automatically')
        result.update(gpu_checked=True, gpu=torch.cuda.get_device_name(0),
                      gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cudnn=torch.backends.cudnn.version(), status='RUNTIME_PASS')
    return result

def authorize(root, path, action):
    a = read(path); policy = read(Path(root)/REPORT/'budget.json')
    expected = dict(action=action, authorized=True, runs=1, run_id='B2_C8_V2_seed42',
                    execution_manifest_sha256=sha256(Path(root)/REPORT/'execution_manifest.json'),
                    session_reserve_minutes=policy['session_minutes'],
                    card_hours_reserved=policy['conservative_card_hours'], gpu_count=1)
    if any(a.get(k) != v for k,v in expected.items()):
        raise ValueError('Explicit matching authorization required')
    if (a.get('quota_remaining_card_hours', -1) < policy['conservative_card_hours'] or
            a.get('session_remaining_minutes', -1) < policy['session_minutes'] or
            not a.get('quota_checked_at') or not a.get('authorized_by') or
            not a.get('local_backup_destination') or a.get('external_backup_destination') != 'D:\\GeoFire-X'):
        raise ValueError('Quota, time and explicit backup destinations must be checked before launch')
    return a

def verify_zip(path, expected=None):
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('Archive SHA differs')
    with zipfile.ZipFile(path) as z:
        m = json.loads(z.read('artifact_manifest.json'))
        names = [r['path'] for r in m['files']]
        if len(names) != len(set(names)) or len(z.namelist()) != len(names)+1 or set(z.namelist()) != set(names)|{'artifact_manifest.json'}:
            raise ValueError('Archive membership differs')
        for row in m['files']:
            h = hashlib.sha256(); n = 0
            with z.open(row['path']) as stream:
                for block in iter(lambda:stream.read(1024**2), b''):
                    h.update(block); n += len(block)
            if h.hexdigest() != row['sha256'] or n != row['bytes']:
                raise ValueError('Archive member differs')
    return digest

def package(out, stage):
    out = Path(out).resolve()
    files = sorted(p for p in out.rglob('*') if p.is_file() and '_scratch' not in p.relative_to(out).parts)
    if any(p.is_symlink() or not p.resolve().is_relative_to(out) for p in files):
        raise ValueError('Own regular outputs only')
    disk_check(out.parent, sum(p.stat().st_size for p in files)+64*1024**2)
    path = out.parent / f'{out.name}-{stage}-{uuid.uuid4().hex[:8]}.zip'
    manifest = dict(stage=stage, files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in files])
    with path.open('xb') as raw:
        with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in files:
                z.write(p,p.relative_to(out).as_posix())
            z.writestr('artifact_manifest.json',json.dumps(manifest))
        raw.flush(); os.fsync(raw.fileno())
    digest = verify_zip(path)
    path.with_suffix('.sha256').write_text(digest+'  '+path.name+'\n',encoding='ascii')
    print('VERIFIED ARCHIVE',path,digest,flush=True)
    return path

def restore(archive,sidecar,destination):
    """Restore only a verified own archive into a new directory; never overwrite."""
    expected=Path(sidecar).read_text(encoding='ascii').split()[0]
    verify_zip(archive,expected)
    destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if '\\' in name or ':' in name or not (destination/name).resolve().is_relative_to(destination):
                raise ValueError('Unsafe archive path')
        destination.mkdir(parents=True,exist_ok=False)
        z.extractall(destination)
    return destination

def safeguarded_evaluation(root,out,callback):
    """The recovery handoff is independently CPU-testable without any fitting."""
    try:
        final_state(out,root)
        package(out,'TRAINING-SAFE')
        return callback()
    finally:
        package(out,'COMPLETE' if (Path(out)/'evaluation/complete.json').exists() else 'PARTIAL')

def preserve(root, out, lock):
    for name in list(lock['files']) + [(REPORT/'execution_manifest.json').as_posix()]:
        src = Path(root)/name; dst = out/'inputs'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as target, src.open('rb') as source:
            shutil.copyfileobj(source,target)

def checkpoint(out, model, optimizer, state, config, lock, normalizer):
    import torch
    path = out/f"checkpoint_u{state['updates']:07d}.pt"
    payload = dict(**state, model=model.state_dict(),optimizer=optimizer.state_dict(),
                   config=config, execution_manifest=lock,normalizer=normalizer,
                   rng=dict(python=random.getstate(),numpy=np.random.get_state(),
                            torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    temp = path.with_suffix('.pt.partial')
    with temp.open('xb') as stream:
        torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    temp.rename(path)
    save(path.with_suffix('.json'),dict(sha256=sha256(path),updates=state['updates'],epoch=state['epoch'],cursor=state['cursor']))
    return path

def final_state(source, root):
    import torch
    c, lock, u = identity(root)
    source = Path(source); receipt = read(source/'training_complete.json')
    name = receipt['checkpoint']
    if Path(name).name != name or receipt['updates'] != u['total_updates']:
        raise ValueError('Incomplete training receipt')
    path = source/name
    if sha256(path) != receipt['checkpoint_sha256'] or sha256(path) != read(path.with_suffix('.json'))['sha256']:
        raise ValueError('Final checkpoint corrupted')
    # Only locally produced/trusted own checkpoints; never arbitrary pickle files.
    state = torch.load(path,map_location='cpu',weights_only=False)
    if (state['config'] != c or state['execution_manifest'] != lock or state['epoch'] != c['epochs'] or
            state['cursor'] != 0 or state['updates'] != u['total_updates']):
        raise ValueError('Final checkpoint identity/cursor differs')
    return state, path

def fit(root, maps, out, authorization):
    import torch
    from geofirex.benchmark_models import StackedUNet
    from geofirex.unet import SegmentationLoss
    c, lock, u = identity(root); policy = read(Path(root)/REPORT/'budget.json')
    authorize(root, authorization, 'fit')
    out = Path(out).resolve()
    if out.exists() or out.parent != Path('/kaggle/working').resolve():
        raise ValueError('New direct Kaggle working directory required')
    receipt = preflight(root,maps,out.parent,'/tmp',gpu=True)
    mounts,_ = verify_mounts(root,maps)
    save(out.parent/'B2_C8_V2_seed42_SINGLE_RUN.lock.json',dict(run_id=c['run_id']))
    out.mkdir(); preserve(root,out,lock); save(out/'preflight.json',receipt)
    shutil.copyfile(authorization,out/'authorization.json')
    train = val = None; last = None; handed_off=False
    started = time.monotonic(); io_seconds = 0.; peak_rss = 0
    try:
        random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
        torch.cuda.manual_seed_all(c['seed']); torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model = StackedUNet(8,3,base_channels=c['base_channels'],depth=c['depth']).to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(),lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])
        loss_fn = SegmentationLoss(**{k:c['loss'][k] for k in ('dice_weight','pos_weight')}).to('cuda')
        train,val = make_v2_loaders('',Path(root)/c['index'],mounts=mounts,batch_size=c['batch_size'],workers=2)
        state = dict(epoch=0,cursor=0,updates=0,history=[])
        last = checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
        with (out/'steps.jsonl').open('x',encoding='utf8') as log:
            for epoch in range(c['epochs']):
                train.sampler.epoch=epoch; model.train(); weighted_loss=0.; valid_pixels=0
                before=time.monotonic()
                for x,y,v in train:
                    io_seconds += time.monotonic()-before
                    if time.monotonic()-started > policy['fit_seconds']:
                        if read(last.with_suffix('.json'))['updates'] != state['updates']:
                            last=checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
                        save(out/'paused.json',dict(status='PAUSED_NOT_COMPLETE',updates=state['updates']))
                        return out
                    optimizer.zero_grad(set_to_none=True)
                    loss=loss_fn(model(x.to('cuda')),y.to('cuda'),v.to('cuda'))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),c['optimizer']['clip_grad_norm'],error_if_nonfinite=True)
                    optimizer.step(); n=int(v.sum()); weighted_loss+=float(loss.detach())*n; valid_pixels+=n
                    state['cursor']+=len(x); state['updates']+=1
                    log.write(json.dumps(dict(epoch=epoch+1,updates=state['updates'],cursor=state['cursor'],loss=float(loss.detach())))+'\n')
                    if state['updates']%1000==0:
                        log.flush(); disk_check(out,policy['minimum_runtime_free_bytes'])
                        last=checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
                    peak_rss=max(peak_rss,host_rss()); before=time.monotonic()
                if state['cursor'] != u['train_tiles']:
                    raise ValueError('Nonexhaustive epoch')
                state['history'].append(dict(epoch=epoch+1,loss=weighted_loss/valid_pixels))
                state['epoch']=epoch+1; state['cursor']=0
                last=checkpoint(out,model,optimizer,state,c,lock,train.dataset.normalization)
        if state['updates'] != u['total_updates']:
            raise ValueError('Incomplete optimizer budget')
        torch.cuda.synchronize()
        save(out/'resources_fit.json',dict(fit_seconds=time.monotonic()-started,loader_wait_seconds=io_seconds,
             peak_host_rss_bytes=peak_rss,peak_vram_bytes=torch.cuda.max_memory_allocated(),
             visible_gpu_count=1,visible_gpu_wall_hours=(time.monotonic()-started)/3600,
             quota_card_hours='Record actual Kaggle session deduction separately'))
        save(out/'training_complete.json',dict(status='COMPLETE',updates=state['updates'],epochs=c['epochs'],
             checkpoint=last.name,checkpoint_sha256=sha256(last)))
        # Full verified checkpoint archive MUST precede any AP work or inference.
        def evaluate_final():
            with evaluation_deadline(policy['evaluation_seconds']):
                return evaluate(root,maps,out,out/'evaluation',checkpoint_source=out)
        handed_off=True
        safeguarded_evaluation(root,out,evaluate_final)
        return out
    finally:
        if train is not None: train.dataset.close()
        if val is not None: val.dataset.close()
        if not handed_off:
            package(out,'PARTIAL')

def evaluate(root,maps,out,destination,checkpoint_source=None,authorization=None):
    """Can be called after fit or explicitly for recovery; no optimizer constructed."""
    import torch
    from geofirex.benchmark_models import StackedUNet
    from geofirex.protocol_v2 import V2CompactTiles
    from geofirex.v2_propagation import PropagationMetrics
    from torch.utils.data import DataLoader
    if checkpoint_source is None:
        authorize(root,authorization,'evaluate-only'); checkpoint_source=out
    if (Path(checkpoint_source)/'evaluation/complete.json').exists():
        raise ValueError('Already fully evaluated; preserve existing result')
    c,_,_=identity(root); policy=read(Path(root)/REPORT/'budget.json')
    preflight(root,maps,Path(destination).parent,'/tmp',gpu=True)
    state,path=final_state(checkpoint_source,root)
    mounts,_=verify_mounts(root,maps)
    destination=Path(destination); destination.mkdir(exist_ok=False)
    save(destination/'checkpoint_identity.json',dict(sha256=sha256(path),training_executed=False))
    ds=None; start=time.monotonic(); inference_seconds=0.; peak_rss=0
    try:
        model=StackedUNet(8,3,base_channels=c['base_channels'],depth=c['depth'])
        model.load_state_dict(state['model']); model.to('cuda'); model.eval()
        ds=V2CompactTiles('',Path(root)/c['index'],'val',mounts=mounts)
        keys=sample_keys(ds); acc=PropagationMetrics(destination/'_scratch',keys,read(Path(root)/REPORT/'propagation_policy.json'))
        offset=0
        with torch.inference_mode():
            for x,y,v in DataLoader(ds,batch_size=16,num_workers=2,shuffle=False):
                if time.monotonic()-start>policy['evaluation_seconds']:
                    raise TimeoutError('Evaluation budget reached; recover checkpoint without fit')
                t=time.monotonic(); p=model(x.to('cuda')).sigmoid().cpu().numpy().astype(np.float32)
                inference_seconds+=time.monotonic()-t
                for i in range(len(x)):
                    acc.update(keys[offset+i],p[i,0],y[i,0].numpy(),v[i,0].numpy(),x[i,-1,c['fire_channel']].numpy())
                offset+=len(x)
                peak_rss=max(peak_rss,host_rss())
                if offset%1024==0:
                    disk_check(destination,policy['minimum_runtime_free_bytes'])
        # result() computes exact pooled AP, then exact new-fire PR/AP.
        scratch_before_merge=sum(p.stat().st_size for p in (destination/'_scratch').rglob('*') if p.is_file())
        result=acc.result()
        save(destination/'evaluation.json',result)
        rows=[]
        for event in result['per_event']:
            row={k:v for k,v in event.items() if k!='new_fire'}
            row.update({'new_'+k:v for k,v in event['new_fire'].items()}); rows.append(row)
        with (destination/'per_event.csv').open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        curve=result['propagation_pr']['curve']
        with (destination/'new_fire_pr.csv').open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)
        save(destination/'resources.json',dict(evaluation_seconds=time.monotonic()-start,
             inference_seconds=inference_seconds,exact_sort_and_accumulation_seconds=time.monotonic()-start-inference_seconds,
             sampled_peak_host_rss_bytes=max(peak_rss,host_rss()),
             scratch_bytes_before_exact_merge=scratch_before_merge,
             scratch_merge_reserved_upper_bytes=policy['scratch_merge_upper_bytes'],
             scratch_bytes_remaining=sum(p.stat().st_size for p in (destination/'_scratch').rglob('*') if p.is_file()),
             peak_vram_bytes=torch.cuda.max_memory_allocated()))
        save(destination/'complete.json',dict(status='COMPLETE',checkpoint_sha256=sha256(path),tiles=offset))
        return result
    finally:
        if ds is not None: ds.close()

def backup(archive,sidecar,local_root,external_root):
    archive=Path(archive); expected=Path(sidecar).read_text(encoding='ascii').split()[0]
    verify_zip(archive,expected)
    roots=[Path(local_root).resolve(),Path(external_root).resolve()]
    if roots[0]==roots[1] or not all(p.is_dir() for p in roots):
        raise ValueError('Two distinct existing backup destinations required')
    receipts=[]
    for root in roots:
        disk_check(root,archive.stat().st_size+64*1024**2)
        destination=root/('v2-preserved-'+uuid.uuid4().hex); destination.mkdir()
        target=destination/archive.name
        with target.open('xb') as dst,archive.open('rb') as src:
            shutil.copyfileobj(src,dst); dst.flush(); os.fsync(dst.fileno())
        verify_zip(target,expected); shutil.copyfile(sidecar,destination/Path(sidecar).name)
        save(destination/'backup_verified.json',dict(sha256=expected,all_members_verified=True,source=str(archive)))
        receipts.append(str(target))
    return receipts
