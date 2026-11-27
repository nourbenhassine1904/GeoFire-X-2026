"""Measured CPU/ADATA resource profile. The scientific V2 freeze stays intact.

Only cache capacities, runtime resource checks and storage orchestration differ.
All inputs, inference, thresholds, masks and exact metric implementations are
imported unchanged from the original final reader. No training capability.
"""
from __future__ import annotations
from pathlib import Path
import ctypes
import json
import math
import os
import shutil
import threading
import time
import re
from datetime import datetime,timezone
from geofirex import final_test_v2 as f

MIB=1024**2
RAW_CACHE=16*MIB
TARGET_CACHE=4*MIB
GDAL_CACHE=32*MIB
HEADROOM=512*MIB
ROUNDING=256*MIB
NEW_FILES=('src/geofirex/final_test_v2_local.py','scripts/run_final_test_v2_local.py',
           'tests/test_final_test_v2_local.py',
           'reports/final_test_v2_local_20261125_v1/measure_memory.py')


def memory_snapshot():
    """Native Windows process peak working set, private commit and system RAM."""
    class PM(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in (
            'peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','commit','peak_commit','private_commit')]
    m=PM();m.cb=ctypes.sizeof(m)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(m),m.cb):raise ctypes.WinError()
    class Global(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in (
            'total','available','commit_limit','available_commit','virtual','available_virtual','extended')]
    g=Global();g.length=ctypes.sizeof(g)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(g)):raise ctypes.WinError()
    installed=ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(installed)):raise ctypes.WinError()
    return dict(peak_rss_bytes=m.peak_rss,rss_bytes=m.rss,private_commit_bytes=m.private_commit,
                peak_private_commit_bytes=m.peak_commit,total_bytes=g.total,available_bytes=g.available,
                available_commit_bytes=g.available_commit,installed_bytes=installed.value*1024)


def volume_info(path):
    anchor=Path(path).resolve().anchor
    label=ctypes.create_unicode_buffer(261);fs=ctypes.create_unicode_buffer(261)
    serial=ctypes.c_ulong();maxlen=ctypes.c_ulong();flags=ctypes.c_ulong()
    ok=ctypes.windll.kernel32.GetVolumeInformationW(anchor,label,261,ctypes.byref(serial),ctypes.byref(maxlen),ctypes.byref(flags),fs,261)
    if not ok:raise ctypes.WinError()
    return dict(anchor=anchor,label=label.value,filesystem=fs.value,serial=serial.value)


def derived_memory_policy(receipts):
    if not receipts or any(r['status']!='PASS' or r.get('test_tiff_reads')!=0 for r in receipts):
        raise ValueError('Successful no-2021 resource measurements required')
    peak=max(r['memory']['peak_rss_bytes'] for r in receipts)
    commit=max(r['memory']['peak_private_commit_bytes'] for r in receipts)
    round_up=lambda n:math.ceil(n/ROUNDING)*ROUNDING
    return dict(measured_peak_rss_bytes=peak,measured_peak_private_commit_bytes=commit,
                available_ram_required_bytes=round_up(2*peak+HEADROOM),
                available_commit_required_bytes=round_up(2*commit+HEADROOM),
                process_rss_ceiling_bytes=round_up(2*peak),system_available_floor_bytes=HEADROOM,
                total_ram_min_bytes=16_000_000_000,
                derivation='2 x measured worst process peak + 512MiB OS reserve; round up to 256MiB',
                raw_cache_bytes=RAW_CACHE,target_cache_bytes=TARGET_CACHE,gdal_cache_bytes=GDAL_CACHE)


class LocalReader(f.FinalReader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reader.budget=RAW_CACHE

    def _target(self,*args):
        result=super()._target(*args)
        while self.target_cache and self.target_cache_bytes>TARGET_CACHE:
            _,(a,b)=self.target_cache.popitem(last=False)
            self.target_cache_bytes-=a.nbytes+b.nbytes
        return result


class MemoryGuard:
    """Fail closed during inference AND external AP merges; never frees other apps."""
    def __init__(self,policy):self.policy=policy;self.stop=threading.Event();self.failure=None
    def __enter__(self):
        import _thread
        def watch():
            while not self.stop.wait(.25):
                try:
                    m=memory_snapshot()
                    if (m['rss_bytes']>self.policy['process_rss_ceiling_bytes'] or m['available_bytes']<self.policy['system_available_floor_bytes']
                            or m['available_commit_bytes']<HEADROOM):
                        self.failure=dict(reason='measured memory envelope exceeded',snapshot=m)
                        _thread.interrupt_main();return
                except Exception as exc:
                    self.failure=dict(reason='memory monitor failed',error=repr(exc));_thread.interrupt_main();return
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=1)


def verify_local(root,profile):
    root,profile=Path(root),Path(profile)
    seal=f.load(profile/'operational_seal.json')
    for name,expected in seal['files'].items():
        if f.sha(root/name)!=expected:raise ValueError('Qualified operational file changed: '+name)
    derived=derived_memory_policy([f.load(profile/name) for name in ('validation_memory.json','synthetic_memory.json')])
    if seal['memory_policy']!=derived:raise ValueError('Memory policy differs from measured derivation')
    frozen=root/seal['scientific_freeze']
    if f.sha(frozen/'selection.json')!=seal['scientific_freeze_sha256']:raise ValueError('Scientific freeze changed')
    science,rows,keys=f.verify_freeze(root,frozen)
    return seal,science,rows,keys,frozen


def preflight(root,profile,output=None):
    import importlib.metadata as md
    seal,science,rows,keys,frozen=verify_local(root,profile)
    paths=seal['paths'];policy=seal['memory_policy'];mem=memory_snapshot()
    _,_,stats=f.inventory(rows,{r['event_id'] for r in rows},paths['rasters'])
    plan=f.resource_plan(stats) # The old 8GiB guard remains unchanged in the old reader.
    plan['original_preparation_ram_available_min_bytes']=plan['ram_available_min_bytes']
    plan['ram_available_min_bytes']=policy['available_ram_required_bytes']
    plan['session_seconds']=seal.get('max_seconds',21600)
    plan['quota_minutes']=0  # CPU only, never a GPU quota reservation.
    runtime={p:md.version(p) for p in ('torch','numpy','rasterio','pyarrow')}
    q=f.load(Path(profile)/'validation_memory.json')
    work,scratch=Path(paths['results']),Path(paths['scratch'])
    # Dedicated directory constraints are part of the operational seal.
    base=Path(paths['adata_root']).resolve()
    for directory in (work,scratch,Path(paths['backups'])):
        if not directory.resolve().is_relative_to(base) or not directory.is_dir():raise ValueError('Invalid dedicated ADATA directory')
    same=os.stat(work).st_dev==os.stat(scratch).st_dev
    d_volume=volume_info(base);c_volume=volume_info(paths['ledger'])
    hw=f.load(Path(profile)/'hardware_initial.json')['volumes']
    volume_ok=d_volume['filesystem']=='NTFS' and d_volume['serial']==hw['D']['serial'] and c_volume['serial']==hw['C']['serial']
    free_d=shutil.disk_usage(base).free;free_c=shutil.disk_usage(paths['ledger']).free
    reserve=100*f.GIB
    disk_ok=free_d>=plan['same_filesystem_required_bytes']+reserve and free_c>=plan['output_required_bytes']
    ram_ok=(mem['total_bytes']>=policy['total_ram_min_bytes'] and mem['available_bytes']>=policy['available_ram_required_bytes']
            and mem['available_commit_bytes']>=policy['available_commit_required_bytes'])
    # Tiny new fsynced probes; no original archive or TIFF is opened for writing.
    probes=[]
    from uuid import uuid4
    for directory in (scratch,work,Path(paths['backups']),Path(paths['ledger'])):
        path=directory/('PREFLIGHT-PROBE-'+uuid4().hex+'.json')
        f.save(path,dict(probe_only=True,authorization=False));assert f.load(path)['probe_only'] is True
        probes.append(str(path))
    parity=q['status']=='PASS' and q['runtime']==runtime and q['validation_parity']['passed']
    test_identity=science['test_identity']
    lock=Path(paths['ledger'])/f'final-test-2021-{test_identity}.lock.json'
    checks=dict(frozen_identities='PASS',tiff_paths_sizes='PASS' if not stats['missing_paths'] and not stats['size_errors'] else 'FAIL',
        c8_archive='PASS' if f.sha(paths['archive'])==f.ARCHIVE_SHA else 'FAIL',ram='PASS' if ram_ok else 'FAIL',disk='PASS' if disk_ok else 'FAIL',
        runtime_validation_parity='PASS' if parity else 'FAIL',volumes='PASS' if volume_ok else 'FAIL',write_and_fsync='PASS',persistent_local_ledger='PASS',
        single_use_available='PASS' if not lock.exists() else 'FAIL',cpu_runtime='PASS',independent_authorization='BLOCKED')
    result=dict(schema='geofirex-final-test-v2-local-preflight-1',checks=checks,
       technical_status='PASS' if all(v=='PASS' for k,v in checks.items() if k!='independent_authorization') else 'FAIL',
       freeze_sha256=f.sha(frozen/'selection.json'),operational_seal_sha256=f.sha(Path(profile)/'operational_seal.json'),
       runtime=runtime,memory=mem,memory_policy=policy,plan=plan,adata_free_bytes=free_d,
       adata_reserve_after_worst_case_bytes=free_d-plan['same_filesystem_required_bytes'],required_adata_reserve_bytes=reserve,
       C_backup_free_bytes=free_c,same_filesystem=same,volumes=dict(adata=d_volume,ledger=c_volume),paths=paths,inventory=stats,
       qualification_sha256=f.sha(Path(profile)/'validation_memory.json'),
       created_utc=datetime.now(timezone.utc).isoformat(),test_pixels_opened=0,gpu_used=False,probes=probes)
    if output:f.save(output,result)
    return result


def seal_profile(root,profile):
    """Operational extension only; the original scientific freeze is never edited."""
    root,profile=Path(root).resolve(),Path(profile).resolve()
    target=profile/'operational_seal.json'
    if target.exists():raise FileExistsError(target)
    frozen=root/'reports/final_test_v2_20261125_v1/freeze'
    science,_,_=f.verify_freeze(root,frozen)
    receipts=[f.load(profile/name) for name in ('validation_memory.json','synthetic_memory.json')]
    if receipts[0]['validation_parity']['passed'] is not True or receipts[1]['hierarchical_merge_exercised'] is not True:
        raise ValueError('Full-batch parity and hierarchical sorting qualification required')
    storage=f.load(profile/'storage_probe.json')
    if storage['status']!='PASS':raise ValueError('Storage probe failed')
    base=Path(storage['root'])
    backup=root/'artifacts/final_test_v2_backups';backup.mkdir(exist_ok=False)
    paths=dict(adata_root=str(base),scratch=str(base/'scratch'),results=str(base/'results'),
      backups=str(base/'backups'),ledger=storage['ledger_root'],C_backups=str(backup),
      archive='D:\\GeoFire-X\\GeoFire-X-V2\\B2_C8_V2_seed42-COMPLETE-48e35c59.zip',
      rasters='D:\\GeoFire-X\\WildfireSpreadTS\\rasters')
    names=list(NEW_FILES)+[(profile/name).relative_to(root).as_posix() for name in (
          'validation_memory.json','synthetic_memory.json','storage_probe.json','hardware_initial.json')]
    record=dict(schema='geofirex-final-v2-local-operational-seal-1',
        scientific_freeze=frozen.relative_to(root).as_posix(),scientific_freeze_sha256=f.sha(frozen/'selection.json'),
        test_identity=science['test_identity'],paths=paths,memory_policy=derived_memory_policy(receipts),
        max_seconds=21600,archive_reserve_seconds=1800,
        duration_policy='6h CPU reservation, 30min archive reserve; supersedes only the unexecuted 4h preparation estimate',
        changes='cache sizes, measured resource guards, dedicated ADATA storage and C backups only; identical scientific math',
        final_test_authorized=False,training_authorized=False,gpu_authorized=False,
        files={name:f.sha(root/name) for name in names},created_utc=datetime.now(timezone.utc).isoformat())
    f.save(target,record)
    template=dict(schema='geofirex-final-v2-local-cpu-authorization-1',authorized=False,training_authorized=False,
        device='cpu',freeze_sha256=record['scientific_freeze_sha256'],operational_seal_sha256=f.sha(target),
        test_identity=science['test_identity'],preflight_sha256=None,qualification_sha256=f.sha(profile/'validation_memory.json'),
        authorization_id=None,authorized_by=None,authorized_at_utc=None,session_seconds_remaining=None,max_seconds=21600,
        ledger_root=paths['ledger'],scratch_root=paths['scratch'],raster_root=paths['rasters'],archive=paths['archive'],
        C_backup_root=paths['C_backups'],output=None,durable_shared_ledger_confirmed=False,runtime_mounts_verified=False,
        cpu_only=True,post_failure='keep permanent consumed lock; no automatic inference retry; human incident review')
    f.save(profile/'authorization_TEMPLATE_DISABLED.json',template)
    return record


def consume_local_authorization(root,profile,authorization,preflight_path,output):
    a=f.load(authorization)
    if a.get('authorized') is not True:raise PermissionError('Independent 2021 CPU authorization absent/disabled')
    seal,science,_,_,frozen=verify_local(root,profile);paths=seal['paths']
    expected=dict(freeze_sha256=f.sha(frozen/'selection.json'),operational_seal_sha256=f.sha(Path(profile)/'operational_seal.json'),
      test_identity=science['test_identity'],preflight_sha256=f.sha(preflight_path),
      qualification_sha256=f.sha(Path(profile)/'validation_memory.json'),output=str(Path(output).resolve()),
      ledger_root=paths['ledger'],scratch_root=paths['scratch'],raster_root=paths['rasters'],archive=paths['archive'],
      C_backup_root=paths['C_backups'],device='cpu')
    if any(a.get(k)!=v for k,v in expected.items()):raise PermissionError('Independent authorization bindings differ')
    if (a.get('schema')!='geofirex-final-v2-local-cpu-authorization-1' or a.get('training_authorized') is not False
        or a.get('cpu_only') is not True or a.get('durable_shared_ledger_confirmed') is not True
        or a.get('runtime_mounts_verified') is not True or a.get('max_seconds')!=seal['max_seconds']
        or not isinstance(a.get('session_seconds_remaining'),(int,float)) or a['session_seconds_remaining']<seal['max_seconds']
        or not a.get('authorized_by') or not a.get('authorized_at_utc')
        or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}',a.get('authorization_id',''))):
        raise PermissionError('Incomplete independent CPU authorization')
    receipt=f.load(preflight_path)
    if (receipt.get('technical_status')!='PASS' or receipt.get('operational_seal_sha256')!=expected['operational_seal_sha256']
        or receipt.get('freeze_sha256')!=expected['freeze_sha256']):raise PermissionError('Qualified preflight required')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(receipt['created_utc'])).total_seconds()
    if not 0<=age<=1800:raise PermissionError('Preflight must be less than 30 minutes old')
    output=Path(output).resolve()
    if output.parent!=Path(paths['results']).resolve() or output.exists():raise PermissionError('Exclusive dedicated result directory required')
    ledger=Path(paths['ledger']).resolve()
    if f.ephemeral_ledger(ledger) or not ledger.is_dir():raise PermissionError('Persistent local ledger required')
    lock=ledger/f"final-test-2021-{science['test_identity']}.lock.json"
    f.save(lock,dict(status='CONSUMED_BEFORE_ANY_TEST_READ',**expected,authorization_sha256=f.sha(authorization),
      authorization_id=a['authorization_id'],created_utc=datetime.now(timezone.utc).isoformat(),retry='never automatically reset'))
    return f.PixelPermit(science['test_identity'],str(lock),a['authorization_id'])


class SafePredictionArchive(f.PredictionArchive):
    """Same lossless binary payload; fsynced second-volume copy after each event."""
    def __init__(self,out,backup):
        super().__init__(out);self.backup=Path(backup);self.backup.mkdir(parents=True,exist_ok=False)
    def _finish(self):
        active=self.stream is not None
        super()._finish()
        if active:
            last=self.receipts[-1];source=self.out/last['path']
            f.copy_exclusive(source,self.backup/source.name)
            f.copy_exclusive(source.with_suffix('.json'),self.backup/source.with_suffix('.json').name)
            if f.sha(self.backup/source.name)!=last['sha256']:raise ValueError('C event backup SHA-256 differs')
    def finish(self):
        super().finish();f.copy_exclusive(self.out/'manifest.json',self.backup/'manifest.json')


def evaluate(root,profile,authorization,preflight_path,output):
    """Future real-test entry point. Disabled grants fail before any model/I/O."""
    if f.load(authorization).get('authorized') is not True:raise PermissionError('2021 CPU evaluation is NOT authorized')
    seal,science,rows,keys,frozen=verify_local(root,profile)
    live=preflight(root,profile)
    if live['technical_status']!='PASS':raise PermissionError('Live CPU resource preflight failed')
    # Check all bindings before consuming the durable one-use ledger.
    permit=consume_local_authorization(root,profile,authorization,preflight_path,output)
    paths=seal['paths'];out=Path(output);out.mkdir(exist_ok=False)
    backup=Path(paths['C_backups'])/permit.nonce;backup.mkdir(exist_ok=False)
    for src,name in ((permit.ledger_file,'CONSUMED.json'),(authorization,'authorization.json'),
                     (frozen/'selection.json','selection.json'),(Path(profile)/'operational_seal.json','operational_seal.json')):
        f.copy_exclusive(src,out/name);f.copy_exclusive(src,backup/name)
    f.save(out/'live_preflight.json',live)
    guard=MemoryGuard(seal['memory_policy']);writer=None;started=time.perf_counter();inference_seconds=0.
    try:
        with guard,f.execution_deadline(seal['max_seconds']-seal['archive_reserve_seconds']):
            import torch,numpy as np,rasterio
            from geofirex.benchmark_models import StackedUNet
            norm=f.load(Path(root)/f.INDEX/'normalization.json');state=f.selected_checkpoint(paths['archive'],norm)
            torch.set_num_threads(2)
            model=StackedUNet(8,3,base_channels=8,depth=3);model.load_state_dict(state['model']);model.eval();model.requires_grad_(False)
            reader=LocalReader(rows,paths['rasters'],norm,permit=permit,test_identity=science['test_identity'])
            writer=SafePredictionArchive(out/'predictions',backup/'predictions')
            acc=f.FinalMetrics(Path(paths['scratch'])/('final-v2-'+permit.nonce),keys)
            with rasterio.Env(GDAL_PAM_ENABLED='NO',GDAL_CACHEMAX=GDAL_CACHE),(out/'progress.jsonl').open('x') as log:
                for offset in range(0,len(keys),16):
                    batch=keys[offset:offset+16];contexts=[reader.context(k) for k in batch]
                    x=torch.from_numpy(np.stack([a for a,_ in contexts]));t=time.perf_counter()
                    with torch.inference_mode():prob=model(x).sigmoid().numpy()
                    inference_seconds+=time.perf_counter()-t
                    for key,(_,token),pp in zip(batch,contexts,prob):
                        reader.record_prediction(token,pp);y,v,o=reader.target_after_prediction(token)
                        writer.add(key,pp,y,v,o);acc.update(key,pp,y,v,o)
                    if offset//16%100==0:
                        log.write(json.dumps(dict(tiles=min(offset+16,len(keys)),elapsed_seconds=time.perf_counter()-started,memory=memory_snapshot()))+'\n');log.flush()
                writer.finish()
            f.save(out/'INFERENCE_COMPLETE.json',dict(keys_sha256=science['keys_sha256'],tiles=len(keys),
                   prediction_manifest_sha256=f.sha(out/'predictions/manifest.json')))
            f.copy_exclusive(out/'INFERENCE_COMPLETE.json',backup/'INFERENCE_COMPLETE.json')
            result=acc.result();result.update(selected=science['selected'],checkpoint_sha256=f.CHECKPOINT_SHA,
                freeze_sha256=f.sha(frozen/'selection.json'),operational_seal_sha256=f.sha(Path(profile)/'operational_seal.json'),
                test_identity=science['test_identity'],device='cpu',gpu_compute_seconds=0,training_seconds=0,
                resources=dict(total_seconds=time.perf_counter()-started,inference_seconds=inference_seconds,
                    loader_io_seconds=reader.io_seconds,memory=memory_snapshot()),
                test_observations_accessed=len(reader.opened_2021))
            f.save(out/'evaluation.json',result)
            import csv
            csv_rows=[dict(event_id=r['event_id'],**{domain+'_'+k:v for domain in ('pooled','new_fire') for k,v in r[domain].items()}) for r in result['per_event']]
            with (out/'per_event.csv').open('x',newline='',encoding='utf-8') as stream:
                w=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
        package=Path(f.result_package(out));f.copy_exclusive(package,backup/package.name)
        if f.sha(package)!=f.sha(backup/package.name):raise ValueError('C result ZIP backup differs')
        f.copy_exclusive(package.with_suffix('.sha256.json'),backup/package.with_suffix('.sha256.json').name)
        return str(package)
    except BaseException as exc:
        if writer is not None and writer.stream is not None:
            writer.stream.flush();os.fsync(writer.stream.fileno());writer.stream.close()
        f.save(out/'PARTIAL_NO_RETRY.json',dict(error=repr(exc),memory_guard=guard.failure,ledger=permit.ledger_file,
               instruction='preserve lock; human incident review; recover metrics only if saved predictions are complete'))
        package=Path(f.result_package(out,'PARTIAL'));f.copy_exclusive(package,backup/package.name)
        raise
