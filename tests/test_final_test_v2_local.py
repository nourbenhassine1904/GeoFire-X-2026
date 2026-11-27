"""Synthetic/temporary fixtures only; no ADATA, real 2021 pixels or GPU."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pytest
from geofirex import final_test_v2 as f
from geofirex import final_test_v2_local as local
from test_final_test_v2 import synthetic


def test_policy_is_derived_not_a_disabled_guard():
    receipt=dict(status='PASS',test_tiff_reads=0,memory=dict(peak_rss_bytes=600*local.MIB,peak_private_commit_bytes=1024*local.MIB))
    policy=local.derived_memory_policy([receipt])
    assert policy['available_ram_required_bytes']==1792*local.MIB
    assert policy['available_commit_required_bytes']==2560*local.MIB
    assert policy['process_rss_ceiling_bytes']==1280*local.MIB
    assert policy['system_available_floor_bytes']==512*local.MIB
    receipt['status']='FAIL'
    with pytest.raises(ValueError):local.derived_memory_policy([receipt])


def test_cache_reduction_is_bit_exact(synthetic):
    root,rows,norm=synthetic
    old=f.FinalReader(rows,root,norm);new=local.LocalReader(rows,root,norm)
    assert old.reader.budget==64*local.MIB and new.reader.budget==16*local.MIB
    for row,col in [(0,0),(64,64)]:
        key=('2020/fire_1','2020-09-06',row,col)
        x,t=old.context(key);xx,tt=new.context(key);assert np.array_equal(x,xx)
        p=np.full((1,64,64),.78,np.float32)
        old.record_prediction(t,p);new.record_prediction(tt,p)
        for a,b in zip(old.target_after_prediction(t),new.target_after_prediction(tt)):assert np.array_equal(a,b)


def test_2021_guard_inherited_before_any_io(tmp_path):
    with pytest.raises(PermissionError,match='2021 reader'):
        local.LocalReader([dict(event_id='2021/fire_1')],tmp_path,{})


def test_event_backup_is_lossless_and_exclusive(tmp_path):
    out=tmp_path/'pred';backup=tmp_path/'backup';writer=local.SafePredictionArchive(out,backup)
    p=np.array([.78,.12345678],np.float32);y=np.array([1,0]);v=np.ones(2);o=np.zeros(2)
    writer.add(('SYNTHETIC_A','2020-01-01',0,0),p,y,v,o)
    writer.add(('SYNTHETIC_B','2020-01-01',0,0),p,y,v,o)
    assert (backup/'event_000.bin').is_file()  # backup made at the event boundary
    writer.finish()
    assert f.sha(out/'event_000.bin')==f.sha(backup/'event_000.bin')
    assert f.sha(out/'manifest.json')==f.sha(backup/'manifest.json')
    assert np.array_equal(np.fromfile(backup/'event_000.bin',dtype=f.PIXEL)['p'],p)


def test_disabled_authorization_before_model_output_or_rasters(tmp_path,monkeypatch):
    disabled=tmp_path/'disabled.json';f.save(disabled,{'authorized':False})
    monkeypatch.setattr(local,'verify_local',lambda *a:pytest.fail('disabled grant must fail first'))
    with pytest.raises(PermissionError):local.evaluate(tmp_path,tmp_path,disabled,tmp_path,tmp_path/'output')
    with pytest.raises(PermissionError):local.consume_local_authorization(tmp_path,tmp_path,disabled,tmp_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_memory_guard_stops_on_low_memory(monkeypatch):
    import _thread,time
    signals=[]
    monkeypatch.setattr(local,'memory_snapshot',lambda:dict(rss_bytes=1,available_bytes=1))
    monkeypatch.setattr(_thread,'interrupt_main',lambda:signals.append(True))
    with local.MemoryGuard(dict(process_rss_ceiling_bytes=10,system_available_floor_bytes=5)) as guard:
        for _ in range(20):
            if signals:break
            time.sleep(.025)
    assert signals and guard.failure['reason']=='measured memory envelope exceeded'


def test_real_lock_never_consumed_by_preflight(tmp_path,synthetic,monkeypatch):
    import importlib.metadata as md
    root,rows,norm=synthetic;_,keys,stats=f.inventory(rows,['2020/fire_1'])
    base=tmp_path/'adata';base.mkdir()
    paths={name:str(base/name) for name in ('scratch','results','backups')}
    paths.update(adata_root=str(base),ledger=str(tmp_path/'ledger'),archive=str(tmp_path/'source.zip'),rasters=str(root))
    Path(paths['ledger']).mkdir()
    for name in ('scratch','results','backups'):Path(paths[name]).mkdir()
    profile=tmp_path/'profile';profile.mkdir();f.save(profile/'operational_seal.json',{'synthetic':True})
    f.save(profile/'hardware_initial.json',{'volumes':{'C':{'serial':1},'D':{'serial':1}}})
    f.save(profile/'validation_memory.json',dict(status='PASS',runtime={k:md.version(k) for k in ('torch','numpy','rasterio','pyarrow')},validation_parity=dict(passed=True)))
    frozen=tmp_path/'freeze';frozen.mkdir();f.save(frozen/'selection.json',{'synthetic':True})
    policy=local.derived_memory_policy([dict(status='PASS',test_tiff_reads=0,memory=dict(peak_rss_bytes=600*local.MIB,peak_private_commit_bytes=1024*local.MIB))])
    monkeypatch.setattr(local,'verify_local',lambda *a:(dict(paths=paths,memory_policy=policy),{'test_identity':'SYNTHETIC'},rows,keys,frozen))
    monkeypatch.setattr(local,'memory_snapshot',lambda:dict(total_bytes=16*f.GIB,available_bytes=2*f.GIB,available_commit_bytes=4*f.GIB))
    monkeypatch.setattr(local,'volume_info',lambda *a:dict(filesystem='NTFS',serial=1))
    class Disk:free=400*f.GIB
    monkeypatch.setattr(local.shutil,'disk_usage',lambda *a:Disk())
    original=f.sha;monkeypatch.setattr(f,'sha',lambda p:f.ARCHIVE_SHA if str(p)==paths['archive'] else original(p))
    r=local.preflight(tmp_path,profile)
    assert r['technical_status']=='PASS' and r['checks']['independent_authorization']=='BLOCKED'
    assert not list(Path(paths['ledger']).glob('final-test-2021-*.lock.json'))
    monkeypatch.setattr(local,'memory_snapshot',lambda:dict(total_bytes=16*f.GIB,available_bytes=f.GIB,available_commit_bytes=4*f.GIB))
    assert local.preflight(tmp_path,profile)['checks']['ram']=='FAIL'


def test_local_independent_lock_is_one_use(tmp_path,monkeypatch):
    ledger=tmp_path/'ledger';ledger.mkdir();results=tmp_path/'results';results.mkdir()
    profile=tmp_path/'profile';profile.mkdir();f.save(profile/'operational_seal.json',{'synthetic':True})
    f.save(profile/'validation_memory.json',{'synthetic':True})
    frozen=tmp_path/'freeze';frozen.mkdir();f.save(frozen/'selection.json',{'synthetic':True})
    paths=dict(results=str(results),ledger=str(ledger),scratch_root='unused',scratch=str(tmp_path),rasters=str(tmp_path),archive=str(tmp_path),C_backups=str(tmp_path))
    seal=dict(paths=paths,max_seconds=21600)
    monkeypatch.setattr(local,'verify_local',lambda *a:(seal,dict(test_identity='SYNTHETIC-NOT-REAL-DATA'),[],[],frozen))
    monkeypatch.setattr(f,'ephemeral_ledger',lambda p:False) # temp fixture only
    pre=tmp_path/'p.json';f.save(pre,dict(technical_status='PASS',operational_seal_sha256=f.sha(profile/'operational_seal.json'),freeze_sha256=f.sha(frozen/'selection.json'),created_utc=datetime.now(timezone.utc).isoformat()))
    out=results/'synthetic'
    a=dict(schema='geofirex-final-v2-local-cpu-authorization-1',authorized=True,training_authorized=False,
      freeze_sha256=f.sha(frozen/'selection.json'),operational_seal_sha256=f.sha(profile/'operational_seal.json'),
      test_identity='SYNTHETIC-NOT-REAL-DATA',preflight_sha256=f.sha(pre),qualification_sha256=f.sha(profile/'validation_memory.json'),
      output=str(out),ledger_root=str(ledger),scratch_root=str(tmp_path),raster_root=str(tmp_path),archive=str(tmp_path),C_backup_root=str(tmp_path),
      device='cpu',cpu_only=True,durable_shared_ledger_confirmed=True,runtime_mounts_verified=True,max_seconds=21600,
      session_seconds_remaining=21600,authorized_by='SYNTHETIC-FIXTURE',authorized_at_utc=datetime.now(timezone.utc).isoformat(),authorization_id='SYNTHETIC-ONLY-0001')
    auth=tmp_path/'SYNTHETIC_ONLY.json';f.save(auth,a)
    token=local.consume_local_authorization(tmp_path,profile,auth,pre,out);assert Path(token.ledger_file).is_file()
    with pytest.raises(FileExistsError):local.consume_local_authorization(tmp_path,profile,auth,pre,out)


def test_cpu_controller_synthetic_end_to_end_with_C_backup(tmp_path,synthetic,monkeypatch):
    import torch
    from geofirex.benchmark_models import StackedUNet
    root,rows,norm=synthetic;_,keys,stats=f.inventory(rows,['2020/fire_1'])
    results=tmp_path/'results';results.mkdir();cback=tmp_path/'C_backups';cback.mkdir()
    paths=dict(scratch=str(tmp_path),rasters=str(root),archive=str(tmp_path/'SYNTHETIC.archive'),C_backups=str(cback))
    seal=dict(paths=paths,max_seconds=21600,archive_reserve_seconds=1800,
              memory_policy=dict(process_rss_ceiling_bytes=4*f.GIB,system_available_floor_bytes=0))
    science=dict(selected='SYNTHETIC-ONLY',test_identity='SYNTHETIC',keys_sha256=f.digest(keys))
    frozen=tmp_path/'freeze';frozen.mkdir();f.save(frozen/'selection.json',science)
    profile=tmp_path/'profile';profile.mkdir();f.save(profile/'operational_seal.json',{'synthetic':True})
    (tmp_path/f.INDEX).mkdir(parents=True);f.save(tmp_path/f.INDEX/'normalization.json',norm)
    auth=tmp_path/'SYNTHETIC-ONLY.json';f.save(auth,{'authorized':True})
    ledger=tmp_path/'SYNTHETIC.lock';f.save(ledger,{'synthetic':True})
    monkeypatch.setattr(local,'verify_local',lambda *a:(seal,science,rows,keys,frozen))
    monkeypatch.setattr(local,'preflight',lambda *a:dict(technical_status='PASS'))
    monkeypatch.setattr(local,'consume_local_authorization',lambda *a:f.PixelPermit('SYNTHETIC',str(ledger),'SYNTHETIC-ONLY'))
    model=StackedUNet(8,3,base_channels=8,depth=3)
    monkeypatch.setattr(f,'selected_checkpoint',lambda *a:dict(model=model.state_dict()))
    monkeypatch.setattr(torch.optim.Optimizer,'step',lambda *a,**k:pytest.fail('training forbidden'))
    out=results/'synthetic';package=Path(local.evaluate(tmp_path,profile,auth,tmp_path/'unused',out))
    backup=cback/'SYNTHETIC-ONLY'
    assert f.sha(package)==f.sha(backup/package.name)
    assert f.sha(out/'predictions/manifest.json')==f.sha(backup/'predictions/manifest.json')
    assert (backup/'INFERENCE_COMPLETE.json').is_file()
    assert f.load(out/'evaluation.json')['gpu_compute_seconds']==0
