"""New forecast contract and guards. No corpus training or inference in tests."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from geofirex.benchmark_models import make_model
from geofirex.fast_compact import FastCompactTiles
from geofirex.propagation_experiment import PropagationUNet,PropagationLoss,state_hash,run,resume_budget
from geofirex.propagation_results import verified
from geofirex.science_results import create_results_app,ROOT
from test_unet_confirmation import request

BASE=ROOT/'reports/propagation_20261112_v1'


def test_main_unet_exact_initial_identity_and_specialist_gating():
    torch.set_num_threads(2);torch.manual_seed(42);old=make_model('B2',8,3)
    torch.manual_seed(42);new=PropagationUNet()
    assert state_hash(new,True)==state_hash(old)
    x=torch.randn(2,3,8,16,24);x[:,-1,7]=0;x[0,-1,7,2:5,3:6]=1
    main,specialist=new.heads(x)
    assert torch.equal(main,old(x))
    with torch.no_grad():
        new.network.head.weight.zero_();new.network.head.bias.fill_(2)
        new.new_head.weight.zero_();new.new_head.bias.fill_(-2)
    output=new(x)
    assert torch.equal(output[x[:,-1,7:8]==1],torch.full_like(output[x[:,-1,7:8]==1],2))
    assert torch.equal(output[x[:,-1,7:8]==0],torch.full_like(output[x[:,-1,7:8]==0],-2))


def test_auxiliary_gradients_ignore_invalid_and_observed_keep_negative_rejection():
    main=torch.zeros(1,1,8,8,requires_grad=True);new=torch.zeros_like(main,requires_grad=True)
    target=torch.zeros_like(main);target[0,0,2,2]=1
    valid=torch.ones_like(main);valid[0,0,0,0]=0
    observed=torch.zeros_like(main);observed[0,0,1,1]=1
    loss,_,_=PropagationLoss()(main,new,target,valid,observed);loss.backward()
    assert new.grad[0,0,0,0]==0 and new.grad[0,0,1,1]==0
    assert new.grad[0,0,2,2]<0 and new.grad[0,0,3,3]>0
    assert main.grad[0,0,0,0]==0 and torch.isfinite(new.grad).all()
    # No new target: BCE still penalizes false alarms. All invalid: zero loss/gradient.
    new2=torch.zeros_like(main,requires_grad=True)
    loss,_,aux=PropagationLoss()(main.detach(),new2,torch.zeros_like(target),valid,observed)
    loss.backward();assert aux>0 and new2.grad[0,0,3,3]>0
    zero,_,_=PropagationLoss()(main,new,target,valid*0,observed)
    assert float(zero.detach())==0


def test_preflight_never_reads_tiles_initializes_model_or_opens_cuda(monkeypatch,tmp_path):
    import geofirex.propagation_experiment as module
    def forbidden(*a,**k):raise AssertionError('Pixels/model/GPU forbidden in metadata-only mode')
    monkeypatch.setattr(FastCompactTiles,'__getitem__',forbidden)
    monkeypatch.setattr(module,'PropagationUNet',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    args=SimpleNamespace(corpus=ROOT/'reports/mission_20261111/compact_full',index=ROOT/'reports/gpu_acceleration_20261111_v1/index',
        protocol=BASE/'protocol.json',baseline32=BASE/'baseline32.json',mode='preflight',out=tmp_path/'unused')
    plan=run(args)
    assert plan['runs']==1 and plan['updates']==23404 and plan['gpu_wall_limit_seconds']==7200
    assert len(plan['calibration_events'])==32 and plan['model_context']==3 and not args.out.exists()
    args.mode='train';args.execute_training=False
    with pytest.raises(ValueError,match='Explicit'):run(args)


def test_saved_diagnostic_partitions_and_real_maps_without_new_inference():
    data,seal=verified()
    assert data['summary']['new_false_negative_share']==pytest.approx(262923/300710)
    assert data['summary']['events_zero_new_true_positives']==19
    assert len(data['per_event'])==169 and len(data['label_checks'])==3
    assert sum(c['tiles'] for c in data['label_checks'])==sum(c['tile_count'] for c in json.loads((ROOT/'reports/unet_confirmation_20261112_v1/confirmation_cpu/B2_w10_holdout169/evaluation.json').read_text())['maps'])
    assert len(seal['assets'])==15
    for k in ('tp','fp','fn','tn'):
        assert data['new_fire'][k]+data['observed_fire'][k]==data['full'][k]
        assert sum(r[k] for r in data['per_event'])==data['new_fire'][k]


def test_propagation_endpoints_images_whitelist_and_integrity(tmp_path):
    app=create_results_app();status,body=asyncio.run(request(app,'/science/propagation'))
    data=json.loads(body);assert status==200 and data['full']['threshold']==.78
    for case in data['cases']:
        for name in case['layers'].values():
            status,body=asyncio.run(request(app,'/science/propagation-assets/'+name))
            assert status==200 and body.startswith(b'\x89PNG')
    assert asyncio.run(request(app,'/science/propagation-assets/diagnosis.json'))[0]==404
    assert asyncio.run(request(app,'/science/events','phase=2021'))[0]==422
    import shutil
    shutil.copy(BASE/'diagnosis.json',tmp_path);shutil.copy(BASE/'assets_manifest.json',tmp_path)
    p=tmp_path/'diagnosis.json';p.write_text(p.read_text().replace('SAVED_OUTPUTS_ONLY','ALTERED'))
    with pytest.raises(ValueError,match='integrity'):verified(tmp_path)


def test_notebook_bundle_is_offline_one_run_disabled_and_exact_scope():
    with zipfile.ZipFile(BASE/'propagation-code.zip') as z:
        for name in z.namelist():
            if name.endswith('.py'):compile(z.read(name),name,'exec')
        assert json.loads(z.read('protocol.json'))['calibration_events']==json.loads((BASE/'protocol.json').read_text())['calibration_events']
        assert b'mission_streaming import' not in z.read('geofirex/fast_compact.py')
    nb=json.loads((BASE/'GEOFIREX_PROPAGATION_NEW_FIRE_HEAD_V1.ipynb').read_text(encoding='utf-8'))
    cells=[c['source'] for c in nb['cells'] if c['cell_type']=='code']
    for source in cells:compile(source,'notebook','exec')
    assert any('RUN_TRAINING=False' in c for c in cells)
    assert any('RUN_SELECTION32=False' in c for c in cells)
    assert any('timeout=remaining' in c for c in cells)


def test_gpu_budget_fails_closed_after_unclean_stop(tmp_path):
    from geofirex.calibration_io import sha
    cp=tmp_path/'checkpoint_latest.pt';cp.write_bytes(b'budget test, no model')
    with pytest.raises(ValueError,match='missing'):resume_budget(cp,{'gpu_wall_seconds':100})
    account=tmp_path/'gpu_budget.json';account.write_text(json.dumps(dict(status='RESERVED',reserved_seconds=7200)))
    with pytest.raises(ValueError,match='reserved/spent'):resume_budget(cp,{'gpu_wall_seconds':100})
    account.write_text(json.dumps(dict(status='PAUSED',checkpoint_sha256=sha(cp),spent_seconds=600)))
    assert resume_budget(cp,{'gpu_wall_seconds':500})==600
    account.write_text(json.dumps(dict(status='PAUSED',checkpoint_sha256=sha(cp),spent_seconds=7180)))
    with pytest.raises(ValueError,match='exhausted'):resume_budget(cp,{'gpu_wall_seconds':500})


def test_training_rejects_two_visible_gpus_without_model_allocation(monkeypatch,tmp_path):
    import geofirex.propagation_experiment as module
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    def forbidden(*a,**kw):raise AssertionError('No model or GPU allocation after device count rejection')
    monkeypatch.setattr(module,'PropagationUNet',forbidden)
    args=SimpleNamespace(execute_training=True,resume=None,out=tmp_path/'unused')
    with pytest.raises(ValueError,match='exactly one GPU'):module.train_run(args,{},None)
    assert not args.out.exists()
