"""Completed ablation and fixed confirmation; tests never infer the corpus."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from geofirex.calibration_io import sha
from geofirex.science_results import ScientificResults,create_results_app,ROOT,DEFAULT_ZIP
from geofirex.unet_confirmation import audit,BASE,ABLATION,compare
from geofirex.unet_results import UnetResults


async def request(app,path,query=''):
    messages=[];delivered=False
    async def receive():
        nonlocal delivered
        if delivered:await asyncio.Event().wait()
        delivered=True
        return dict(type='http.request',body=b'',more_body=False)
    async def send(m):messages.append(m)
    await app(dict(type='http',asgi=dict(version='3.0',spec_version='2.4'),http_version='1.1',method='GET',scheme='http',
        path=path,raw_path=path.encode(),query_string=query.encode(),headers=[],client=('127.0.0.1',1),server=('localhost',80),root_path=''),receive,send)
    return next(m['status'] for m in messages if m['type']=='http.response.start'),b''.join(m.get('body',b'') for m in messages)


def test_three_real_unet_runs_verify_and_choose_dice_without_exaggeration():
    result=audit()
    assert result['status']=='VERIFIED'
    assert len(result['variants'])==3
    assert result['winner']['pos_weight']==10 and result['winner']['threshold']==.78
    assert result['delta_dice_10_vs_30']==pytest.approx(.0008651086943657349)
    assert result['delta_ap_10_vs_30']<0
    assert len({r['initial_state_sha256'] for r in result['variants']})==1
    assert len({r['checkpoint_sha256'] for r in result['variants']})==3


def test_altered_threshold_in_saved_selection_is_rejected(tmp_path):
    output=tmp_path/'altered.zip'
    with zipfile.ZipFile(ABLATION) as source,zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            data=source.read(name)
            if name=='B2_posweight_10/selection32.json':
                row=json.loads(data);row['threshold']=.8;data=json.dumps(row).encode()
            target.writestr(name,data)
    with pytest.raises(ValueError,match='maximum'):audit(output)


def test_confirmation_uses_one_fixed_threshold_and_same_historical_pixels():
    store=UnetResults(ScientificResults());confirmed=store.confirmed()
    assert confirmed is not None,'The one real CPU confirmation must finish before this check'
    evaluation=confirmed['evaluation']
    assert len(evaluation['curves'])==1 and evaluation['curves'][0]['threshold']==.78
    assert len(evaluation['per_event'])==169 and evaluation['tiles']==65012
    assert confirmed['execution']['device']=='cpu' and not confirmed['execution']['gpu_used']
    assert confirmed['execution']['test_2021_pixels_opened']==0
    assert not set(evaluation['events'])&set(store.audited['protocol']['calibration_events'])
    baseline=confirmed['manifest']['comparison']
    assert len({r['valid_pixels'] for r in baseline})==1
    assert len({r['positive_pixels'] for r in baseline})==1
    assert compare(evaluation,store.audited)['rows']==baseline
    for key in ('tp','fp','fn','tn'):
        assert confirmed['new_fire']['new_fire_candidates'][key]+confirmed['new_fire']['observed_fire_candidates'][key]==evaluation['curves'][0][key]


def test_results_launch_does_not_import_historical_predictor_or_torch():
    code='''import sys,builtins,types,importlib.util
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'src'))
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=='torch' or name.startswith('torch.') or name=='geofirex.serving':
        raise AssertionError('Historical inference dependency loaded: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.modules['uvicorn']=types.SimpleNamespace(run=lambda app,**kwargs: print('RESULTS_ONLY_LAUNCH_PASS'))
spec=importlib.util.spec_from_file_location('serve_demo',root/'scripts/serve_demo.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.main(['--results-only','--port','8014'])
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    assert 'RESULTS_ONLY_LAUNCH_PASS' in result.stdout


def test_confirmed_api_maps_assets_and_2021_lock():
    original=sha(DEFAULT_ZIP);app=create_results_app()
    status,body=asyncio.run(request(app,'/science/unet-ablation'));data=json.loads(body)
    assert status==200 and data['confirmation']['status']=='COMPLETE'
    assert data['selected']['threshold']==.78 and data['selected']['pos_weight']==10
    assert len(data['confirmation']['historical_new_fire'])==2
    assert len(data['confirmation']['per_event_new_fire'])==169
    status,body=asyncio.run(request(app,'/science/retained-model'))
    assert status==200 and json.loads(body)['test_2021_locked']
    status,_=asyncio.run(request(app,'/science/unet-assets/confirmation_locked.json'));assert status==404
    for case in data['confirmation']['maps']:
        for layer in case['layers'].values():
            status,body=asyncio.run(request(app,'/science/unet-assets/'+case['directory']+'/'+layer))
            assert status==200 and body.startswith(b'\x89PNG')
    status,_=asyncio.run(request(app,'/science/events','phase=2021'));assert status==422
    status,body=asyncio.run(request(app,'/science/big-data'));spark=json.loads(body)
    assert status==200 and spark['master']=='local[2]' and not spark['distributed_cluster']
    assert sha(DEFAULT_ZIP)==original


def test_confirmation_integrity_failure_is_not_silently_served(tmp_path):
    root=tmp_path/'tampered';(root/'inputs').mkdir(parents=True)
    shutil.copyfile(ABLATION,root/'inputs'/ABLATION.name)
    shutil.copyfile(BASE/'ablation_verified.json',root/'ablation_verified.json')
    source=BASE/'confirmation_cpu';out=root/'confirmation_cpu';out.mkdir()
    for name in ('status_complete.json','confirmation_locked.json','app_manifest.json'):
        shutil.copyfile(source/name,out/name)
    status=json.loads((out/'status_complete.json').read_text());status['manifest_sha256']='0'*64
    (out/'status_complete.json').write_text(json.dumps(status))
    with pytest.raises(ValueError,match='integrity'):UnetResults(ScientificResults(),root).payload()
