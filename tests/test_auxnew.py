"""Synthetic CPU gradients, never an optimizer or model update."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
import torch
from geofirex.auxnew_model import AuxiliaryUNet,AuxiliaryLoss
from geofirex.benchmark_models import StackedUNet
from geofirex.propagation_experiment import state_hash,EXPECTED_MAIN_INITIAL_SHA
from geofirex.unet import SegmentationLoss
from geofirex import auxnew_experiment as exp

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_optimizer(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',lambda *a,**kw:pytest.fail('No optimizer in CPU tests'))


def test_exact_main_initial_weights_and_primary_only_forecast():
    torch.manual_seed(42);base=StackedUNet(8,3)
    torch.manual_seed(42);candidate=AuxiliaryUNet()
    assert state_hash(candidate,True)==state_hash(base)
    # Cross-version CPU initialization is not the archived GPU qualification.
    original=json.loads((ROOT/exp.REPORT/'original_initial_identity.json').read_text())
    assert original['model_state_sha256']==EXPECTED_MAIN_INITIAL_SHA
    assert sum(p.numel() for p in candidate.parameters())-sum(p.numel() for p in base.parameters())==9
    x=torch.randn(2,3,8,16,24)
    assert torch.equal(base(x),candidate(x))
    with torch.no_grad():candidate.new_head.bias.fill_(1000);candidate.new_head.weight.fill_(1000)
    assert torch.equal(base(x),candidate(x))
    with pytest.raises(ValueError):candidate(torch.zeros(1,3,13,16,16))


def test_original_initial_digest_gate_refuses_a_different_initialization():
    torch.manual_seed(123);wrong=AuxiliaryUNet()
    with pytest.raises(ValueError,match='no optimizer'):exp.verify_initial(wrong)


def test_coefficient_zero_exact_primary_value_and_gradients():
    main=torch.randn(2,1,8,8,requires_grad=True);aux=torch.zeros_like(main,requires_grad=True)
    y=torch.zeros_like(main);v=torch.ones_like(main);o=torch.zeros_like(main)
    a=AuxiliaryLoss(0)(main,aux,y,v,o)[0];b=SegmentationLoss(.5,10)(main,y,v)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,main,retain_graph=True)[0],torch.autograd.grad(b,main)[0])


@pytest.mark.parametrize('empty_mask',[False,True])
def test_auxiliary_gradients_valid_new_domain_and_negatives(empty_mask):
    main=torch.zeros(1,1,8,8,requires_grad=True);aux=torch.zeros_like(main,requires_grad=True)
    y=torch.zeros_like(main);y[0,0,2,2]=1;v=torch.ones_like(main);v[0,0,0,0]=0
    o=torch.zeros_like(main);o[0,0,1,1]=1
    if empty_mask:v.zero_()
    loss,_,extra=AuxiliaryLoss()(main,aux,y,v,o)
    grad=torch.autograd.grad(loss,aux)[0]
    assert grad[0,0,0,0]==0 and grad[0,0,1,1]==0 and torch.isfinite(grad).all()
    if empty_mask:assert not grad.any() and extra==0
    else:assert grad[0,0,2,2]<0 and grad[0,0,3,3]>0


def test_no_positive_batch_still_penalizes_false_alarms():
    x=torch.zeros(1,1,8,8,requires_grad=True)
    loss=AuxiliaryLoss()(x,x,torch.zeros_like(x),torch.ones_like(x),torch.zeros_like(x))[2]
    assert torch.autograd.grad(loss,x)[0].gt(0).all()
    with pytest.raises(ValueError):AuxiliaryLoss(.5)


def test_preregistered_one_factor_budget_and_unchanged_science():
    config,_,protocol=exp.identity(ROOT)
    base=exp.old.identity(ROOT)[0]
    for key in ['channel_names','loss','sampler','normalization_sha256','index_sha256','event_manifest_sha256','epochs','total_updates','threshold']:
        assert config[key]==base[key]
    assert config['channels_per_day']==8 and config['model_input_channels']==24
    assert config['auxiliary_task']['inference']=='primary_head_only'
    assert not protocol['training_authorized'] and not protocol['launch_enabled']
    feasibility=exp.old.read(ROOT/exp.REPORT/'cpu_feasibility.json')
    assert feasibility['CPU_time_ratio']<2 and not feasibility['GPU_measured'] and not feasibility['optimizer_created']
    assert exp.old.read(ROOT/exp.REPORT/'acceptance.json')==exp.old.read(ROOT/'reports/fire13_20261122_v1/acceptance.json')


@pytest.mark.parametrize('field,value',[('authorized',False),('action','fire13-fit'),('device','cpu'),
    ('execution_manifest_sha256','0'*64),('session_remaining_minutes',299),('quota_remaining_card_hours',9.99),('runs',2)])
def test_bad_grants_refused_before_cuda_or_output(tmp_path,monkeypatch,field,value):
    a=exp.old.read(ROOT/exp.REPORT/'authorization_auxnew-fit_DISABLED.json')
    a.update(authorized=True,authorized_by='CPU fixture',authorized_at='fixture',quota_checked_at='fixture',authorization_id='fixture',
        output_parent=str(tmp_path),session_remaining_minutes=300,quota_remaining_card_hours=10,card_hours_reserved=10,
        local_backup_destination='local',external_backup_destination='external')
    a[field]=value;p=tmp_path/'grant.json';p.write_text(json.dumps(a))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('No CUDA'))
    with pytest.raises(ValueError):exp.authorize(ROOT,p,'auxnew-fit',tmp_path)


def test_notebook_flags_disabled_and_payload_isolated_transitive_imports(tmp_path):
    nb=exp.old.read(ROOT/'notebooks/GEOFIREX_AUXNEW_V2_DISABLED.ipynb');data=None
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source']);compile(source,'AUXNEW-notebook','exec')
        assert not cell['outputs'] and cell['execution_count'] is None
        if 'RUN_FIT = False' in source:assert 'RUN_GPU_PREFLIGHT = False' in source
        for node in ast.parse(source).body:
            if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='DATA' for n in node.targets):
                data=base64.b64decode(node.value.args[0].value)
    build=exp.old.read(ROOT/exp.REPORT/'notebook_build.json')
    assert hashlib.sha256(data).hexdigest()==build['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert not any(n.endswith(('.pt','.tif','.bin')) for n in z.namelist());z.extractall(tmp_path)
    code="import sys,importlib;sys.path.insert(0,sys.argv[1]);[importlib.import_module(m) for m in sys.argv[3:]];from geofirex import auxnew_experiment as c;print(c.identity(sys.argv[2])[0]['run_id'])"
    process=subprocess.run([sys.executable,'-I','-c',code,str(tmp_path/'src'),str(tmp_path),*build['modules']],
        cwd=tmp_path,capture_output=True,text=True,timeout=180)
    assert process.returncode==0,process.stderr
    assert exp.RUN in process.stdout
