"""CPU-only feasibility and disabled offline notebook for one auxiliary task."""
from copy import deepcopy
import base64
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import zipfile
import numpy as np
import torch
from geofirex import v2_execution as old
from geofirex.protocol_v2 import sha256
from geofirex.auxnew_model import AuxiliaryUNet,AuxiliaryLoss
from geofirex.auxnew_experiment import REPORT,RUN
from geofirex.benchmark_models import StackedUNet
from geofirex.unet import SegmentationLoss
from prepare_fire13_recovery import closure,cell

ROOT=Path(__file__).resolve().parents[1]


def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')


def profile():
    torch.set_num_threads(2);torch.manual_seed(42)
    baseline=StackedUNet(8,3);torch.manual_seed(42);candidate=AuxiliaryUNet()
    x=torch.randn(16,3,8,64,64);x[:,-1,7]=0
    target=torch.zeros(16,1,64,64);target[0,0,32,32]=1;valid=torch.ones_like(target)
    timings={}
    for name,model in [('C8',baseline),('AUXNEW',candidate)]:
        samples=[]
        for i in range(4):
            model.zero_grad(set_to_none=True);start=time.perf_counter()
            if name=='C8':loss=SegmentationLoss(.5,10)(model(x),target,valid)
            else:loss=AuxiliaryLoss()(*model.heads(x),target,valid,x[:,-1,7:8])[0]
            # Derivative calculation on random, untrained fixtures; no optimizer/update.
            torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
            if i:samples.append(time.perf_counter()-start)
        timings[name]=dict(median_seconds=statistics.median(samples),samples_seconds=samples,
            parameters=sum(p.numel() for p in model.parameters()))
    with zipfile.ZipFile('D:/GeoFire-X/GeoFire-X-V2/B2_C8_V2_seed42-COMPLETE-48e35c59.zip') as z:
        resources=json.loads(z.read('resources_fit.json'))
    return dict(status='GO_CPU_PREPARATION_ONLY_GPU_LAUNCH_NOT_AUTHORIZED',timings=timings,
        CPU_time_ratio=timings['AUXNEW']['median_seconds']/timings['C8']['median_seconds'],
        additional_parameters=9,additional_parameter_float32_bytes=36,
        one_auxiliary_logit_batch_bytes=16*64*64*4,GPU_additional_memory_reserve_bytes=256*1024**2,
        archived_C8_fit_resources=resources,GPU_measured=False,optimizer_created=False,
        training_executed=False,gradient_checks_only=True,
        uncertainty='CPU timing does not predict GPU timing; measured C8 GPU fit and conservative reserves are the reference',
        session_minutes=300,card_hours_reserved=10,fit_deadline_seconds=5400,evaluation_deadline_seconds=8100,
        RAM_total_min_bytes=16*1024**3,RAM_available_min_bytes=8*1024**3,disk_min_bytes=20*1024**3,
        launch_conditions=['new explicit user authorization','current quota >= 10 card-hours',
            'at least 300 session minutes','pinned T4 runtime preflight','only one fresh seed42 run'],
        scientific_risks=['validation80 already observed, exploratory only','one seed, no guaranteed gain',
            'prior specialist-gated head failed on legacy32; this primary-only auxiliary task is distinct',
            'eight causal C8 channels unchanged; no new weather or fuel information'])


def main():
    out=ROOT/REPORT;out.mkdir(exist_ok=True)
    base,original,_=old.identity(ROOT)
    if not (out/'cpu_feasibility.json').exists():write(out/'cpu_feasibility.json',profile())
    feasibility=old.read(out/'cpu_feasibility.json')
    if feasibility['CPU_time_ratio']>2:
        write(out/'decision.json',dict(status='NO_GO',reason='CPU derivative overhead exceeds conservative 2x bound'))
        print('NO_GO: no notebook generated');return
    config=dict(base,run_id=RUN,status='EXPERIMENTAL_DISABLED',
        auxiliary_task=dict(coefficient=.25,loss='BCEWithLogits',pos_weight=10.,domain='valid & observed_J==0',
            inference='primary_head_only',additional_parameters=9))
    write(out/'protocol.json',dict(schema='geofirex-auxnew-preregistration-1',config=config,
        hypothesis='Auxiliary supervision may improve main-head new-fire ranking without routing scores to a specialist',
        changed_factor='shared-decoder auxiliary BCE supervision only; lambda=.25 fixed before any fit',
        comparison='same V2 C8 baseline, 80 events, 30882 tiles, threshold .78; primary head only',
        no_sweep=True,one_fit_only=True,no_resume=True,epoch_selection=4,
        original_scientific_identities=77,training_authorized=False,launch_enabled=False,
        scientific_scope='Exploratory on already-observed 2020 validation; test2021 remains closed',
        prior_experiment='UNET_NEW_FIRE_HEAD_V1 failed legacy32 criteria; different split, gated specialist and auxiliary Dice; no score transfer',
        acceptance='Same eight frozen V2 gates; no automatic promotion',
        initial_main_state_sha256=old.read(out/'original_initial_identity.json')['model_state_sha256'],
        initialization='fresh seed42 with exact C8 initial backbone digest checked before optimizer; never initialize from any trained checkpoint'))
    for target,source in [('acceptance.json','reports/fire13_20261122_v1/acceptance.json'),
                          ('baseline_reference.json','reports/c8_propagation_20261121_v1/baseline_reference.json')]:
        (out/target).write_bytes((ROOT/source).read_bytes())
    modules,external=closure(ROOT,['geofirex.auxnew_experiment','geofirex'])
    files=dict(original['files'])
    for path in modules.values():
        digest=sha256(ROOT/path)
        if path in files and files[path]!=digest:raise ValueError('Frozen source changed: '+path)
        files[path]=digest
    for name in ['protocol.json','cpu_feasibility.json','acceptance.json','baseline_reference.json','original_initial_identity.json']:
        files[(REPORT/name).as_posix()]=sha256(out/name)
    files[(old.REPORT/'execution_manifest.json').as_posix()]=sha256(ROOT/old.REPORT/'execution_manifest.json')
    write(out/'execution_manifest.json',dict(schema='geofirex-auxnew-execution-1',run_id=RUN,files=files,
        original_scientific_identities=77,training_authorized=False))
    actions=['auxnew-gpu-preflight','auxnew-fit','auxnew-evaluate']
    for action in actions:
        write(out/('authorization_'+action+'_DISABLED.json'),dict(authorized=False,action=action,experiment_id=RUN,
            device='cuda',runs=1,execution_manifest_sha256=sha256(out/'execution_manifest.json'),
            authorized_by='',authorized_at='',authorization_id='',output_parent='/kaggle/working',
            session_remaining_minutes=None,quota_remaining_card_hours=None,card_hours_reserved=None,quota_checked_at='',
            local_backup_destination='',external_backup_destination='D:\\GeoFire-X\\GeoFire-X-V2'))
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        names=set(files)|{(REPORT/'execution_manifest.json').as_posix()}
        names|={(REPORT/('authorization_'+a+'_DISABLED.json')).as_posix() for a in actions}
        for name in sorted(names):z.writestr(name,(ROOT/name).read_bytes())
    data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
    (out/'auxnew_code.zip').write_bytes(data)
    cells=[cell('markdown','# AUXNEW C8 V2 — EXPERIMENTAL, DISABLED\n\nUn seul candidat : supervision auxiliaire BCE .25, tête primaire seule en inférence. Huit canaux C8 inchangés, quatre époques, seed42, seuil .78. Aucun fit automatique. Autorisation utilisateur distincte requise avant toute activation. Onze partitions existantes, Internet OFF. Test 2021 fermé. Aucun appel FIRE13.\n',0),
        cell('code',"RUN_GPU_PREFLIGHT = False\nRUN_FIT = False\nimport os,sys\nif 'torch' in sys.modules: raise RuntimeError('Restart kernel first')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n",1),
        cell('code',"from pathlib import Path\nimport base64,hashlib,io,zipfile,json\n"+f"DATA=base64.b64decode({base64.b64encode(data).decode()!r})\nassert hashlib.sha256(DATA).hexdigest()=={digest!r}\n"+
            "ROOT=Path('/kaggle/working/geofirex-auxnew-code')\nROOT.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(DATA)) as z:\n    for n in z.namelist():assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())\n    z.extractall(ROOT)\nsys.path.insert(0,str(ROOT/'src'))\nfrom geofirex import auxnew_experiment as c\nprint(c.identity(ROOT)[2])\n",2),
        cell('code',"EXPECTED={r['map_sha256'] for r in c.old.read(ROOT/c.old.REPORT/'partitions.json').values()}\nMAPS=[p for p in Path('/kaggle/input').rglob('mount-map.json') if c.sha256(p) in EXPECTED]\nassert len(MAPS)==11\nprint(c.old.preflight(ROOT,MAPS))\n",3),
        cell('markdown','Ne pas créer de grant sans nouvelle autorisation utilisateur explicite. Copier les trois templates DISABLED dans /kaggle/working, renseigner opérateur/date/identifiant unique/quota actuel/deux sauvegardes. Réserver 300 minutes, 10 heures-carte, 16/8 GiB RAM totale/disponible et 20 GiB disque. Le temps CPU ne qualifie pas le GPU. Contrôler T4, versions exactes et cuDNN ; aucun pip automatique. Ne pas supprimer les verrous.\n',4),
        cell('code',"GPU_AUTH=Path('/kaggle/working/auxnew_gpu_authorization.json')\nFIT_AUTH=Path('/kaggle/working/auxnew_fit_authorization.json')\nEVAL_AUTH=Path('/kaggle/working/auxnew_evaluate_authorization.json')\nOUT=Path('/kaggle/working')/c.RUN\nif RUN_GPU_PREFLIGHT:\n    c.authorize(ROOT,GPU_AUTH,'auxnew-gpu-preflight',OUT.parent)\n    print(c.preflight(ROOT,MAPS,OUT.parent,gpu=True))\nelse: print('GPU disabled')\n",5),
        cell('code',"if RUN_FIT:\n    assert RUN_GPU_PREFLIGHT\n    c.fit(ROOT,MAPS,OUT,FIT_AUTH,EVAL_AUTH,GPU_AUTH)\nelse: print('Fit disabled; no optimizer created')\n",6),
        cell('code',"from IPython.display import display,FileLink\nfor p in sorted(Path('/kaggle/working').glob(c.RUN+'-*.zip')):\n    print(c.old.verify_zip(p,p.with_suffix('.sha256').read_text().split()[0]))\n    display(FileLink(str(p)));display(FileLink(str(p.with_suffix('.sha256'))))\n",7)]
    notebook=ROOT/'notebooks/GEOFIREX_AUXNEW_V2_DISABLED.ipynb'
    write(notebook,dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3'))))
    write(out/'notebook_build.json',dict(payload_sha256=digest,notebook_sha256=sha256(notebook),
        modules=modules,external_import_roots=external,all_flags_disabled=True,published=False))
    write(out/'decision.json',dict(status='GO_CPU_PREPARATION_ONLY',GPU_launch='NO_GO_WITHOUT_NEW_USER_AUTHORIZATION_CURRENT_QUOTA_AND_PREFLIGHT',
        no_training_executed=True,no_performance_obtained=True,only_one_candidate=True))
    print(json.dumps(dict(decision=old.read(out/'decision.json'),feasibility=feasibility),indent=2))


if __name__=='__main__':main()
