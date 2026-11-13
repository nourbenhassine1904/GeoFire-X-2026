"""Reuse frozen scientific metadata; prepare seven CPU notebooks and GPU templates.

No raw pixels, uploads, model imports, scientific audits, or historical writes.
"""
import argparse
import ast
import base64
from collections import Counter
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import zlib

BASE=Path('reports/recovery_20261107_v3/finalized/deployment/bundle')
OUT=Path('reports/kaggle_remaining_20261108_v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_notebook(folder,source,title,sources,*,gpu=False,small_control=True):
    ast.parse(source)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    nb=dict(cells=[dict(cell_type='markdown',metadata={},source=[f'# {title}\n','Prepared locally; never executed by preparation.\n']),
        dict(cell_type='code',metadata={},source=source.splitlines(keepends=True),execution_count=None,outputs=[])],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    payload=json.dumps(nb,indent=2).encode()
    if small_control and len(payload)>1_000_000: raise ValueError('CPU notebook control payload too large')
    (folder/'certify.ipynb').write_bytes(payload)
    kernel='nnnnbhss/'+title.lower().replace(' ','-')
    meta=dict(id=kernel,title=title,code_file='certify.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=gpu,enable_tpu=False,enable_internet=False,dataset_sources=sources,competition_sources=[],kernel_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    return dict(folder=str(folder),sha256=sha(folder/'certify.ipynb'),kernel=kernel,bytes=len(payload))


def source_blob(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,content in files.items(): archive.writestr(name,content)
    return base64.b64encode(stream.getvalue()).decode()


def training_notebooks(out,plan):
    """Generate templates now; enable execution only after all CPU receipts exist."""
    import yaml
    from kaggle_cpu_job import validate_receipt
    from record_lot001_certificate import verify_evidence
    out=Path(out)
    bundle=out/'training/bundle'
    gates=[]
    lot001=json.loads((bundle/'metadata/lot-001.json').read_text())
    verify_evidence(plan['lot001_evidence'],lot001)
    gates.append(dict(number=1,handle=lot001['handle'],identity=lot001['identity'],
        certificate=json.loads((Path(plan['lot001_evidence'])/'certificate.json').read_text())))
    pending=[]
    for spec in plan['lots']:
        candidates=[out/'execution'/spec['tag']/'cpu']+[p/'execution'/spec['tag']/'cpu' for p in sorted(out.glob('resume-*')) if p.is_dir()]
        evidence=next((p for p in candidates if (p/'controller-result.json').is_file()
            and json.loads((p/'controller-result.json').read_text()).get('status')=='REMOTE_CPU_RECEIPT_VERIFIED'),None)
        cert=None
        if evidence is not None: cert=validate_receipt(evidence,spec)
        else: pending.append(spec['tag'])
        gates.append(dict(number=spec['number'],handle=spec['handle'],identity=spec['identity'],certificate=cert))
    (bundle/'certification-gates.json').write_text(json.dumps(dict(lots=gates,pending=pending),indent=2))
    files={p.as_posix():p.read_bytes() for p in Path('src/geofirex').rglob('*.py')}
    for name in ('run_benchmark.py','profile_expanded_gpu.py','run_expanded_benchmark.py','mount_certified_lots.py'):
        files['scripts/'+name]=Path('scripts',name).read_bytes()
    for p in bundle.rglob('*'):
        if p.is_file(): files['bundle/'+p.relative_to(bundle).as_posix()]=p.read_bytes()
    blob=source_blob(files)
    package_sha=hashlib.sha256(base64.b64decode(blob)).hexdigest()
    bootstrap=("import base64,hashlib,io,json,os,sys,zipfile\nfrom pathlib import Path\n"
        "work=Path('/kaggle/working/geofirex-training')\nwork.mkdir(exist_ok=False)\n"
        f"data=base64.b64decode({blob!r})\n"
        f"if hashlib.sha256(data).hexdigest()!={package_sha!r}: raise ValueError('Code/metadata package changed')\n"
        "with zipfile.ZipFile(io.BytesIO(data)) as archive:\n    archive.extractall(work)\n"
        "os.chdir(work)\nsys.path.insert(0,str(work/'src'))\nos.environ['PYTHONPATH']=str(work/'src')\n"
        "os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')\n"
        "sys.path.insert(0,str(work/'scripts'))\nfrom mount_certified_lots import mount\nmount(work/'bundle')\n")
    sources=[lot001['handle']]+[x['handle'] for x in plan['lots']]
    artifacts=[]
    for model in ('B0','B1','B2','B3'):
        source=bootstrap+"import subprocess\n"
        config=f'/kaggle/working/geofirex-training/bundle/configs/{model}_T3_seed42.yaml'
        if model=='B0':
            source+="os.environ['CUDA_VISIBLE_DEVICES']=''\n"+f"subprocess.run([sys.executable,'scripts/run_benchmark.py','train','--config',{config!r}],check=True,timeout=7200)\n"
        else:
            source+=(f"subprocess.run([sys.executable,'scripts/profile_expanded_gpu.py','--config',{config!r},'--out','/kaggle/working/profile-{model}','--seconds','900'],check=True,timeout=930)\n"
                f"subprocess.run([sys.executable,'scripts/run_expanded_benchmark.py','--config',{config!r},'--profile','/kaggle/working/profile-{model}/profile.json','--out','/kaggle/working/supervisor-{model}','--seconds','7200'],check=True,timeout=7230)\n")
        folder=out/'training/notebooks'/model
        if folder.exists():
            # Regeneration only for these explicitly prepared, unsubmitted templates.
            for name in ('certify.ipynb','kernel-metadata.json'):
                if (folder/name).exists(): (folder/name).unlink()
            folder.rmdir()
        artifacts.append(dict(model=model,**make_notebook(folder,source,f'GeoFireX Expanded {model} T3 Seed42 20261108 V1',sources,gpu=model!='B0',small_control=False)))
    result=dict(status='PREPARED_TEMPLATES_WAITING_FOR_CERTIFICATES' if pending else 'READY_FOR_SEPARATE_TRAINING_AUTHORIZATION',
        pending_cpu_certificates=pending,models=artifacts,train_events=250,validation_events=201,
        test_events_attached=0,normalizer_sha256=plan['normalizer_sha256'],context=3,seed=42,
        encoded_channels=8,learned_model_epochs=1,learned_samples_per_epoch=93616,
        expected_updates_per_learned_model=5851,baseline_device='cpu',gpu_training_started=False,
        mlflow_enabled=False,tracking='Runner JSON/CSV/checkpoints retained; optional MLflow disabled in new copies because Kaggle CPU image lacks it',
        metrics=['dice','iou','precision','recall','average_precision'],
        selection='best validation AP; decision threshold selected on validation Dice; 2021 final-test commands omitted',
        limitation='One complete epoch per learned model is the existing operational budget, not a convergence claim')
    (out/'training/readiness.json').write_text(json.dumps(result,indent=2))
    return result


def prepare(out):
    import pyarrow.parquet as pq
    import yaml
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    evidence=Path('reports/kaggle_lot001_certified_20261108_v1')
    from record_lot001_certificate import verify_evidence
    lots=[json.loads((BASE/f'metadata/lot-{i:03d}.json').read_text()) for i in range(1,9)]
    verify_evidence(evidence,lots[0])
    with (BASE/'metadata/fold_00.csv').open() as stream: split={r['event_id']:r['split'] for r in csv.DictReader(stream)}
    rows=pq.read_table(BASE/'metadata/observations.parquet').to_pylist()
    bypath={r['relative_path']:r for r in rows}
    index=json.loads((BASE/'metadata/index.json').read_text())
    indexed={r['path']:r for r in index['entries']}
    normal=json.loads((BASE/'metadata/normalization.json').read_text())
    if (normal['provenance']['train_events'],normal['provenance']['validation_inputs'],normal['provenance']['test_inputs'])!=(250,0,0):
        raise ValueError('Train-only normalizer provenance differs')
    if sha(BASE/'metadata/normalization.json')!='aa30df4e3d6d895760f04e35d74bcbbac0d739cea045d236517401f5f9f4c7f6':
        raise ValueError('Pinned normalizer changed')
    binding_paths=dict(manifest='observations.parquet',window_manifest='windows.parquet',split='fold_00.csv',index='index.json')
    if normal['bindings']!={k:sha(BASE/'metadata'/p) for k,p in binding_paths.items()}: raise ValueError('Normalizer input bindings changed')
    seen_events=set();seen_files=set();specs=[]
    bundle=out/'training/bundle';(bundle/'metadata').mkdir(parents=True)
    for path in (BASE/'metadata').iterdir():
        if path.is_file(): shutil.copy2(path,bundle/'metadata'/path.name)
    role_counts=Counter()
    for lot in lots:
        identity=hashlib.sha256(json.dumps({k:v for k,v in lot.items() if k!='identity'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if identity!=lot['identity'] or seen_events & set(lot['events']): raise ValueError('Lot identity/event collision')
        if any(split[e]!=lot['role'] or e.startswith('2021/') for e in lot['events']): raise ValueError('Lot role/test leakage')
        expected={f['path'] for f in lot['files']}
        if len(expected)!=lot['observations'] or seen_files & expected: raise ValueError('Lot file collision/count mismatch')
        if sum(f['bytes'] for f in lot['files'])!=lot['bytes']: raise ValueError('Lot byte total mismatch')
        if expected!={e['path'] for e in index['entries'] if e['event_id'] in lot['events']}: raise ValueError('Incomplete event against original index')
        for f in lot['files']:
            r=bypath[f['path']];e=indexed[f['path']]
            if (r['status']!='ok' or r['sha256']!=f['sha256'] or r['bytes']!=f['bytes']
                or e['geotiff_bytes']!=f['bytes'] or e['crc32']!=f['crc32']): raise ValueError('Frozen QC/size/SHA/CRC mismatch')
        seen_events.update(lot['events']);seen_files.update(expected);role_counts[lot['role']]+=len(lot['events'])
        if lot['number']==1: continue
        years=Counter(str(dt.date.fromisoformat(Path(f['path']).stem).year) for f in lot['files'])
        allowed={'2018','2019'} if lot['role']=='train' else {'2020'}
        if not set(years)<=allowed: raise ValueError('Observation outside planned role years')
        tag=f"lot-{lot['number']:03d}"
        canonical=bundle/f'metadata/{tag}.json'
        specs.append(dict(number=lot['number'],tag=tag,handle=lot['handle'],role=lot['role'],events=len(lot['events']),
            files=lot['observations'],raw_bytes=lot['bytes'],identity=identity,observation_year_counts=dict(years),
            canonical=str(canonical),canonical_sha256=sha(canonical),archive_ceiling_bytes=lot['bytes']+10_000_000,
            staging=f'artifacts/kaggle_remaining_20261108_v1/{tag}',state='MANIFEST_READY_NOT_STAGED_NOT_UPLOADED'))
    if role_counts!={'train':250,'val':201} or len(seen_files)!=9490 or set(bypath)!=seen_files:
        raise ValueError('Expected exactly 250 train / 201 validation events and 9490 observations')
    fit=normal['normalization']['fit_paths']
    if len(fit)!=4948 or len(set(fit))!=4948 or len({str(Path(p).parent) for p in fit})!=250 or any(split[str(Path(p).parent).replace('\\','/') ]!='train' for p in fit):
        raise ValueError('Normalizer exact train fit set mismatch')
    modules=('__init__','runner_data','benchmark','corpus','remote_archive','unet_data','windows','real_constants')
    source_files={f'geofirex/{name}.py':Path(f'src/geofirex/{name}.py').read_bytes() for name in modules}
    codeblob=source_blob(source_files)
    helper=Path('scripts/kaggle_certify_inputs.py').read_text()
    runtime=Path('scripts/certify_kaggle_planned_lot_runtime.py').read_text()
    for spec in specs:
        lot=lots[spec['number']-1]
        observations=[{k:r[k] for k in ('event_id','date','relative_path','height','width')} for r in rows if r['event_id'] in lot['events']]
        payload=base64.b64encode(zlib.compress(json.dumps(dict(reference=lot,observations=observations,normalization=normal,split=split)).encode())).decode()
        source=("import base64,io,json,os,sys,zipfile,zlib\nfrom pathlib import Path\n"
            "os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')\n"
            f"work=Path('/kaggle/working/{spec['tag']}-certification')\nwork.mkdir(exist_ok=False)\n"
            f"archive=zipfile.ZipFile(io.BytesIO(base64.b64decode({codeblob!r})))\narchive.extractall(work/'source')\n"
            "sys.path.insert(0,str(work/'source'))\nhelpers={'__name__':'cpu_helpers'}\n"
            f"exec(compile({helper!r},'kaggle_certify_inputs.py','exec'),helpers)\n"
            "namespace={'__name__':'cpu_lot_runtime'}\n"
            f"exec(compile({runtime!r},'certify_kaggle_planned_lot_runtime.py','exec'),namespace)\n"
            f"payload=json.loads(zlib.decompress(base64.b64decode({payload!r})))\n"
            "namespace['run'](**payload,helpers=helpers,output=work/'results')\n")
        spec['cpu_notebook']=make_notebook(out/'cpu-certification'/spec['tag'],source,
            f"GeoFireX Certify {spec['tag'].replace('-',' ')} CPU 20261108 V1",[spec['handle']])
    canary_source=("import importlib,json,os\nfrom pathlib import Path\n"
        "os.environ['CUDA_VISIBLE_DEVICES']=''\n"
        "if list(Path('/dev').glob('nvidia[0-9]*')): raise RuntimeError('CPU session required')\n"
        "for name in ('numpy','rasterio','torch','psutil','pyarrow','yaml'): importlib.import_module(name)\n"
        "Path('/kaggle/working/environment.json').write_text(json.dumps({'status':'CPU_AUTOMATION_READY','gpu':False,'training':False,'datasets':[]}))\n")
    canary=make_notebook(out/'cpu-certification/canary',canary_source,'GeoFireX CPU Automation Gate 20261108 V1',[])
    plan=dict(schema='remaining-kaggle-sequential-v1',status='PREPARED_NOT_AUTHORIZED',lot001_closed=True,lot001_evidence=str(evidence),
        raw_root='D:/GeoFire-X/WildfireSpreadTS/rasters',lots=specs,canary=canary,
        remaining_raw_bytes=sum(x['raw_bytes'] for x in specs),archive_ceiling_bytes=sum(x['archive_ceiling_bytes'] for x in specs),
        remaining_body_ceiling_bytes=31_000_000_000,physical_network_planning_envelope_bytes=32_000_000_000,
        budget_scope='New remaining-lots phase only; historical lot001 body ledger is preserved separately; no retries',
        journal=str(out/'http-body.jsonl'),automatic_retries=0,normalizer_sha256=sha(BASE/'metadata/normalization.json'),
        cpu_api_note='Private CPU automation gate MUST complete before any raster staging/upload',
        train_events=250,validation_events=201,test_events=0)
    (bundle/'configs').mkdir()
    work='/kaggle/working/geofirex-training'
    checked_configs=[]
    common_keys=('context','seed','fold','feature_policy','samples_per_epoch','batch_size','patch_size','learning_rate','epochs','positive_tile_fraction')
    configs=[]
    for path in sorted((BASE/'configs').glob('*.yaml')):
        c=yaml.safe_load(path.read_text());configs.append(c)
        c.update(**{k:f'{work}/bundle/metadata/{p}' for k,p in binding_paths.items()},
            normalization_artifact=f'{work}/bundle/metadata/normalization.json',
            experiment='geofirex-expanded-first-epoch-20261108',output_root=f'{work}/runs',
            tracking_uri=f'sqlite:///{work}/runs/tracking.db',artifact_root=f'file://{work}/runs/mlflow_artifacts',
            data_root='/kaggle/input/RESOLVED_FROM_PINNED_INVENTORIES',data_roots=[],device='cpu' if c['model']=='B0' else 'cuda')
        c['mlflow_enabled']=False
        if c['corpus_scope']!='train_validation' or c['normalization_sha256']!=plan['normalizer_sha256'] or c['epochs']!=1:
            raise ValueError('Existing run protocol differs')
        (bundle/'configs'/path.name).write_text(yaml.safe_dump(c,sort_keys=False))
        checked_configs.append(path.name)
    if len(configs)!=4 or any({k:c[k] for k in common_keys}!={k:configs[0][k] for k in common_keys} for c in configs[1:]):
        raise ValueError('Models do not share the reviewed comparison protocol')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    training=training_notebooks(out,plan)
    print(json.dumps(dict(status=plan['status'],lots=specs,remaining_raw_bytes=plan['remaining_raw_bytes'],
        body_ceiling_bytes=plan['remaining_body_ceiling_bytes'],training_status=training['status'],checked_configs=checked_configs),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--refresh-training',action='store_true')
    a=p.parse_args()
    if a.refresh_training:
        print(json.dumps(training_notebooks(a.out,json.loads((a.out/'plan.json').read_text())),indent=2))
    else: prepare(a.out)
