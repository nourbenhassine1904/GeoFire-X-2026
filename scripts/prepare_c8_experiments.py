"""Freeze additive A/B configs and two self-contained DISABLED notebooks.

Preparation only: never calls an inference, optimizer, CUDA API or Kaggle mutation.
"""
from copy import deepcopy
from pathlib import Path
import base64
import hashlib
import io
import json
import zipfile
from geofirex import v2_execution as old
from geofirex.protocol_v2 import save,sha256

ROOT=Path(__file__).resolve().parents[1]
REPORT=Path('reports/c8_propagation_20261121_v1')
A='C8_V2_THRESHOLD_CROSSFIT_seed42_v1';B='B2_C8_V2_NEWPOS4_seed42'


def code(source,i):return dict(cell_type='code',id=f'c8-{i}',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))
def md(source,i):return dict(cell_type='markdown',id=f'c8-{i}',metadata={},source=source.splitlines(True))


def main(refresh_draft=False):
    out=ROOT/REPORT
    base,oldlock,_=old.identity(ROOT)
    if (out/'execution_manifest.json').exists():
        if not refresh_draft or (out/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed A/B preparation')
        # Preserve every prior *new draft file* in a revision. Never move a
        # directory or any original C8/C36 input. All paths checked in workspace.
        import uuid
        revision=out/'draft_revisions'/uuid.uuid4().hex;revision.mkdir(parents=True,exist_ok=False)
        targets=[ROOT/'configs/c8_newpos4_v1.json',ROOT/'configs/c8_thresholds_v1.json',
            ROOT/'notebooks/GEOFIREX_C8_THRESHOLDS_DISABLED.ipynb',ROOT/'notebooks/GEOFIREX_C8_NEWPOS4_DISABLED.ipynb']
        targets += [out/n for n in ('budget.json','baseline_reference.json','execution_manifest.json','c8_experiments_code.zip',
            'notebook_build.json','kernel_metadata_A_NOT_PUBLISHED.json','kernel_metadata_B_NOT_PUBLISHED.json')]
        targets += list(out.glob('authorization_*_DISABLED.json'))
        for p in targets:
            p=p.resolve()
            if not p.is_relative_to(ROOT) or p.relative_to(ROOT).as_posix() in oldlock['files']:
                raise ValueError('Draft refresh would touch a protected input')
            if p.is_file():
                dst=revision/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);p.rename(dst)
    proposal=old.read(ROOT/'configs/c8_propagation_proposal_v1.json')
    candidate=deepcopy(base)
    candidate.update(run_id=B,loss=dict(base['loss'],name='NewPositiveLoss',new_positive_weight=4.),
        status='PREPARED_DISABLED',base_config_sha256=sha256(ROOT/'configs/protocol_v2/baseline_c8.json'))
    save(ROOT/'configs/c8_newpos4_v1.json',candidate)
    analysis=dict(schema='c8-threshold-crossfit-1',analysis_id=A,status='PREPARED_DISABLED',
        launch_enabled=False,training_authorized=False,validation_events=80,validation_tiles=30882,
        baseline_checkpoint_sha256=proposal['baseline_checkpoint_sha256'],baseline_threshold=.78,
        events_sha256=base['event_manifest_sha256'],index_sha256=base['index_sha256'],
        normalization_sha256=base['normalization_sha256'],fpr_budget=.00028,
        fpr_budget_rational=[7,25000],folds=5,fold_assignment=proposal['priority_no_fit_analysis']['fold_assignment'],
        calibration_events=64,assessment_events=16,threshold_lock_before_assessment=True,
        interpretation='exploratory validation after observing baseline aggregates; not a new blind test',
        AP_and_Brier_cannot_improve_from_threshold=True,test_2021_reserved=True)
    save(ROOT/'configs/c8_thresholds_v1.json',analysis)
    budget=dict(schema='c8-ab-budget-1',A_session_minutes=180,B_session_minutes=225,
        A_cpu_inference_session_minutes=240,A_cpu_analysis_session_minutes=150,
        A_card_hours=6,B_card_hours=7.5,one_visible_T4=True,possible_reserved_cards=2,
        A_inference_seconds_cuda=1800,A_inference_seconds_cpu=10800,A_analysis_seconds=5400,
        B_fit_seconds=3600,B_evaluation_seconds=5400,required_free_bytes=18*old.GIB,
        minimum_runtime_free_bytes=2*old.GIB,min_available_ram_cuda_bytes=8*old.GIB,
        min_available_ram_cpu_bytes=2*old.GIB,prediction_record_bytes=7,
        padded_validation_pixels=30882*4096,prediction_uncompressed_bytes=30882*4096*7,
        two_exact_AP_sorters_merge_upper_bytes=30882*4096*24*4,
        threshold_calibration='two scans per calibration fold; exact top-(budget+1) negatives, no full threshold sort',
        threshold_negative_working_array_upper_bytes=2*(104184280*7//25000+1)*4,
        cpu_ram_working_estimate_bytes=2*old.GIB,checkpoint_envelope_bytes=33*2*1024**2,
        measured_baseline_fit_seconds=1133.55283755,measured_baseline_evaluation_seconds=1213.553304316,
        GPU_measured_this_mission=False,wall_times_are_reserves_not_guarantees=True,
        A_cuda_breakdown_minutes=dict(setup=15,inference_and_save=30,CPU_analysis=90,archives=30,margin=15),
        B_cuda_breakdown_minutes=dict(setup=15,fit=60,evaluation=90,comparison=5,archives=30,margin=25))
    save(out/'budget.json',budget)
    archives=old.read(ROOT/'reports/c36_feasibility_20261121_v1/c8_archive_verification.json')['archives']
    complete=next(a for a in archives if a['stage']=='COMPLETE')
    old.verify_zip(complete['path'],complete['sha256'])
    with zipfile.ZipFile(complete['path']) as z:reference=json.loads(z.read('evaluation/evaluation.json'))
    save(out/'baseline_reference.json',reference)
    files={**oldlock['files']}
    names=['reports/protocol_v2_execution/execution_manifest.json','configs/c8_propagation_proposal_v1.json',
        'configs/c8_newpos4_v1.json','configs/c8_thresholds_v1.json',str(REPORT/'budget.json'),str(REPORT/'baseline_reference.json'),
        'src/geofirex/c8_newpos_loss.py','src/geofirex/c8_predictions.py','src/geofirex/c8_thresholds.py',
        'src/geofirex/c8_comparison.py','src/geofirex/c8_experiments.py','scripts/run_c8_experiments.py',
        'tests/test_c8_propagation.py']
    files.update({n.replace('\\','/'):sha256(ROOT/n) for n in names})
    save(out/'execution_manifest.json',dict(schema='c8-ab-execution-1',files=files,experiments=[A,B],
        baseline_execution_manifest_sha256=sha256(ROOT/old.REPORT/'execution_manifest.json')))
    auths=[]
    for action,experiment,device in [('threshold-infer',A,'cuda'),('threshold-infer',A,'cpu'),('threshold-analyze',A,'cpu'),
        ('newpos4-fit',B,'cuda'),('newpos4-evaluate',B,'cuda'),('gpu-preflight',A,'cuda'),('gpu-preflight',B,'cuda')]:
        cpu_suffix='_cpu' if action=='threshold-infer' and device=='cpu' else ''
        name=f'authorization_{action}_{"A" if experiment==A else "B"}{cpu_suffix}_DISABLED.json'
        save(out/name,dict(authorized=False,action=action,experiment_id=experiment,device=device,runs=1,
            execution_manifest_sha256=sha256(out/'execution_manifest.json'),authorized_by='',authorized_at='',
            output_parent='/kaggle/working',session_remaining_minutes=None,quota_remaining_card_hours=None,
            card_hours_reserved=None,quota_checked_at='',
            local_backup_destination=str(ROOT/'artifacts/c8_propagation_runs'),external_backup_destination='D:\\GeoFire-X'))
        auths.append(str(REPORT/name))
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(set(files)|set(auths)|{str(REPORT/'execution_manifest.json')}):
            info=zipfile.ZipInfo(name.replace('\\','/'),date_time=(2026,11,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(ROOT/name).read_bytes())
    blob=payload.getvalue();digest=hashlib.sha256(blob).hexdigest()
    with (out/'c8_experiments_code.zip').open('xb') as f:f.write(blob)
    notebooks=[]
    for which,name in [('A','GEOFIREX_C8_THRESHOLDS_DISABLED'),('B','GEOFIREX_C8_NEWPOS4_DISABLED')]:
        heading='Analyse exploratoire des seuils C8' if which=='A' else 'C8 NEWPOS4, une seule nouvelle loss'
        cells=[md('# '+heading+'\n\nInternet OFF. Noyau neuf. Aucun pixel2021. Tous les drapeaux sont désactivés. Une autorisation externe par action est nécessaire.\n',0),
            code("RUN_GPU_PREFLIGHT = False\nRUN_INFERENCE = False\nRUN_ANALYSIS = False\nRUN_FIT = False\nRUN_RECOVERY = False\nimport os,sys\nif 'torch' in sys.modules: raise RuntimeError('Redémarrer le noyau avant de fixer CUDA')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n",1),
            code("from pathlib import Path\nimport base64,hashlib,io,zipfile,json\n"+
                f"DATA=base64.b64decode({base64.b64encode(blob).decode()!r})\nassert hashlib.sha256(DATA).hexdigest()=={digest!r}\n"+
                f"ROOT=Path('/kaggle/working/geofirex-c8-{which.lower()}-code')\nROOT.mkdir(exist_ok=False)\n"+
                "with zipfile.ZipFile(io.BytesIO(DATA)) as z:\n    for n in z.namelist():\n        assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())\n    z.extractall(ROOT)\nsys.path.insert(0,str(ROOT/'src'))\nfrom geofirex import c8_experiments as c\nprint(c.preflight(ROOT))\n",2),
            code("EXPECTED=set(v['map_sha256'] for v in c.read(ROOT/c.old.REPORT/'partitions.json').values())\nMAPS=[p for p in Path('/kaggle/input').rglob('mount-map.json') if c.sha256(p) in EXPECTED]\nprint(c.preflight(ROOT,MAPS))\nprint(json.dumps(c.read(ROOT/c.REPORT/'budget.json'),indent=2))\n",3),
            md('Après autorisation explicite, créer une COPIE du template désactivé sous /kaggle/working, renseigner opérateur, date, quota réel, durée et destinations. Ne jamais modifier le template dans le payload. Un seul run neuf ; télécharger chaque archive vérifiée.\n',4),
            code(f"AUTH=Path('/kaggle/working/c8_{which.lower()}_authorization.json')\nGPU_AUTH=Path('/kaggle/working/c8_{which.lower()}_gpu_authorization.json')\nif RUN_GPU_PREFLIGHT:\n    c.authorize(ROOT,GPU_AUTH,'gpu-preflight','cuda',Path('/kaggle/working'))\n    result=c.preflight(ROOT,MAPS,'/kaggle/working','cuda',gpu=True)\n    c.save(Path('/kaggle/working/c8_{which.lower()}_gpu_preflight.json'),result)\n    print(result)\nelse: print('Précontrôle GPU désactivé')\n",5)]
        if which=='A':
            cells += [code("ARCHIVES=[p for p in Path('/kaggle/input').rglob('B2_C8_V2_seed42-COMPLETE-48e35c59.*') if p.suffix in ('.zip','.bin')]\nOUT=Path('/kaggle/working/'+c.A_ID)\nif RUN_INFERENCE:\n    assert RUN_GPU_PREFLIGHT and not RUN_FIT and not RUN_RECOVERY\n    assert len(ARCHIVES)==1, 'Attacher le ZIP COMPLETE officiel intact (extension .zip ou .bin) en entrée privée'\n    c.infer_thresholds(ROOT,MAPS,ARCHIVES[0],OUT,AUTH,'cuda')\nelse: print('Inférence exhaustive désactivée')\n",6),
                code("ANALYSIS_AUTH=Path('/kaggle/working/c8_a_analysis_authorization.json')\nif RUN_ANALYSIS:\n    assert not RUN_FIT and not RUN_RECOVERY\n    c.analyze_thresholds(ROOT,OUT/'predictions',Path('/kaggle/working/'+c.A_ID+'_ANALYSIS'),ANALYSIS_AUTH)\nelse: print('Analyse des seuils désactivée')\n",7)]
        else:
            cells += [code("OUT=Path('/kaggle/working/'+c.B_ID)\nif RUN_FIT:\n    assert RUN_GPU_PREFLIGHT and not RUN_INFERENCE and not RUN_ANALYSIS and not RUN_RECOVERY\n    c.fit(ROOT,MAPS,OUT,AUTH)\nelse: print('Fit NEWPOS4 désactivé')\n",6),
                code("RECOVERY_SOURCE=Path('/kaggle/working/restored_newpos4')\nRECOVERY_OUT=Path('/kaggle/working/'+c.B_ID+'_EVALUATION_RECOVERY')\nRECOVERY_AUTH=Path('/kaggle/working/c8_b_recovery_authorization.json')\nRECOVERY_ARCHIVE=Path('/kaggle/input/REPLACE_WITH_PRIVATE_MOUNT/newpos4-training-safe.bin')\nRECOVERY_SIDECAR=RECOVERY_ARCHIVE.with_suffix('.sha256')\nif RUN_RECOVERY:\n    assert RUN_GPU_PREFLIGHT and not RUN_FIT\n    c.authorize(ROOT,RECOVERY_AUTH,'newpos4-evaluate','cuda',RECOVERY_OUT.parent)\n    c.restore_candidate(ROOT,RECOVERY_ARCHIVE,RECOVERY_SIDECAR,RECOVERY_SOURCE)\n    c.evaluate_candidate(ROOT,MAPS,RECOVERY_SOURCE,RECOVERY_OUT,RECOVERY_AUTH)\nelse: print('Récupération évaluation seule désactivée')\n",7)]
        cells.append(code("from IPython.display import display,FileLink\nfor p in sorted(Path('/kaggle/working').glob('*.zip')):\n    if p.name.startswith((c.A_ID,c.B_ID)):\n        print(c.old.verify_zip(p));display(FileLink(str(p)));display(FileLink(str(p.with_suffix('.sha256'))))\n",8))
        path=ROOT/'notebooks'/(name+'.ipynb')
        save(path,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells))
        notebooks.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path)))
        save(out/f'kernel_metadata_{which}_NOT_PUBLISHED.json',dict(id=f'nnnnbhss/geofirex-c8-{which.lower()}-prepared',title=heading,
            code_file=path.name,language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,
            dataset_sources=old.read(ROOT/old.REPORT/'dataset-sources.json')))
    save(out/'notebook_build.json',dict(notebooks=notebooks,payload_sha256=digest,payload_bytes=len(blob),
        runtime_flags_enabled=False,published=False))
    print(json.dumps(dict(updates=28976,payload_bytes=len(blob),notebooks=notebooks),indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--refresh-draft',action='store_true')
    main(p.parse_args().refresh_draft)
