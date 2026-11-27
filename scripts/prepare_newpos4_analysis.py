"""Freeze CPU-only analysis payload and prospective NEWPOS2; never executes it."""
from copy import deepcopy
from pathlib import Path
import base64
import hashlib
import io
import json
import zipfile
from geofirex import c8_experiments as c
from geofirex import newpos4_posthoc as n

ROOT=Path(__file__).resolve().parents[1]
REPORT=Path('reports/newpos4_thresholds_20261121_v1')


def cell(kind,text,i):
    result=dict(cell_type=kind,id=f'newpos4-cpu-{i}',metadata={},source=text.splitlines(True))
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def main():
    out=ROOT/REPORT
    if (out/'analysis_inputs.json').exists():raise FileExistsError('Do not replace frozen analysis inputs')
    if c.read(out/'prediction_verification.json')['status']!='PASS':raise ValueError('Prediction verification required')
    baseline=c.read(ROOT/'configs/protocol_v2/baseline_c8.json');candidate=deepcopy(baseline)
    candidate.update(run_id='B2_C8_V2_NEWPOS2_seed42',loss=dict(baseline['loss'],name='NewPositiveLoss',new_positive_weight=2.),
        status='PROSPECTIVE_PROTOCOL_NOT_EXECUTABLE',base_config_sha256=c.sha256(ROOT/'configs/protocol_v2/baseline_c8.json'))
    c.save(ROOT/'configs/c8_newpos2_prereg_20261121_v1.json',candidate)
    c.save(ROOT/'configs/newpos4_threshold_costs_v1.json',n.cost_policy())
    files=dict(c.read(ROOT/c.REPORT/'execution_manifest.json')['files'])
    names=[(c.REPORT/'execution_manifest.json').as_posix(),'src/geofirex/newpos4_posthoc.py','src/geofirex/newpos4_analysis.py',
        'src/geofirex/newpos4_studio.py','scripts/run_newpos4_thresholds.py','tests/test_newpos4_posthoc.py',
        'configs/newpos4_threshold_costs_v1.json']
    names += [(REPORT/f).as_posix() for f in ('baseline_reference.json','newpos4_evaluation.json','c8_crossfit.json',
        'prediction_verification.json','archive_verification.json','threshold_policy.json','c8_threshold_lock.json')]
    files.update({name:c.sha256(ROOT/name) for name in names})
    source=next(a for a in c.read(out/'archive_verification.json')['archives'] if 'NEWPOS4' in a['path'] and '-COMPLETE-' in a['path'])
    c.save(out/'analysis_inputs.json',dict(schema='saved-newpos4-cpu-inputs-1',files=files,
        source_archive_sha256=source['sha256'],source_archive_bytes=source['bytes'],checkpoint_sha256=n.CHECKPOINT,
        analysis_id='NEWPOS4_V2_THRESHOLDS_COSTS_CPU_v1',primary_policy='fpr_0.00028',
        GPU_authorized=False,training_authorized=False,inference_authorized=False,
        streaming_resource_policy=dict(start_available_ram_bytes=768*1024**2,reserve_available_bytes=512*1024**2,
            rss_increment_limit_bytes=128*1024**2,block_pixels=65536,no_exact_AP_resort=True,no_prediction_extraction=True)))
    payload=io.BytesIO()
    with zipfile.ZipFile(ROOT/c.REPORT/'c8_experiments_code.zip') as old,zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
        names=sorted(set(old.namelist())|set(files)|{(REPORT/'analysis_inputs.json').as_posix()})
        for name in names:
            data=(ROOT/name).read_bytes() if (ROOT/name).is_file() else old.read(name)
            info=zipfile.ZipInfo(name,date_time=(2026,11,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
    blob=payload.getvalue();digest=hashlib.sha256(blob).hexdigest()
    with (out/'newpos4_cpu_code.zip').open('xb') as f:f.write(blob)
    cells=[cell('markdown','# NEWPOS4 — seuils et coûts, CPU uniquement\n\nAnalyse exploratoire sur 80 événements de validation. Aucun GPU, fit, inférence ou pixel2021. Réutilise les scores sauvegardés et AP exactes archivées. Les 15 seuils sont verrouillés avant toute estimation des scénarios. Aucun choix automatique du scénario gagnant.\n',0),
        cell('code',"RUN_ANALYSIS = False\nimport os,sys\nif 'torch' in sys.modules: raise RuntimeError('Redémarrer le noyau avant de masquer CUDA')\nos.environ['CUDA_VISIBLE_DEVICES']=''\nos.environ['OMP_NUM_THREADS']='2'\nos.environ['MKL_NUM_THREADS']='2'\n",1),
        cell('code',"from pathlib import Path\nimport base64,hashlib,io,zipfile,json\n"+
            f"DATA=base64.b64decode({base64.b64encode(blob).decode()!r})\nassert hashlib.sha256(DATA).hexdigest()=={digest!r}\n"+
            "ROOT=Path('/kaggle/working/geofirex-newpos4-cpu-code')\nROOT.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(DATA)) as z:\n    for name in z.namelist():\n        assert (ROOT/name).resolve().is_relative_to(ROOT.resolve())\n    z.extractall(ROOT)\nsys.path.insert(0,str(ROOT/'src'))\nfrom geofirex.newpos4_analysis import ResourceGuard,analysis\nfrom geofirex import c8_experiments as c\nPINS=c.read(ROOT/'reports/newpos4_thresholds_20261121_v1/analysis_inputs.json')\nc.old.verify_manifest(ROOT,PINS['files'])\nprint(ResourceGuard('/kaggle/working').receipt())\n",2),
        cell('markdown','Attacher les sorties existantes du notebook NEWPOS4 comme entrée, avec son archive COMPLETE intacte. Aucune des11 partitions de rasters et aucun checkpoint séparé ne sont nécessaires. Si un upload privé décompresse le ZIP, conserver uniquement une copie de transport .bin aux mêmes octets/SHA ; préserver le ZIP original. Ne jamais télécharger ou régénérer les prédictions par inférence.\n',3),
        cell('code',"ARCHIVES=[p for p in Path('/kaggle/input').rglob('B2_C8_V2_NEWPOS4_seed42-COMPLETE-*') if p.suffix in ('.zip','.bin') and p.stat().st_size==PINS['source_archive_bytes']]\nOUT=Path('/kaggle/working/NEWPOS4_V2_THRESHOLDS_COSTS_CPU_v1')\nif RUN_ANALYSIS:\n    assert len(ARCHIVES)==1, 'Attacher exactement une archive COMPLETE intacte du run NEWPOS4'\n    analysis(ROOT,ARCHIVES[0],OUT)\nelse: print('Analyse désactivée ; aucun GPU, entraînement ou inférence')\n",4),
        cell('code',"from IPython.display import display,FileLink\nif RUN_ANALYSIS:\n    result=c.read(OUT/'results.json')\n    for row in result['scenarios']:\n        print(row['scenario'],row['summary'],row['threshold_operating_gates'])\n    print('Exploratoire validation ; aucune promotion ni sélection automatique')\n    for p in sorted(Path('/kaggle/working').glob('NEWPOS4_V2_THRESHOLDS_COSTS_CPU_v1-*.zip')):\n        print(c.old.verify_zip(p));display(FileLink(str(p)));display(FileLink(str(p.with_suffix('.sha256'))))\n",5)]
    notebook=ROOT/'notebooks/GEOFIREX_NEWPOS4_THRESHOLDS_CPU_DISABLED.ipynb'
    c.save(notebook,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells))
    c.save(out/'kernel_metadata_CPU_NOT_PUBLISHED.json',dict(id='nnnnbhss/geofirex-newpos4-thresholds-cpu-prepared',
        title='GeoFire-X NEWPOS4 saved-score CPU thresholds',code_file=notebook.name,language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_internet=False,dataset_sources=[],kernel_sources=[],
        note='Attach existing NEWPOS4 notebook output containing COMPLETE; not published or launched here'))
    c.save(out/'notebook_build.json',dict(payload_bytes=len(blob),payload_sha256=digest,notebook_sha256=c.sha256(notebook),
        analysis_enabled=False,GPU_enabled=False,published=False))
    print(json.dumps(dict(payload_bytes=len(blob),notebook=str(notebook),source_archive=source['path']),indent=2))


if __name__=='__main__':main()
