"""Package exactly one disabled, offline Kaggle notebook. Never executes it."""
from pathlib import Path
import base64
import io
import json
import math
import zipfile
from geofirex.protocol_v2 import sha256
from geofirex.v2_execution import read, verify_manifest, package, REPORT, GIB

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/REPORT

def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def main():
    # No new cutoff calculations: verify and carry the final decision only.
    split=read(OUT/'split_lock.json'); verify_manifest(ROOT,split['protected_sha256'])
    c=read(ROOT/'configs/protocol_v2/baseline_c8.json')
    import numpy as np
    n=len(np.load(ROOT/c['index']/'train.npy',mmap_mode='r',allow_pickle=False))
    updates=math.ceil(n/c['batch_size'])*c['epochs']
    old=23404; measured=887.069
    # Two exact sorters coexist until pooled AP is reduced; conservative 2x merge
    # space for BOTH full domains, plus checkpoints, two ZIPs and 4GiB margin.
    val_tiles=len(np.load(ROOT/c['index']/'val.npy',mmap_mode='r',allow_pickle=False))
    upper_pixels=val_tiles*64*64
    sort_bytes=2*2*24*upper_pixels
    checkpoint_bytes=2*1024**2*(1+updates//1000+c['epochs'])
    required=math.ceil((sort_bytes+3*checkpoint_bytes+4*GIB)/GIB)*GIB
    budget=dict(schema='v2-budget-1',torch='2.11.0+cu130',numpy='2.1.3',cuda_build='13.0',
        historical_reference='reports/focal_go_final_v1/BILAN_FINAL.md',historical_seconds=measured,
        historical_updates=old,update_ratio=updates/old,
        scenario_fit_minutes=[measured*(updates/old)*f/60 for f in (.75,1.5)],
        scenario_warning='Related U-Net with extra head; not V2 throughput, prediction or guaranteed upper bound',
        setup_minutes=15,fit_seconds=3600,evaluation_seconds=5400,archive_transfer_minutes=30,margin_minutes=15,
        session_minutes=210,conservative_card_hours=7,gpu_count_used=1,possible_cards_reserved_by_kaggle=2,
        min_host_ram_bytes=16*GIB,min_available_ram_bytes=8*GIB,
        upper_validation_pixels=upper_pixels,exact_sort_record_bytes=24,sorters=2,
        scratch_merge_upper_bytes=sort_bytes,checkpoint_envelope_bytes=checkpoint_bytes,
        required_free_bytes=required,minimum_runtime_free_bytes=2*GIB,
        gpu_throughput_measured_this_mission=False,quota_check='Current Kaggle UI before launch; record card-hours and timestamp',
        actual_quota_deduction='Record after session; CUDA visibility does not determine billing')
    assert budget['session_minutes']==15+60+90+30+15
    write(OUT/'budget.json',budget)
    policy=dict(schema='v2-propagation-pr-preregistration-1',fpr_budgets=[0,.00001,.0001,.001,.01],
        recall_curve_intervals=100,official_threshold=.78,threshold_selection_enabled=False,
        denominator='negatives among valid & observed_J==0, repeated pixel-windows, not unique area',
        budget_rule='floor(FPR_budget * negative_count); ties indivisible; reject-all always feasible',
        selected_point='maximum recall at or below FP budget; earliest highest threshold on recall ties',
        empty_policy='no positives: recall/AP null; no negatives: FPR null',
        pr_curve='Exact full global-tie scan; output downsampled at fixed recall crossings plus final point',
        distinction='0.78 fixed official metrics; PR/budget envelope descriptive; operational threshold selection disabled',
        optimism='Choosing and evaluating threshold on same panel is optimistically biased. Report envelope as descriptive, never prospective operating performance.',
        future_operational_selection='Requires separate preregistration before results: event-disjoint development calibration and assessment, or nested event-level CV; lock before final 2021. Current 80 already observed, not blind.',
        test2021='Never used for threshold choice; no test reader in this kit')
    write(OUT/'propagation_policy.json',policy)
    partitions={}
    for p in sorted((ROOT/'reports/gpu_acceleration_20261111_v1/publication').glob('p??/mount-map.json')):
        partitions[read(p)['identity']]=dict(partition=p.parent.name,map_sha256=sha256(p))
    assert len(partitions)==11
    write(OUT/'partitions.json',partitions)
    sources=read(ROOT/'reports/focal_go_final_v1/kit/dataset-sources.json')
    write(OUT/'dataset-sources.json',sources)
    files=list((ROOT/'src/geofirex').glob('*.py'))
    files += [ROOT/'scripts/run_v2_baseline.py']
    files += list((ROOT/'configs/protocol_v2').glob('*.json'))
    files += [ROOT/c['index']/s for s in ('index.json','train.npy','val.npy','normalization.json')]
    files += [ROOT/c['event_manifest']]
    files += [OUT/name for name in ('split_lock.json','budget.json','propagation_policy.json','partitions.json','dataset-sources.json')]
    manifest=dict(schema='v2-execution-manifest-1',run_id=c['run_id'],
                  files={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(files)})
    write(OUT/'execution_manifest.json',manifest)
    authorization=dict(action='fit',authorized=False,runs=1,run_id=c['run_id'],
        execution_manifest_sha256=sha256(OUT/'execution_manifest.json'),session_reserve_minutes=210,
        card_hours_reserved=7,gpu_count=1,quota_remaining_card_hours=None,session_remaining_minutes=None,
        quota_checked_at='',authorized_by='',local_backup_destination=str(ROOT/'artifacts/v2_runs'),
        external_backup_destination='D:\\GeoFire-X')
    write(OUT/'authorization_TEMPLATE_DISABLED.json',authorization)
    # ZIP payload with pinned sources/config/indices only, never pixels/checkpoints.
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'execution_manifest.json',OUT/'authorization_TEMPLATE_DISABLED.json']:
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,11,19,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; z.writestr(info,p.read_bytes())
    blob=payload.getvalue()
    import hashlib
    digest=hashlib.sha256(blob).hexdigest()
    (OUT/'v2-code.zip').write_bytes(blob)
    def md(source): return dict(cell_type='markdown',metadata={},source=source.splitlines(True))
    def code(source): return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))
    cells=[md('# B2 C8 V2 — désactivé\nUne seule baseline, cutoff 23 août figé. Aucun test 2021. Internet OFF. Noyau neuf.\nRéserve 210 min / jusqu’à 7 heures-carte. Les contrôles GPU et le fit exigent des activations distinctes.\n'),
        code("RUN_GPU_PREFLIGHT = False\nRUN_TRAINING = False\nRUN_EVALUATION_RECOVERY = False\nimport os, sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Redémarrer le noyau : le masque GPU doit précéder Torch')\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"),
        code("import base64, hashlib, io, zipfile\nfrom pathlib import Path\n"+
             f"PAYLOAD = {base64.b64encode(blob).decode()!r}\nEXPECTED_SHA = {digest!r}\n"+
             "data = base64.b64decode(PAYLOAD)\nassert hashlib.sha256(data).hexdigest() == EXPECTED_SHA\nROOT = Path('/kaggle/working/geofirex-v2-code')\nROOT.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(data)) as z:\n    for name in z.namelist():\n        target = (ROOT/name).resolve()\n        assert target.is_relative_to(ROOT.resolve())\n    z.extractall(ROOT)\nsys.path.insert(0, str(ROOT/'src'))\nfrom geofirex import v2_execution as v2\nprint(v2.preflight(ROOT))\n"),
        code("# Les 11 datasets compacts existants p00..p10 doivent être attachés.\nMAPS = sorted(Path('/kaggle/input').rglob('mount-map.json'))\nmounts, receipt = v2.verify_mounts(ROOT, MAPS)\nprint(receipt)\nprint(v2.preflight(ROOT, MAPS))\n"),
        code("# Quota à lire dans l’interface Kaggle, pas à déduire de CUDA.\nimport json\nprint(json.dumps(v2.read(ROOT/v2.REPORT/'budget.json'), indent=2))\nif RUN_GPU_PREFLIGHT:\n    result = v2.preflight(ROOT, MAPS, '/kaggle/working', '/tmp', gpu=True)\n    v2.save(Path('/kaggle/working/v2_runtime_preflight.json'), result)\n    print(result)\nelse:\n    print('Contrôle GPU désactivé')\n"),
        code("# Après autorisation explicite, éditer une COPIE du template externe.\nAUTHORIZATION = Path('/kaggle/working/v2_authorization.json')\nOUT = Path('/kaggle/working/B2_C8_V2_seed42')\nif RUN_TRAINING:\n    assert RUN_GPU_PREFLIGHT\n    v2.fit(ROOT, MAPS, OUT, AUTHORIZATION)\nelse:\n    print('Entraînement désactivé')\n"),
        code("# Télécharger TRAINING-SAFE dès qu’il existe ; COMPLETE après succès.\nfrom IPython.display import display, FileLink\nfor archive in sorted(Path('/kaggle/working').glob('B2_C8_V2_seed42*.zip')):\n    print(v2.verify_zip(archive))\n    display(FileLink(str(archive)))\n    display(FileLink(str(archive.with_suffix('.sha256'))))\n"),
        code("# Récupération séparée : checkpoint epoch4 complet, aucun nouvel entraînement.\n# Nouvelle session : cellules 1 à 4 ; fournir authorization action='evaluate-only'.\nRECOVERY_ARCHIVE = Path('/kaggle/input/v2-training-safe/TRAINING-SAFE.zip')\nRECOVERY_SIDECAR = Path('/kaggle/input/v2-training-safe/TRAINING-SAFE.sha256')\nRECOVERY_SOURCE = Path('/kaggle/working/restored/B2_C8_V2_seed42')\nRECOVERY_OUT = Path('/kaggle/working/B2_C8_V2_seed42_recovery')\nif RUN_EVALUATION_RECOVERY:\n    v2.restore(RECOVERY_ARCHIVE, RECOVERY_SIDECAR, RECOVERY_SOURCE)\n    import subprocess\n    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))\n    subprocess.run([sys.executable, str(ROOT/'scripts/run_v2_baseline.py'), 'evaluate-only',\n        '--root', str(ROOT), '--maps', *map(str, MAPS), '--training-out', str(RECOVERY_SOURCE),\n        '--out', str(RECOVERY_OUT), '--authorization', str(AUTHORIZATION)], check=True, env=env)\nelse:\n    print('Récupération désactivée')\n"),
        md('Après téléchargement : vérifier le SHA externe et tous les membres du ZIP, puis exécuter la commande backup documentée sur Windows. Ne pas fermer la session avant les deux reçus de copie locale et D:\\GeoFire-X. Aucune copie D: ne peut être faite depuis Kaggle.\n')]
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
                  cells=[dict(cell,id=f'v2-cell-{i:02d}') for i,cell in enumerate(cells)])
    write(ROOT/'notebooks/GEOFIREX_V2_DISABLED.ipynb',notebook)
    write(OUT/'notebook_build.json',dict(payload_sha256=digest,notebook_sha256=sha256(ROOT/'notebooks/GEOFIREX_V2_DISABLED.ipynb'),
         execution_manifest_sha256=sha256(OUT/'execution_manifest.json'),training_enabled=False))
    print(json.dumps(dict(updates=updates,budget=budget,payload_bytes=len(blob)),indent=2))

if __name__=='__main__': main()
