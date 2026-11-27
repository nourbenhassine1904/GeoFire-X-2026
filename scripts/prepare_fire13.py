"""Build an additive, offline, disabled FIRE13 delivery; never executes GPU work."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from geofirex.protocol_v2 import save, sha256
from geofirex import v2_execution as old

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path('reports/fire13_20261122_v1')
RUN = 'B2_C13_FIRE_V2_seed42'
CONFIG = 'configs/c8_fire13_v1.json'


def cell(kind, source, i):
    result = dict(cell_type=kind, id=f'fire13-{i}', metadata={}, source=source.splitlines(True))
    if kind == 'code': result.update(execution_count=None, outputs=[])
    return result


def main(refresh_draft=False):
    out = ROOT / REPORT
    if (out/'execution_manifest.json').exists():raise FileExistsError('Preserve built FIRE13 preparation')
    if old.read(out/'parity_full.json')['status']!='PASS':raise ValueError('Real exhaustive parity required before packaging')
    base, lock, _ = old.identity(ROOT)
    assert len(lock['files']) == 77
    from geofirex.fire13_features import NAMES
    candidate=dict(base,run_id=RUN,status='PREPARED_DISABLED',channels_per_day=13,model_input_channels=39,
        channel_names=base['channel_names']+NAMES,feature_policy='C8_plus_FIRE13',
        derived_normalization_sha256=sha256(out/'derived_normalization.json'),
        scene_geometry_sha256=sha256(out/'scene_geometry.json'),
        base_config_sha256=sha256(ROOT/'configs/protocol_v2/baseline_c8.json'))
    save(ROOT/CONFIG, candidate)
    for name, source in [('baseline_reference.json', 'reports/c8_propagation_20261121_v1/baseline_reference.json'),
                         ('newpos4_reference.json', 'reports/newpos4_thresholds_20261121_v1/newpos4_evaluation.json')]:
        save(out/name, old.read(ROOT/source))
    save(out/'newpos2_reference.json',old.read(ROOT/'reports/prop_hn2_20261122_v1/newpos2_reference.json'))
    budget = old.read(ROOT/'reports/c8_propagation_20261121_v1/budget.json')
    budget={k:v for k,v in budget.items() if not k.startswith(('A_','threshold_'))}
    budget.update(schema='fire13-budget-1', estimation_source='C8 measured resources and NEWPOS4 conservative envelope',
                  source_budget_sha256=sha256(ROOT/'reports/c8_propagation_20261121_v1/budget.json'),
                  full_session_archive_copies=3, local_and_external_copies_required=True)
    budget.update(B_session_minutes=300,B_card_hours=10.,B_fit_seconds=5400,B_evaluation_seconds=8100,
        B_cuda_breakdown_minutes=dict(setup=15,fit=90,inference_and_AP=120,comparison=15,archives=30,margin=30),
        input_batch_bytes=16*3*13*64*64*4,worker_descriptor_cache_bytes=64*1024**2,
        corpus_overlay_required=False,derived_storage_bytes=0,
        cost_pilot_sha256=sha256(out/'cost_pilot.json'))
    save(out/'budget.json', budget)
    save(out/'acceptance.json', dict(AP_new_min=0.11512837301201887,
         AP_pooled_min=0.34023390370637985, Dice_pooled_min=0.4339254871567759,
         recall_new_min=0.08138351147320427, Dice_new_min=0.1106963425412671,
         FP_new_max=29134, macro_Dice_min=0.22778327086319602, Brier_max=0.0031547185297208474,
         threshold=0.78, bootstrap_repetitions=2000, bootstrap_seed=20261121,
         exact_domain_required=True, authorized_budget_required=True, promotion_authorized=False))
    save(out/'comparable_fp_policy.json',dict(FPR_rational=[25334,103727130],folds=5,calibration_events=64,assessment_events=16,
        assignment='SHA256 geofirex-threshold-v1 event rank modulo5',budget='floor(rate * calibration negatives)',
        tie_rule='global score ties indivisible, maximum recall and highest threshold at equal recall',
        lock_all_thresholds_before_assessment=True,operational_threshold_selected=False,acceptance_unchanged=True))
    files = dict(lock['files'])
    extra = [CONFIG, 'src/geofirex/fire13_features.py','scripts/prepare_fire13_data.py','docs/FIRE13_LANCEMENT.md',
         'reports/newpos4_thresholds_20261121_v1/PROTOCOLE_NEWPOS2_ET_CANAUX.md',
         'reports/protocol_v2_execution/execution_manifest.json',
         'src/geofirex/c8_newpos_loss.py', 'src/geofirex/c8_predictions.py',
         'src/geofirex/c8_comparison.py', 'src/geofirex/fire13_experiment.py',
         'src/geofirex/fire13_comparison.py', 'scripts/run_fire13.py',
         'scripts/prepare_fire13.py', 'tests/test_fire13.py']
    extra += [(REPORT/n).as_posix() for n in ('baseline_reference.json', 'newpos4_reference.json','newpos2_reference.json','budget.json','acceptance.json','comparable_fp_policy.json',
        'scene_geometry.json','derived_normalization.json','parity_full.json','cost_pilot.json','inventory.json','raw_band_decisions.json')]
    files.update({n: sha256(ROOT/n) for n in extra})
    save(out/'execution_manifest.json', dict(schema='fire13-execution-1', files=files,
         run_id=RUN, original_scientific_identities=77, training_authorized=False))
    auths = []
    for action in ('gpu-preflight', 'fire13-fit', 'fire13-evaluate'):
        name = f'authorization_{action}_DISABLED.json'
        save(out/name, dict(authorized=False, action=action, experiment_id=RUN, device='cuda', runs=1,
             execution_manifest_sha256=sha256(out/'execution_manifest.json'), authorized_by='', authorized_at='',
             output_parent='/kaggle/working', session_remaining_minutes=None,
             quota_remaining_card_hours=None, card_hours_reserved=None, quota_checked_at='',
             local_backup_destination='', external_backup_destination='D:\\GeoFire-X\\GeoFire-X-V2'))
        auths.append((REPORT/name).as_posix())
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(set(files) | set(auths) | {(REPORT/'execution_manifest.json').as_posix()}):
            info = zipfile.ZipInfo(name, date_time=(2026,11,22,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (ROOT/name).read_bytes())
    blob = stream.getvalue(); digest = hashlib.sha256(blob).hexdigest()
    with (out/'fire13_code.zip').open('xb') as f: f.write(blob)
    cells = [cell('markdown', '# GeoFire-X C8 FIRE13 — préparation désactivée\n\nUn seul fit neuf seed42, quatre époques, seuil 0,78. Internet OFF. Aucun pixel 2021. Attacher les onze datasets p00 à p10 existants. Aucun seuil NEWPOS4 supplémentaire.\n', 0),
        cell('code', "RUN_GPU_PREFLIGHT = False\nRUN_FIT = False\nRUN_RECOVERY = False\nimport os,sys\nif 'torch' in sys.modules: raise RuntimeError('Redémarrer le noyau avant CUDA_VISIBLE_DEVICES')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n", 1),
        cell('code', "from pathlib import Path\nimport base64,hashlib,io,zipfile,json\n"+
             f"DATA=base64.b64decode({base64.b64encode(blob).decode()!r})\nassert hashlib.sha256(DATA).hexdigest()=={digest!r}\n"+
             "ROOT=Path('/kaggle/working/geofirex-fire13-code')\nROOT.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(DATA)) as z:\n    for n in z.namelist():\n        assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())\n    z.extractall(ROOT)\nsys.path.insert(0,str(ROOT/'src'))\nfrom geofirex import fire13_experiment as c\nprint(c.preflight(ROOT))\n", 2),
        cell('code', "EXPECTED={v['map_sha256'] for v in c.read(ROOT/c.old.REPORT/'partitions.json').values()}\nMAPS=[p for p in Path('/kaggle/input').rglob('mount-map.json') if c.sha256(p) in EXPECTED]\nassert len(MAPS)==11, 'Onze partitions p00 à p10 distinctes requises'\nprint(c.preflight(ROOT,MAPS))\nprint(json.dumps(c.read(ROOT/c.REPORT/'budget.json'),indent=2))\n", 3),
        cell('markdown', "Autorisation GPU distincte obligatoire : copier les templates DISABLED dans /kaggle/working, renseigner authorized=True, opérateur/date UTC, quota actuel, durée disponible et deux destinations de sauvegarde. Le SHA du manifeste doit rester identique. Réserver 300 minutes et 10 heures-carte. Exécuter le précontrôle puis un seul fit. Ne jamais supprimer le verrou pour recommencer ; pas de resume.\n", 4),
        cell('code', "GPU_AUTH=Path('/kaggle/working/fire13_gpu_authorization.json')\nAUTH=Path('/kaggle/working/fire13_fit_authorization.json')\nOUT=Path('/kaggle/working/'+c.B_ID)\nif RUN_GPU_PREFLIGHT:\n    c.authorize(ROOT,GPU_AUTH,'gpu-preflight','cuda',OUT.parent)\n    result=c.preflight(ROOT,MAPS,OUT.parent,'cuda',gpu=True)\n    c.save(Path('/kaggle/working/fire13_gpu_preflight.json'),result)\n    print(result)\nelse: print('Précontrôle GPU désactivé')\n", 5),
        cell('code', "if RUN_FIT:\n    assert RUN_GPU_PREFLIGHT and not RUN_RECOVERY\n    c.fit(ROOT,MAPS,OUT,AUTH)\nelse: print('Fit FIRE13 désactivé')\n", 6),
        cell('markdown', "Après interruption de l’évaluation seulement : restaurer TRAINING-SAFE avec son SHA, autorisation fire13-evaluate distincte. Aucun entraînement supplémentaire. Un fit incomplet reste PARTIAL.\n", 7),
        cell('code', "RECOVERY_SOURCE=Path('/kaggle/working/restored_fire13')\nRECOVERY_OUT=Path('/kaggle/working/'+c.B_ID+'_EVALUATION_RECOVERY')\nRECOVERY_AUTH=Path('/kaggle/working/fire13_evaluation_authorization.json')\nRECOVERY_ARCHIVE=Path('/kaggle/input/REPLACE_PRIVATE_MOUNT/fire13-training-safe.bin')\nif RUN_RECOVERY:\n    assert RUN_GPU_PREFLIGHT and not RUN_FIT\n    c.authorize(ROOT,RECOVERY_AUTH,'fire13-evaluate','cuda',RECOVERY_OUT.parent)\n    c.restore_candidate(ROOT,RECOVERY_ARCHIVE,RECOVERY_ARCHIVE.with_suffix('.sha256'),RECOVERY_SOURCE)\n    c.evaluate_candidate(ROOT,MAPS,RECOVERY_SOURCE,RECOVERY_OUT,RECOVERY_AUTH)\nelse: print('Récupération évaluation seule désactivée')\n", 8),
        cell('code', "from IPython.display import display,FileLink\nfor p in sorted(Path('/kaggle/working').glob(c.B_ID+'*.zip')):\n    print(c.old.verify_zip(p))\n    display(FileLink(str(p)));display(FileLink(str(p.with_suffix('.sha256'))))\n", 9)]
    cells[0]=cell('markdown', '# GeoFire-X FIRE13 - preparation desactivee\n\n13 canaux/jour, 39 entrees empilees : C8 + distance, densite, apparitions et evolution observees, masque de disponibilite. Loss C8 et LocalitySampler inchanges. Source : trois jours observes jusqu\'a J ; aucune cible J+1 dans les derivees. Un seul fit neuf seed42, quatre epoques, seuil0,78. Internet OFF, p00-p10 existants, aucun pixel2021. Protocole complet embarque dans docs/FIRE13_LANCEMENT.md.\n',0)
    notebook = ROOT/'notebooks/GEOFIREX_C13_FIRE_DISABLED.ipynb'
    save(notebook, dict(nbformat=4, nbformat_minor=5, cells=cells,
         metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'))))
    save(out/'notebook_build.json', dict(payload_sha256=digest, payload_bytes=len(blob),
         notebook_sha256=sha256(notebook), all_flags_disabled=True, published=False))
    save(out/'kernel_metadata_NOT_PUBLISHED.json', dict(id='nnnnbhss/geofirex-c8-fire13-prepared',
         title='GeoFire-X C8 FIRE13 DISABLED', code_file=notebook.name, language='python',kernel_type='notebook',
         is_private=True, enable_gpu=False,enable_internet=False,
         dataset_sources=old.read(ROOT/old.REPORT/'dataset-sources.json')))
    print(json.dumps(dict(status='PREPARED_DISABLED',payload_bytes=len(blob),payload_sha256=digest)))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.parse_args()
    main()
