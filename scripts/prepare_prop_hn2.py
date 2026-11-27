"""Build an additive, offline, disabled PROP-HN2 delivery; never executes GPU work."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from geofirex.protocol_v2 import save, sha256
from geofirex import v2_execution as old

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path('reports/prop_hn2_20261122_v1')
RUN = 'B2_C8_V2_PROP_HN2_seed42'
CONFIG = 'configs/c8_prop_hn2_v1.json'


def cell(kind, source, i):
    result = dict(cell_type=kind, id=f'prop_hn2-{i}', metadata={}, source=source.splitlines(True))
    if kind == 'code': result.update(execution_count=None, outputs=[])
    return result


def main(refresh_draft=False):
    out = ROOT / REPORT
    if (out/'execution_manifest.json').exists():raise FileExistsError('Preserve built HN2 delivery')
    out.mkdir(exist_ok=True)
    base, lock, _ = old.identity(ROOT)
    assert len(lock['files']) == 77
    # Preserve existing scientific inputs and every existing source/config/notebook.
    names = set(lock['files'])
    for folder in ('src', 'scripts', 'configs', 'notebooks', 'splits'):
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    for folder in ('reports/c8_propagation_20261121_v1', 'reports/newpos4_thresholds_20261121_v1','reports/newpos2_20261122_v1'):
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).iterdir() if p.is_file())
    names -= {'src/geofirex/prop_hn2_sampling.py','scripts/qualify_prop_hn2_cpu.py','scripts/verify_newpos2_adata.py','scripts/prepare_prop_hn2.py','scripts/run_prop_hn2.py','tests/test_prop_hn2.py',
              'src/geofirex/prop_hn2_experiment.py','src/geofirex/prop_hn2_comparison.py',CONFIG,
              'notebooks/GEOFIREX_C8_PROP_HN2_DISABLED.ipynb'}
    save(out/'preservation_before.json', dict(files={n: sha256(ROOT/n) for n in sorted(names)},
         original_scientific_identities=77, policy='All changes additive; no existing file overwritten'))
    candidate=dict(base,run_id=RUN,status='PREPARED_DISABLED',
        base_config_sha256=sha256(ROOT/'configs/protocol_v2/baseline_c8.json'),
        sampler=dict(name='PropHNSampler',seed=42,block_size=256,epochs=[0,1,2,3],
        policy='all_positive_once_hard_negative_twice_drop_equal_easy_without_replacement_reserve_each_event'))
    save(ROOT/CONFIG, candidate)
    for name, source in [('baseline_reference.json', 'reports/c8_propagation_20261121_v1/baseline_reference.json'),
                         ('newpos4_reference.json', 'reports/newpos4_thresholds_20261121_v1/newpos4_evaluation.json')]:
        save(out/name, old.read(ROOT/source))
    budget = old.read(ROOT/'reports/c8_propagation_20261121_v1/budget.json')
    budget.update(schema='prop_hn2-budget-1', estimation_source='C8 + verified NEWPOS2 observed throughput; conservative reserve for changed locality',
                  source_budget_sha256=sha256(ROOT/'reports/c8_propagation_20261121_v1/budget.json'),
                  full_session_archive_copies=3, local_and_external_copies_required=True)
    budget.update(B_session_minutes=255,B_card_hours=8.5,B_fit_seconds=5400,
        B_cuda_breakdown_minutes=dict(setup=15,fit=90,evaluation=90,comparison=10,archives=30,margin=20),
        measured_newpos2_fit_seconds=old.read(out/'newpos2_resources_fit.json')['fit_seconds'],
        cpu_hard_negative_qualification_seconds=old.read(out/'train_category_qualification.json')['seconds'])
    save(out/'budget.json', budget)
    save(out/'acceptance.json', dict(AP_new_min=0.11512837301201887,
         AP_pooled_min=0.34023390370637985, Dice_pooled_min=0.4339254871567759,
         recall_new_min=0.08138351147320427, Dice_new_min=0.1106963425412671,
         FP_new_max=29134, macro_Dice_min=0.22778327086319602, Brier_max=0.0031547185297208474,
         threshold=0.78, bootstrap_repetitions=2000, bootstrap_seed=20261121,
         exact_domain_required=True, authorized_budget_required=True, promotion_authorized=False))
    files = dict(lock['files'])
    extra = [CONFIG, 'src/geofirex/prop_hn2_sampling.py','scripts/qualify_prop_hn2_cpu.py','scripts/verify_newpos2_adata.py',
         'reports/newpos4_thresholds_20261121_v1/PROTOCOLE_NEWPOS2_ET_CANAUX.md',
         'docs/PROP_HN2_PROTOCOLE.md',
         'reports/protocol_v2_execution/execution_manifest.json',
         'src/geofirex/c8_newpos_loss.py', 'src/geofirex/c8_predictions.py',
         'src/geofirex/c8_comparison.py', 'src/geofirex/prop_hn2_experiment.py',
         'src/geofirex/prop_hn2_comparison.py', 'scripts/run_prop_hn2.py',
         'scripts/prepare_prop_hn2.py', 'tests/test_prop_hn2.py']
    extra += [(REPORT/n).as_posix() for n in ('baseline_reference.json','newpos4_reference.json','newpos2_reference.json',
        'newpos2_archive_verification.json','train_category_qualification.json','budget.json','acceptance.json')]
    files.update({n: sha256(ROOT/n) for n in extra})
    save(out/'execution_manifest.json', dict(schema='prop_hn2-execution-1', files=files,
         run_id=RUN, original_scientific_identities=77, training_authorized=False))
    auths = []
    for action in ('gpu-preflight', 'prop_hn2-fit', 'prop_hn2-evaluate'):
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
    with (out/'prop_hn2_code.zip').open('xb') as f: f.write(blob)
    cells = [cell('markdown', '# GeoFire-X C8 PROP-HN2 — préparation désactivée\n\nUn seul facteur : sampler HN2 ; loss historique C8 inchangée. Tous les positifs une fois ; négatifs avec feu observé à J et cible nulle deux fois ; autant de négatifs faciles retirés. Un seul fit neuf seed42, quatre époques, seuil 0,78. Internet OFF. Aucun pixel 2021. Attacher les onze datasets p00 à p10 existants. Aucun seuil supplémentaire.\n', 0),
        cell('code', "RUN_GPU_PREFLIGHT = False\nRUN_FIT = False\nRUN_RECOVERY = False\nimport os,sys\nif 'torch' in sys.modules: raise RuntimeError('Redémarrer le noyau avant CUDA_VISIBLE_DEVICES')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n", 1),
        cell('code', "from pathlib import Path\nimport base64,hashlib,io,zipfile,json\n"+
             f"DATA=base64.b64decode({base64.b64encode(blob).decode()!r})\nassert hashlib.sha256(DATA).hexdigest()=={digest!r}\n"+
             "ROOT=Path('/kaggle/working/geofirex-prop_hn2-code')\nROOT.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(DATA)) as z:\n    for n in z.namelist():\n        assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())\n    z.extractall(ROOT)\nsys.path.insert(0,str(ROOT/'src'))\nfrom geofirex import prop_hn2_experiment as c\nprint(c.preflight(ROOT))\n", 2),
        cell('code', "EXPECTED={v['map_sha256'] for v in c.read(ROOT/c.old.REPORT/'partitions.json').values()}\nMAPS=[p for p in Path('/kaggle/input').rglob('mount-map.json') if c.sha256(p) in EXPECTED]\nassert len(MAPS)==11, 'Onze partitions p00 à p10 distinctes requises'\nprint(c.preflight(ROOT,MAPS))\nprint(json.dumps(c.read(ROOT/c.REPORT/'budget.json'),indent=2))\n", 3),
        cell('markdown', "Autorisation GPU distincte obligatoire : copier les templates DISABLED dans /kaggle/working, renseigner authorized=True, opérateur/date UTC, quota actuel, durée disponible et deux destinations de sauvegarde. Le SHA du manifeste doit rester identique. Réserver 255 minutes et 8,5 heures-carte. Exécuter le précontrôle puis un seul fit. Ne jamais supprimer le verrou pour recommencer ; pas de resume.\n", 4),
        cell('code', "GPU_AUTH=Path('/kaggle/working/prop_hn2_gpu_authorization.json')\nAUTH=Path('/kaggle/working/prop_hn2_fit_authorization.json')\nOUT=Path('/kaggle/working/'+c.B_ID)\nif RUN_GPU_PREFLIGHT:\n    c.authorize(ROOT,GPU_AUTH,'gpu-preflight','cuda',OUT.parent)\n    result=c.preflight(ROOT,MAPS,OUT.parent,'cuda',gpu=True)\n    c.save(Path('/kaggle/working/prop_hn2_gpu_preflight.json'),result)\n    print(result)\nelse: print('Précontrôle GPU désactivé')\n", 5),
        cell('code', "if RUN_FIT:\n    assert RUN_GPU_PREFLIGHT and not RUN_RECOVERY\n    c.fit(ROOT,MAPS,OUT,AUTH)\nelse: print('Fit PROP-HN2 désactivé')\n", 6),
        cell('markdown', "Après interruption de l’évaluation seulement : restaurer TRAINING-SAFE avec son SHA, autorisation prop_hn2-evaluate distincte. Aucun entraînement supplémentaire. Un fit incomplet reste PARTIAL.\n", 7),
        cell('code', "RECOVERY_SOURCE=Path('/kaggle/working/restored_prop_hn2')\nRECOVERY_OUT=Path('/kaggle/working/'+c.B_ID+'_EVALUATION_RECOVERY')\nRECOVERY_AUTH=Path('/kaggle/working/prop_hn2_evaluation_authorization.json')\nRECOVERY_ARCHIVE=Path('/kaggle/input/REPLACE_PRIVATE_MOUNT/prop_hn2-training-safe.bin')\nif RUN_RECOVERY:\n    assert RUN_GPU_PREFLIGHT and not RUN_FIT\n    c.authorize(ROOT,RECOVERY_AUTH,'prop_hn2-evaluate','cuda',RECOVERY_OUT.parent)\n    c.restore_candidate(ROOT,RECOVERY_ARCHIVE,RECOVERY_ARCHIVE.with_suffix('.sha256'),RECOVERY_SOURCE)\n    c.evaluate_candidate(ROOT,MAPS,RECOVERY_SOURCE,RECOVERY_OUT,RECOVERY_AUTH)\nelse: print('Récupération évaluation seule désactivée')\n", 8),
        cell('code', "from IPython.display import display,FileLink\nfor p in sorted(Path('/kaggle/working').glob(c.B_ID+'*.zip')):\n    print(c.old.verify_zip(p))\n    display(FileLink(str(p)));display(FileLink(str(p.with_suffix('.sha256'))))\n", 9)]
    notebook = ROOT/'notebooks/GEOFIREX_C8_PROP_HN2_DISABLED.ipynb'
    save(notebook, dict(nbformat=4, nbformat_minor=5, cells=cells,
         metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'))))
    save(out/'notebook_build.json', dict(payload_sha256=digest, payload_bytes=len(blob),
         notebook_sha256=sha256(notebook), all_flags_disabled=True, published=False))
    save(out/'kernel_metadata_NOT_PUBLISHED.json', dict(id='nnnnbhss/geofirex-c8-prop_hn2-prepared',
         title='GeoFire-X C8 PROP-HN2 DISABLED', code_file=notebook.name, language='python',kernel_type='notebook',
         is_private=True, enable_gpu=False,enable_internet=False,
         dataset_sources=old.read(ROOT/old.REPORT/'dataset-sources.json')))
    print(json.dumps(dict(status='PREPARED_DISABLED',payload_bytes=len(blob),payload_sha256=digest)))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.parse_args()
    main()
