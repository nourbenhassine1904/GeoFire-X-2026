"""Offline, one-time notebook logistics update. Never imports torch or data readers."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'reports/focal_preparation_v1'
OUT = ROOT/'reports/focal_go_final_v1'


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False); f.write('\n')


def main():
    helper = ROOT/'scripts/focal_run_logistics.py'
    spec = importlib.util.spec_from_file_location('logistics', helper)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    c = json.loads((PREP/'focal_config.json').read_bytes())
    report = ROOT/'reports/focal_repro_precheck_v1/precheck_report_kaggle_PASS.json'
    m.checked_pass(report, c)
    assert digest(PREP/'focal_config.json') == m.CONFIG_SHA
    assert digest(PREP/'focal-code.zip') == m.CODE_SHA
    assert (ROOT/'.git/HEAD').read_text().strip() == 'ref: refs/heads/feature/scientific-improvements-v1'
    # Sources and UI fingerprints only. No checkpoint/corpus rescan.
    protected = [p for folder in ('src', 'configs', 'frontend')
                 for p in (ROOT/folder).rglob('*') if p.is_file()
                 and not {'node_modules', '__pycache__', '.next', 'dist'} & set(p.parts)]
    protected += [PREP/n for n in ('focal_config.json','focal-code.zip','baseline_evaluation32.json',
                                  'baseline_config.json','baseline_selection32.json')]
    OUT.mkdir(exist_ok=False)
    save(OUT/'preservation_before.json', {p.relative_to(ROOT).as_posix():digest(p) for p in protected})
    notebook = PREP/'GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb'
    shutil.copyfile(notebook, OUT/'GEOFIREX_FOCAL_SINGLE_DISABLED.before.ipynb')
    shutil.copyfile(report, OUT/report.name)
    with (OUT/(report.name+'.sha256')).open('x') as f:
        f.write(m.PASS_SHA+'  '+report.name+'\n')
    n = json.loads(notebook.read_bytes())
    codes = [cell for cell in n['cells'] if cell['cell_type'] == 'code']
    assert len(codes) == 4
    n['cells'][0]['source'] = '''# GeoFire-X — unique Focal, préparation finale après PASS
Le PASS Kaggle valide environnement, initialisation et corpus. Aucun entraînement effectué.
Paquet scientifique et configuration inchangés ; seule la logistique du notebook évolue.
Internet OFF. Attacher les 11 datasets privés EXISTANTS, aucune réimportation/publication.
GPU T4 x2 peut être proposé par Kaggle ; CUDA_VISIBLE_DEVICES=0 expose UNE seule T4.
Ne pas lancer Run All/Save Version avec exécution avant autorisation explicite.
Réserve de session proposée : 120 min, dont 45 min maximum de fit/session (pause existante).
Contrôle : 12 GiB libres sur working et tmp ; AP exacte non approximée.
RUN_TRAINING=False ; aucun fichier d'autorisation active n'est livré.
'''
    codes[0]['source'] = codes[0]['source'].replace('import base64,',
        'import os\nos.environ["CUDA_VISIBLE_DEVICES"]="0"  # BEFORE torch/CUDA import\nimport base64,', 1)
    for path, name in ((helper,'focal_run_logistics.py'),(report,report.name)):
        data = path.read_bytes()
        codes[0]['source'] += (f'\nextra=base64.b64decode({base64.b64encode(data).decode()!r})\n'
            f'assert hashlib.sha256(extra).hexdigest()=={hashlib.sha256(data).hexdigest()!r}\n'
            f'with (workspace/{name!r}).open("xb") as f: f.write(extra)\n')
    # Retain the existing scientific mount cell and runner metadata checks exactly.
    codes[1]['source'] += '''
import focal_run_logistics as logistics
logistics.identity(workspace)
print("Disk free /working:", logistics.disk_check("/kaggle/working"))
print("Disk free /tmp:", logistics.disk_check("/tmp"))
import torch, numpy as np
assert torch.__version__=="2.11.0+cu130" and np.__version__=="2.1.3", "Environment drift; do not train"
print("PASS accepted; frozen science unchanged; training remains disabled.")
'''
    for cell in n['cells']:
        if cell['cell_type'] == 'markdown' and cell['source'].startswith('## Explicit'):
            cell['source'] = '''## Autorisation explicite requise
Conserver RUN_TRAINING=False jusqu'à l'accord explicite portant sur UN run et le budget complet.
Ensuite seulement créer le fichier externe selon authorization_TEMPLATE_DISABLED.json :
hashes config/code/logistique, runs=1, max_training_seconds=2700, session_reserve_minutes=120,
gpu_count=1, destinations locale/externe et authorized_by. Ce modèle n'autorise rien.
Le verrou de session interdit un second lancement. Aucun resume/retrain automatique.
'''
        if cell['cell_type'] == 'markdown' and cell['source'].startswith('## Preserve'):
            cell['source'] = '''## Sauvegarde et récupération
TRAINING-SAFE.zip + SHA est écrit et relu AVANT l'évaluation32/AP exacte inchangée.
COMPLETE.zip contient sélection32, comparaison historique, métriques, historique et checkpoint.
PARTIAL.zip est tenté en finally en cas d'erreur/pause ; un arrêt brutal du noyau ne peut garantir finally.
Télécharger ZIP + SHA, garder les outputs Kaggle puis exécuter le script backup local/externe.
Ne jamais clôturer la session avant le téléchargement et sa vérification.
Si AP échoue après 23404 updates, utiliser evaluate-only sur CE checkpoint, jamais retrain.
Le guide KAGGLE_CELLULES_ET_SAUVEGARDE.md donne les commandes exactes.
'''
    codes[-2]['source'] = '''RUN_TRAINING=False
AUTHORIZATION_FILE=None
OUT=None
if RUN_TRAINING:
    if AUTHORIZATION_FILE is None: raise RuntimeError("Explicit authorization required")
    import uuid
    OUT=Path("/kaggle/working")/("geofirex-focal-"+uuid.uuid4().hex)
    execution=[sys.executable,str(workspace/"focal_run_logistics.py"),"train",
               "--workspace",str(workspace),"--corpus",str(corpus),"--out",str(OUT),
               "--authorization",str(AUTHORIZATION_FILE)]
    subprocess.run(execution,check=True,env=os.environ.copy())
else:
    print("DISABLED: no training/evaluation/GPU call.")
'''
    codes[-1]['source'] = '''if RUN_TRAINING and OUT is not None:
    from IPython.display import display, FileLink
    archives=sorted(OUT.parent.glob(OUT.name+"-*.zip"))
    for archive in archives:
        logistics.verify_zip(archive,archive.with_suffix(".sha256").read_text().split()[0])
        display(FileLink(str(archive)))
        display(FileLink(str(archive.with_suffix(".sha256"))))
    if not archives: raise RuntimeError("No verified archive: preserve working directory manually")
'''
    for cell in codes:
        compile(cell['source'], 'disabled-notebook', 'exec')
        cell['execution_count']=None; cell['outputs']=[]
    notebook.write_text(json.dumps(n,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    save(OUT/'authorization_TEMPLATE_DISABLED.json', dict(training_authorized=False, runs=1,
        experiment=c['experiment'], config_sha256=m.CONFIG_SHA, code_sha256=m.CODE_SHA,
        logistics_sha256=digest(helper), max_training_seconds=2700,session_reserve_minutes=120,
        gpu_count=1,checkpoint_backup_destination='',local_backup_destination='',authorized_by=''))
    save(OUT/'decision.json', dict(preparation='READY',execution='BLOCKED_PENDING_EXPLICIT_AUTHORIZATION',
        kaggle_report_received=True,kaggle_prerequisites='PASS',pass_sha256=m.PASS_SHA,
        config_sha256=m.CONFIG_SHA,code_sha256=m.CODE_SHA,logistics_sha256=digest(helper),
        notebook_sha256=digest(notebook),RUN_TRAINING=False,training_authorized=False,
        training_executed=False,forward_calls=0,optimizer_steps=0,gpu_jobs_submitted=0,
        test2021_pixels_opened=0,scientific_payload_changed=False,
        full_session_reserve_minutes=120,training_pause_cap_seconds=2700,
        timings='planning reserve; setup/evaluation/AP/backup not measured; no guaranteed wall-time bound',
        minimum_free_bytes_each_working_tmp=12*1024**3,
        remaining_user_actions=['explicit authorization of one run, full budget and backup destinations',
         'use existing 11 mounts; verify current session identity and free space',
         'download verified training-safe and complete ZIP+SHA before ending session',
         'verified local and external GeoFire-X copies after download'],
        next_science='asset-level availability by J, then comparable multimodal U-Net; ResNet18-U-Net then U-TAE'))
    before = json.loads((OUT/'preservation_before.json').read_bytes())
    assert all(digest(ROOT/p)==h for p,h in before.items())
    save(OUT/'preservation_after.json',dict(files=len(before),all_unchanged=True,
        protected_sha256=before,checkpoint_or_data_files_written=False,git_commands_executed=False))
    print('READY preparation; execution disabled. PASS SHA256:',m.PASS_SHA)


if __name__ == '__main__': main()
