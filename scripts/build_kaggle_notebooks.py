"""Generate guarded CPU/GPU operator notebooks. Does not execute any cell."""
from pathlib import Path
from geofirex.corpus import json_new


def md(text):
    import hashlib
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:12], "metadata": {}, "source": text.strip()+"\n"}


def code(text):
    import hashlib
    return {"cell_type": "code", "id": hashlib.sha256(text.encode()).hexdigest()[:12], "metadata": {}, "source": text.strip()+"\n", "outputs": [], "execution_count": None}


SETUP = r'''
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
CODE_SOURCE = Path('/kaggle/input/datasets/nnnnbhss/geofirex-code-v3/GeoFire-X-2026')  # adapter au dataset de code v3
CODE = Path('/kaggle/working/GeoFire-X-2026-v3')
assert CODE_SOURCE.is_dir(), 'Renseigner le chemin du code v3 attaché au notebook'
if not CODE.exists():
    shutil.copytree(CODE_SOURCE, CODE)
manifest = json.loads((CODE_SOURCE/'code_manifest.json').read_text())
assert all(hashlib.sha256((CODE/name).read_bytes()).hexdigest() == digest for name, digest in manifest.items()), 'Code local différent du paquet v3'
os.chdir(CODE)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', '.'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'kagglehub>=1.0.3,<2'], check=True)
sys.path.insert(0, str(CODE/'src'))
from kagglehub.auth import get_username
assert get_username() == 'nnnnbhss', 'Compte Kaggle authentifié inattendu'
print('Compte vérifié :', get_username())
'''


def build(*, update=False):
    cpu = [md("""# CPU : adopter batch001, puis publier les lots pilotes
Notebook de procédure, sans résultats préremplis. Première session : `PHASE='adopt'`.
Conserver le JSON d'adoption hors de la VM. Ouvrir ensuite un **nouveau notebook CPU**,
réutiliser ce notebook avec `PHASE='verify_and_plan'` et coller ce JSON.
Le dataset manuel n'est ni recréé ni retéléchargé. Ne copier aucun marqueur de session.
"""), code(SETUP), code(r'''
PHASE = 'adopt'  # puis 'verify_and_plan' dans un NOUVEAU notebook CPU
MANUAL_ROOT = Path('/kaggle/input/datasets/nnnnbhss/geofire-x-wsts-batch001')
MANUAL_HANDLE = 'nnnnbhss/geofire-x-wsts-batch001'
INDEX_PATH = Path('data/manifests/real/corpus_zip_index_v1.json')
INDEX = json.loads(INDEX_PATH.read_text())
ADOPTION_PROOF_JSON = ''  # coller le JSON de la première session, pas un reçu d'upload inventé
def save_once(path, value):
    from geofirex.corpus import json_new
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Choisir une nouvelle sortie pour un résultat différent'
    else:
        json_new(path, value)
'''), md("## Session A — adoption en lecture seule"), code(r'''
if PHASE == 'adopt':
    from geofirex.kaggle_persistence import adopt_manual, session_id
    proof = adopt_manual(INDEX, MANUAL_ROOT, MANUAL_HANDLE, session=session_id())
    save_once(Path('/kaggle/working/manual-adoption.json'), proof)
    print(json.dumps(proof, indent=2))
    print('Conserver ce JSON hors de la VM, puis ouvrir un nouveau notebook CPU.')
'''), md("## Session B — vérification indépendante et certificat bootstrap"), code(r'''
if PHASE == 'verify_and_plan':
    from geofirex.kaggle_persistence import verify_fresh_session, session_id, validate_bootstrap
    assert ADOPTION_PROOF_JSON.strip(), 'Coller le JSON d’adoption provenant de la session A'
    proof = json.loads(ADOPTION_PROOF_JSON)
    certificate = verify_fresh_session(proof, session=session_id())
    validate_bootstrap(certificate, INDEX)
    BOOTSTRAP = Path('/kaggle/working/bootstrap-manual-verified.json')
    save_once(BOOTSTRAP, certificate)
    print(json.dumps(certificate, indent=2))
    print('Conserver également ce certificat hors de la VM.')
'''), md("## Tester l'API automatique avant plusieurs Gio\nCette sonde publie 60 minuscules fichiers texte (aucune donnée scientifique), teste aussi le ZIP automatique et relit tous les hashes. Elle ne remplace pas le certificat du corpus manuel. Activer seulement cette petite publication pour valider les permissions de création réelles avant les gros lots."), code(r'''
RUN_PUBLICATION_PROBE = False
if PHASE == 'verify_and_plan' and RUN_PUBLICATION_PROBE:
    import uuid
    subprocess.run([sys.executable, '-m', 'scripts.kaggle_publication_probe',
        '--job', '/kaggle/working/publication-probe-v3', '--handle', 'nnnnbhss/geofirex-publication-probe-v3',
        '--out', f'/kaggle/working/publication-probe-{uuid.uuid4().hex[:8]}.json', '--execute'], check=True)
'''), md("## Quotas réels et plans sans transfert\nRenseigner le stockage **privé encore disponible** dans le compte Kaggle. Les 3 Gio de transfert par lot sont un plafond à approuver, pas un quota supposé. Le disque est mesuré et le ZIP d'upload est budgété."), code(r'''
PRIVATE_FREE_GIB = None  # OBLIGATOIRE : valeur vérifiée dans votre compte Kaggle
PLAN_DIR = Path('/kaggle/working/pilot-v3')
if PHASE == 'verify_and_plan':
    from scripts.prepare_kaggle_pilot import prepare
    from geofirex.cloud_corpus import validate_plan
    assert PRIVATE_FREE_GIB is not None and PRIVATE_FREE_GIB > 0, 'Renseigner le quota privé disponible'
    private_bytes = int(PRIVATE_FREE_GIB*1024**3)
    free = shutil.disk_usage('/kaggle/working').free
    allocation = min(free, 8*1024**3)
    if not PLAN_DIR.exists():
        report = prepare(INDEX, PLAN_DIR, private_free_bytes=private_bytes,
                         storage_bytes=allocation, roots=[MANUAL_ROOT])
    else:
        report = json.loads((PLAN_DIR/'pilot.json').read_text())
        assert report['private_quota_declared_bytes'] == private_bytes, 'Nouveaux quotas : choisir un autre PLAN_DIR'
        for label in ['train2018', 'val2020', 'extension2019']:
            validate_plan(json.loads((PLAN_DIR/f'{label}.json').read_text()), INDEX)
    print(json.dumps(report, indent=2))
'''), md("## Commandes exactes — affichées avant exécution"), code(r'''
PUBLICATION_MODE = 'create'  # 'resume' seulement pour vérifier une publication possiblement déjà réussie
def acquisition_command(label):
    plan = json.loads((PLAN_DIR/f'{label}.json').read_text())
    limits = plan['limits']
    minimum, maximum = (30, 40) if label == 'train2018' else (20, 30)
    args = [sys.executable, '-m', 'scripts.kaggle_acquire_publish',
        '--index', str(INDEX_PATH), '--job', f'/kaggle/working/jobs-v3/{label}',
        '--handle', f'nnnnbhss/geofire-x-wsts-{label}-pilot01', '--years', plan['events'][0][:4],
        '--min-events', str(minimum), '--max-events', str(maximum), '--plan', str(PLAN_DIR/f'{label}.json'),
        '--bootstrap-verification', str(BOOTSTRAP), '--acquired-roots', str(MANUAL_ROOT),
        '--publication-mode', PUBLICATION_MODE]
    for key, flag in [('transfer_bytes','transfer'), ('storage_bytes','storage'),
                      ('upload_bytes','upload'), ('private_free_bytes','private-free')]:
        args += [f'--{flag}-bytes', str(limits[key])]
    return args
if PHASE == 'verify_and_plan':
    import shlex
    for label in ['train2018', 'val2020', 'extension2019']:
        print(label, ':', shlex.join(acquisition_command(label)+['--execute']))
'''), md("## Exécution autorisée — une cellule par lot\nPasser uniquement le booléen du lot approuvé à `True`. La cellule publie et relit la version privée avant de se terminer. Conserver le reçu affiché hors de la VM. Après une interruption d'acquisition, relancer le même job ; après une réponse d'upload perdue, utiliser `PUBLICATION_MODE='resume'`. Aucun fichier local n'est supprimé."), code(r'''
RUN_TRAIN_2018 = False
if PHASE == 'verify_and_plan' and RUN_TRAIN_2018:
    subprocess.run(acquisition_command('train2018')+['--execute'], check=True)
'''), code(r'''
RUN_VAL_2020 = False
if PHASE == 'verify_and_plan' and RUN_VAL_2020:
    subprocess.run(acquisition_command('val2020')+['--execute'], check=True)
'''), code(r'''
RUN_EXTENSION_2019 = False
if PHASE == 'verify_and_plan' and RUN_EXTENSION_2019:
    subprocess.run(acquisition_command('extension2019')+['--execute'], check=True)
'''), md("## Récupérer les reçus pour le notebook GPU indépendant\nCes petits JSON doivent être conservés hors de `/kaggle/working`. Le dataset privé contient les TIFF ; les reçus servent à vérifier les versions et la nouvelle session."), code(r'''
if PHASE == 'verify_and_plan':
    receipts = []
    for label in ['train2018', 'val2020', 'extension2019']:
        folder = Path('/kaggle/working/jobs-v3')/label
        for path in sorted(folder.glob('publication-*.json')):
            receipt = json.loads(path.read_text())
            if receipt['status'] == 'REMOTE_VERIFIED_FRESH_SESSION_PENDING':
                receipts.append(receipt)
    print(json.dumps(receipts, indent=2))
''')]
    gpu = [md("""# GPU T4 : données persistées, mesure de débit, premières epochs
Ouvrir un **nouveau notebook GPU**, activer une Tesla T4 et Internet pour les sauvegardes.
Coller les vrais reçus CPU ; aucun score expérimental n'est prérempli.
Les cellules coûteuses sont désactivées par défaut. Le total reste fixé à 30 epochs,
avec un premier arrêt après deux epochs pour mesurer le coût réel et reprendre ensuite.
"""), code(SETUP), md("## Vérifier les lots CPU depuis cette nouvelle session et construire le manifeste cumulatif"), code(r'''
BATCH_RECEIPTS_JSON = ''  # coller la liste JSON des reçus train2018 et val2020 (et 2019 si disponible)
from geofirex.kaggle_persistence import verify_fresh_session, session_id, KaggleBackend
from geofirex.cloud_corpus import prepare_mounted
from geofirex.corpus import json_new
assert BATCH_RECEIPTS_JSON.strip(), 'Les vrais reçus CPU sont requis'
receipts = json.loads(BATCH_RECEIPTS_JSON)
assert len(receipts) >= 2
backend = KaggleBackend()
roots = []
for receipt in receipts:
    certificate = verify_fresh_session(receipt, session=session_id(), backend=backend)
    root = backend.mount(receipt['versioned_handle'])
    if root not in roots:
        roots.append(root)
    print(certificate['status'], receipt['versioned_handle'])
manual_root = Path('/kaggle/input/datasets/nnnnbhss/geofire-x-wsts-batch001')
if manual_root.is_dir() and manual_root not in roots:
    from geofirex.kaggle_persistence import inspect_dataset
    assert inspect_dataset('nnnnbhss/geofire-x-wsts-batch001')['state'] == 'EXISTING_PRIVATE'
    roots.append(manual_root)  # joindre batch001 dans Add Input pour l’inclure
index_path = Path('data/manifests/real/corpus_zip_index_v1.json')
index = json.loads(index_path.read_text())
prepared = Path('/kaggle/working/prepared-v3')
assert not prepared.exists(), 'Utiliser une nouvelle sortie ou reprendre le manifeste existant identique'
report = prepare_mounted(index, roots, prepared)
print(json.dumps(report, indent=2))
'''), md("## Configurations comparables, quotas de checkpoints et GPU réel"), code(r'''
PRIVATE_CHECKPOINT_FREE_GIB = None  # disponible réel du compte, APRÈS acquisition
APPROVED_CHECKPOINT_UPLOAD_GIB = None  # budget total de payload pour dix runs, à approuver
assert PRIVATE_CHECKPOINT_FREE_GIB and APPROVED_CHECKPOINT_UPLOAD_GIB
from scripts.kaggle_campaign import configure
import torch
assert torch.cuda.is_available(), 'CUDA indisponible ; ne pas remplacer le build PyTorch CUDA'
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
campaign = Path('/kaggle/working/campaign-v3')
configs = configure(prepared, index_path, campaign, 'nnnnbhss/geofirex-pilot-v3',
    int(APPROVED_CHECKPOINT_UPLOAD_GIB*1024**3), int(PRIVATE_CHECKPOINT_FREE_GIB*1024**3))
print([(c['model'], c['seed'], c['planned_optimizer_updates']) for c in configs])
'''), md("## Première mesure de débit GPU — modèle jeté, aucune métrique scientifique\n100 pas mesurés + 10 pas d'échauffement. Lire les temps et la VRAM avant d'autoriser les epochs."), code(r'''
RUN_THROUGHPUT = False
MODEL = 'B1'  # puis B2/B3 pour mesurer leur coût ; B1 utilise spatial_cnn_v2 explicite
config_path = campaign/f'{MODEL}_s42.yaml'
if RUN_THROUGHPUT:
    subprocess.run([sys.executable, '-m', 'scripts.measure_gpu_throughput', '--config', str(config_path),
        '--steps', '100', '--warmup', '10', '--out', f'/kaggle/working/throughput-{MODEL}.json'], check=True)
    print(Path(f'/kaggle/working/throughput-{MODEL}.json').read_text())
'''), md("## B0 une fois, puis deux vraies epochs par modèle entraînable\nDeux epochs représentent 1 024 updates planifiées par modèle, avant les pas éventuellement sautés par AMP. B0 n'a pas d'optimiseur. Chaque epoch et la fin de session produisent une sauvegarde privée vérifiée côté producteur."), code(r'''
RUN_B0 = False
if RUN_B0:
    subprocess.run([sys.executable, 'scripts/run_benchmark.py', 'train', '--config', str(campaign/'B0_s42.yaml')], check=True)
'''), code(r'''
RUN_TWO_EPOCHS = False
if RUN_TWO_EPOCHS:
    subprocess.run([sys.executable, 'scripts/run_benchmark.py', 'train', '--config', str(config_path),
        '--stop-after-epochs', '2'], check=True)
'''), md("## Sauvegardes et reprise dans une autre session\nConserver le reçu `backup_0002_final.json` hors de la VM. Dans une nouvelle session, vérifier ce reçu avec `verify_fresh_session`, puis monter sa version Kaggle. Recréer le même manifeste et les mêmes configurations à partir du même paquet de code. `stop-after-epochs` ne modifie pas l'identité scientifique du run. Une session partielle porte l'état `PAUSED_AT_EPOCH_BOUNDARY` et ne peut pas être classée comme un run terminé."), code(r'''
print('Reçus de sauvegarde à conserver :')
for path in Path('/kaggle/working/geofirex-runs').glob('*/backup_*_final.json'):
    print(path)
    print(path.read_text())
'''), code(r'''
RUN_RESUME = False
CHECKPOINT_RECEIPT_JSON = ''  # reçu d'une AUTRE session, à vérifier ici
if RUN_RESUME:
    receipt = json.loads(CHECKPOINT_RECEIPT_JSON)
    print(verify_fresh_session(receipt, session=session_id(), backend=backend))
    snapshot = backend.mount(receipt['versioned_handle'])
    checkpoints = sorted((snapshot/'run').glob('epoch_*.pt'))
    assert checkpoints, 'Aucun checkpoint complet dans le snapshot vérifié'
    subprocess.run([sys.executable, 'scripts/run_benchmark.py', 'train', '--config', str(config_path),
        '--resume', str(checkpoints[-1]), '--stop-after-epochs', '2'], check=True)
'''), md("## Étape Big Data suivante — après les lots, dans CPU\nExporter les métadonnées réelles montées en Parquet, puis lancer Spark avec au moins 30 événements et 500 observations. Une exécution `local[2]` est un traitement Spark local, pas un cluster. PySpark/JVM ne sont pas installés ni validés par les tests locaux de ce notebook."), code(r'''
import shlex
metadata = '/kaggle/working/metadata-pilot-v3'
print(shlex.join([sys.executable, '-m', 'scripts.export_observation_metadata', '--manifest', str(prepared/'manifest.csv'),
    '--out-dir', metadata, '--data-roots', *map(str, roots)]))
print(shlex.join([sys.executable, '-m', 'scripts.spark_metadata', '--input', metadata+'/observations.parquet',
    '--output', '/kaggle/working/spark-pilot-v3', '--master', 'local[2]', '--min-events', '30',
    '--min-observations', '500', '--out-report', '/kaggle/working/spark-pilot-v3.json']))
''')]
    for name, cells in (("kaggle_cpu_unblock_v3.ipynb", cpu), ("kaggle_gpu_pilot_v3.ipynb", gpu)):
        payload = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
        target = Path("notebooks")/name
        if update:
            import json
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            json_new(target, payload)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Explicitly replace the generated notebook sources, including any outputs")
    build(update=parser.parse_args().update)
