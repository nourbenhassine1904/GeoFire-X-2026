# Exécuter le runner sur Kaggle GPU et Runpod

Cette procédure est reproductible à partir d'un snapshot du dépôt, d'un dataset
Kaggle versionné et du YAML archivé dans chaque run. Elle n'effectue aucun
téléchargement automatique de WildfireSpreadTS. Le PyTorch CUDA fourni par
l'hôte doit être conservé.

## 1. Préparer les entrées et l'accélérateur

Créer un notebook GPU, activer l'accélérateur dans ses paramètres, puis attacher
séparément le snapshot du code et le dataset. Le nom exact des montages varie :
les inspecter avant de configurer les chemins. La présence d'un index des 607
événements n'est pas la présence de leurs TIFF.

```python
from pathlib import Path
import os, sys, subprocess, torch
print(sys.executable, torch.__version__, torch.version.cuda)
print('CUDA:', torch.cuda.is_available())
print([p.name for p in Path('/kaggle/input').iterdir()])
assert torch.cuda.is_available(), 'Activer le GPU ; ne pas réinstaller torch CPU'
print(torch.cuda.get_device_name(0))
```

Copier le snapshot du code dans `/kaggle/working/GeoFire-X-2026`, sans copier
un venv Windows, les données brutes ou d'anciens artefacts. Le dépôt lui-même
n'est pas téléchargé par le runner. Si le snapshot est un ZIP, vérifier son hash
et l'extraire dans ce nouveau dossier. Le répertoire de travail utilisé dans
les cellules suivantes contient `pyproject.toml`, `scripts/` et `src/`.

```python
repo = Path('/kaggle/working/GeoFire-X-2026')
assert (repo / 'pyproject.toml').is_file()
os.chdir(repo)
```

## 2. Vérifier les dépendances sans remplacer PyTorch

```python
subprocess.run([sys.executable, 'scripts/gpu_preflight.py', '--require-cuda',
                '--tests', '--out', '/kaggle/working/gpu_preflight.json'], check=True)
```

Si des dépendances manquent, relancer explicitement avec `--install-missing`
(Internet nécessaire pour ces packages). Ce mode contraint `torch` à sa version
déjà installée ; il ne demande ni son installation ni son upgrade. Un conflit
de dépendances doit échouer, pas entraîner un remplacement de PyTorch.
L'installation du projet est `pip install --no-deps --no-build-isolation -e .`.
Ne pas lancer `pip install torch`, une réinstallation des requirements entiers,
ou un upgrade global. Si setuptools/build backend manque, provisionner ce
composant séparément en conservant la contrainte torch.

Le script compare version/build CUDA dans un nouvel interpréteur, fait un
forward/backward CUDA minuscule, importe les dépendances, exécute les tests si
`--tests` et écrit un inventaire. Un environnement non compatible s'arrête avec
une erreur. Le rapport a un nom neuf à chaque tentative (pas d'écrasement).

## 3. Résoudre les chemins du dataset

Les `relative_path` du manifeste sont résolus **sous `data_root`**, pas sous le
dossier du notebook. Si une ligne est
`data/raw/WildfireSpreadTS/real_sample/2018/fire_.../2018-05-03.tif`, le montage
doit contenir cette arborescence, ou il faut créer un **nouveau manifeste** avec
les chemins relatifs corrects. Ne pas éditer silencieusement le manifeste épinglé.

```python
import csv, yaml
config = yaml.safe_load((repo/'configs/runner_kaggle.yaml').read_text())
config['data_root'] = '/kaggle/input/VOTRE-DATASET'
config['manifest'] = '/kaggle/input/VOTRE-DATASET/manifest.csv'
with open(config['manifest'], newline='') as f:
    rows = list(csv.DictReader(f))
missing = [r['relative_path'] for r in rows
           if not (Path(config['data_root'])/r['relative_path']).is_file()]
assert not missing, missing[:5]
config_path = Path('/kaggle/working/runner_session.yaml')
with config_path.open('x') as f:
    yaml.safe_dump(config, f, sort_keys=False)
subprocess.run([sys.executable, 'scripts/run_benchmark.py', 'inspect',
                '--config', str(config_path)], check=True)
```

Utiliser les folds annuels du dépôt, jamais un split pixel ou fenêtre aléatoire.
Les cinq événements du sample ne servent qu'au débogage. Au fold 0 leur test
est absent : un entraînement diagnostique reste possible, l'évaluation finale
échouera explicitement.

## 4. Entraînement progressif et borné

Après les tests, lancer **un** modèle pour valider la chaîne. Chaque run B1–B3
fait son propre mini-overfit train puis repart de poids neufs. Le YAML Kaggle
fourni est un diagnostic d'un epoch avec 32 patches (4 mises à jour batch 8).
Les autres modèles se lancent explicitement après contrôle du premier run.

```python
subprocess.run([sys.executable, 'scripts/run_benchmark.py', 'train',
                '--config', str(config_path), '--model', 'B1',
                '--context', '3', '--seed', '42', '--device', 'cuda'], check=True)
# Puis, individuellement selon le budget validé : --model B0, B2, B3.
```

Les batchs se règlent dans le YAML. Si OOM, conserver le run en échec, créer un
nouveau YAML avec batch réduit, puis appliquer le même batch/budget aux modèles
comparés. Ne pas publier une comparaison où le nombre de mises à jour change
sans le signaler. AMP est CUDA uniquement ; l'inférence de validation reste
FP32. Cette version utilise un seul GPU (device 0), pas DDP/multi-GPU.

Le tri AP exact et l'I/O peuvent dominer la durée, même pour un epoch court.
Surveiller l'espace de `/kaggle/working`, conserver une marge pour les tris
sur disque. Ne pas lancer la matrice annuelle complète automatiquement.

## 5. Exporter avant fermeture

Chaque epoch complet a un checkpoint et une synchronisation MLflow. Une
coupure brutale avant cette publication oblige à reprendre l'epoch. Ne pas
attendre la fin de session pour faire les sauvegardes.

```python
from datetime import datetime, timezone
run_root = Path('/kaggle/working/geofirex-runs')
export_root = Path('/kaggle/working/exports')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
for run in sorted(run_root.iterdir()):
    if run.is_dir() and (run/'summary.json').exists():
        subprocess.run([sys.executable, 'scripts/verify_and_export_benchmark.py',
                        '--runs', str(run), '--export-dir', str(export_root/stamp),
                        '--report', str(export_root/f'{run.name}-{stamp}.verification.json')], check=True)
```

Le script versionné vérifie les métadonnées, les sources archivées, le statut
MLflow et les SHA256 de tous les fichiers exportés, sans valeur locale fixe
(pas de noms de runs, de 42 fenêtres ou de quatre mises à jour imposés).
Pour un contrôle groupé, fournir les dossiers après `--runs` et éventuellement
`--require-comparable`. Pour revérifier sans réécrire, utiliser `--archive-dir`
à la place de `--export-dir`. `--skip-mlflow` est réservé au contrôle hors ligne
et indique explicitement que le statut MLflow n'a pas été vérifié.

Exporter également un run interrompu ayant des checkpoints complets via
`scripts/run_benchmark.py export --run <run> --out <nouvelle-archive.tar.gz>` :
le vérificateur complet exige un résumé de run terminé. Les `.tmp`
et `.bin` de scratch AP sont exclus de l'archive ; les états optimizer/scaler/RNG
et JSON sont conservés. Vérifier les SHA256 générés et lister le contenu des
archives. Après fermeture des processus MLflow, conserver aussi `tracking.db`,
`mlflow_artifacts`, `gpu_preflight.json`, le YAML et tout verrou de sélection.

Sauvegarder la version Kaggle avec les outputs activés et récupérer les fichiers
ou les copier vers un stockage persistant autorisé. Un fichier présent dans
`/kaggle/working` seul ne prouve pas une sauvegarde durable. Cette mission n'a
pas réalisé ces actions cloud et ne prétend pas les avoir validées en session.

## 6. Reprise et benchmark annuel

Monter les données et sorties sur les mêmes chemins stables, puis :

```bash
python scripts/run_benchmark.py train --config /kaggle/working/runner_session.yaml --model B1 --resume /kaggle/working/geofirex-runs/<parent>/epoch_0001.pt
```

Les sources exactes sont disponibles dans l'archive sous `source/`. La reprise
refuse une identité différente. Pour un vrai benchmark, préparer le manifeste
complet et un YAML scientifique sur le modèle de `runner_annual.yaml`, puis
suivre [RUNNER_ANNUAL.md](RUNNER_ANNUAL.md). La sélection et le test sont des
commandes distinctes ; ne pas utiliser le test pour ajuster les paramètres.

## Runpod / autre serveur GPU

Utiliser une image déjà équipée de PyTorch CUDA et lancer le même préflight.
Monter **données, sorties, base MLflow et artifact store** sur des volumes
persistants. Le YAML annuel donne un exemple `/workspace` à adapter au vrai
volume. Vérifier le montage, espace, droits et rétention avant le pilote. Pour
un object store distant, suivre [RUNNER_MLFLOW.md](RUNNER_MLFLOW.md). Aucun
provisionnement, achat de GPU ou transfert massif n'est effectué par ces scripts.
