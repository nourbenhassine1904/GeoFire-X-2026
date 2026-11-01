# Acquisition CPU et campagnes GPU Kaggle

> Procédure actualisée après le blocage HTTP403 : suivre
> [KAGGLE_UNBLOCK_QUICKSTART.md](KAGGLE_UNBLOCK_QUICKSTART.md) et les notebooks v3.
> Ils remplacent les anciennes instructions de première publication ci-dessous.

Le code ne considère jamais `/kaggle/working` comme une sauvegarde. Aucun
transfert réel, upload Kaggle ou entraînement GPU prolongé n'a été exécuté dans
la préparation locale. Les tests du backend cloud utilisent un service simulé.

## 1. Installer le code dans un notebook CPU privé

Activer Internet, choisir CPU, et placer **le code courant** du projet dans un
dataset de code privé attaché au notebook (ou une archive de code sans données
ni secrets). Ne pas utiliser un checkout distant ancien : aucun commit/push
n'est réalisé par cette mission. Copier seulement le code vers le dossier
de travail, puis se placer à sa racine. Il faut `pyproject.toml`, `src/`,
`scripts/`, `configs/`, `tests/` et l'index officiel
`data/manifests/real/corpus_zip_index_v1.json`.

L'archive locale `artifacts/kaggle/geofirex-cloud-code-v2.zip` contient ces
fichiers de code et l'index, sans les données brutes ni les expériences locales.
La joindre à un dataset de code privé dans Kaggle, puis copier le dossier
`GeoFire-X-2026` décompressé depuis `/kaggle/input` vers `/kaggle/working`.
Elle contient un inventaire des hashes du code. Pour la reconstruire après
modification :

```powershell
.\.venv\Scripts\python.exe -m scripts.package_kaggle_code --out artifacts/kaggle/geofirex-cloud-code-v3.zip
```

Dans une cellule notebook, après `%cd /kaggle/working/GeoFire-X-2026` :

```python
%pip install -e '.[kaggle]'
import kagglehub
# L'authentification implicite Kaggle doit permettre la CREATION de datasets.
# Sinon utiliser Kaggle Secrets / kagglehub.login(), jamais un token dans le code.
```

L'installation complète peut résoudre PyTorch : vérifier le build CUDA dans le
notebook GPU avec `scripts/gpu_preflight.py` avant une campagne. Les dépendances
cloud sont optionnelles, les tests locaux n'exigent aucun compte Kaggle.

Sources officielles consultées :
[upload/download Kagglehub](https://github.com/Kaggle/kagglehub/blob/main/README.md),
[création privée et versionnement](https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/datasets_helpers.py),
[spécifications Kaggle Datasets](https://www.kaggle.com/docs/datasets).
Les limites réelles du compte, de la session et le stockage déjà occupé doivent
être vérifiés dans Kaggle ; aucun chiffre de quota disponible n'est supposé.

## 2. Première publication contrôlée : un événement complet

Commande **sans transfert**, exécutable localement ou dans Kaggle :

```bash
python -m scripts.kaggle_acquire_publish --job /kaggle/working/wsts-batch001 --handle VOTRE_IDENTIFIANT/geofirex-wsts-batch001 --transfer-bytes 67108864 --storage-bytes 2147483648 --upload-bytes 67108864 --private-free-bytes 67108864
```

Ce scénario choisit `2018/fire_21458798` : **14 observations**, 44 352 359 octets
ZIP compressés, **45 061 097 octets GeoTIFF**. Les 64 Mio de transfert Zenodo
incluent métadonnées ZIP et réservations de retries. Les 2 Gio de stockage
alloué incluent une réserve de 1 Gio. Les 64 Mio d'upload et d'espace privé
sont des plafonds de **payload**, à remplacer si le compte a moins de place.
L'API Kaggle gère ses retries : le trafic réseau réel d'upload n'est pas mesuré
ni plafonné octet par octet par ce code. Le readback Kaggle s'effectue dans son
cache partagé monté ; ne pas exécuter le job cloud sur Windows.

Après validation des quotas et approbation du transfert, la commande suivante
acquiert, valide, publie en privé et relit la version créée :

```bash
python -m scripts.kaggle_acquire_publish --job /kaggle/working/wsts-batch001 --handle VOTRE_IDENTIFIANT/geofirex-wsts-batch001 --transfer-bytes 67108864 --storage-bytes 2147483648 --upload-bytes 67108864 --private-free-bytes 67108864 --execute
```

Dans une cellule notebook, préfixer la commande par `!`. Les fichiers sont
stockés sous `payload/2018/fire_.../YYYY-MM-DD.tif`. La validation compare
taille+CRC à l'index, puis calcule SHA256. `inventory.json` décrit chaque fichier
et la liste exacte des événements. Le MD5 de l'archive entière est une référence
de provenance : il n'est **pas recalculé** par les lectures Range.

Une erreur conserve les TIFF, les `.part` et un rapport d'acquisition. Relancer
la même commande avec le même `--job` reprend au fichier manquant. Un TIFF
existant corrompu est conservé et bloque la suite. Aucun effacement automatique.
Un job conserve son plan et ses quotas ; utiliser un autre job pour les changer.
Une nouvelle publication du même contenu ne crée pas une version inutile.
Un contenu différent sur un handle existant est refusé par le job d'acquisition.

L'upload seul ne suffit pas : le code refuse un dataset existant public, épingle
`owner/slug/versions/N`, relit **tous** les fichiers et leurs SHA256/CRC. État
attendu du reçu : `REMOTE_VERIFIED_FRESH_SESSION_PENDING`. Une panne d'auth,
de permission, de traitement ou de relecture laisse `FAILED_FILES_PRESERVED`.
Après échec, ne pas inférer l'absence de dataset : relancer permet de vérifier
une publication ayant réussi côté serveur avant l'interruption.

## 3. Vérifier depuis un NOUVEAU notebook

Le job affiche son reçu JSON et l'enregistre dans
`/kaggle/working/wsts-batch001/publication-*.json`. **Conserver ce petit reçu
hors de la VM** (téléchargement du JSON, copie du texte affiché, ou outputs d'une
version de notebook effectivement sauvegardée). Le reçu n'est pas un secret.
Les données elles-mêmes sont dans le dataset privé épinglé.

Dans un nouveau notebook CPU, réinstaller le même code, placer le reçu dans
`/kaggle/working/receipt.json` puis lancer :

```bash
python -m scripts.kaggle_cloud verify-session --receipt /kaggle/working/receipt.json --out /kaggle/working/bootstrap-verified.json
```

Le dataset est attaché via `kagglehub.dataset_download(handle_versionne)` et
contrôlé sans copie intégrale. Le marqueur de session dans `/kaggle/working`
doit différer. L'état final est `PERSISTENCE_VERIFIED_NEW_SESSION`. Copier ce
certificat hors de la VM également. Ne pas copier `.geofirex-session-id` entre
notebooks. Un nouveau processus dans la même session ne satisfait pas ce contrôle.

Une interruption détruisant la VM avant publication perd les fichiers non
publiés : ils seront réacquis. La reprise entre VM repose sur les datasets
validés, pas sur une promesse de conservation des répertoires temporaires.

## 4. Lots de plusieurs Gio et couverture complète

Les plans hors réseau générés dans `reports/cloud_campaign_v1/` couvrent
**607 événements / 13 607 observations**, en 15 lots annuels, selon un scénario
de 4 Gio de transfert Zenodo par lot, 6 Gio de stockage temporaire, 5 Gio de
payload d'upload et 64 Gio de disponibilité privée **hypothétique**. Rien n'a
été acquis par cette planification. `expected_observations.csv` est la table
attendue ; elle ne prétend pas être un inventaire de données présentes.

| Rôle | Années | Événements attendus | Observations attendues |
|---|---|---:|---:|
| train | 2018–2019 | 250 | 5 198 |
| validation | 2020 | 201 | 4 292 |
| test verrouillé | 2021 | 156 | 4 117 |

Pour régénérer un scénario avec les quotas réellement validés :

```bash
python -m scripts.plan_cloud_campaign --index data/manifests/real/corpus_zip_index_v1.json --out /kaggle/working/cloud-plans --transfer-bytes 4294967296 --storage-bytes 6442450944 --upload-bytes 5368709120 --private-free-bytes 68719476736
```

Premier ensemble intermédiaire : au moins 30 événements train et 20 validation
avec fenêtres alignées. Le premier lot 2018 planifié contient 48 événements,
le premier 2020 en contient 55. Ils ne représentent pas le corpus complet.
Pour une extension après approbation de ces nouveaux budgets :

```bash
python -m scripts.kaggle_acquire_publish --job /kaggle/working/wsts-train02 --handle VOTRE_IDENTIFIANT/geofirex-wsts-train02 --years 2018 2019 --max-events 40 --transfer-bytes 4294967296 --storage-bytes 6442450944 --upload-bytes 5368709120 --private-free-bytes 10737418240 --acquired-roots /kaggle/input/CHEMIN_EXACT_DU_PREMIER_LOT --bootstrap-verification /kaggle/working/bootstrap-verified.json --execute
```

Utiliser un handle/job distinct par lot. Même commande avec `--years 2020`,
un autre handle/job et sans les racines 2018 pour un lot validation. Les chemins
de montage peuvent varier : employer le chemin retourné par Kagglehub, contenant
directement `inventory.json` et les répertoires annuels. `--acquired-roots`
vérifie CRC/hashes puis exclut les événements acquis. Un certificat de première
publication vérifiée est obligatoire pour exécuter un job de plusieurs événements.

Les plans annuels fixes sont également exécutables avec `kaggle_cloud acquire`
(`--plan`, `--index`, `--destination`, `--out`, `--execute`), puis
`kaggle_cloud publish` (`--root`, `--handle`, les deux quotas d'upload,
`--out`, `--execute`). Ces commandes unitaires servent à réparer/reprendre une
étape ; le job combiné est l'entrée recommandée.

## 5. Préparer les montages puis les configurations GPU

Depuis le notebook GPU attachant les lots privés vérifiés :

```bash
python -m scripts.kaggle_cloud prepare --index data/manifests/real/corpus_zip_index_v1.json --roots /kaggle/input/LOT_TRAIN /kaggle/input/LOT_VALIDATION --out /kaggle/working/prepared
python -m scripts.kaggle_campaign --prepared /kaggle/working/prepared --index data/manifests/real/corpus_zip_index_v1.json --out /kaggle/working/campaign --checkpoint-prefix VOTRE_IDENTIFIANT/geofirex-pilot01 --upload-budget-bytes 10737418240 --private-free-bytes 10737418240
```

Le générateur crée B0 seed42, B1/B2/B3 seeds42/43/44, T=3, même manifeste,
normalisation train uniquement, sampling et domaine de validation exhaustif.
Il refuse moins de 30 événements train ou 20 validation utilisables. Ajouter
`--scientific` uniquement après montage du corpus complet ; sinon `pilot`.
Les 10 Gio ci-dessus constituent encore un **scénario à faire approuver** pour
les sauvegardes des dix runs. Chaque run reçoit 1/10 de ces budgets, qui diminue
à chaque version complète, sans supposer une déduplication côté serveur.

Après approbation de la campagne longue et du budget de sauvegarde :

```bash
python scripts/run_benchmark.py inspect --config /kaggle/working/campaign/B1_s42.yaml
python scripts/run_benchmark.py train --config /kaggle/working/campaign/B0_s42.yaml
python scripts/run_benchmark.py train --config /kaggle/working/campaign/B1_s42.yaml
```

Répéter pour les autres YAML, séquentiellement et dans la limite de la session.
30 epochs × 4096 patches / batch8 = **15 360 updates planifiées** par modèle
entraînable. L'arrêt anticipé peut réduire ce nombre ; le résumé compte les
updates effectivement exécutées, y compris les pas sautés par AMP. B0 n'entraîne
aucun paramètre. Le rapport indique aussi les événements/observations du
manifeste, ceux des fenêtres éligibles et la couverture réellement échantillonnée
par epoch. Un corpus partiel n'est jamais déclaré complet.

Les mécanismes existants conservent BCE pondérée+Dice, AMP CUDA, gradients
clippés, courbes, AP/Dice/IoU/Precision/Recall et nouvelles ignitions, sélection
du meilleur checkpoint sur validation, suivi MLflow. Le sampling mélange 50 %
de tirages de tuiles positives train et 50 % de tirages uniformes. Les labels
validation/test ne participent jamais au sampling ou à la normalisation.

La lecture est fenêtrée depuis les montages, avec un LRU borné par worker.
La normalisation parcourt seulement les entrées train uniques. L'AP exacte
utilise des tris temporaires sur disque : son coût disque/temps sur 13 607 TIFF
n'a pas été validé. Mesurer le premier lot GPU et la taille des sorties avant
de lancer une année entière ; ce code ne prouve pas qu'une epoch complète
tient dans une session gratuite. Aucun cache dense de tout le corpus n'est créé.

## 6. Sauvegardes d'epochs et reprise dans une autre session

Les configurations générées publient après chaque epoch, puis après fermeture
MLflow. Chaque snapshot contient dernier+meilleur checkpoint, états optimiseur,
scaler et RNG, normalisation, code, audits, métriques, copie SQLite cohérente,
export portable du run MLflow et historiques de métriques. Le scratch `.bin`
de l'AP n'est pas une entrée de reprise et n'est pas publié. Aucun fichier
source n'est supprimé. Les budgets épuisés arrêtent la publication explicitement
en conservant les checkpoints locaux. Chaque reçu doit encore être vérifié
depuis une nouvelle session comme au §3 avant d'affirmer cette vérification.

Dans un nouveau notebook, monter la version du dataset de checkpoints indiquée
dans le reçu, vérifier ses hashes, reconstruire le même manifeste à partir des
mêmes lots de données, puis réutiliser **le même budget d'epochs** :

```bash
python scripts/run_benchmark.py train --config /kaggle/working/campaign/B1_s42.yaml --data-roots /kaggle/input/NOUVEAU_CHEMIN_TRAIN /kaggle/input/NOUVEAU_CHEMIN_VAL --resume /kaggle/input/CHECKPOINT_SNAPSHOT/run/epoch_0005.pt
```

Les chemins de montage et de sortie sont des paramètres d'exécution. Les hashes
des données, du code et les paramètres scientifiques doivent rester identiques.
Le snapshot est compatible avec un autre chemin de montage ; la reprise crée
un nouveau run MLflow. `tracking.db` sauvegarde l'historique d'audit, ses URI
d'artefacts peuvent viser l'ancienne VM. Pour une consultation portable utiliser
`mlflow_run.json`, `mlflow_metric_history.json` et les fichiers `run/` ; ne pas
présenter cette copie SQLite comme un serveur MLflow restauré automatiquement.

Les configurations régénérées doivent conserver leurs paramètres scientifiques
et leur manifeste, sinon la reprise est refusée. Pour un snapshot manuel :

```bash
python -m scripts.kaggle_cloud snapshot-training --run /kaggle/working/geofirex-runs/RUN --tracking-db /kaggle/working/geofirex-runs/tracking.db --out /kaggle/working/manual-backup
python -m scripts.kaggle_cloud publish --root /kaggle/working/manual-backup --handle VOTRE_IDENTIFIANT/geofirex-checkpoints --upload-bytes 1073741824 --private-free-bytes 1073741824 --allow-version --out /kaggle/working/manual-receipt.json --execute
```

Le test 2021 reste réservé au `lock` puis `final-test` existants du runner.
Acquérir/vérifier les fichiers 2021 ne déclenche aucune évaluation de modèle.

## 7. Audit des 23 bandes et ablations explicites

```bash
python -m scripts.audit_multimodal --files /kaggle/input/LOT/2018/fire_ID/DATE.tif --out /kaggle/working/audit23.json
```

Le script vérifie l'ordre réel, accepte l'alias connu `surface_last` à la bande
17, et compte NaN/Inf/valeurs finies et bornes par bande, par tuiles de 256.
`reports/multimodal_real_audit_v1.json` est l'audit local effectué sur les
observations existantes, sans nouvelle acquisition.

| Profil explicite | Variables utilisées | Canaux encodés |
|---|---|---:|
| conservative_fire_terrain | feu, pente/altitude normalisées, aspect sin/cos, indicateurs manquants | 8 |
| terrain_vegetation_v1 | précédent + réflectances M11/I2/I1, NDVI/EVI2, landcover catégoriel | 36 |
| multimodal_observed_j_v1 | précédent + pluie, vitesse vent, Tmin/Tmax, ERC, humidité, PDSI observés | 50 |

La direction du vent brute (index7) est exclue : l'auteur signale un ancien bug
de quadrant dans le [code de création](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset).
Les bandes GFS17–21 restent exclues sans preuve issue-time/valid-time. Les
statistiques train, les masques de valeurs manquantes et les catégories LC sont
calculés/encodés explicitement ; aucun canal de la baseline n'est changé.

Pour une ablation, dupliquer les YAML de campagne, changer `feature_policy` et
pointer `feature_availability` vers une déclaration **revue** (modèle dans
`configs/multimodal_availability.example.json`). Le fichier doit établir la
disponibilité à la fin de J de chaque bande : latence GRIDMET, composites
NDVI/EVI2 et produit LC annuel compris. Le template `reviewed:false` bloque
l'exécution. Les TIFF seuls ne prouvent pas une disponibilité opérationnelle.
Comparer les profils avec mêmes splits, seeds, budgets et cibles ; leurs canaux
et preuves sont hashés et enregistrés. Ne pas mélanger les politiques dans le
classement de baselines qui exige des features identiques.

## Validation locale

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Les tests ajoutés couvrent événements complets, budgets, ZIP Range simulé,
reprise idempotente, refus public, versionnement, erreurs d'upload, corruption,
vérification dans une autre session simulée, montage de GeoTIFF synthétiques,
reprise avec chemins modifiés, sampling, arrêt anticipé, multimodalité et export
MLflow/SQLite. Seule une exécution authentifiée dans Kaggle validera les quotas,
permissions, délais de traitement, montages réels et performances CPU/GPU.
