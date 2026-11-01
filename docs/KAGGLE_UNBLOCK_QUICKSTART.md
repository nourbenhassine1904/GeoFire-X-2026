# Déblocage Kaggle — exécution v3

Entrée recommandée : `notebooks/kaggle_cpu_unblock_v3.ipynb`, puis
`notebooks/kaggle_gpu_pilot_v3.ipynb`. Importer ces notebooks dans Kaggle avec
le paquet de code courant. **Aucun résultat ni certificat réel n'est prérempli.**
L'archive à attacher est `artifacts/kaggle/geofirex-cloud-code-v3.zip` ; adapter
uniquement `CODE_SOURCE` dans la première cellule au chemin du dataset de code.

## CPU, cellule par cellule

1. Installer le code v3 et vérifier le compte `nnnnbhss` (cellule 1).
2. Garder `PHASE='adopt'`, puis exécuter l'adoption de batch001. Lecture seule :
   confidentialité, version épinglée, 14 TIFF, CRC/SHA256 et index officiel.
   Le reçu porte `kind=manual_adoption`, `upload_called=false` et reste en attente
   d'une vérification indépendante. Conserver ce petit JSON hors de la VM.
3. Ouvrir un **nouveau notebook CPU**, passer `PHASE='verify_and_plan'` et coller
   ce JSON. Le nouveau contrôle produit le certificat bootstrap de provenance
   `manual`. Un processus supplémentaire dans la même VM ne suffit pas.
4. Exécuter d'abord la cellule `RUN_PUBLICATION_PROBE=True` : 60 fichiers texte
   minuscules, création privée explicite, ZIP automatique et contrôle complet.
   Cela teste les permissions réelles avant de transférer plusieurs Gio, sans
   toucher au lot manuel. Cette sonde ne peut pas remplacer le bootstrap corpus.
5. Renseigner `PRIVATE_FREE_GIB` avec le stockage privé réellement disponible
   sur le compte. Le disque de travail est mesuré. La planification réserve aussi
   le ZIP temporaire que l'uploader officiel peut créer au-delà de 50 fichiers.
6. Lire les trois plans et les commandes intégrales affichées, avec leurs quotas.
   Après approbation, activer une cellule d'exécution à la fois : train2018,
   val2020, puis extension2019. La publication privée et la relecture sont incluses.
7. Conserver les reçus affichés hors de `/kaggle/working` pour le notebook GPU.

Commande d'adoption équivalente, à la racine du code v3 dans Kaggle CPU :

```bash
python -m scripts.kaggle_cloud adopt-manual --index data/manifests/real/corpus_zip_index_v1.json --root /kaggle/input/datasets/nnnnbhss/geofire-x-wsts-batch001 --handle nnnnbhss/geofire-x-wsts-batch001 --out /kaggle/working/manual-adoption.json
```

Dans le nouveau notebook, avec ce JSON conservé :

```bash
python -m scripts.kaggle_cloud verify-session --receipt /kaggle/working/manual-adoption.json --out /kaggle/working/bootstrap-manual-verified.json
```

Les deux commandes d'acquisition effectives dans le notebook, **après lecture des
budgets affichés et autorisation du transfert**, sont :

```python
subprocess.run(acquisition_command('train2018') + ['--execute'], check=True)
subprocess.run(acquisition_command('val2020') + ['--execute'], check=True)
```

La fonction de la cellule précédente développe tous les arguments : handles
`nnnnbhss/geofire-x-wsts-train2018-pilot01` et
`nnnnbhss/geofire-x-wsts-val2020-pilot01`, index, plan immuable, exclusion du lot
manuel, certificat bootstrap, quotas et répertoires de reprise. Ces commandes
ne reposent pas sur une estimation fictive du quota privé.

Scénario **hors réseau seulement** généré dans `reports/kaggle_pilot_v2/` avec
12 Gio privés déclarés, 8 Gio de disque alloué et 3 Gio de transfert par lot :

| Plan | Événements complets éligibles par dates | Observations | GeoTIFF |
|---|---:|---:|---:|
| train 2018, batch001 exclu | 37 | 714 | 2 844 119 383 octets |
| validation 2020 | 30 | 546 | 2 134 551 752 octets |
| extension train 2019 | 30 | 599 | 2 331 203 739 octets |

Ce scénario n'atteste pas 12 Gio disponibles sur votre compte. Le notebook le
recalcule avec votre valeur. Les plans historiques du corpus complet sont
conservés ; leurs quotas antérieurs n'incluaient pas tous l'espace du ZIP de
publication, donc employer les nouveaux plans pour exécuter ces lots.

## Modes de publication et reprise

`create` appelle explicitement `CreateDataset(is_private=True)` sans `GetDataset`
préalable et sans fallback vers une nouvelle version. Un conflit ne peut être
accepté que si la relecture prouve un contenu déjà identique. `resume` vérifie
une publication existante sans mutation et fonctionne depuis un montage partagé,
même sans quota pour un nouvel upload. `version` doit être explicite : corpus
inchangé conservé, ou même identité d'entraînement pour les checkpoints ; aucune
ancienne version n'est supprimée. Le propriétaire doit être le compte authentifié.

HTTP403 signifie autorisation refusée ou existence non divulguée, **jamais 404**.
`inspect-dataset --handle ...` expose cette distinction. Une erreur conserve les
fichiers et un reçu d'échec. Après une réponse perdue côté serveur, utiliser
`PUBLICATION_MODE='resume'`. Une récupération ne prétend pas avoir observé
l'upload d'origine ; sa provenance est `recovered_existing`, jamais un upload
automatique inventé. Pour le lot manuel utiliser sa procédure d'adoption dédiée.

## GPU T4

Dans un nouveau notebook GPU, coller les vrais reçus CPU. Les cellules vérifient
la nouvelle session, les versions privées, tous les hashes et construisent un
manifeste cumulatif sans doublons. Joindre batch001 à Add Input pour l'inclure.
Le générateur refuse moins de 30 événements train ou 20 validation utilisables.
Le test 2021 reste verrouillé.

Renseigner les deux budgets réels de sauvegarde des checkpoints. Mesurer ensuite
B1/B2/B3 avec `RUN_THROUGHPUT=True` : 100 pas mesurés, 10 d'échauffement, aucun
score de validation. B1 est explicitement `spatial_cnn_v2` (trois convolutions) ;
l'ancien B1 à une couche reste chargeable avec sa configuration historique.

Après approbation, lancer B0 une fois et B1/B2/B3 avec `--stop-after-epochs 2`.
Cela représente 1 024 updates planifiées par modèle entraînable. Le budget total
reste 30 ; reprendre avec le même YAML et un checkpoint publié vérifié, sans
changer `epochs`. L'état partiel est `PAUSED_AT_EPOCH_BOUNDARY` et n'est pas classé
comme un run terminé. Étendre ensuite les sessions et les seeds42/43/44 selon
les temps mesurés. Loss, AP, Dice/IoU/Precision/Recall, seuil, temps, VRAM,
patches/updates, comptes de données et hashes sont enregistrés dans MLflow.

## Suite Big Data et application

La dernière cellule GPU affiche l'export Parquet des métadonnées réellement
montées et la commande Spark `local[2]` avec minimum 30 événements/500 observations.
Installer PySpark et une JVM compatibles dans le notebook CPU avant cette
exécution. Elle n'a pas été réalisée ici ; aucun cluster n'est prétendu validé.

Backend préparé : `geofirex.serving.create_app`. Il vérifie le snapshot par son
identité d'inventaire issue du reçu, charge le **vrai meilleur checkpoint B1/B2/B3**,
refuse B0 comme modèle applicatif et contrôle les observations utilisées.
Les endpoints sont `/health`, `/events`, `/predict?event_id=...&as_of=YYYY-MM-DD`.
L'inférence utilise seulement les dates ≤J et prédit J+1, sans lire son label.
2021 est exclu par défaut. Sur un environnement local de démonstration :

```powershell
# Définir GEOFIREX_SNAPSHOT, GEOFIREX_INVENTORY_IDENTITY et GEOFIREX_DATA_ROOTS (liste JSON).
.\.venv\Scripts\python.exe -m uvicorn geofirex.serving:app_from_env --factory --host 127.0.0.1 --port 8000
```

`frontend/src/EventForecast.mjs` définit le composant React : sélection événement
et date, feu observé J, probabilités J+1 et masque prédit. Il appelle ces endpoints
et n'invente aucune prédiction. Il reste à l'intégrer dans une application React
avec un proxy vers l'API ; ce composant n'est pas une application web déployée.

Sources officielles : [requêtes de création/versionnement Kaggle](https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/datasets_helpers.py),
[uploader et staging ZIP](https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/gcs_upload.py).
Les tests locaux simulent ces API. La création automatique et les entraînements
sur votre compte Kaggle restent à exécuter, sans annonce anticipée de résultats.
