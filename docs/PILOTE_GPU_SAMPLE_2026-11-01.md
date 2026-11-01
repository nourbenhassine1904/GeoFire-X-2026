# Pilote sample T4 — prêt pour validation, non lancé

> État historique de préparation. Le pilote a depuis été autorisé et exécuté : voir
> `reports/GPU_PILOT_2026-11-01.md` pour les résultats réels et le nouveau lancement de démonstration.

## Données vérifiées

Dataset privé `nnnnbhss/geofire-x-wildfirespreadts-sample`, v1 selon l'inventaire de reprise.
L'inventaire authentifié du 1er novembre contient exactement les 108 chemins et tailles attendus.
Contrôle CPU indépendant terminé : `nnnnbhss/geofirex-sample-integrity-nov01`, version 1,
GPU et Internet désactivés, 74,96 secondes de lecture effective. Les 108 TIFF sont décodés,
23 bandes float32, sans Inf ; SHA-256 identiques aux fichiers locaux, tailles et CRC32
conformes à l'index officiel. Aucun TIFF rapatrié sur le PC.
Volume : **447 927 526 octets**. Preuves : `reports/sample_*20261101.json`.

| Partition | Événement | Observations | Dates incluses |
|---|---|---:|---|
| train | 2018/fire_21748813 | 14 | 2018-05-03 → 2018-05-16 |
| train | 2018/fire_21997864 | 18 | 2018-08-02 → 2018-08-19 |
| train | 2019/fire_23300839 | 24 | 2019-09-03 → 2019-09-26 |
| validation | 2020/fire_23654679 | 28 | 2020-01-01 → 2020-01-28 |
| validation | 2020/fire_24104674 | 24 | 2020-06-11 → 2020-07-04 |

Dates quotidiennes sans trou ni doublon selon l'audit existant et la construction des fenêtres.
Le nouvel audit local contrôle SHA, géométries et absence de duplication entre partitions.
Les cinq événements sont complets selon l'index officiel : `qualification: pilot` sera accepté.
Train : 56 observations / **41 cibles** ; validation : 52 observations / **42 cibles**.
Aucun événement test. Ne pas utiliser `splits/real/smoke_test_split.csv` : utiliser le fold annuel 00.
La certification Batch001, Spark et les acquisitions annulées n'ont pas été relancés.

## Protocole prêt

`configs/runner_sample_gpu_pilot.yaml` et `notebooks/kaggle_sample_gpu_pilot_20261101.ipynb`.
Le notebook génère quatre YAML comparables dans `/kaggle/working/sample-prepared`.

- B0 : Persistence exacte sur le feu observé à J ; aucun apprentissage.
- B1 : CNN spatial v2, largeur 16, trois convolutions.
- B2 : U-Net, base 8, profondeur 3, décodeur `nearest_v1`.
- B3 : encodeur + ConvLSTM spatial, hidden 8, décodeur `nearest_v1`.

Correction indispensable : le backward bilinéaire CUDA historique est incompatible avec
`torch.use_deterministic_algorithms(True)`. Les variantes nearest sont explicitement versionnées ;
les modèles et checkpoints historiques gardent leur comportement. CPU vérifié ; les deux tests
CUDA sont ignorés localement faute de GPU. Premier préflight T4 obligatoire dans le runner.

Pour tous : seed 42, contexte 3 jours, cible J+1, mêmes 41/42 cibles et domaine valide T=5,
patches 64², batch 16, AdamW 0,001, AMP désactivé, 10 epochs × 2 048 patches.
**128 updates/epoch, 1 280 par modèle, 3 840 au total**, plus 30 updates de mini-overfit
jetées pour chacun des trois modèles (90 supplémentaires). 61 440 patches d'entraînement au total.
Le DataLoader utilise 2 workers, pinned memory, cache physique borné à 256 Mio/processus,
échantillonnage déterministe avec 50 % de tirages dans le pool de tuiles positives train.
BCE pondérée (ratio train, plafonné à 100) + Dice 0,5. Normalisation ajustée uniquement
sur les chemins train uniques du contexte maximal, partagée par toutes les architectures.
Profil conservateur feu/terrain : ne pas présenter ce pilote comme un benchmark multimodal complet.

Validation exhaustive sans suréchantillonnage : micro AP, Dice, IoU, précision, rappel,
FP/FN et AP des nouvelles ignitions. Meilleur checkpoint par AP validation ; seuil choisi
sur validation uniquement. Les hashes des cibles/comparaison doivent être égaux entre les quatre runs.
Une seule seed, cinq événements : **pilote**, sans classement scientifique ni généralisation test.

## Exécution exacte après validation

Préparation locale réalisée sans publication :

```powershell
.venv/Scripts/python.exe scripts/package_kaggle_code.py --out artifacts/kaggle/geofirex-sample-pilot-20261101-v2.zip
```

Dans Kaggle, attacher le nouveau paquet de code décompressé (contenant `code_manifest.json`),
le sample privé et la sortie du notebook CPU `nnnnbhss/geofirex-sample-integrity-nov01`.
Utiliser le **nouveau** notebook sample ; les anciens notebooks v3 et l'archive v4.2 ne sont pas ce pilote.
Vérifier les dépendances avant allocation GPU : `torch`, `rasterio`, `numpy`, `mlflow`,
`pyyaml`, `scikit-learn`, `matplotlib`. Conserver le PyTorch CUDA de Kaggle.
Le notebook échoue explicitement si une dépendance manque ; il n'installe rien automatiquement.
Choisir Tesla T4, limiter la session à **3 600 s**, puis passer `RUN_APPROVED=False` à `True`
uniquement après validation. La cellule appelle les commandes suivantes, avec timeout contrôlé :

```bash
python scripts/run_benchmark.py train --config /kaggle/working/sample-prepared/B0.yaml
python scripts/run_benchmark.py train --config /kaggle/working/sample-prepared/B1.yaml
python scripts/run_benchmark.py train --config /kaggle/working/sample-prepared/B2.yaml
python scripts/run_benchmark.py train --config /kaggle/working/sample-prepared/B3.yaml
```

Exécuter ces commandes via la boucle bornée du notebook. Elles seules ne portent pas de timeout.
Budget proposé : **300 s B0 + 900 s B1 + 900 s B2 + 900 s B3 + 600 s préparation/sauvegarde = 3 600 s**.
Ces plafonds comprennent lecture/QC, normalisation, préflight, validation et MLflow ; ce ne sont
pas des prédictions de durée. Les 10 epochs ne sont pas garanties sous ce plafond : arrêt et reprise
si nécessaire, sans classement de runs incomplets/inégaux. Aucun nouveau budget n'est implicite.
GPU utilisé : device 0 seulement. Réserver au maximum 1 GPU-h si une T4 est allouée,
**2 GPU-h physiques si Kaggle alloue deux T4 pendant l'heure** ; le débit du quota Kaggle
doit être confirmé dans le compte, ce calcul n'en présume pas les règles. Prévoir 8 Gio libres.
Acquisition Zenodo : 0 octet ; réutilisation du sample monté, aucun transfert des 943 TIFF.

Le notebook journalise toutes les GPU allouées avec `nvidia-smi` chaque seconde,
puissance, utilisation et mémoire ; intégration trapézoïdale dans `energy.json` (kWh GPU mesurés,
hors CPU/RAM/PUE ; indisponible si capteurs absents). Le runner enregistre temps CUDA,
temps muraux, débit, pic mémoire, temps I/O/validation et updates effectivement réalisées.

## Conservation, reprise et démonstration

MLflow SQLite + artefacts, versions des packages, sources/hashes, configuration, normalisation,
prior train, échantillonnage par epoch, métriques et checkpoints sont sous
`/kaggle/working/sample-pilot`. Le notebook conserve une archive finale sans scratch `.bin/.tmp`.
Exporter **aussi le dossier complet et les YAML préparés** avant de fermer la session.
Une extinction forcée de la VM ne garantit pas la persistance des fichiers non exportés.
Pas de publication automatique de checkpoints privés : aucun budget d'upload n'a été autorisé.

Reprise, depuis le même paquet de code immuable et les sorties conservées :

```bash
python scripts/run_benchmark.py train --config /chemin/conserve/prepared-configs/B2.yaml --resume /chemin/conserve/B2-run/epoch_0003.pt --data-roots /kaggle/input/datasets/nnnnbhss/geofire-x-wildfirespreadts-sample --manifest /chemin/conserve/prepared-configs/manifest.csv --output-root /kaggle/working/sample-pilot-resume --stop-after-epochs 1
```

Remplacer le checkpoint par la dernière epoch complète, jamais `.tmp` ; conserver aussi son meilleur
checkpoint et ses sorties de validation. Le budget total reste 10 epochs, optimizer/RNG repris,
nouveau run MLflow lié au parent. Adapter seulement les chemins de montage ; ne pas modifier
les hyperparamètres ni le code. Chaque session de reprise nécessite son budget validé.

Pour brancher un run GPU fini à l'application existante :

```bash
python scripts/kaggle_cloud.py snapshot-training --run /chemin/run-fini --tracking-db /chemin/tracking.db --out /chemin/nouveau-snapshot
```

Créer ensuite un nouveau launch record scellé avec `snapshot`, `data_roots` et `inventory_identity`
du nouveau snapshot ; préserver `artifacts/demo-serving/local.json` et l'ancien snapshot.
Le backend vérifie sources, empreinte du checkpoint, normalisation et données avant inférence.

Application actuelle :

```powershell
.venv/Scripts/python.exe scripts/serve_demo.py --config artifacts/demo-serving/local.json --port 8000
```

Ouvrir `http://127.0.0.1:8000`. Le bundle React est déjà construit. CNN CPU réel à 8 updates
conservé et rechargé après correction : prédiction validation du 6 janvier 2020 à partir du 5 janvier.
Vérité terrain séparée (`/ground-truth`), jamais transmise à `/predict`. Le rendu navigateur
n'a pas été refait ; preuve HTTP historique préservée. Nouveau contrôle direct Predictor/FastAPI réussi.
TestClient indisponible localement (`httpx2` absent), sans incidence sur le lancement Uvicorn.

## Préservation Git et limites

Aucun commit, push, suppression ni staging. Modifications historiques conservées.
Scan des candidats Git : aucune signature de secret à forte confiance, aucun fichier >10 Mio ;
cela ne constitue pas une garantie exhaustive. Données, checkpoints, SQLite, artefacts et frontend
compilé sont ignorés : **un futur commit seul ne les sauvegardera pas**.
Les rapports CPU courts, notebooks v3, ancienne archive v4.2 et contrôles des sorties annulées
restent historiques ; ils ne prouvent pas les résultats du nouveau pilote.

Tests ciblés finaux : 5 réussis, 2 CUDA ignorés. Audit des normalisations train/max-contexte,
échantillonnage, MLflow et reprise existante : première sélection 5 réussis également.
Pas de suite complète relancée. Aucun entraînement GPU lancé. Validation demandée avant allocation.
