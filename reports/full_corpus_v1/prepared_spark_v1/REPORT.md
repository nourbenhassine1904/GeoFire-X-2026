# Corpus complet WildfireSpreadTS — fold 0

Certification indépendante : **607 événements / 13607 GeoTIFF**.
CRC, SHA-256, lecture des 23 bandes, géométrie et dates ; 0 erreur(s).
Comparaison SHA-256 au certificat d’extraction : EXACT_SHA256_CRC_SIZE_MATCH.

Train 2018–2019 : 250 événements ; validation 2020 : 201 ; test 2021 : 156, verrouillé.

Événements réellement représentés dans les fenêtres : train 250, validation 201.

| Split | Fenêtres T1 naturelles | T3 naturelles | T5 naturelles | Cibles alignées réellement chargées par T | Patches utilisables à 64 px |
|---|---:|---:|---:|---:|---:|
| train | 4948 | 4448 | 3948 | 3948 | 93603 |
| val | 4091 | 3689 | 3287 | 3287 | 78771 |

Parquet fenêtres : 21705 lignes. Traitement : SPARK_EXECUTED.
Les fenêtres alignées partagent les cibles et le masque conjoint T5 ; aucune cible de 2021 dans ces tables.
Normalisation : 4948 fichiers contextes uniques de 250 événements train ; 8 canaux encodés.

Mode du normalisateur : REUSED_VERIFIED_TRAIN_ONLY.

## Mesures CPU de lecture et encodage

| Split | T | Événements échantillonnés | Patches | Secondes | Patches/s |
|---|---:|---:|---:|---:|---:|
| train | 1 | 4 | 48 | 2.382 | 20.15 |
| train | 3 | 4 | 48 | 1.333 | 36.00 |
| train | 5 | 4 | 48 | 3.211 | 14.95 |
| val | 1 | 4 | 48 | 6.353 | 7.56 |
| val | 3 | 4 | 48 | 3.293 | 14.58 |
| val | 5 | 4 | 48 | 3.763 | 12.76 |

Spark 4.0.2, local[2] : 41.88 secondes pour SQL/collect/matérialisation, hors démarrage JVM et contrôles préalables des certificats.

Dépendances installées : {'pyspark': '4.0.2', 'py4j': '0.10.9.9'}. Corps HTTP consommés : 434421883 / 600000000 octets ; réservations conservatrices : 436510100 octets.
Aucune donnée supplémentaire du corpus téléchargée.

Mesures bornées, workers=0, deux threads CPU, cache lecteur 64 Mio. Le cache système n’a pas été vidé.
Les sorties B0/B1/B2/B3 ont été contrôlées sur CPU, sans gradient ni mise à jour.

## GPU — commandes futures, aucune exécution autorisée

Proposition : calibration B1/B2/B3 T3, 20 étapes + 3 préchauffages, enveloppe totale 0,5 heure GPU incluant la préparation.
Puis première campagne T3 seed42 : plafond proposé 2 heures par modèle entraînable, soit 6 heures GPU. B0 peut fonctionner sur CPU.
Ces enveloppes demandent approbation et calibration réelle ; aucun temps GPU de ce corpus n’a été mesuré.
Le garde-fou du runner agit aux frontières des époques : prévoir aussi un timeout externe dur.
Prévoir GPU 16 Go, RAM 16 Go et volume libre pour le corpus train/validation, les sorties et une réserve de 20 Go.

```powershell
.\.venv\Scripts\python.exe scripts/measure_gpu_throughput.py --config reports/full_corpus_v1/prepared_spark_v1/gpu_configs/B1_T3_seed42.yaml --steps 20 --warmup 3 --out artifacts/full-corpus-probes-v1/B1.json
.\.venv\Scripts\python.exe scripts/measure_gpu_throughput.py --config reports/full_corpus_v1/prepared_spark_v1/gpu_configs/B2_T3_seed42.yaml --steps 20 --warmup 3 --out artifacts/full-corpus-probes-v1/B2.json
.\.venv\Scripts\python.exe scripts/measure_gpu_throughput.py --config reports/full_corpus_v1/prepared_spark_v1/gpu_configs/B3_T3_seed42.yaml --steps 20 --warmup 3 --out artifacts/full-corpus-probes-v1/B3.json
```

Baseline B0 sur CPU (commande préparée, non exécutée) :

```powershell
.\.venv\Scripts\python.exe scripts/run_benchmark.py train --config reports/full_corpus_v1/prepared_spark_v1/gpu_configs/B0_T3_seed42.yaml --device cpu
```

Après calibration et autorisation de la campagne :

```powershell
.\.venv\Scripts\python.exe scripts/run_benchmark.py train --config reports/full_corpus_v1/prepared_spark_v1/gpu_configs/B1_T3_seed42.yaml
.\.venv\Scripts\python.exe scripts/run_benchmark.py train --config reports/full_corpus_v1/prepared_spark_v1/gpu_configs/B2_T3_seed42.yaml
.\.venv\Scripts\python.exe scripts/run_benchmark.py train --config reports/full_corpus_v1/prepared_spark_v1/gpu_configs/B3_T3_seed42.yaml
```

Les chemins configurés ciblent le PC actuel. Pour une autre machine : monter le corpus déjà autorisé,
adapter les chemins de configuration ou les options --data-root/--manifest/--window-manifest/--split/--index.
Les chemins stockés dans les tables restent relatifs au corpus. Aucune publication cloud n’est déclenchée.

Nouveau namespace MLflow : `geofirex-full-corpus-fold0-v1`. Les autres profils de variables gardent leur verrou de disponibilité.
Tests : {"tests_delivery.xml": {"tests": 55, "errors": 0, "failures": 0, "skipped": 1, "passed": 54, "sha256": "489f928961c586ff8f5a91c0100ec0ad00c620909971950494898e245748712b"}, "tests_contracts_final.xml": {"tests": 13, "errors": 0, "failures": 0, "skipped": 1, "passed": 12, "sha256": "ed2b16ffab72e2467c57091f86d73d639a7ec910176b3ed3d67f4927d56495cf"}, "tests_spark_installer.xml": {"tests": 3, "errors": 0, "failures": 0, "skipped": 0, "passed": 3, "sha256": "b0000a5b5ba623468a979d2e3c624404ba7c5a48bec4d916834dc6fa854d4b6f"}, "tests_spark_execution_v1.xml": {"tests": 1, "errors": 0, "failures": 0, "skipped": 0, "passed": 1, "sha256": "87b211b4851dff77be505487a0998c2f447d1018b3f84fcc5e2dccabb40d0f1b"}, "tests_spark_reuse.xml": {"tests": 15, "errors": 0, "failures": 0, "skipped": 0, "passed": 15, "sha256": "453ce55b6a553b298e255bb677d3a94172e2946293edfaffef5c8cd65fdd55e2"}, "tests_spark_final.xml": {"tests": 58, "errors": 0, "failures": 0, "skipped": 0, "passed": 58, "sha256": "5a65fcb157d27b0e2b06a5691f9e09af0ab89c5a4158e35977f2064ae82c87db"}}.
Les résultats, checkpoints et démonstrations historiques n’ont pas été modifiés.
