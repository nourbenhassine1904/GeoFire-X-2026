# Suivi 3 — préparation du corpus complet et du benchmark multi-modèles

Livraison du 30 octobre 2026, branche `feature/unet-baseline`.
**119 tests passent**, sans skip, en 34,88 s au dernier passage. Les 84 tests
du début de mission ont été relancés ; 35 tests préparatoires s'y ajoutent.
Les warnings concernent rasterio/Affine et les fixtures historiques.

Les sept livrables prioritaires sont préparés. Le corpus complet et le
benchmark entraîné ne sont **pas** déclarés exécutés. Aucun GeoTIFF distant
n'a été téléchargé pendant cette mission ; seuls 1,61 Mo de métadonnées ZIP
ont été lus. Aucun optimizer n'a mis à jour un modèle pendant cette mission.

## Livrables

| Besoin | Document / preuve | État |
| --- | --- | --- |
| Audit précis de la U-Net et des causes démontrées/hypothèses | [Audit U-Net V2](UNET_AUDIT_V2.md), [mesures JSON](unet_diagnostics_v2/diagnostics.json) | Exécuté, validation seulement |
| Acquisition et traitement des 607 événements | [Plan et budget](../docs/FULL_CORPUS_PLAN.md), [budget mesuré](corpus_budget_v1.json) | Scripts prêts ; acquisition massive en attente d'accord |
| Matrice B0/B1/B2/B3/B4 | [Protocole](../docs/BENCHMARK_PROTOCOL.md), [profil CPU](benchmark_cpu_profile_v1.json) | B0–B3 prototypes testés ; B4 différé |
| Splits, métriques et MLflow | [Protocole](../docs/BENCHMARK_PROTOCOL.md), `splits/benchmark/annual_v1/` | 12 folds metadata et dates T harmonisées générés |
| Structure modulaire du code | Tableau d'état dans le protocole ; modules ci-dessous | Composants préparatoires fonctionnels, runner scientifique restant à construire |
| Coûts calcul et stockage | [Plan](../docs/FULL_CORPUS_PLAN.md), [matériel](benchmark_hardware_v1.json) | Mesures et extrapolations distinguées |
| Scripts/tests non destructifs | CLI ci-dessous, tests `test_corpus_preparation.py` et `test_benchmark_contracts.py` | Exécutés localement, hors Spark et transfert de TIFF |

## Conclusions vérifiées

L'ancien checkpoint et ses sources archivées ont été revérifiés. Les mauvaises
métriques sont cohérentes avec le replay, pas des poids non entraînés. Le scoring
a reçu une petite garde contre labels non binaires/NaN, probabilités invalides,
formes incohérentes et nom de prédicteur inconnu ; le replay validation reste
exactement identique aux résultats historiques pour les entrées valides.

Le modèle utilise le feu J : l'AP validation passe de 0,0008373 à 0,0005619 si
ce canal est retiré. Il produit néanmoins beaucoup de faux positifs et les
scores sont mal calibrés sur l'événement tenu à part. Les raccords et changements
d'origine des tuiles affectent les scores. Aucune ablation ne prouve que le LR,
la taille d'U-Net ou l'absence de météo soient la cause principale de l'échec.

La reconstruction des tirages montre 381 tuiles uniques sur 1 695, 94 positives
sur 98, et 0,366 % de pixels feu effectivement vus. Un quota 50/50 de patches
n'est pas un équilibrage 50/50 des pixels. Le gradient soft Dice d'un exemple
vide est mesuré presque nul ; ce mécanisme est distingué des hypothèses de
convergence du modèle.

Le répertoire distant confirme **607 événements, 13 607 observations**,
48,356 Go de charges compressées et **49,294 Go de TIFF extraits**.
Avec environ 129 Go libres, ZIP+TIFF+cache+marge dépasse l'espace actuel ;
l'extraction Range incrémentale et un cache borné sont proposés. Le scratch des
AP exactes doit être ajouté au budget, pas ignoré.

Le fold 0 annuel est train 2018/2019, val 2020, test 2021 :
250/201/156 événements attendus. Il ne dispose localement que de 41/42/0 cibles
alignées T=1/3/5. Les 12 folds sont des définitions complètes d'événements,
pas des datasets complets présents sur disque. `complete_fold_metadata=false`
et `scientific_execution_enabled=false` évitent de les confondre avec un
benchmark prêt. `ready_for_local_training` n'indique que la présence d'exemples
dans les trois rôles pour un éventuel pilote ; ce n'est pas une validation de
complétude scientifique.

Le pilote memmap réel de 14 jours occupe 19,68 Mo. L'inventaire Parquet comprend
108 observations et a été relu à l'identique. L'AP sur disque respecte les
égalités globales et retrouve la référence sklearn réelle à 10⁻¹² ; sur les
nouveaux pixels de feu, l'AP U-Net reste inférieure à la référence constante.

## Code préparé

- `corpus.py` : client Range plafonné, index détaillé, CRC et acquisition par
  fichiers, refus des chemins invalides et des fichiers existants corrompus.
- `event_cache.py` : cache memmap indépendant du fold, sources/tableaux hashés,
  lecture de patches et conservation des valeurs physiques/NaN.
- `benchmark.py` : folds annuels et intersection des cibles réellement valides.
- `benchmark_models.py` : interface commune, Persistence exacte, convolution
  logistique, U-Net empilée et ConvLSTM miniature ; B4 explicitement non implémenté.
- `benchmark_metrics.py` : AP externe exacte, nouvelles ignitions et sélection
  sur validation avec comparabilité et seeds préenregistrées.
- CLI d'inventaire, dry-run/reprise, conversion, Parquet et contrôles ; exemple
  Spark d'agrégation des métadonnées, non exécuté faute de Spark/JVM configurés.

Le futur trainer commun doit encore brancher l'adapter temporel/cache, les
statistiques du fold, le checkpoint choisi sur AP et un verrou avant test.
L'ancien trainer reste un outil T=1 historique ; il n'est pas présenté comme
capable de lancer le benchmark complet avec ce seul YAML préparatoire.

## Vérifications exécutées et commandes reproductibles

Les sorties doivent être **nouvelles** lors d'une réexécution : les outils
refusent volontairement d'écraser rapports et répertoires déjà présents.

```powershell
# Relecture du run historique (sources archivées et bruts inchangés).
.\.venv\Scripts\python.exe scripts/verify_unet_run.py --run-dir artifacts/unet/20261030T091804Z_3277e4f4 --out artifacts/reverification_nouvelle.json
# Diagnostics sans entraînement, validation uniquement (~75 s observées).
.\.venv\Scripts\python.exe scripts/diagnose_unet.py --checkpoint artifacts/unet/20261030T091804Z_3277e4f4/best.pt --out-dir reports/diagnostics_nouveaux
# Plan offline depuis l'index déjà enrichi.
.\.venv\Scripts\python.exe scripts/plan_corpus.py --index data/manifests/real/corpus_zip_index_v1.json --out reports/budget_nouveau.json
# Vérification CRC de l'échantillon réel (aucun réseau).
.\.venv\Scripts\python.exe scripts/acquire_corpus.py --index data/manifests/real/corpus_zip_index_v1.json --destination data/raw/WildfireSpreadTS/real_sample --events 2018/fire_21748813 2018/fire_21997864 2019/fire_23300839 2020/fire_23654679 2020/fire_24104674 --max-transfer-bytes 100000000 --verify-only --out reports/integrite_nouvelle.json
# Génération annuelle et harmonisation des dates T.
.\.venv\Scripts\python.exe scripts/prepare_benchmark.py --index data/manifests/real/corpus_zip_index_v1.json --manifest data/manifests/real/manifest_wildfirespreadts.csv --out-dir splits/benchmark/annual_nouveau
# Pilote cache d'un seul événement déjà local.
.\.venv\Scripts\python.exe scripts/prepare_event_cache.py --manifest data/manifests/real/manifest_wildfirespreadts.csv --index data/manifests/real/corpus_zip_index_v1.json --events 2018/fire_21748813 --out-dir data/processed/cache_nouveau
# Parquet de métadonnées, lecture locale des rasters.
.\.venv\Scripts\python.exe scripts/export_observation_metadata.py --manifest data/manifests/real/manifest_wildfirespreadts.csv --out-dir data/interim/metadata_nouvelles
# Profils sans optimizer, pas des expériences de prédiction.
.\.venv\Scripts\python.exe scripts/profile_benchmark_models.py --manifest data/manifests/real/manifest_wildfirespreadts.csv --normalizer artifacts/unet/20261030T091804Z_3277e4f4/normalization.json --out reports/profil_nouveau.json
# Contrôle AP exacte sur validation réelle (~34 s observées).
.\.venv\Scripts\python.exe scripts/check_reference_ap.py --run-dir artifacts/unet/20261030T091804Z_3277e4f4 --out-dir artifacts/ap_reference_nouvelle
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Tests nouveaux : CRC malgré taille identique, interruption sans publication de
TIFF partiel, plafond avant réseau et retries, refus HTTP 200 sans lecture,
borne de lecture, chemins/dates, intégrité/cache et parité de features, Parquet,
12 folds, gaps et dates communes, contrats/grads B1–B3 à T=1/3/5, dépendance
temporelle et Persistence, AP externe avec égalités et merges, nouvelles
ignitions, sélection indépendante du test, seeds identiques et labels invalides.

## Conservation et décision suivante

Les anciens manifestes/splits, modèles, checkpoints et expériences MLflow sont
conservés. Pas de commit, push, publication, modification de `main`, migration
des données ni frontend/backend nouveau. Le seul ajout de dépendance déclaré
est l'extra optionnel `benchmark-metadata=[pyarrow]`, déjà disponible localement ;
aucune installation HDF5/Zarr/Spark/CUDA n'a été réalisée.

La décision suivante est l'accord sur **une sélection d'événements, un plafond
de transfert et une destination dotée de l'espace prévu**, puis sur un pilote
de calcul court. Le plan complet de 324 runs hors HPO doit rester une cible
progressive avec ressources validées, pas une exécution automatique.
