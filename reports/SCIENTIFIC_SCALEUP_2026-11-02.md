# GeoFire-X — journal scientifique du 2 novembre 2026

État final de cet incrément : code développé, audit local exécuté, préparation hors ligne testée. Aucune acquisition réseau, aucun job GPU, aucun commit/push/merge. Test 2021 non lu.

Branche `feature/scientific-scaleup-v1`, HEAD `7428561f607914eb2f7cd286ee50435d142f8e09` = `origin/main` lors du contrôle. La branche frontend reste disponible. La modification antérieure du README et les fichiers locaux historiques sont conservés.

## A — Audit technique : réalisé

| Composant / fichiers inspectés | Preuve / qualification | Priorité |
|---|---|---|
| `README.md`, `FULL_CORPUS_PLAN.md`, `BENCHMARK_PROTOCOL.md`, `benchmark_protocol.yaml` | Certaines sections décrivent une préparation ancienne ; le pilote GPU et Spark ultérieurs ont été exécutés. Ne pas utiliser leurs descriptions « non exécuté » comme état actuel. | P0 documentation |
| `runner_data.py`, `windows.py`, `benchmark.py` | T=1/3/5 consécutifs, intersection des dates cibles, train/val par années. RasterReader LRU par worker borné en octets ; fallback lecture de tuiles. Non validé sur 451 événements. | P0 données |
| `feature_profiles.py`, `unet_data.py` | 8 / 36 / 50 canaux encodés, ordre déterministe, sin/cos aspect, LC 17 classes + unknown ; normalisation sur entrées train uniques. Extensions bloquées sans déclaration revue. | P0 disponibilité |
| `benchmark_models.py`, `deterministic_models.py`, `runner.py` | B0–B3 réellement utilisés par le pilote ; B1 spatial v2 5 809 paramètres, B2 nearest 123 977, B3 nearest 5 297. BCE pondérée + Dice, AdamW, clipping, sampling positif train, sélection AP validation, reprise optimizer/RNG. | P1 profilage |
| `benchmark_metrics.py`, `runner_evaluation.py` | AP micro exacte externe, seuil validation, TP/FP/FN, nouvelles ignitions, par événement ; outils présents/testés, aucune nouvelle amélioration mesurée ici. | P1 analyse |
| `cloud_corpus.py`, `corpus.py`, scripts acquisition/Kaggle | CRC, SHA, événements entiers, budgets de session, `.part`, préservation des fichiers divergents ; scripts présents ≠ acquisitions toutes exécutées. | P0 budget |
| `spark_index_demo.py`, preuves sous `artifacts/orchestrator/` | Exécution historique réussie sur 13 607 lignes + QC export 108 lignes selon `CODEX_REPRISE_2026-11-01.md`. Aucun Spark relancé ; ce travail prépare les contrats Parquet locaux. | P1 intégration |
| `serving.py`, `DEMO_SUIVI3.md`, `GPU_PILOT_2026-11-01.md`, `frontend/` | Démo B1 certifiée historique ; routes health/events/predict/ground-truth conservées. Aucun changement applicatif ; aucune nouvelle validation visuelle revendiquée. | stable |
| `FoldData` qualification scientific | Exige 607 événements et inventaire test ; les paliers 80/200/451 ne peuvent actuellement pas être qualifiés scientific. Configurations produites = pilot. Garde non modifiée silencieusement. | P0 avant benchmark définitif |

## B — Rareté : exécuté sur les 108 TIFF du pilote

Durée du workflow final : 65.123 s, CPU local, aucune exécution distribuée. CRC et SHA vérifiés, 5 événements complets, 108 fichiers lisibles, 0 corrompu/manquant dans ce manifeste. Géométries cohérentes par événement, pas de trou de dates dans ce pilote. Aucun pixel déclaré analysé pour un fichier absent.

| Partition | Observations | Jours positifs | Sans détection | Pixels positifs / valides | Prévalence |
|---|---:|---:|---:|---:|---:|
| train | 56 | 33 | 23 | 2223 / 4629690 | 0.048016 % |
| val | 52 | 23 | 29 | 1736 / 5189208 | 0.033454 % |

NaN dans active-fire signifie absence de détection, pas NoData du footprint. Aucun pixel invalidé par masque explicite/NoData fini dans cet échantillon ; cela ne prouve pas une qualité parfaite des labels VIIRS. Les jours sans détection doivent rester dans la validation et comptabiliser les faux positifs.

| Split | Fenêtres T=1 | T=3 | T=5 | Cibles harmonisées | Positifs / pixels du domaine commun |
|---|---:|---:|---:|---:|---:|
| train | 53 | 47 | 41 | 41 | 1957 / 3353015 |
| val | 50 | 46 | 42 | 42 | 1736 / 4190163 |

Les comptes bruts T=1/3/5 sont des fenêtres géométriquement/chronologiquement candidates ; le domaine commun intersecte réellement les masques des 5 contextes et de la cible. Une fenêtre entièrement invalide est distinguée et non présentée comme un jour négatif valide. Le domaine val harmonisé retrouve exactement 1 736 / 4 190 163, soit environ 0,04143 %. Aucune nouvelle métrique de modèle calculée.

À inspecter dans la démo : `2018/fire_21997864` (densité de détection la plus élevée du pilote), `2020/fire_23654679` (nombreux jours sans détection) et `2020/fire_24104674` (validation plus positive). La somme des pixels détectés sur plusieurs jours ne constitue pas une surface brûlée unique. Morphologie et surfaces géodésiques détaillées restent différées.

## C — Trois paliers : préparés, non acquis

Sélection déterministe indépendante des labels/scores : stratification proportionnelle par année, seed 42, événements locaux complets au sens des octets conservés en premier, paliers emboîtés. Les 100 TIFF supplémentaires de `corpus_v1` ont un inventaire CRC/SHA séparé ; ce contrôle ne remplace pas leur QC raster ni une certification distante indépendante.

| Événements | 2018 / 2019 / 2020 | TIFF attendus | Charge compressée supplémentaire (octets) | TIFF supplémentaires (octets) | Plafond réseau proposé (octets) |
|---:|---|---:|---:|---:|---:|
| 80 | 31 / 13 / 36 | 1768 | 5741850510 | 5853750530 | 6611516695 |
| 200 | 78 / 33 / 89 | 4219 | 14539037757 | 14826857372 | 16728282029 |
| 451 | 176 / 74 / 201 | 9490 | 33536994440 | 34189635708 | 38575932214 |

Les charges ZIP et tailles TIFF proviennent exactement de l’index épinglé ; le réseau réel inclut en-têtes, répertoire central, requêtes et retries. Le plafond proposé ajoute 15 % et 8 Mio, sans garantie de finir si la source échoue. Il doit être approuvé séparément pour chaque session. Les chiffres supplémentaires sont relatifs à l’inventaire local actuel, pas la somme à payer successivement. Les champs `incremental_*` détaillent les étapes après achèvement du palier précédent.

| Palier | TIFF totaux (octets) | Réutilisés / manquants | Disque conservateur proposé (octets) | Transfert idéal 50 Mbit/s (heures) |
|---:|---:|---|---:|---:|
| 80 | 6634312839 | 202 / 1566 | 32488063369 | 0.255 |
| 200 | 15607419681 | 202 / 4017 | 50434277053 | 0.646 |
| 451 | 34995352430 | 208 / 9282 | 89184988138 | 1.491 |

Le disque proposé = nouveaux TIFF + une copie de staging du palier entier + réserve 20 Go. Le wrapper local ne crée pas cette copie ; elle représente un scénario conservateur de publication ultérieure. Le cache actif du chemin d’audit est borné à 64 Mio ; les runs préparés réservent 256 Mio de cache lecteur par worker (workers=0 initialement). Le scratch AP dépend des scores et reste à mesurer ; le plan historique propose 20–60 Go pour un grand split, hors tableau. Ne pas lancer si l’espace libre et les quotas réels ne sont pas vérifiés. Coût monétaire, débit réel et quotas Kaggle actuels : inconnus.

Batch001 : preuve historique indépendante pour `2018/fire_21458798`, 14 TIFF, 45 061 097 octets, certificat indiqué dans `reports/CODEX_REPRISE_2026-11-01.md`. Les traces 483/460 TIFF des jobs annulés ne sont pas créditées. Aucune tentative de récupération/publication distante, aucune réutilisation d’un ancien crédit réseau. Le nouvel outil opère localement uniquement ; une reprise cloud sans transit PC exige un inventaire authentifié et des quotas récents. La disponibilité actuelle des sorties annulées n’est pas vérifiée.

APPROBATION ACQUISITION REQUISE — choisir le palier, la destination et un nouveau plafond de session ; aucun accord implicite.

Simulation exacte du premier palier (déjà exécutée, sans réseau) :

```powershell
.\.venv\Scripts\python.exe scripts/acquire_scientific_stage.py --plan reports/scientific_scaleup_v4/acquisition_80.json --destination data/raw/WildfireSpreadTS/scientific_stage080_v1 --reuse-root data/raw/WildfireSpreadTS/real_sample --reuse-root data/raw/WildfireSpreadTS/corpus_v1 --out reports/acquisition080_review_NEW.json
```

Après accord sur ces octets uniquement, ajouter `--execute` avec une nouvelle sortie. Le wrapper revalide les CRC sur toutes les racines, refuse une perte de réutilisation qui augmenterait la charge approuvée, active un ledger durable des octets/retries et conserve les `.part`. Il ne publie rien. Après transfert : QC raster, géométrie/dates, manifeste multi-racines, certification indépendante et nouvelle autorisation distincte avant publication.

## D/E — Modalités et Big Data : contrats développés et testés

`feature_contracts.json` / `feature_contracts_release.json` exposent 8/36/50 canaux ordonnés et leur identité. `bands_23.csv` distingue les 23 bandes, sources documentées dans le dépôt, fréquence, unité connue ou non certifiée, transformation et blocage temporel. Aucune déclaration de disponibilité n’est fabriquée. Forecasts exclus ; direction du vent historique exclue ; LC annuel et composites exigent une revue de leur date de production.

Contrat : index officiel → jointure événement/date et QC des fichiers acquis → `observations.parquet` → `windows.parquet` harmonisé → `load_window_manifest` → `TemporalStore` / Dataset PyTorch existant. Schémas Parquet explicites, nullables pour erreurs et tables vides ; chemins et listes temporelles typés, SHA, CRS, transform, split, statut QC, comptages. Les sidecars sont hashés et sérialisés dans `sidecars_json`. Aucun tenseur raster en Parquet.

Spark peut lire ces tables via `spark.read.parquet(...)`, grouper par année/split et joindre l’index officiel. Une ligne de l’index officiel n’est pas une observation acquise. Les statistiques de rareté ne portent que sur `status=ok`. Le chargeur valide les manifestes contre `aligned_windows`, sans seconde logique de fenêtrage ; un test Parquet → TemporalStore vérifie `[T=3,C=8,H,W]`. L’adaptateur est disponible ; le runner historique conserve son chemin existant et n’est pas migré automatiquement. Aucun nouveau cluster Spark exécuté.

## F/G — GPU et évaluation : préparés, non approuvés

16 YAML validés et modèles instanciés sur CPU : niveau 1 = B0 + 3 runs appris seed42 ; niveau 2 = B0 + 9 runs appris seeds42/43/44. Pour chaque run appris : T=3, 10 epochs maximum, 2 048 patches/epoch, batch16 = 128 updates/epoch = 1 280 maximum, FP32, patches64, même profil feu/terrain. Ce budget historique est un contrôle de comparabilité, pas une preuve de convergence du corpus élargi. Le plafond de session hérité de 780 s est une coupure de protection à replanifier, pas une durée estimée suffisante. Changer uniquement les données au niveau1, puis uniquement les seeds au niveau2. Ne pas agréger ces niveaux comme une seule expérience.

Manifeste de chaque YAML à remplacer par le manifeste certifié du palier : aucun YAML préparatoire ne peut trouver ce fichier actuellement. Sorties et MLflow utilisent de nouveaux chemins sous `artifacts/scientific-scaleup-runs-v1` ; aucun accès à la base historique dans ce workflow. Les tests du runner utilisent des fixtures temporaires CPU, jamais les données historiques pour entraîner. Reprise prévue : `run_benchmark.py train --config <yaml> --resume <checkpoint_derniere_epoch>` ; le runner crée un nouveau run et vérifie l’identité. Prévoir best checkpoint AP + état optimizer/RNG + normalisation/config/sources/MLflow ; volume et débit réels à mesurer avant allocation.

Niveau3 modalités et T=1/3/5 : différé tant que disponibilité, manifeste certifié et hypothèse d’ablation ne sont pas gelés. AP exacte, courbes PR/calibration, nouvelles ignitions et agrégations existent dans les outils inspectés ; aucune nouvelle courbe de modèle inventée. Le test final exige un verrou de sélection et un accord explicite. Aucune métrique test calculée.

APPROBATION ENTRAÎNEMENT GPU REQUISE — aucune soumission. Durée/coût/quota/checkpoint storage du corpus élargi inconnus : il faut d’abord proposer un profil de débit borné sur le palier certifié, puis un budget chiffré révisé. Les 21 minutes du pilote ne sont pas une estimation des centaines d’événements.

## H/I — Application, tests, préservation et démonstration

Application conservée : suivre `docs/DEMO_SUIVI3.md` et son launch record `gpu-pilot-local.json`. Aucun besoin de reconstruction frontend pour ce lot. Présenter `rarity.png`, `events.csv`, les cas proposés, les plans exacts et les scores GPU historiques déjà certifiés. Ne pas confondre catalogue pilote et futurs 80/200/451 événements.

Validation exécutée : suite élargie 64 passés / 2 ignorés (`scientific_scaleup_tests_v3.xml`) ; tests finaux du nouveau module dans `scientific_scaleup_tests_release.xml`. Cas couverts : sans feu, entièrement invalide, données incomplètes, CRC divergent, fichier illisible avec CRC concordant, dates discontinues, doublons, verrou2021, Parquet vide, dictionnaires sidecars vides, manifestes altérés, contrat canaux, réutilisation multi-racines et plans emboîtés. Avertissements de dépendances documentés dans les XML ; pas d’échec revendiqué comme réussite.

Premières tentatives `scientific_scaleup_v1/v2` conservées comme diagnostics partiels (collision de champ puis schéma Parquet vide corrigés). `v3` est une exécution complète antérieure ; **`v4` est le livrable de référence**. `preservation_and_delivery.json` vérifie les hashes des anciens fichiers non suivis, le diff README initial, les 108 SHA historiques et les sorties finales. Figure PNG inspectée visuellement : libellés, unités et légende lisibles. Contrôle limité des motifs de secrets dans les nouvelles sources, aucun secret détecté ; aucune garantie exhaustive. Les données, checkpoints et SQLite historiques n’ont pas été modifiés par les commandes de cette mission.

Fichiers développés : `src/geofirex/scientific_scaleup.py`, `scripts/scientific_scaleup.py`, `scripts/acquire_scientific_stage.py`, `scripts/report_scientific_scaleup.py`, `tests/test_scientific_scaleup.py`. Tous les autres changements sont des sorties nouvelles sous `reports/scientific_scaleup*`. Aucun fichier du cœur historique n’a été modifié.

Reproduction dans un nouveau répertoire :

```powershell
.\.venv\Scripts\python.exe scripts/scientific_scaleup.py --out reports/scientific_scaleup_REPRO_NEW --reuse-root data/raw/WildfireSpreadTS/corpus_v1
.\.venv\Scripts\python.exe -m pytest tests/test_scientific_scaleup.py -q
git -c safe.directory=C:/Users/nourb/Projects/GeoFire-X-2026 diff --check
```

Autorisation pour A/B/D/E/H/I : non, travaux locaux sûrs exécutés ou limites explicitées. Autorisation pour C/F : oui, acquisition et GPU non approuvés. Prochaine décision concrète : destination et budget du premier palier ; la préparation locale ne donne aucun droit à consommer les anciens plafonds. Aucun commit/push/merge sans accord séparé.
