# Suivi 3 — runner scientifique commun

Rapport du **30 octobre 2026**, branche vérifiée `feature/unet-baseline`.
Travail local uniquement. Aucun commit/push ; aucun fichier historique suivi
modifié ; aucune action sur `main`. Aucun téléchargement de corpus, achat cloud
ou entraînement GPU effectué.

## Livrables

| Livrable | Fichiers |
| --- | --- |
| CLI unique | `scripts/run_benchmark.py` : inspect/train/lock/final-test/export |
| Entraînement et traçabilité | `src/geofirex/runner.py` |
| Adaptateur temporel et cache RAM borné | `src/geofirex/runner_data.py` |
| Évaluation exacte et agrégations | `src/geofirex/runner_evaluation.py` |
| Configurations reproductibles | `configs/runner_debug.yaml`, `runner_kaggle.yaml`, `runner_annual.yaml` |
| Tests nouveaux | `tests/test_runner.py` : 15 tests, complétant les 119 historiques |
| Préflight cloud préservant PyTorch | `scripts/gpu_preflight.py` |
| Vérification/export réutilisable (préparation du 31 octobre) | `scripts/verify_and_export_benchmark.py` |
| Guide annuel / contrats / limites | [RUNNER_ANNUAL.md](../docs/RUNNER_ANNUAL.md) |
| Guide GPU / export / Runpod | [KAGGLE_GPU_RUNNER.md](../docs/KAGGLE_GPU_RUNNER.md) |
| Expériences et stockage | [RUNNER_MLFLOW.md](../docs/RUNNER_MLFLOW.md) |
| Résultats détaillés vérifiés | [RUNNER_VALIDATION_2026-10-30.json](RUNNER_VALIDATION_2026-10-30.json) |

## Audit initial

Les fichiers `BENCHMARK_PROTOCOL.md`, `FULL_CORPUS_PLAN.md`, les modèles/métriques
benchmark et les scripts U-Net ont été examinés avant l'implémentation.

- B0–B3 étaient des adaptateurs forward/backward ; aucun trainer commun n'était
  branché. B1 est une convolution logistique 3×3 à 217 paramètres pour T=3,
  conservée comme CNN minimal. B2 empile T aux canaux ; B3 possède une récurrence
  ConvLSTM. Ils ne constituent pas des architectures à capacité égale.
- L'ancien trainer/store est T=1 et lit `context_paths[0]` : une nouvelle chaîne
  temporelle était nécessaire. Les anciens scripts et résultats sont conservés.
- Les 12 folds annuels et l'intersection réelle des cibles T=1/3/5 existaient.
  Le CSV de split couvre les 607 événements ; le manifeste local en couvre cinq.
  Fold 0 local : 41 cibles train, 42 validation, zéro test harmonisé.
- L'AP exacte externe existante a été réutilisée, avec métriques poolées globales
  et par événement ; les anciennes moyennes d'AP par fenêtres ne sont pas
  utilisées comme AP principale ni comme AP événement poolée.
- Les NaN de la bande active-fire signifient absence de détection. Les traiter
  comme labels invalides supprimerait les négatifs. Les nouveaux masques
  distinguent cette convention des empreintes invalides et du padding.
- Le cache disque préparatoire ne conserve pas l'empreinte de validité. Il
  n'est pas réutilisé silencieusement. Le nouveau LRU RAM conserve bandes
  physiques et masques, avec borne mémoire et repli sur lecture par fenêtres.

Le protocole exécuté est nommé **annual-v2-runner**, distinct du préparatoire
annual-v1. Il déclare le sampling uniforme avec remise, l'empreinte commune T=5,
la validation complète à chaque epoch et ses règles de sélection.

## Vérifications réellement exécutées

| Vérification | Résultat observé |
| --- | --- |
| Tests historiques avant modification | 119 passed, 39,36 s |
| Suite complète après runner et cache borné | **134 passed**, 642 warnings, 74,76 s |
| Forward/backward B1/B2/B3, T=1/3/5, dimensions impaires | Réussis dans la suite |
| Mini-overfit B1/B2/B3 sur fixtures GeoTIFF | Loss diminuée, gradients finis |
| Smoke end-to-end B0/B1/B2/B3 synthétiques | Checkpoints, métriques et figures produits |
| Reprise CPU depuis epoch 1 d'un run à deux epochs | Poids finaux de l'epoch 2 identiques bit à bit |
| Garde anti-ouverture test pendant train | Test automatisé réussi |
| Verrou, final-test et refus de modifications | Réussis sur fixtures artificielles |
| AP scènes complètes et égalités de scores | Comparées à sklearn ; tests réussis |
| Validité commune T=1/3/5, NaN, padding, cache | Tests réussis, lecture avec/sans cache identique |
| MLflow local | Statut FINISHED, tags, métrique et artefact vérifiés par client |
| Préflight dépendances local | PyTorch `2.15.0+cpu` conservé, CUDA indisponible ; imports et installation editable sans résolution de dépendances réussis |
| Archives des quatre diagnostics réels | Présence et SHA256 des checkpoints vérifiés dans les TAR, scratch `.bin`/`.tmp` exclu |

Les warnings observés concernent Rasterio (dépréciation de transformation,
fixtures sans géoréférencement) et SQLAlchemy/MLflow (dépréciation), sans échec.
Le préflight local n'a pas relancé la suite via `--tests` : celle-ci avait été
exécutée séparément. Aucun préflight CUDA n'a été exécuté ici.

La vérification visuelle a montré que les lignes d'une figure à un seul epoch
étaient invisibles sans marqueur. Le tracé a été corrigé pour les futurs runs
et vérifié sur une **copie séparée** sous `artifacts/benchmark/figure_review/B1`.
Les figures et snapshots originaux des diagnostics restent inchangés. La suite
134 tests précède uniquement cette correction de marqueurs ; la version corrigée
a été importée et la figure rendue/inspectée directement.

## Diagnostics sur les cinq événements réels

Même fold 0, T=3, seed 42, CPU, patches 64, batch 8. B1–B3 : 32 patches par
epoch, **4 mises à jour** AdamW ; mini-overfit de 30 pas jeté avant le vrai run.
Normalisation identique des entrées train T=5 uniques, contrôlée par hash.
Les listes de patches de l'epoch sont identiques pour B1/B2/B3.

Tous les modèles ont évalué les mêmes **42 scènes validation**, totalisant
**4 190 163 pixels valides**. Seuils choisis sur cette validation uniquement.

| Modèle | AP micro validation exacte | Dice au seuil validation | Seuil | Mises à jour | Durée totale enregistrée (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 Persistence | 0,23194838 | 0,48135198 | 0,9 | 0 | 19,89 |
| B1 CNN minimal | 0,00356119 | 0,01019108 | 0,7 | 4 | 65,66 |
| B2 U-Net empilée | 0,00028386 | 0,00083126 | 0,02 | 4 | 79,65 |
| B3 ConvLSTM | 0,00026665 | 0,00082826 | 0,3 | 4 | 71,55 |

**Ce tableau est un diagnostic technique, pas un classement scientifique.**
Une seule seed, cinq événements et quatre mises à jour ne démontrent ni
convergence ni généralisation. Aucun modèle n'est déclaré meilleur pour le
benchmark des 607 événements. B2 et B3 ont tourné simultanément sur CPU : leurs
temps incluent de la contention et ne mesurent pas une comparaison matérielle.
La durée totale enregistrée précède l'upload final/fermeture MLflow et l'export.

Pour B1, l'epoch a pris 44,96 s, dont 1,99 s pour la boucle train ; la validation
a notamment consacré 37,43 s au scoring/tri AP. Ce constat montre que la durée
d'une epoch GPU historique ne permet pas de prévoir le coût du runner complet.
Les chiffres sont des mesures locales, pas une projection de coût cloud.

Les JSON enregistrent aussi IoU, Precision, Recall, TP/FP/FN, prévalence,
Brier, nouvelles ignitions, métriques poolées par événement et macros avec
nombres définis. Aucun score test réel n'a été calculé. Le fold 0 local ne
possède pas de test ; aucun verrou scientifique final n'a été créé.

Runs MLflow locaux, tous vérifiés **FINISHED** :

| Modèle | Run ID | Dossier sous `artifacts/benchmark/` |
| --- | --- | --- |
| B0 | `6f63c191bd0a420f86862c7ff35e1273` | `B0-fold00-T3-seed42-train-20261030T224923Z-d43290c8fe` |
| B1 | `b42cb6d998bb40f4bc2805201a13b178` | `B1-fold00-T3-seed42-train-20261030T224824Z-cb05372ac7` |
| B2 | `382c4fb73dc24d7787d485db62f301c2` | `B2-fold00-T3-seed42-train-20261030T225026Z-55c1cce64f` |
| B3 | `a335064cd7e349188de86b8489bae38f` | `B3-fold00-T3-seed42-train-20261030T225026Z-db807cc16e` |

Expérience : `geofirex-benchmark-diagnostic-v2`, base
`artifacts/benchmark/tracking.db`, artefacts sous `mlflow_artifacts/`.
Les archives sont sous `artifacts/benchmark/exports/`, avec SHA256 séparés.
Les chemins et hashes détaillés figurent dans le rapport JSON lié plus haut.
Les sources exactes de chaque run sont archivées ; utiliser ce snapshot si
le code courant a changé pour une reprise ou une évaluation finale.

Un premier diagnostic B1 sans cache a été interrompu avant tout checkpoint
d'epoch après constat de relectures/decompressions redondantes des TIFF. Son
dossier `B1-fold00-T3-seed42-train-20261030T224124Z-29eb55a96c` et ses fichiers
restent conservés ; le run `b78a1b318d924461bbd957012e96cfc8` est **KILLED**, avec
`interruption.json`. Aucun résultat d'epoch ne lui est attribué.

## Ce qui reste avant un benchmark scientifique

1. Acquisition autorisée et contrôle du corpus complet ; manifeste versionné
   des 607 événements. Le runner refuse la qualification scientifique sur un
   inventaire partiel et contrôle taille/CRC des fichiers lus.
2. Revue des recouvrements géographiques entre événements de splits différents.
   Identités de fichiers et réutilisations de chemins sont bloquées ; les
   recouvrements spatiaux non identiques exigent encore une revue.
3. Préflight **réel** Kaggle/Runpod : CUDA/AMP, stabilité, mémoire, multiworker,
   débit I/O, tri AP et export/rétention durable. Le code existe, ces opérations
   cloud n'ont pas été exécutées ni prétendues validées.
4. Pilote GPU autorisé, budget/convergence/HPO communs préenregistrés,
   réplications 42/43/44 puis sélection validation et verrou scientifique.
   `runner_annual.yaml` est une proposition de budget, pas une autorisation.
5. Tests finaux des folds après sélection, puis agrégations annuelles sans
   choisir le HPO à partir des tests. Aucun intervalle de confiance n'est inventé.
6. Cache disque préservant les empreintes, débit corpus complet et stockage
   distant S3/serveur MLflow à profiler/valider avant montée en charge.
7. Modalités additionnelles avec audit de disponibilité temporelle, Spark
   distribué, FastAPI et React restent des étapes du système complet.

Commande de départ recommandée pour reproduire le diagnostic, sans GPU :

```powershell
.\.venv\Scripts\python.exe scripts/run_benchmark.py inspect --config configs/runner_debug.yaml
.\.venv\Scripts\python.exe scripts/run_benchmark.py train --config configs/runner_debug.yaml --model B1 --device cpu
```

Changer explicitement `--model` pour B0/B2/B3. Chaque invocation crée un nouveau
dossier et un nouveau run ; aucun lancement de la matrice complète n'est implicite.
