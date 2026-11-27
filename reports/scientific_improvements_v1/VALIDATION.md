# Validation et décision — audit scientifique du 13 novembre 2026

Branche : `feature/scientific-improvements-v1`, HEAD `8c939f6`. Aucun commit, push, merge, reset ou clean. Le [rapport de décision complet](../../docs/SCIENTIFIC_IMPROVEMENTS_V1.md) distingue les constats prouvés, les limites des preuves et les hypothèses.

## Décision

Préparer en premier A : une nouvelle branche Focal sur le même U-Net léger T=3, les huit canaux et les exemples certifiés, comparée aux résultats historiques poids10. Le changement inclut le retrait de Dice : il teste une famille de loss, pas uniquement le facteur de focalisation. Aucun gain n'est mesuré dans cette mission.

Le [contrat de préparation](../../configs/scientific_improvements_v1.json) bloque toute exécution et toute promotion Studio. Avant A : implémentation réelle de Focal et réduction masquée vérifiées, environnement historique et initialisation appariés, données/source/ordre des batches figés, budget GPU et sauvegarde des nouveaux checkpoints définis. Les critères AP/Dice/rappel/FP et les seuils de développement sont explicités dans le contrat. La confirmation169 déjà consultée reste descriptive.

Ordre proposé : A loss, B temporalité, C modalités dont la disponibilité à J est prouvée, D ResNet18-U-Net, E U-TAE. C reste bloquée ; les schémas36/50 ne constituent pas des modalités présentes dans le compact8. La direction du vent GFS, les composites et les produits consolidés nécessitent des preuves particulières.

Le [budget](gpu_budget_scenarios.json) sépare la mesure réelle de 887,069 secondes sur T4 pour un U-Net apparenté et le scénario choisi de 11,1–22,2 minutes par nouveau run A/B. Ce scénario n'est ni une mesure de Focal ni une borne garantie.

## Vérifications réalisées

- **51 tests réussis**, zéro échec, en 6,739 secondes : [JUnit final](tests-delivery.xml). Sept avertissements `PendingDeprecationWarning` proviennent de l'usage d'`Affine` dans la lecture de rasters synthétiques ; ils ne sont pas des échecs.
- Couverture ciblée : schémas23→8/36/50, normalisation train-only, données manquantes/aspect, chronologie et exhaustivité des preuves, refus2021, alignement T1/T3, masque cinq jours+cible, contrat d'expériences bloqué, calcul Focal synthétique, AP sur données synthétiques, comptages sauvegardés, routes scientifiques et gardes du Prediction Lab.
- Aucun entraînement, backward d'entraînement, forward sur données réelles, évaluation de modèle ou décodage de GeoTIFF/NPZ réel. Les petits rasters des fixtures sont entièrement synthétiques.
- **123 fichiers protégés inchangés**, dont les sources originales, le frontend, les configurations existantes et les artefacts certifiés sélectionnés : [avant](preservation_before.json), [après](preservation_after.json). Cette vérification ne prétend pas rehacher les 9,8 Go du corpus : les archives du corpus sont référencées par leurs certificats existants.
- Une seconde lecture des seules sorties enregistrées reproduit exactement les six JSON/CSV générés par le script d'audit. Les sorties existantes sont refusées avant lecture ; aucune sortie scientifique antérieure n'est remplacée.
- Frontend, routes de production, modèle Studio, seuil0,78, checkpoints, normaliseurs et archives certifiées conservés. Aucun nouveau build/frontend ou calcul Spark n'est nécessaire puisque leur code n'a pas changé.

Commande de vérification exécutée depuis la racine :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_readiness.py tests/test_science_results.py tests/test_prediction_lab.py::test_catalogue_2021_and_invalid_dates_rejected_before_pixels tests/test_prediction_lab.py::test_integrity_failure_closes_lab_instead_of_repinning tests/test_benchmark_contracts.py::test_contexts_share_exact_target_dates_and_consecutive_inputs tests/test_benchmark_contracts.py::test_alignment_handles_gaps_without_short_context_advantage tests/test_benchmark_contracts.py::test_external_ap_matches_sklearn_across_chunks_ties_and_merge_levels tests/test_unet_data.py::test_normalization_fits_only_train_inputs_not_targets_or_heldout tests/test_unet_data.py::test_nan_inf_angles_and_categorical_unknowns_have_explicit_encoding -q -p no:cacheprovider --basetemp=artifacts/scientific-improvements-v1/pytest-delivery --junitxml=reports/scientific_improvements_v1/tests-delivery.xml
```

Cette commande est la trace de l'exécution ; pour une nouvelle exécution, employer un **nouveau** répertoire temporaire et un **nouveau** nom JUnit.

## Fichiers préparés

| Fichier | Rôle |
|---|---|
| `docs/SCIENTIFIC_IMPROVEMENTS_V1.md` | Audit, sources primaires, métriques, protocole A→E, limites et décision |
| `configs/scientific_improvements_v1.json` | Contrat inactif, critères et prérequis préenregistrés |
| `src/geofirex/experiment_readiness.py` | Validations isolées de métadonnées ; aucun branchement au trainer ou Studio |
| `scripts/audit_scientific_improvements.py` | Agrégation déterministe de résultats sauvegardés, refus d'écrasement |
| `tests/test_experiment_readiness.py` | Tests légers nouveaux, sans données réelles |
| `tests/test_unet_data.py` | Une ligne de fixture corrigée : racine explicite du manifeste synthétique |
| `reports/scientific_improvements_v1/` | Preuves et livrables de cette mission uniquement |

Dans ce dossier : [métriques micro/macro et diagnostics](saved_metrics_audit.json), [507 lignes événement/modèle](per_event_saved.csv), [23 variables et disponibilité](feature_availability_matrix.csv), [état des données](readiness.json), [coût](gpu_budget_scenarios.json), [sources locales hashées](sources.json), figure [performance_saved.png](performance_saved.png) et son code [plot_saved.py](plot_saved.py), empreintes de préservation et JUnit. Les deux autres JUnit réussis conservent la trace des contrôles intermédiaires ; le résultat final est `tests-delivery.xml`.

Les quatre fichiers non suivis présents avant la mission dans `reports/gpu_operational_v1/` et `reports/recovery_20261107_v1/` sont conservés. Aucun staging n'a été effectué.

## Reproduire les agrégations sans nouvelle évaluation

Choisir des destinations nouvelles :

```powershell
.\.venv\Scripts\python.exe scripts/audit_scientific_improvements.py --out reports/scientific_improvements_review_next
.\.venv\Scripts\python.exe reports/scientific_improvements_v1/plot_saved.py --audit reports/scientific_improvements_review_next --output reports/scientific_improvements_review_next/performance_saved.png
```

Ces commandes lisent des métriques et métadonnées enregistrées. Elles n'ouvrent aucun pixel réel, ne chargent aucun checkpoint et ne lancent aucun entraînement. Le contrôle de chronologie vérifie la structure des preuves, pas leur véracité externe ; sa réussite n'active aucun profil.
