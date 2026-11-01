# Préparation de publication du runner — 31 octobre 2026

Branche : `feature/unet-baseline`. Aucun commit, push, merge, téléchargement de
corpus ou entraînement long effectué. Les seuls entraînements de cette étape
sont les fixtures minuscules de la suite automatisée, sans données réelles.

## Audit du script ignoré

`artifacts/benchmark/verify_and_export.py` est un script ponctuel exécutant son
travail dès l'import. Il importe `json`, `tarfile`, `Path`, `hashlib`, le client
MLflow, `geofirex.corpus.sha256_file` et `geofirex.runner.export_run`. Il ciblait
quatre noms de runs exacts, 42 fenêtres validation, quatre mises à jour, l'absence
de test 2021 et un rapport historique daté. Une seconde exécution aurait refusé
les exports déjà présents. Aucun autre script ou notebook ne l'appelait ; ce
n'était pas une dépendance de `run_benchmark.py`.

Son contrôle réutilisable a été porté dans
`scripts/verify_and_export_benchmark.py`, avec fonctions importables, garde
`__main__` et arguments CLI. Le prototype original est conservé dans les artefacts
locaux, toujours ignorés. `.gitignore` n'a pas été modifié et aucun ajout forcé
d'artefacts à Git n'est nécessaire.

Le nouveau script accepte les runs choisis par l'utilisateur ; il n'impose
aucun nom de run, année, seed, nombre de fenêtres ou budget local. Il vérifie
checkpoint, cohérence résumé/validation, inventaire des sources archivées,
provenance et statut MLflow. Il ne lit aucun raster, ne désérialise pas les
checkpoints et ne modifie pas les runs MLflow. Il fonctionne aussi hors ligne
avec `--skip-mlflow`, statut alors explicitement non vérifié.

`--export-dir` crée des archives neuves ; `--archive-dir` vérifie celles qui
existent déjà. Taille et SHA256 de **chaque fichier** sont contrôlés en streaming,
sans extraction. Les rapports/export/sidecars existants sont refusés avant
écriture ; les collisions d'un lot sont contrôlées avant sa création. La commande
simple `run_benchmark.py export` reste disponible pour les runs interrompus.

Commandes et références ont été adaptées dans les guides annuel, Kaggle et MLflow.
Les rapports publics du vérificateur contiennent basenames, métriques et hashes,
sans recopier les configs privées ou leurs chemins absolus. Les archives de
reprise conservent leur provenance intégrale et restent hors du dépôt public.

## Vérifications exécutées

- Suite complète : **142 passed, 642 warnings, 39,72 s**. Les 134 tests existants
  sont conservés ; huit cas supplémentaires vérifient l'export générique, les
  corruptions, fichiers manquants, changements de sources/checkpoint, collisions,
  comparabilité et le mode MLflow hors ligne.
- Quatre archives historiques B0/B1/B2/B3 : **381 fichiers** contrôlés, sans
  réécriture. Résultats dans
  [RUNNER_EXPORT_RECHECK_2026-10-31.json](RUNNER_EXPORT_RECHECK_2026-10-31.json).
- Quatre nouveaux exports créés sous
  `artifacts/publication-prep/2026-10-31/exports` : 381 fichiers contrôlés,
  rapport privé `new_exports_verification.json` dans le dossier parent.
- Les SHA256 des quatre archives originales et des quatre checkpoints restent
  égaux à ceux du rapport historique du 30 octobre. Les métriques entre les
  vérifications des anciens/nouveaux exports sont identiques ; les quatre runs
  MLflow sont toujours FINISHED. Aucun test réel n'a été réévalué.
- Les avertissements sont ceux de Rasterio/SQLAlchemy déjà observés, sans échec.
  `git diff --check` a réussi.

Exemple reproductible depuis la racine du checkout, après installation du projet :

```bash
python scripts/verify_and_export_benchmark.py --runs artifacts/benchmark/<run> --export-dir artifacts/exports/session-unique --report artifacts/exports/verification-unique.json
python scripts/verify_and_export_benchmark.py --runs artifacts/benchmark/<run> --archive-dir artifacts/exports/session-unique
```

Sur Kaggle, remplacer les chemins par les montages `/kaggle/working` documentés.
Le vérificateur ne sélectionne ni architecture ni checkpoint : il vérifie les
résultats existants. Les chiffres historiques restent diagnostiques, sans
classement scientifique sur les 607 événements.

## Confidentialité et préservation

Les fichiers texte de `configs/` et `reports/` ont été recherchés pour les chemins
de profils personnels, affectations de credentials, clés privées et signatures
de tokens de services. Aucun secret ou chemin personnel résiduel détecté après
correction. Les chemins `/workspace` et `/kaggle` sont des montages génériques
utiles et restent dans les exemples. Aucun secret n'est nécessaire dans les YAML.

Une seule modification d'un ancien rapport suivi : `output_dir` dans
`reports/unet_baseline_v1_results.json` est devenu relatif au dépôt. Tous ses
autres champs et métriques sont identiques. Sa copie originale est conservée
sous `artifacts/publication-prep/2026-10-31/unet_baseline_v1_results.original.json`.
Les checkpoints, dossiers de runs, anciennes archives, bases historiques et
prototype ignoré n'ont pas été modifiés. Le rapport historique `RUNNER_SUIVI_3.md`
conserve ses résultats d'origine et référence le nouvel outil versionné.

Le contrôle des fichiers à publier ne constitue pas un audit des anciens
commits Git ou des backends cloud. Aucun stockage distant n'a été contacté.

## Liste exacte des fichiers à commiter

Liste complète pour publier le runner préparé lors du suivi 3 et cette correction
(18 nouveaux fichiers et un rapport historique modifié). Aucun fichier n'a été
stagé par cette intervention.

```text
configs/runner_annual.yaml
configs/runner_debug.yaml
configs/runner_kaggle.yaml
docs/KAGGLE_GPU_RUNNER.md
docs/RUNNER_ANNUAL.md
docs/RUNNER_MLFLOW.md
reports/RUNNER_EXPORT_RECHECK_2026-10-31.json
reports/RUNNER_PUBLICATION_PREP.md
reports/RUNNER_SUIVI_3.md
reports/RUNNER_VALIDATION_2026-10-30.json
reports/unet_baseline_v1_results.json
scripts/gpu_preflight.py
scripts/run_benchmark.py
scripts/verify_and_export_benchmark.py
src/geofirex/runner.py
src/geofirex/runner_data.py
src/geofirex/runner_evaluation.py
tests/test_benchmark_export.py
tests/test_runner.py
```

Ne pas inclure `artifacts/`, checkpoints, archives, bases MLflow, venv, données
brutes ou secrets locaux. Aucun changement de `.gitignore` n'est nécessaire.
