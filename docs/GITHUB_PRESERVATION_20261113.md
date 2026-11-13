# Sauvegarde scientifique avant refonte — 13 novembre 2026

Cette préparation ne vaut ni commit, ni push, ni sauvegarde externe réalisée.
La publication vise exclusivement `feature/scientific-scaleup-v1`.

## Ce que GitHub conservera

Les sources Python (prétraitement, DataLoader, modèles, entraînement, évaluation,
acquisition et orchestration), FastAPI, Prediction Lab, React, les configurations,
tests, notebooks, rapports JSON/CSV/Markdown, figures et petites tables Parquet.
Les fichiers déjà suivis et inchangés, dont les splits annuels et notebooks
historiques, restent dans l'historique. Les sources exactes d'expériences sont
conservées dans de petites archives de code auditées. L'archive des résultats de
calibration est également publiée : elle contient les résultats/cartes, aucun
checkpoint. Les index NumPy train/validation sont de petites métadonnées requises
par l'inférence, pas les données d'entraînement.

La liste exhaustive à publier est `reports/github_preservation_20261113/publication_files.txt`.
`publication_manifest.csv` fournit tailles et SHA-256 des contenus sélectionnés
(hors manifestes qui se décrivent eux-mêmes). `excluded_files.csv` inventorie
les exclusions avec leurs raisons. Les fichiers existants ne sont pas supprimés.

`.gitattributes` désactive la conversion des fins de ligne pour les rapports,
configurations et artefacts de comparaison sélectionnés : les SHA-256 doivent
rester identiques après staging et sur un autre système. Les fichiers historiques
inchangés ne sont pas renormalisés durant cette sauvegarde.

## Sauvegarde hors Git indispensable

Un clone seul ne contient pas les poids ni le corpus. Conserver une copie
indépendante avant toute refonte risquant de toucher ces fichiers. Aucun transfert
Kaggle, stockage cloud ou nouvel entraînement n'est effectué par cet audit.
Le support/dossier de sauvegarde externe reste à choisir et sa copie à vérifier.

`off_git_manifest.csv` recense les fichiers scientifiques exclus avec SHA-256.
Les environnements, caches, répertoires temporaires de tests et builds ne sont
pas des résultats scientifiques à restaurer. Les copies de publication sont
signalées ; ne pas les supprimer pendant cette mission.

Éléments indispensables à l'application actuelle :

| Élément | Chemin attendu après restauration |
| --- | --- |
| Poids U-Net, trois ablations et décisions originales | `reports/unet_confirmation_20261112_v1/inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip` |
| Expérience de propagation complète | `artifacts/propagation-comparison/GeoFireX_UNET_PROPAGATION_COMPLETE.zip` |
| Corpus compact et grands manifestes | `reports/mission_20261111/compact_full/` |
| Autres checkpoints, états d'optimiseur, résultats numériques historiques | chemins consignés dans `off_git_manifest.csv` |

Le ZIP d'ablation doit rester intact. Le checkpoint retenu est
`B2_posweight_10/checkpoint_latest.pt`, SHA-256
`7295eaa6f957e59a31b7155403881a9b3bf19933a35b6872768896bda1f1b14e`.
Les trois variantes ont quatre époques et le modèle retenu 23 404 mises à jour.
Le seuil figé est 0,78. `checkpoint_members.json` décrit aussi les autres poids.
Ne jamais remplacer les empreintes dans les configurations pour contourner
une erreur d'intégrité.

Le manifeste de publication du compact
`reports/gpu_acceleration_20261111_v1/publication/plan.json` contient 11 partitions,
les chemins, tailles, empreintes et identités. Il conserve les chemins absolus
historiques comme provenance ; ils ne constituent pas des instructions à exécuter.
Les alias `.bin` sous `publication/p00` à `p10` et le montage de vérification
dupliquent des objets du corpus. Restaurer les chemins logiques attendus par
`configs/prediction_lab_v1.json`. Les données 2021 restent verrouillées.

Pour une restauration à partir d'un dossier de sauvegarde conservant les chemins
relatifs, copier uniquement les fichiers manquants, puis vérifier les empreintes :

```powershell
# Depuis le dépôt cloné ; remplacer ce chemin par le support réellement vérifié.
$backupRoot = 'E:\GeoFire-X-preserved-20261113'
$repoRoot = (Get-Location).Path
$rows = Import-Csv reports/github_preservation_20261113/off_git_manifest.csv
foreach ($row in $rows) {
    if ([IO.Path]::IsPathRooted($row.path) -or $row.path.Split('/') -contains '..') {
        throw "Chemin de manifeste non relatif : $($row.path)"
    }
    $sourcePath = Join-Path $backupRoot $row.path
    $targetPath = Join-Path $repoRoot $row.path
    if (Test-Path -LiteralPath $targetPath) {
        if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLower() -ne $row.sha256) {
            throw "Fichier existant différent : $($row.path)"
        }
        continue
    }
    if (-not (Test-Path -LiteralPath $sourcePath)) { throw "Sauvegarde manquante : $($row.path)" }
    if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLower() -ne $row.sha256) {
        throw "Empreinte source incorrecte : $($row.path)"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath -ErrorAction Stop
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLower() -ne $row.sha256) {
        throw "Empreinte restaurée incorrecte : $($row.path)"
    }
}
```

Ce protocole conserve aussi les données brutes et anciens résultats inventoriés.
La restauration minimale du Lab utilise le corpus compact et les deux archives
ci-dessus ; les grands manifestes `samples.json`/`shards.json` doivent être repris
avec le corpus. L'environnement virtuel se recrée, il ne se copie pas depuis Git.

## Relance et contrôles

`environment_versions.json` et `requirements-environment.txt` décrivent
l'environnement local constaté, sans identifiants. Les dépendances propres au
projet restent déclarées dans `pyproject.toml`. La version CUDA historique des
entraînements est une provenance ; la démonstration utilise PyTorch sur CPU.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,api,benchmark-metadata,gpu-operational]'
Push-Location frontend
npm ci
npm run build
Pop-Location
.\.venv\Scripts\python.exe -m pytest tests/test_prediction_lab.py tests/test_science_results.py tests/test_unet_confirmation.py -q
.\.venv\Scripts\python.exe scripts/serve_demo.py --results-only --port 8014
```

La reproduction des anciennes requêtes Spark demande Java et PySpark ; lire
`reports/full_corpus_v1/spark-installation.json` et le script
`scripts/mission_spark_demo.py`. Consulter les résultats existants ne redémarre
pas Spark. Aucun entraînement ni téléchargement n'est nécessaire pour les
contrôles de l'application, après restauration des artefacts.

## Publication soumise à accord

L'audit distant a constaté `main` et `feature/unet-baseline`, sans branche
`feature/scientific-scaleup-v1`. `HEAD` et `origin/main` sont au même commit
`7428561f607914eb2f7cd286ee50435d142f8e09` (divergence 0/0).
Cette observation est datée ; recontrôler les références avant publication.
Si la branche distante apparaît, s'arrêter et comparer avant tout push.
Une avancée ultérieure de `main` ne justifie pas de modifier `main` ni de forcer
le push de la branche de travail.

Les commandes exactes proposées sont dans
`reports/github_preservation_20261113/COMMANDES_APRES_ACCORD.ps1`.
Elles ne sont exécutables qu'après accord explicite pour le commit et le push.
Le push doit créer/configurer l'amont de la seule branche de travail.

Le contrôle de secrets est une recherche de motifs et une revue des alertes,
incluant le texte des notebooks et archives légères. Il ne prouve pas l'absence
de tout secret dans des données binaires ou chiffrées. Les valeurs suspectes ne
sont jamais reproduites dans les rapports.
