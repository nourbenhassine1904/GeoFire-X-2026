# Préparation CPU du candidat C36 V2

Les extensions sont isolées : `v2_c36_readiness.py`, `v2_c36_overlay.py`, `prepare_c36_cpu.py` et `test_v2_c36_cpu.py`. Aucun fichier du manifeste C8 n’est modifié. Il n’existe pas de commande de fit ou de préflight GPU dans cette extension.

`V2Contract.from_repository` contrôle les77 SHA C8, puis dérive les clés et les chemins autorisés des index et observations certifiés. Les dates2017 d’un événement2018 sont admises par leur appartenance réelle. Le rôle provient du manifeste V2. Les chemins absents de l’union des entrées train/val, les traversées et2021 sont refusés avant ouverture.

La commande suivante inspecte les en-têtes, sans lire les pixels. Le dossier de sortie doit être nouveau et la branche dédiée doit être active :

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
./.venv/Scripts/python.exe scripts/prepare_c36_cpu.py --headers --out reports/c36_cpu_NOUVELLE_VERSION
```

Le contrôle des6913 en-têtes a déjà été réalisé dans `reports/c36_cpu_20261121_v1/geometry_headers.json` ; il n’est pas nécessaire de le répéter sans changement des sources. Omettre `--headers` permet une vérification des contrats, chemins et preuves seulement.

## Dossier causal attendu

Le JSON de schéma `geofirex-c36-causal-evidence-1` contient les SHA de `index.json` et `events.csv`, et `records`, exhaustif sur les110106 clés `(event_id, as_of_J, input_date, band)` des six bandes ajoutées. J est la veille de la date cible. Les quatre bandes de base sont héritées sous la justification historique conservée ; aucune nouvelle disponibilité opérationnelle du feu n’est prétendue.

Chaque ligne fournit : `decision_at`, `observation_start`, `observation_end`, `available_at` en UTC explicite ; `kind` (`observed`, `composite` ou `annual`) ; `source_product` (`VNP09GA`, `VNP13A1`, `MCD12Q1`), `product_version`, `asset_id`, `asset_sha256`, `raster_sha256`, `evidence_path`, `evidence_sha256`, `source_uri`, `reviewed=true`, `reviewer`, `reviewed_at`, `justification`. LC déclare aussi `product_year` et son intervalle annuel complet. Le fichier de preuve JSON, effectivement présent et haché, doit confirmer les identifiants, empreintes, produit/version, intervalle, publication et référence primaire. Sa forme doit être reliée aux pièces primaires archivées lors de la revue ; une déclaration synthétique ou une URL de catalogue seule n’est pas recevable.

Un même chemin/bande garde une identité d’asset canonique pour tous ses J référents. La date de disponibilité doit être compatible avec le plus ancien J de cette entrée. Les SHA de rasters sont liés aux certificats existants ; le futur fit statistique vérifie aussi les octets du fichier effectivement lu.

Chaque ligne doit aussi référencer un document primaire archivé **distinct** de sa synthèse JSON : `primary_record_path`, `primary_record_sha256`, `primary_record_locator` (page, ligne ou enregistrement précis). Le validateur vérifie ses octets et la liaison de cette empreinte à la synthèse. Les formats acceptés pour cette pièce sont PDF, HTML, TXT ou JSON de métadonnées, jamais un TIFF ou une archive contenant des pixels. L’archivage et le hash d’une pièce ne prouvent pas à eux seuls la vérité de la date ; la revue nominative doit justifier l’extraction et la pertinence historique de la pièce.

La commande de contrôle d’un dossier disponible est :

```powershell
./.venv/Scripts/python.exe scripts/prepare_c36_cpu.py --evidence CHEMIN_DU_DOSSIER_CAUSAL.json --out reports/c36_evidence_NOUVELLE_VERSION
```

Un résultat `REVIEWED_EVIDENCE_CONSISTENCY_PASS` valide la cohérence et la couverture de pièces préalablement revues ; il ne permet pas au logiciel de certifier leur vérité à partir d’un booléen. Aucun exemple rempli de faux assets réels n’est livré. Les exemples des tests sont explicitement synthétiques et ne qualifient aucune donnée scientifique.

## Primitives disponibles et portes encore fermées

`require_exact_geometry` compare dimensions, CRS normalisé, affine complète, résolution et emprise. Il refuse les écarts ; il ne reprojette pas silencieusement une source. Pour les TIFF déjà sur la grille certifiée, aucune reprojection n’est nécessaire. Pour une autre source, une conversion et son manifeste géométrique/versionné restent nécessaires avant qualification.

`TrainMoments` calcule en float64 des moments de population par blocs, sans padding et une fois par chemin train. `fit_extra_statistics` valide d’abord les preuves causales et un reçu de qualification d’overlay lié au même index et dossier, puis ouvre uniquement les chemins train. Il vérifie les SHA bruts et la géométrie ; des scales/offsets non identitaires exigent une conversion explicite qualifiée. Le résultat sauvegardable contient mean/std/counts, chemins et SHA des sources, clip±8 et SHA des preuves et du normaliseur de base. Aucun normaliseur réel n’a été calculé tant que ces portes sont fermées.

`assemble_c36` conserve les huit canaux C8 bit à bit, ajoute cinq z-scores/missing et18 classes LC, et conserve l’inconnu pour les classes fractionnaires. `parity_stream` vérifie toutes les cibles, masques, feu observé et canaux communs, refuse une couverture incomplète, et reproduit le digest de domaine de `V2Metrics` sans modèle ni probabilités. La parité réelle exhaustive attend toujours un overlay causal et un normaliseur qualifiés.

Le runner C36 n’est pas préparé : cette étape dépend des contrôles réels précédents. Les tests CPU passants ne changent pas ces dépendances.

## Régression CPU immédiatement exécutable

Cette commande crée un dossier de tests unique pour éviter tout nettoyage d’un ancien `--basetemp` :

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
$c36TestRun = Join-Path 'reports' ('c36_tests_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $c36TestRun | Out-Null
./.venv/Scripts/python.exe -m pytest tests/test_v2_c36_cpu.py -p no:cacheprovider --basetemp "$c36TestRun/tmp" --junitxml "$c36TestRun/tests.xml" -q
```
