# GeoFire-X

Multimodal Spatio-Temporal Deep Learning for Next-Day Wildfire Spread Forecasting under Geographic and Temporal Domain Shift.

## Etat actuel : Prediction Studio V2

GeoFire-X est un prototype de recherche pour prevoir les detections de feu actif
a J+1 a partir de trois observations passees. Il etudie la generalisation
temporelle et le compromis rappel/fausses alertes ; ce n'est pas un service
operationnel d'alerte, de prediction d'ignition ou de surface brulee.

### Architecture et donnees

React affiche les grilles et appelle FastAPI ; PyTorch execute les modeles sur
CPU. Les lecteurs d'archives verifient les identites SHA-256 avant de presenter
les preuves scientifiques. Les poids, TIFF, compacts et grandes archives restent
externes au depot. Aucun entrainement n'est lance au demarrage.

WildfireSpreadTS fournit 23 bandes brutes. Le protocole compact C8 utilise huit
canaux par jour (terrain, indicateurs de donnees manquantes et feu observe),
empiles sur J-2/J-1/J. Le split V2 chronologique contient 318 evenements train,
53 purges et 80 de validation 2020 ; le test independant 2021 compte 156
evenements. Le normaliseur historique train-only 2018-2019 est fige ; il n'est
pas reajuste sur la validation ou le test. Ces populations V2 ne doivent pas
etre confondues avec les anciens protocoles 250 train / 201 validation.

### Modeles et evaluation

C8, NEWPOS2 et NEWPOS4 sont des U-Net entraines (epoque 4, 28 976 updates,
123 977 parametres, seuil 0,78). NEWPOS2/4 changent la ponderation BCE des
nouveaux positifs, pas la famille d'architecture. FIRE13 est aussi entraine,
mais reste **archive et desactive en inference directe** : deux scenes echouent
a la tolerance de parite CPU initiale `atol=rtol=1e-6`, non assouplie.
Les variantes n'ont pas ete promues et n'ont pas ete evaluees sur 2021.
Le U-Net historique reste une option separee avec son propre protocole.

| C8, seuil 0,78 | Validation 2020 | Test independant 2021 |
| --- | ---: | ---: |
| Evenements / tuiles | 80 / 30 882 | 156 / 75 540 |
| AP globale | 0,3352339037 | 0,4550013955 |
| Dice global | 0,4389254872 | 0,5348317337 |
| Rappel global | 0,4358386785 | 0,5512253057 |
| AP nouveaux feux | 0,1101283730 | 0,1754763634 |
| Dice nouveaux feux | 0,1006963425 | 0,1576373262 |
| Rappel nouveaux feux | 0,0613835115 | 0,1040384448 |

Le test 2021 est **COMPLETE**, consulte en lecture seule depuis RESULTS verifie.
Il ne doit pas etre relance. Les ecarts entre annees sont descriptifs, pas un
gain apparie. Le faible rappel des nouveaux feux reste une limite majeure.

### Interfaces

Cinq espaces : Overview, Studio (individuel ou comparaison appariee),
Performances, Comparaison et Methodologie. Les observations communes, les vrais
calculs CPU, la revelation separee de J+1, les erreurs TP/FP/FN et les exports
PNG/JSON/CSV sont controles. Les cartes sont des grilles de pixels sans CRS
exploitable : pas d'hectares ni de probabilites calibrees revendiques.

### CRISP-DM et Big Data

La [matrice de cloture CRISP-DM](docs/CRISP_DM_CLOTURE_20261127.md) distingue
objectifs, experiences, entrees/sorties, preuves et criteres d'acceptation.
Les cinq phases scientifiques sont PASS dans le perimetre experimental fige ;
Deployment est PARTIAL (installation vierge et plein ecran interactif a valider).

Spark 4.0.2 a execute les requetes SQL de metadonnees en **local[2]**, sur une
machine. Cinq petits agregats ont ete materialises en Parquet apres `collect`
par PyArrow. L'ingestion utilise les inventaires d'observations et de fenetres,
pas un nouvel examen des pixels 2021. Ce pipeline historique 250/201/156 aide
l'audit des donnees du Deep Learning ; ce n'est pas un entrainement distribue
ni la preuve d'un traitement Spark des pixels du protocole V2.
Preuve : [rapport execute](reports/mission_20261111/spark/report.json).

### Installation, artefacts et tests

Environnement valide : Python 3.11.6, Node.js 22.12.0. Les dependances Python
sont figees dans [requirements-studio-cpu.txt](requirements-studio-cpu.txt),
les dependances React dans [frontend/package-lock.json](frontend/package-lock.json).
Installer puis restaurer les artefacts exacts avant de qualifier les modeles :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-studio-cpu.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
Push-Location frontend
npm.cmd ci
Pop-Location
# Restaurer ADATA et les metadonnees externes selon le guide Deployment.
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe scripts/qualify_studio_v2.py
.\Start-GeoFireX.ps1 -Port 8028
```

Ouvrir http://127.0.0.1:8028/#/overview. Le port par defaut est 8026 ; ne pas
confondre une instance plus ancienne avec la release courante. La qualification
sur 2020 doit donner `PASS_WITH_ARCHIVED_FIRE13` dans l'environnement valide.
Configurer `GEOFIREX_V2_ROOT`, `GEOFIREX_COMPACT_ROOT` et, si necessaire,
`GEOFIREX_RESULTS_ARCHIVE`. Sans ADATA les calculs indisponibles echouent
explicitement ; aucune prediction archivee ne remplace une inference.

Le [guide Deployment](docs/DEPLOYMENT_STUDIO_V2.md) donne les chemins, empreintes,
dependances supplementaires, endpoints, erreurs, sauvegardes et maintenance.
Une installation sur machine vierge **n'a pas ete effectuee** ; le clone seul
ne contient ni corpus, ni poids, ni RESULTS ni verrou durable local 2021.

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe -m pytest tests/test_studio_v2_deployment.py tests/test_studio_fire13.py tests/test_studio_populations.py tests/test_ui_performance.py tests/test_studio_spatial.py tests/test_prediction_lab.py tests/test_jury_finalization.py tests/test_science_results.py::test_big_data_actual_parquet_values_and_provenance -q
$env:GEOFIREX_TEST_URL = 'http://127.0.0.1:8028'
Push-Location frontend
npm.cmd run build
npm.cmd run test:studio
Pop-Location
```

Les tests couvrent CPU 2020, provenance, refus 2021, ADATA absent, navigation et
exports. Ils ne relancent pas l'evaluation 2021. Voir le
[bilan de publication](docs/PUBLICATION_20261127.md) pour les recus et reserves.
Perspectives : qualification numerique FIRE13 sans changer le critere,
validation d'installation restauree, calibration et generalisation sur de
nouveaux jeux autorises, sans reutiliser 2021 pour selectionner un modele.

## Archives historiques

Les sections repliees ci-dessous preservent les anciennes commandes et les
anciens statuts (dont « test ferme »). Elles ne decrivent pas la release actuelle.

<details>
<summary>Historique du projet et anciennes procedures</summary>

## Prediction Studio V2 — premiere documentation de finalisation

```powershell
.\Start-GeoFireX.ps1
```

Ouvrir **http://127.0.0.1:8026/#/overview**. Le lanceur compile React puis démarre
FastAPI. Il ne lance aucun entraînement, test 2021 ou calcul Spark.
En cas de port occupé : `.\Start-GeoFireX.ps1 -Port 8028`.

Studio exécute réellement sur CPU **C8 V2**, **NEWPOS2**, **NEWPOS4** et conserve
le **U-Net historique** comme option séparée. Les variantes restent expérimentales,
sans promotion scientifique. V2 est limité aux 80 événements de validation 2020 ;
le catalogue historique de 201 événements conserve son ancien protocole.

Les trois observations J−2/J−1/J, les probabilités J+1, les nouveaux pixels de feu,
la vérité révélée séparément, les erreurs TP/FP/FN, le zoom et les exports sont
disponibles dans Studio. La comparaison V2 permet trois recalculs CPU sur une même
scène ; les sorties enregistrées NEWPOS2/NEWPOS4/FIRE13 sont présentées séparément.
Overview utilise `feu.png` comme illustration décorative, sans valeur prédictive.

**Le test indépendant C8 2021 est terminé (COMPLETE)** : 156 événements et
75 540 tuiles. Son panneau est en lecture seule et vérifie le SHA-256 officiel de
RESULTS.zip. AP globale **0,4550013955**, Dice **0,5348317337**, rappel
**0,5512253057** ; nouveaux feux : AP **0,1754763634**, Dice **0,1576373262**,
rappel **0,1040384448**. Aucun modèle expérimental n'est déclaré testé sur 2021.
Le test ne doit jamais être relancé et son verrou durable est conservé.

La qualification compare les vraies inférences 2020 aux références archivées,
avec classes identiques à 0,78. Pour C8, la référence disponible est une tuile
diagnostique scellée ; son archive ne contient pas de matrices complètes.
NEWPOS2/NEWPOS4 sont aussi vérifiés sur trois scènes complètes. Les limites de
ces preuves et les versions utilisées figurent dans les reçus de qualification.

Installation, chemins ADATA, commandes de tests, exploitation et maintenance :
[guide Deployment Studio V2](docs/DEPLOYMENT_STUDIO_V2.md).
Les dépendances CPU sont figées dans `requirements-studio-cpu.txt`, celles du
frontend dans `frontend/package-lock.json`. Les modèles ne sont activés qu'avec
une qualification CPU correspondant aux sources et aux versions Torch/NumPy.

Les grilles compactes n'ont pas de CRS/affine disponible. Aucune conversion en
hectares et aucune probabilité calibrée ne sont revendiquées. Le faible rappel
des nouveaux feux reste une limite scientifique majeure.

Les sections et procédures plus anciennes ci-dessous décrivent leurs états
historiques. Le guide Deployment V2 ci-dessus fait autorité pour le lancement
actuel ; les anciennes mentions « test fermé » ne décrivent plus son statut.

## Référence historique — Studio avant la finalisation jury

GeoFire-X dispose d'une application **FastAPI + React fonctionnelle** avec une
inférence réelle J+1 sur les événements historiques de validation 2020.
Le modèle principal est **U-Net poids 10, seuil figé 0,78**. Les entraînements
compacts utilisent **250 événements train (2018–2019)**, quatre époques et
23 404 mises à jour par entraînement. Les 201 événements de validation 2020
se répartissent en 32 de réglage et 169 de confirmation désormais observés.
L'inventaire officiel couvre 607 événements ; **aucun pixel du test 2021 n'est ouvert**.

L'interface Studio V2 « Flamme solaire » conserve les routes et calculs existants et organise les
parcours en cinq espaces : **Overview**, **Prediction Studio**, **Model Analytics**,
**Big Data Observatory** et **Research**. Elle conserve les cartes raster,
agrandissements et exports PNG/JSON/CSV, comparaisons CNN/U-Net, ablations,
propagation et résultats Spark/Parquet déjà calculés. React 19.3.1, esbuild et
le CSS classique restent en place ; les effets React Bits respectent la réduction
des animations et disposent également d'une commande dans la barre supérieure.
Le thème est clair par défaut ; les composants officiels Aurora, SplitText,
CountUp, SpotlightCard, FadeContent, Stepper et AnimatedList sont activés selon
les préférences de mouvement. Les grilles et largeurs validées en V1 sont conservées.

Les cartes du compact sont des **grilles de pixels sans CRS disponible**, pas des
cartes géographiques. La vérité J+1 est lue séparément après l'inférence.
La tête spécialisée de propagation est une expérience comparative **non retenue**.
La plateforme présente des résultats de recherche historiques, sans prétention
de système opérationnel d'alerte ni d'incertitude probabiliste calibrée.

### Lancer l'application actuelle, sans entraînement

Après restauration des artefacts locaux requis :

```powershell
Push-Location frontend
npm ci
npm run build
Pop-Location
.\.venv\Scripts\python.exe scripts/serve_demo.py --results-only --port 8014
```

Ouvrir **http://127.0.0.1:8014/#/overview**. Le mode `--results-only` conserve
l'inférence réelle du Prediction Lab ; il ne requiert pas le snapshot TIFF du
Predictor historique. Il ne déclenche ni entraînement ni calcul Spark.

- [Résultats U-Net confirmés](reports/unet_confirmation_20261112_v1/RESULTATS_CONFIRMES.md)
- [Diagnostic des modèles historiques](reports/calibration_20261112_v1/DIAGNOSTIC.md)
- [Préservation des modèles et restauration](docs/GITHUB_PRESERVATION_20261113.md)
- [Architecture et validation de Studio V1](docs/UI_STUDIO_V1.md)
- [Studio V2 : animations, tests et checklist Chrome](docs/UI_STUDIO_V2.md)

## Historique — préparation, premiers modèles et pilote 108 TIFF

Les étapes suivantes documentent l'état des premières phases, avant les
entraînements compacts et l'application actuelle décrits ci-dessus. Leurs preuves
et résultats sont conservés ; leurs commandes ne sont pas nécessaires à la
démonstration actuelle.

**Suivi 3 — GPU pilot completed (1 November 2026).** B1 CNN, B2 U-Net and B3 ConvLSTM
each completed 10 epochs / 1,280 updates on the certified 108-TIFF sample.
B1 is the demonstration candidate (validation AP 0.3535); Persistence retains
slightly higher Dice (0.4814 vs 0.4800). These are pilot results, one seed,
three training events and two validation events; test 2021 remains locked.

- [Demonstration, installation and restored MLflow](docs/DEMO_SUIVI3.md)
- [GPU results and verified backup](reports/GPU_PILOT_2026-11-01.md)
- [GitHub publication and integration plan](docs/GITHUB_SUIVI3.md)

The following preparation and CPU sections describe historical work. Do not
rerun their acquisition or training commands to launch the current demonstration.

Benchmark preparation now extends the project to all 607 official events:
[delivery report](reports/BENCHMARK_PREPARATION_REPORT.md),
[verified U-Net audit](reports/UNET_AUDIT_V2.md),
[acquisition and storage budget](docs/FULL_CORPUS_PLAN.md), and
[multi-model protocol](docs/BENCHMARK_PROTOCOL.md).
Annual splits and B0–B3 prototypes are prepared; the full corpus has not been
downloaded or trained. Large acquisition and long training require a reviewed budget.

Suivi 3 — First trained lightweight 2D U-Net for J+1 active-fire segmentation.
The five-event sample supports technical verification only. The first short
training performs substantially worse than Persistence; all measured results
are retained in [the training report](reports/UNET_BASELINE_REPORT.md).

The preceding real-data spatio-temporal pipeline remains documented in
`reports/REAL_PIPELINE_REPORT.md`.

### U-Net baseline (real data, CPU or CUDA)

Use the project environment on Windows (Python 3.11), not the global Python:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -u scripts/train_unet.py --config configs/unet_baseline.yaml --device cpu
```

See [the audited protocol and inference commands](docs/UNET_BASELINE.md).
Training creates a unique artifact directory and a new run in
`geofirex-unet2d-real-v1`. Raw rasters, historical manifests/splits and previous
experiments are preserved. Checkpoints and MLflow artifacts stay outside Git.
The conservative input policy uses current fire and terrain; annual land cover
has an explicit optional one-hot encoder and is excluded from this first run.

See [Project Context](docs/PROJECT_CONTEXT.md) for the scientific objective,
input and target contracts, scope, and experimental rules. See
[Dataset Version](docs/DATASET_VERSION.md) and
[Data Audit](reports/DATA_AUDIT.md) for the verified real-data facts (23 raw
bands, not 40; active-fire binarization convention; official split
recommendation), and [Split Strategy](docs/SPLIT_STRATEGY.md) for why the
current 5-event real sample only supports a smoke-test split.

### Reproducing the real pipeline (Phase 3)

```bash
pip install -e .

# Fetch a real, deterministic 5-event sample (no full 48.36 GB download)
python scripts/inspect_remote_archive.py --out data/raw/WildfireSpreadTS/archive_index.json
python scripts/download_real_sample.py --dest data/raw/WildfireSpreadTS/real_sample --num_events 5 --seed 42
python scripts/build_real_manifest.py --data_dir data/raw/WildfireSpreadTS/real_sample --out data/manifests/real/manifest_wildfirespreadts.csv

# Persistence baseline on the real DataLoader (no neural network trained)
python scripts/run_real_persistence_baseline.py \
  --manifest data/manifests/real/manifest_wildfirespreadts.csv \
  --split splits/real/smoke_test_split.csv --split_name train --context_days 1 \
  --figures_dir reports/figures/real_baseline --out reports/real_baseline_results.json

pytest -q
```

## Primary Dataset

WildfireSpreadTS

## Application de démonstration — novembre 2026

```powershell
.\.venv\Scripts\python.exe scripts/serve_demo.py --results-only --port 8014
```

Ouvrir http://127.0.0.1:8014 : **Prediction Lab** → événement2020 → date J → **Prédire J+1**.
Le vrai U-Net poids10 est exécuté sur CPU au seuil figé0,78. Le checkpoint,
les sources d'entraînement, le normaliseur et l'archive de l'événement sont vérifiés.
J+1 est lu séparément après la prédiction pour afficher vérité terrain et erreurs.
Les cartes sont des grilles de pixels du compact, sans CRS disponible.

**Propagation J+1** présente l'expérience à tête spécialisée comme comparaison sur32,
non retenue : le modèle principal et son seuil restent inchangés. Les vues historiques,
l'ablation et Spark/Parquet restent disponibles. Le test2021 reste verrouillé.
Le mode du Predictor TIFF historique conserve ses propres contrôles d'intégrité.

Le Lab propose des cartes sur deux colonnes, un agrandissement avec zoom et légende,
la navigation entre dates et les exports PNG par carte, JSON avec provenance et
CSV des pixels de l'aperçu. Le pas spatial est exporté ; les cellules invalides
sont `null` en JSON et vides en CSV. Les coordonnées sont des lignes/colonnes de
pixels, pas des coordonnées géographiques.

Vérification fonctionnelle locale, sans entraînement ni évaluation agrégée :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_prediction_lab.py -q
cd frontend
node verify-prediction-lab.mjs
```

Le test fonctionnel utilise la même routine de requête, la même vue React et le
même rendu des pixels que l'interface. Il vérifie le cas central du 3 septembre
et trois dates sur chacun de deux autres événements. Il ne relance aucune
évaluation agrégée et ne calcule pas de nouveaux scores de validation.

Pour étudier et présenter l'application :

1. **Prediction Lab** : `2020/fire_24332880`, J=`2020-09-03`, cible=`2020-09-04`.
   Distinguer feu observé, feu futur total et nouveaux pixels. Expliquer TP/FP/FN
   avec les légendes ; ouvrir une carte agrandie et exporter son JSON de provenance.
2. **Propagation J+1** : sur les32, la tête spécialisée gagne5,61 points de rappel
   nouveau, mais ajoute13 055 FP nouveaux. Son Dice global et son AP baissent :
   critère d'acceptation faux, modèle principal conservé. Les statistiques169
   plus bas restent un diagnostic séparé.
3. **Ablation U-Net** : même corpus, seed et budget ; faible écart entre poids10
   et30. La sélection utilise les32 ; la confirmation169 a déjà été consultée
   et ne constitue plus un test aveugle pour de nouvelles décisions.
4. **CNN / U-Net historiques** : lire Dice, rappel et fausses alertes ensemble.
   Le seuil modifie la segmentation, pas les probabilités, l'AP ou le Brier.
5. **Big Data** : ouvrir les véritables requêtes Spark SQL et fichiers Parquet.
   Distinguer observations, fenêtres temporelles et tuiles. Montrer ensuite les
   onze partitions, l'index compact et leur lien avec l'inférence. Spark a été
   exécuté en `local[2]`, pas sur un cluster distribué.

Limites à connaître : détection active quotidienne ≠ surface brûlée cumulée ;
pas de vent/météo dans les huit canaux compacts ; masques de J+1 exclus de
l'inférence, ajoutés seulement à la comparaison historique ; test2021 fermé.
La durée du Lab couvre le calcul et la lecture des tuiles après chargement du
modèle, sans représenter le temps complet du navigateur ni un benchmark général.

## Team

4 Data Science / AI engineering students

## Scientific Pipeline

Data Audit → Event-Level Split → DataLoader → Persistence Baseline → Metrics → ID/OOD Evaluation → Multimodal & Temporal Modeling → Robustness

ID/OOD refers to in-distribution and out-of-distribution evaluation.

## Repository Structure

| Directory | Purpose |
| --- | --- |
| `data/` | Local raw, interim, and processed datasets; tracked metadata in `manifests/`. Dataset contents are ignored by Git. |
| `notebooks/` | Exploratory analysis and data audit notebooks. |
| `src/geofirex/` | Shared Python package for reusable project code. |
| `scripts/` | Reproducible command-line workflows. |
| `tests/` | Checks for data contracts, leakage prevention, and project behavior. |
| `splits/` | Versioned event-level train/validation/test assignments. |
| `configs/` | Shared data and experiment configurations. |
| `reports/` | Evaluation reports and figures in `figures/`. |
| `docs/` | Scientific context and project documentation. |

Empty directories contain `.gitkeep` placeholders so Git preserves the structure.
Keep manifests and split definitions small and versioned; keep datasets and
model artifacts outside Git.

`data/manifests/demo/` and `splits/demo/` hold synthetic reference fixtures
(file names end in `_synthetic`) used to develop and test the pipeline before
real WildfireSpreadTS data is downloaded — see `docs/DATASET_VERSION.md`. Real
manifests and splits must use non-`demo` paths so the two are never confused.

## Collaboration Workflow

- Keep `main` stable and ready for the whole team to use.
- Develop each contribution on a focused feature branch created from `main`.
- Submit contributions through a Pull Request with a clear description and relevant validation.
- Have another student review the Pull Request before merging.
- Never commit large datasets, checkpoints, or experiment artifacts to Git.

Assign each wildfire event to exactly one split: no day from a training event
may appear in validation or test. Every experiment must record its dataset
version, split version, configuration, random seed, and Git commit.

## Historique et provenance

Cette version poursuit les travaux du projet GeoFire-X
initialement developpe dans :
https://github.com/nourbenhassine1904/GeoFire-X

Le code et les resultats des suivis precedents constituent
la base de cette nouvelle phase de developpement.
Les contributions historiques restent consultables dans
le depot source.

## Sauvegarde avant refonte frontend

Consulter le [plan de publication et de restauration](docs/GITHUB_PRESERVATION_20261113.md)
et la [selection exhaustive soumise a accord](reports/github_preservation_20261113/PLAN_PUBLICATION.md).
Les checkpoints et corpus restent hors Git ; leurs empreintes sont conservees dans les manifestes.
## GeoFire-X Studio — version finale du 2 décembre 2026

Lancement local : `./Start-GeoFireX.ps1`, puis <http://127.0.0.1:8026/#/overview>.
Le [guide final de six minutes](docs/SOUTENANCE_FINALE_20261202.txt) décrit le lecteur historique, la révélation séparée J+1, les exports et la distinction avec les résultats V2 sauvegardés.
Les contrôles et le manifeste de publication sont dans `reports/final_studio_20261124/`. Aucun commit, push ou merge automatique.

</details>
