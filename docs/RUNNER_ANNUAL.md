# Runner commun B0/B1/B2/B3 — annual-v2-runner

Ce runner ajoute une chaîne entraînable au protocole préparatoire annual-v1.
Les anciens scripts, checkpoints et résultats restent distincts. Aucun modèle
n'est déclaré meilleur. Le backend FastAPI, React, Spark distribué et les
modalités non auditées ne sont pas implémentés par cette étape.

## Contrats et audit

| Élément | Contrat réellement implémenté |
| --- | --- |
| Entrée/sortie | `[B,T,8,H,W]` chronologique → logits `[B,1,H,W]`, probabilités via `predict_proba` |
| B0 | Dernier masque feu exact, aucun optimizer ; évaluation déterministe |
| B1 | CNN minimal : régression logistique convolutionnelle 3×3, temps empilé ; architecture historique conservée |
| B2 | LightUNet spatiale avec T concaténé aux canaux, paramètres `model_kwargs.base_channels/depth` |
| B3 | Encodeur stride 2, cellule ConvLSTM récurrente, tête et interpolation ; paramètre `hidden_channels` |
| Modalités | Pente/altitude normalisées + indicateurs manquants, sinus/cosinus/aspect manquant, feu J ; LC/prévisions exclues |
| Géométrie | Tailles variables entre événements ; forme/CRS/transform constants au sein d'un événement |
| NaN feu | Absence de détection **valide**, conformément aux données historiques ; un code fini est une détection |
| Validité | Masques GDAL explicites dataset/alpha et nodata fini de la bande feu ; padding exclu |
| Masque commun | Intersection des empreintes cible + contexte maximal T=5, même domaine pour T=1/3/5 |
| Normalisation | Moments stables en float64 sur les pixels valides des chemins d'entrée train T=5 uniques ; jamais les cibles seules/val/test |
| Sampling | Grille uniforme avec remise, seed/epoch/index indépendants des workers ; coordonnées sauvegardées |
| Déséquilibre | `pos_weight=auto` : négatifs/positifs des cibles train harmonisées, plafonné explicitement ; BCE+Dice masquée |

Différences avec le préparatoire : sampling uniforme déclaré plutôt qu'une
fraction de patches positifs proposée ; nouveau domaine de validité explicite ;
évaluation à chaque epoch ; identifiant `annual-v2-runner` distinct. B1 est une
baseline de très faible capacité, pas un CNN profond. B2 n'est pas récurrente.
Les capacités différentes empêchent d'attribuer un écart à la seule temporalité.

L'ancien `WindowStore` n'utilise que `context_paths[0]` et l'ancien trainer est
T=1 : ils ne peuvent pas être utilisés comme adaptateur temporel. Ils sont
conservés. Le nouveau reader lit uniquement les bandes nécessaires, avec
un LRU RAM de bandes physiques et masques, borné à `reader_cache_mb: 64` par
store/processus. Les rasters plus grands que ce budget utilisent des fenêtres
Rasterio ; `reader_cache_mb: 0` désactive le cache. Aucun tenseur des 23 bandes
ni du corpus entier n'est mis en cache. Le cache physique sur disque
existant n'est pas branché : son format ne conserve pas les masques de validité.
Un cache futur doit préserver ces masques avant d'être interchangeable.

## Démarrage local borné

Depuis la racine, dans l'environnement installé (`pip install -e . --no-deps`) :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_benchmark.py inspect --config configs/runner_debug.yaml
.\.venv\Scripts\python.exe scripts/run_benchmark.py train --config configs/runner_debug.yaml --model B1 --device cpu
```

Les modèles se sélectionnent par `--model B0|B1|B2|B3`, avec `--fold`, `--seed`,
`--context 1|3|5`, `--device cpu|cuda|auto`. `--fold` choisit automatiquement le
CSV annuel correspondant, sauf `--split` explicite. Le fold est vérifié contre
les 12 définitions annuelles. Les autres hyperparamètres sont dans le YAML ;
les clés inconnues sont refusées. Les chemins relatifs CLI/YAML sont résolus
depuis la racine du dépôt, indépendamment du répertoire courant.

`inspect` lit les métadonnées et détecte le device, sans entraînement ni lecture
des pixels. `train` vérifie les fichiers train/val et leur provenance avant les
gradients. Le mini-overfit travaille sur un patch positif train, puis son modèle
est jeté et la seed réinitialisée. Il vérifie une diminution de loss, **pas** une
convergence ni une généralisation. Les tests automatisés exercent les trois
architectures ; aucun entraînement GPU n'est implicite.

Les cinq événements locaux servent au diagnostic. Au fold 0 : 41 cibles train,
42 validation, **0 test** après alignement. Un run diagnostique peut entraîner et
valider, mais `final-test` refusera ce test absent. Ne pas reconstruire un split
aléatoire pour contourner ce manque.

## Entraînement et reprise

Chaque epoch effectue le même nombre de mises à jour pour B1/B2/B3 lorsque les
configurations ont le même budget/batch. Le coût, les paramètres, la prévalence
effectivement vue et la couverture des événements sont rapportés séparément.
Les derniers batches partiels comptent comme une mise à jour. Pas de scheduler,
early stopping ou HPO implicite. La validation complète a lieu à chaque epoch.

AMP FP16 + GradScaler est activé uniquement en CUDA ; CPU reste FP32. Les
réductions de loss sont FP32. Le mode déterministe PyTorch est demandé et une
opération non déterministe non supportée échoue explicitement. Des versions ou
matériels différents peuvent donner des résultats différents.

```powershell
python scripts/run_benchmark.py train --config configs/runner_debug.yaml --model B1 --resume "artifacts/benchmark/<run>/epoch_0001.pt"
```

La reprise crée un **nouveau run**. Elle restaure poids, optimizer, scaler, états
Python/NumPy/PyTorch/CUDA, meilleur epoch et historique. Une interruption au
milieu d'un epoch redémarre cet epoch depuis le dernier checkpoint complet ;
les gradients partiels sont abandonnés. Les checkpoints sont publiés par rename
atomique. Source/configuration scientifique/données/normalisation doivent
correspondre exactement. Le budget d'epochs ne peut pas être augmenté en reprise.
Les chemins des données doivent rester stables. Un changement CPU/CUDA ou AMP
est enregistré dans la nouvelle provenance et ne garantit pas l'identité
numérique ; seul le scénario CPU identique est vérifié bit à bit localement.
Charger uniquement des checkpoints locaux de confiance (`torch.load`).

## Évaluation et sélection

Chaque scène est couverte une seule fois par des tuiles non recouvrantes ancrées
en `(0,0)`, padding éliminé. Les pixels valides de **toutes** les tuiles sont
poolés : exactement la même population que les scènes reconstruites. Les joints
et l'effet GroupNorm font partie du protocole ; aucune équivalence au forward
global ou au recouvrement n'est prétendue.

L'AP principale est l'Average Precision micro exacte par tri/fusion sur disque,
avec regroupement global des égalités. Aucun calcul ne moyenne les AP de
patches pour sélectionner le checkpoint. Le meilleur AP validation gagne ; le
premier epoch gagne une égalité. Pour ce checkpoint, le seuil maximise Dice
micro sur la grille validation du YAML ; le seuil le plus élevé gagne une
égalité. Une validation sans positifs échoue au lieu de sélectionner au hasard.

JSON : AP micro, Dice, IoU, Precision/Recall, TP/FP/FN, pixels/prévalence/Brier,
AP poolée par événement puis macro avec nombres définis, macro par fenêtre et
sous-ensembles positifs. Sans positifs, AP/Recall sont null ; sans prédiction,
Precision est null ; deux masques vides donnent Dice/IoU=1. Les faux positifs
des fenêtres vides restent dans l'AP globale. Nouvelles ignitions : domaine
valide `fire_J=0`, mêmes scores et seuil verrouillé, AP et confusion séparées.
Persistence y a toujours TP=0 ; son AP constante vaut la prévalence si des
positifs existent, son rappel est nul.

Le test n'est ouvert que par la commande explicite `final-test` après verrou :

```powershell
python scripts/run_benchmark.py lock --runs <runB1seed42> <runB1seed43> <runB1seed44> <runB2seed42> <runB2seed43> <runB2seed44> --out artifacts/selection_fold00_v2.json
python scripts/run_benchmark.py final-test --run <run_selectionne> --lock artifacts/selection_fold00_v2.json
```

Ajouter B3 et B0 à la liste réelle des candidats avant de conclure sur les quatre
modèles. Le verrou ne découvre pas automatiquement les runs absents. En mode
scientifique : seeds déclarées 42/43/44 par architecture/T (surcharge explicite
`--seeds` possible pour un protocole préenregistré), même budget, données,
normalisation, source et cibles. Classement par moyenne d'AP validation et
écart-type, jamais par test. B0 est exécuté une fois ; son score déterministe
est répété uniquement pour l'agrégation des seeds (écart-type nul). Fournir un
seul B0 par contexte, pas trois répliques artificielles. Un verrou diagnostique
à un candidat est permis et conserve sa qualification.

Le verrou stocke classement, candidats, coût, hashes des artefacts sélectionnés
et checkpoint/config/seuil/normalisation. Le test vérifie verrou, hashes,
sources exécutées et identités des manifestes, puis audite les TIFF test. Les
statistiques ne sont jamais réajustées. Les résultats test ont un nouveau dossier
et un nouveau run MLflow. Ce verrou est une barrière opérationnelle, pas une
signature cryptographique empêchant un utilisateur de modifier volontairement
le protocole. Après consultation du test, ne plus ajuster cette comparaison.

## Passage aux 607 événements et aux douze folds

1. Acquérir uniquement avec budget autorisé, conserver l'index épinglé et
   construire un **nouveau** manifeste complet versionné, sans remplacer le sample.
2. Vérifier les 607 événements/13 607 dates, tailles/CRC de l'archive,
   descriptions et géométries. Le mode `scientific` refuse un manifeste partiel
   et exige les contrôles taille/CRC des fichiers lus ; le test est audité après
   sélection. Les SHA256 effectifs et fichiers auxiliaires GDAL sont enregistrés.
3. Examiner les recouvrements spatiaux d'événements différents à partir des
   CRS/transforms/shapes du rapport. Les chemins et fichiers identiques entre
   splits sont refusés, **pas tous les recouvrements géographiques**. Ce dernier
   contrôle reste une revue scientifique nécessaire : les folds annuels ne
   garantissent pas une indépendance spatiale.
4. Exécuter les tests et un pilote court par architecture/T sur le GPU cible,
   mesurer débit I/O, tri AP, RAM/disque et convergence. La configuration
   `runner_annual.yaml` propose un budget, sans le prétendre optimal ou autorisé.
5. Fixer budget/HPO identiques, grille, canaux, seeds et modalités avant les
   comparaisons. Pour chaque fold, repartir de poids neufs et recalculer ses
   statistiques ; ne jamais transférer les poids de l'ancienne U-Net.
6. Lancer explicitement les combinaisons autorisées avec `--fold N --model Bn
   --context T --seed S`. Stocker les sorties sur volume persistant. Aucun CLI
   n'effectue automatiquement les 324 entraînements.
7. Verrouiller les candidats sélectionnés sur validation par fold, exécuter les
   seeds sélectionnées sur test, conserver toutes les sorties et rapporter la
   distribution annuelle. Les folds partagent des événements entre rôles : ils
   ne sont pas douze répétitions indépendantes. Ne pas choisir le HPO à partir
   des douze tests. Les intervalles éventuels rééchantillonnent les événements,
   pas les pixels ; aucun intervalle n'est fabriqué par ce runner.

Le tri AP peut coûter beaucoup plus que les gradients ; les fichiers `.bin`
restent dans chaque run local pour audit, sans upload MLflow/export par défaut.
La mémoire dépend du batch de patches, pas du nombre total de pixels ; les listes
de tuiles et chemins de runs de tri restent proportionnelles au nombre de tuiles.
Le débit du corpus complet, multiworker et des volumes cloud reste à mesurer.
Compter un cache par store et par worker ; les tableaux chauds et handles GDAL
ne sont pas sérialisés dans les workers. Les tests vérifient l'identité des
patches avec/sans cache, l'éviction et la borne en octets.

## Vérifier et exporter les résultats

```bash
python scripts/verify_and_export_benchmark.py --runs artifacts/benchmark/<run> --export-dir artifacts/exports/session-unique --report artifacts/exports/session-unique.json
python scripts/verify_and_export_benchmark.py --runs artifacts/benchmark/<run> --archive-dir artifacts/exports/session-unique
```

Le script est versionné et générique : les fenêtres et paramètres sont lus dans
les métadonnées du run, sans dépendre des quatre diagnostics locaux. Il contrôle
les sources sauvegardées (pas le code courant), les checkpoints par SHA256,
MLflow et chaque fichier de l'archive, sans ouvrir les TIFF ou le test. Les
exports, sidecars SHA256 et rapports existants ne sont jamais écrasés. Un
rapport public ne copie pas les chemins absolus des configurations privées.
Les archives privées conservent cependant la provenance intégrale : ne pas les
ajouter à GitHub sans revue spécifique. Voir [RUNNER_MLFLOW.md](RUNNER_MLFLOW.md).
