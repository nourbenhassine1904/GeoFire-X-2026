# Benchmark scientifique WildfireSpreadTS — protocole préparatoire annual-v1

Ce document prépare les expériences ; **aucun benchmark multi-modèles entraîné
sur le corpus complet n'a été exécuté**. La configuration déclarative est
`configs/benchmark_protocol.yaml`. La première U-Net reste un résultat technique
distinct, pas le modèle de référence du corpus complet.

## Matrice de comparaison

Tous les modèles reçoivent `[B,T,C,H,W]` chronologique et produisent des logits
`[B,1,H,W]`. `predict_proba` applique la sigmoid ; B0 renvoie exactement le
masque binaire du dernier jour. Même géométrie, canaux, cibles et protocole
d'inférence au sein d'une comparaison. Aucun classement de qualité n'est présumé.

| ID | Architecture | T à comparer | Paramètres avec C=8 | État réel |
| --- | --- | --- | --- | --- |
| B0 | Persistence, dernier masque J | Même résultat après alignement T=1/3/5 | 0 | Baseline existante ; adaptateur commun testé |
| B1 | Régression logistique convolutionnelle 3×3, T concaténé | 1,3,5 | 73 / 217 / 361 | Prototype forward/backward testé, non entraîné |
| B2 | U-Net 2D, T concaténé aux canaux | 1,3,5 | 122 825 / 123 977 / 125 129 | Prototype commun testé ; seul ancien run T=1 entraîné sur 5 événements |
| B3 | Encodeur spatial stride 2, cellule ConvLSTM, tête spatiale | 1,3,5 | 5 297, état caché 8 | Prototype récurrent testé, non entraîné |
| B4 | Temporal U-Net ou UTAE | À décider après profil | Non mesuré | Différé ; factory refuse explicitement B4 |

B2 à T>1 est une U-Net spatiale sur canaux empilés, pas une U-Net récurrente ni
UTAE. B3 est volontairement petit ; sa capacité diffère de B2. Une comparaison
de ces baselines ne permet pas d'attribuer tout écart au seul mécanisme temporel.
La dépendance au passé de B3 et ses gradients sont testés ; son utilité
prédictive n'est pas démontrée.

Microprofil réel CPU, batch 2, patches 64, 2 threads, trois répétitions après
warm-up, sans optimizer ni mise à jour des poids :

| Modèle | T=1 | T=3 | T=5 |
| --- | ---: | ---: | ---: |
| B1 forward+loss+backward | 1,262 ms | 1,537 ms | 1,786 ms |
| B2 forward+loss+backward | 12,685 ms | 14,335 ms | 13,682 ms |
| B3 forward+loss+backward | 3,062 ms | 4,910 ms | 8,607 ms |

Ces différences de petits timings incluent du bruit ; elles ne constituent pas
un résultat de performance scientifique ni une projection fiable du coût I/O.
Source : `reports/benchmark_cpu_profile_v1.json`.

## Splits annuels et comparabilité temporelle

Les 12 folds suivent **l'ordre exact** de
[FireSpreadDataModule.split_fires des auteurs](https://github.com/SebastianGer/WildfireSpreadTS/blob/main/src/dataloader/FireSpreadDataModule.py) :
chaque paire de deux années sert au train, puis les deux années restantes
permutent entre validation et test. Aucun événement ne traverse un split.

Fold 0 : train 2018–2019 (**250 événements**), validation 2020 (**201**),
test 2021 (**156**). Les assignations des 607 événements sont générées dans
`splits/benchmark/annual_v1/fold_00.csv` à `fold_11.csv` et hashées dans
`folds.json`. Il s'agit d'un inventaire metadata, pas d'une preuve de présence
locale du corpus. Le fold 0 n'est pas prêt : aucun événement 2021 local.

Les auteurs prévoient aussi un ajustement des observations test selon T. Ici,
l'intersection est explicite : construire les fenêtres consécutives T=1/3/5,
puis intersecter les clés `(event_id,target_date)`, en tenant compte des trous
et tailles. Garder la même liste triée pour tous les modèles. Ce principe est
appliqué à **train aussi** pour l'ablation temporelle, afin de ne pas confondre
T et nombre d'exemples d'entraînement ; c'est une décision GeoFire-X déclarée,
pas une reproduction exacte des crops aléatoires de validation des auteurs.

Le manifeste local donne 83 cibles communes sur les cinq événements, contre
103 cibles T=1 non harmonisées. Fold 0 local : 41 train, 42 validation, 0 test.
Les CSV `*_aligned_targets.csv` en donnent les chemins et dates pour chaque T.
Lors de l'acquisition suivante, régénérer dans un **nouveau dossier versionné**.

Avant un run scientifique : vérifier l'intégralité des fichiers attendus pour
les années du fold, les CRC/SHA, descriptions, CRS/transform, dates et dimensions.
Une séparation annuelle ne garantit pas à elle seule l'absence de recouvrement
spatial de feux différents ; rapporter les bbox et examiner les doublons spatiaux.
Ne pas transférer les poids de l'ancienne U-Net vers un fold où ses événements
d'entraînement seraient dans le test. Tous les modèles partent d'une seed neuve.

## Prétraitement, entraînement et inférence communs

Premier profil : feu J et relief, C=8, LC exclu, réserve de livraison du masque
J entier explicitée. Le profil toutes modalités est une ablation distincte
exigeant un audit des horodatages de disponibilité, notamment cartes annuelles,
composites et prévisions. Il ne doit pas être comparé à une architecture ayant
moins de modalités comme si seul le modèle changeait.

Pour chaque fold, ajuster une seule normalisation sur l'union des **chemins
d'entrée train T=5 uniques** des cibles harmonisées ; la partager entre T=1/3/5.
Les paramètres restent identiques quel que soit le nombre de jours fourni.
Ni statistiques val/test ni cibles seules n'y contribuent. Un cache indépendant
des folds conserve unités physiques et NaN ; l'encodage est appliqué après lecture.

Patches de taille fixée 64, validité du padding, seed, listes d'échantillonnage
et budget de mises à jour enregistrés. La fraction de patches positifs doit
être complétée par la prévalence pixel effectivement vue et la couverture des
événements. Le label peut guider le sampling du train uniquement. Évaluation
sans sélection de régions selon la vérité terrain.

Choisir un budget comparable de mises à jour et un plan de HPO identique pour
B1–B3, puis rapporter séparément temps et paramètres. La loss/pondération/LR ne
sont pas déclarés optimaux : les ablations BCE, BCE pondérée et BCE+Dice seront
préenregistrées. Tout pos_weight ou prior est calculé sur le train uniquement.
Le petit budget historique n'est pas reconduit comme une preuve de convergence.

Sélection du checkpoint : **AP micro validation sur scènes complètes**, maximum,
avec epoch la plus ancienne en cas d'égalité. Déclarer la fréquence d'évaluation
à l'avance ; un checkpoint non évalué n'est pas admissible. Loss et Dice restent
des diagnostics secondaires. Toute early stopping utilise la validation.

Le protocole v1 fixe les tuiles sans recouvrement et leur origine ; les scènes
entières sont reconstruites sans padding dans les scores. Les raccords font
partie des limites mesurées. Un mode à recouvrement ou forward global exige un
nouvel identifiant de protocole commun à tous les modèles. Ne pas choisir le
découpage sur test. L'égalité « fully convolutional » n'implique pas l'égalité
des scores entre une scène entière et des patches avec GroupNorm.

## Métriques standardisées

Métrique principale : **Average Precision exacte micro** sur tous les pixels
valides des cibles harmonisées. Ce n'est ni la moyenne des AP par fenêtre ni
l'aire trapézoïdale sous PR. Les scores ne sont pas seuillés pour l'AP ; les
égalités de score sont regroupées globalement, y compris entre événements.

`ExternalAveragePrecision` fournit une référence de tri/fusion sur disque,
mémoire bornée et fan-in borné. L'exactitude est testée contre sklearn, sur
égalités/chunks et sur la validation réelle. Son débit complet reste à mesurer.
Les fichiers de tri sont nouveaux et conservés ; prévoir leur place. Une AP
approchée par histogramme ne peut pas être publiée sous le même nom sans
qualification et mesure d'erreur.

Rapporter conjointement :

- Dice/F1, IoU, Precision, Recall, TP/FP/FN, nombre de pixels et prévalence.
- Micro global ; **AP poolée par événement**, puis macro d'événements avec
  nombre de scores définis. Ne pas appeler « AP par événement » une simple
  moyenne de ses AP par fenêtre.
- Macro par fenêtre, toutes et positives seulement, en métrique secondaire.
- Événements à au moins une cible positive sur la liste évaluée, avec décompte ;
  garder les fenêtres vides dans leur score poolé pour compter leurs FP.
- Événements/fenêtres sans positifs : AP et Recall `null`, Precision `null`
  sans prédiction ; Dice/IoU=1 si les deux masques sont vides. Ne pas exclure
  leurs faux positifs de l'AP globale lorsqu'il existe des positifs ailleurs.
- Nouvelles ignitions : domaine candidat `fire_J=0`, cible `fire_J+1=1`.
  AP, Precision, Recall et TP/FN conditionnels, séparés de l'évaluation complète.
  Persistence a un score constant zéro dans ce domaine et un rappel nul.

Choisir le seuil de décision sur Dice micro validation parmi la grille déclarée
au YAML, avec règle d'égalité déterministe ; publier la courbe et signaler une
borne de grille atteinte. Le seuil reste secondaire à l'AP pour le classement.
Brier et fiabilité sont des diagnostics de calibration. Si un calibrateur est
ajouté, réserver un sous-ensemble d'événements de validation avant toute sélection ;
aucun calibrateur n'est ajusté dans cette préparation.

Les pixels et fenêtres recouvrantes sont corrélés. Les intervalles éventuels
doivent rééchantillonner les événements, pas les pixels ; avec un seul événement
local de validation historique, pas d'intervalle crédible de généralisation.

## MLflow et sélection des modèles

Expérience future unique : **`geofirex-benchmark-annual-v1`**. Elle n'est pas
créée ici pour éviter de présenter des prototypes non entraînés comme des runs
scientifiques. Les anciennes expériences restent intactes.

Convention de nom : `B2-fold00-T3-seed42-<UTC>-<config_hash>` ; tags obligatoires
`protocol_version`, `model`, `fold_id`, `context_days`, `seed`,
`qualification=scientific|pilot|diagnostic`, `phase=train|final_test`.
Un pilote n'est pas admissible à la sélection du benchmark complet.

Chaque run devra enregistrer : architecture et nombre de paramètres ; dataset,
DOI/version, manifeste/hash et index/hash ; split/hash et hash des cibles
harmonisées ; canaux et ordre, disponibilité temporelle, T, preprocessing/hash ;
seed et générateurs ; configuration complète ; commit **et snapshot des sources
non commitées** ; versions Python/packages, CPU/RAM/GPU/CUDA ; optimizer, LR,
loss/poids, tailles/budget/sampling ; losses et AP validation par epoch évaluée ;
temps d'I/O/calcul/tri AP séparés ; checkpoint/hash, figures et toutes les
agrégations/conventions standardisées. Ne pas logger 49 Go de bruts en artefacts.

Sélection : filtrer seulement les runs FINISHED de même protocole, dataset,
fold, canaux et hash de dates cibles ; agréger les **mêmes seeds déclarées**
(42/43/44), classer les architectures/T par moyenne d'AP validation, fournir
écart-type et coût. `rank_validation_candidates` et `rank_architectures` testent
ces contrôles ; aucune métrique test n'est lue pour le classement. Pas de choix
du meilleur seed après consultation des scores.

Avant le test : produire un verrou de sélection (run IDs, checkpoint/config,
statistiques, seuils, protocole et hashes), puis lancer une évaluation finale
explicite. Un runner de production devra vérifier ce verrou et séparer les
artefacts test de la recherche d'hyperparamètres. Le CLI historique de replay
reste un outil de vérification du run déjà exposé, pas cette nouvelle barrière.
Ne pas entraîner/modifier le protocole à partir des futurs résultats test.

Pour 12 folds, réinitialiser les poids et statistiques à chaque fold. Ne pas
choisir la famille du modèle ou son HPO à partir de l'agrégation des 12 tests.
Rapporter leur distribution avec un protocole fixé ; ces folds partagent des
événements entre rôles et ne sont pas 12 répétitions indépendantes.

## Structure du code et état de préparation

| Composant | Réalisé et testé | Étape avant entraînement complet |
| --- | --- | --- |
| `corpus.py` et CLI acquisition | Index, budget réseau, CRC, reprise par fichier, dry-run | Accord de budget puis acquisition/QC complets |
| `event_cache.py` | Cache memmap, NaN physiques, intégrité, lecture patches | Profiler LRU multiworker et débit sur davantage d'événements |
| `benchmark.py` | 12 folds, intersection réelle T=1/3/5 | Manifeste complet, contrôle géospatial exhaustif |
| `benchmark_models.py` | Interfaces B0–B3, gradients, dépendance temporelle B3 | Entraînement commun et mesure GPU ; B4 reste différé |
| `benchmark_metrics.py` | AP externe exacte, nouvelles ignitions, classement validation/seeds | Brancher accumulateurs globaux/par événement au runner commun |
| Trainer commun | Contrats et configuration proposés ; trainer historique T=1 conservé | Construire adapter temporel/cache, normalisation par fold, checkpoint AP et verrou test |
| MLflow commun | Schéma, noms, sélection et artefacts définis | Brancher le futur runner, sans toucher aux anciennes expériences |
| Parquet/Spark | Export local réel ; agrégation Spark préparée | Provisionner Spark/JVM et mesurer une exécution réellement distribuée |

La préparation ne prétend pas fournir déjà un entraînement scientifique
multi-modèles de bout en bout. Les composants manquants sont des portes
explicites avant une exécution longue, pas des résultats supposés acquis.

## Coût de calcul et progression proposée

Le microprofil sans I/O donne un plancher : pour B2 T=5, batch 2,
10 000 mises à jour représenteraient environ **137 s** de forward/backward
seuls au débit du microtest. Ce chiffre exclut optimizer, données, validation,
tri AP, checkpoint, échauffement CPU et toute évolution de taille : ce n'est pas
une estimation de durée de run complet.

La vérification réelle de quatre AP sur 23 scènes a pris **34,0 s**, lectures et
tris compris. Une simple extrapolation à ~3 300 scènes donne ~81 minutes par
évaluation similaire sur ce chemin de référence. Elle est illustrative : un
cache différent, les merges supplémentaires et la distribution des scores
changent fortement le coût. Mesurer séparément ces étapes sur un pilote est
nécessaire avant de fixer le nombre d'epochs ou la fréquence de validation.

Trois modèles entraînables × trois T × trois seeds × douze folds = **324 runs**,
hors HPO ; B0 requiert seulement 12 évaluations déterministes. Même à une heure
par run (hypothèse, pas mesure), cela représenterait 324 h ; le benchmark complet
n'est pas une tâche raisonnable à lancer automatiquement sur ce portable avant
le 4 novembre. Un GPU plus grand/serveur peut être requis, à profiler avant
d'annoncer une accélération ou un tarif. La MX550 2 Go n'est pas validée par le venv.

Ordre proposé après accord : acquisition/QC progressive ; pilote court B1 puis
B2/B3 sur un seul fold et une seed, avec vrai profil I/O/RAM ; correction du
runner et gel du protocole ; fold 0 complet sous budget approuvé ; réplications
et autres folds lorsque les ressources et la convergence sont établies.
Il n'y a aucun entraînement supplémentaire lancé par cette mission préparatoire.
