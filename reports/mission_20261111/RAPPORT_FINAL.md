# GeoFire-X — exécution scientifique du 11 novembre 2026

Les cinq livrables sont produits localement. B0 couvre toute la validation ;
le corpus compact couvre désormais les 250 train et 201 validation. B1 a été
entraîné réellement sur CPU ; le notebook GPU est préparé, sans job GPU soumis.

## 1. B0 réellement évalué sur la validation entière

**201 événements 2020, 3 287 fenêtres alignées, 263 630 731 pixels valides**,
dont 644 306 positifs. Exécution : **561,68 secondes** depuis D:, lecteur
progressif et cache borné. Contexte T=3 ; cibles communes T=1/3/5 et intersection
des footprints des cinq jours de contexte et de J+1.

| Micro sur les pixels valides | Dice | IoU | Précision | Rappel | AP |
|---|---:|---:|---:|---:|---:|
| Persistence B0 | 0,441082 | 0,282941 | 0,438351 | 0,443848 | 0,195920 |

TP=285 974, FP=366 412, FN=358 332, TN=262 620 013. AP exacte non interpolée,
avec regroupement global des deux scores binaires ; confrontée à sklearn.
Les macros événement/fenêtre et conventions des cas vides sont explicites dans
[results.json](b0_v2/results.json). Dice macro événement : 0,215798.
Les masques GDAL/alpha et NoData finis sont appliqués ; NaN active-fire signifie
absence de détection. Aucun pixel invalide dans ce domaine selon ces masques,
ce qui ne prouve pas l'absence d'erreurs de détection VIIRS.

Cartes réelles : [scène B0 complète](scene_maps/case_00_B0.png),
[contexte temporel d'un autre événement](scene_maps/case_01_B0.png),
[prédiction du B1 entraîné](scene_maps/B1_actual_prediction.png).
PNG, tableaux NPZ, dates, événements, CRS et transforms sont conservés.
Ces cas à FN élevés sont des illustrations déclarées, pas un échantillon pour
estimer la performance. Les métriques ci-dessus portent sur toutes les fenêtres.

## 2. Corpus compact complet produit et mesuré

| Propriété | Résultat réel |
|---|---:|
| Événements train / validation | 250 / 201 |
| Observations sources | 9 490 |
| Fenêtres T=3 alignées train / val | 3 948 / 3 287 |
| Tuiles temporelles train / val | 93 603 / 78 771 |
| Fragments jour/tuile | 226 859 |
| Archives partitionnées par événement | 451 |
| GeoTIFF bruts correspondants | 34 995 352 430 octets |
| Archives compressées | 9 595 846 733 octets |
| Métadonnées nécessaires au chargement | 249 738 339 octets |
| Charge + rapport + manifeste de transfert | **9 845 667 110 octets** |

Format : NPZ compressés, regroupés dans des ZIP par événement sans seconde
compression. Chaque jour/tuile est stocké une fois ; les manifestes reconstruisent
T=3 et J+1. Huit canaux encodés existants en float32, masques valides, contexte
du domaine commun, dates, événements et métadonnées spatiales conservés.
Il s'agit d'une réduction vers le profil feu/terrain existant, **pas d'une
compression sans perte des 23 bandes brutes**. Pas de quantification supplémentaire.

Toutes les tuiles sont conservées : train = 85 940 négatives, 6 136 positives,
1 527 négatives difficiles ; val = 70 625 / 6 551 / 1 595. « Difficile » signifie
cible négative avec feu observé à J dans la tuile. Aucun rééquilibrage validation.

Le pilote initial de 16 événements / 340 jours a mesuré 351 826 630 octets,
456,06 s. Le pilote optimisé par archives a mesuré 352 997 802 octets,
132,28 s et 469 581 824 octets de pic RAM. Ses 7 850 fragments sont identiques
au premier pilote. Les premiers rapports RAM à zéro étaient instrumentés
incorrectement : ils sont conservés comme diagnostics, sans être utilisés.

Extension effectuée en trois partitions disjointes : **1 111,41 s observées**
de la création du premier artefact à la dernière vérification de partition ;
**1 199,58 s avec consolidation**, hors démarrage Python avant le premier artefact.
Pics RAM de conversion par processus : 621 744 128 / 619 548 672 / 613 076 992 octets.
Le contrôle final avec chargement des métadonnées complètes a culminé à
**1 234 247 680 octets**, sans charger les tenseurs de tout le corpus.

Vérifications : premier fragment de chacun des 9 490 jours relu à l'identique ;
DataLoaders de chaque partition et du manifeste consolidé validés ; dix
comparaisons exactes X/Y/masque avec D: sur le corpus final, incluant les deux
splits, positifs, négatifs difficiles et bordures. Les SHA originaux viennent des
certificats existants ; taille/CRS/transform ont été vérifiés pendant conversion.
Aucun nouvel audit général CRC/SHA des 35 Go n'est revendiqué.

Preuves : [rapport compact](compact_full/report.json),
[comparaisons originales](full-original-comparison.json),
[budget exact](compact_full/transfer-budget.json),
[liste des fichiers et SHA](compact_full/transfer-manifest.json).
Le budget inclut 456 fichiers à transmettre, hors surcoût réseau/retries.
Les métadonnées intermédiaires des trois partitions ne sont pas nécessaires au
transfert : suivre exactement le manifeste, pas une copie récursive du dossier.

**Décision technique : utiliser ce corpus complet pour B1/B2/B3 T=3.**
Les 451 événements sont disponibles localement au format PyTorch ; transférer
ce volume mesuré évite de transférer les 35 Go bruts pour le profil de huit
canaux. Le transfert massif reste soumis à autorisation et n'a pas commencé.

## 3. B1 entraîné et notebook GPU exploitable

B1 existant `spatial_cnn_v2`, seed42, T=3, loss existante BCE pondérée + Dice :
**deux époques complètes**, dix événements train, six val, **6 900 tuiles vues,
432 mises à jour**, 324,32 s CPU. Loss train 0,666318 → 0,597948 ; validation
0,605275 → 0,583692. [Courbe réelle](b1_cpu_pilot/loss.png).

| Même validation de six événements | Dice | AP |
|---|---:|---:|
| B0 | 0,430987 | 0,187116 |
| B1 après deux époques | 0,316374 | 0,180658 |

B1 ne dépasse pas B0. Cette expérience n'est pas le benchmark neural des 201
événements. Des coutures entre tuiles sont visibles dans la carte B1 ; prévoir
une expérience contrôlée avec halo/recouvrement avant toute conclusion spatiale.
Checkpoints modèle/optimiseur/RNG conservés ; rechargement avec prédictions
identiques vérifié. Normalisateur train-only inchangé, SHA lié au run et copie
préservée avec reçu de provenance. Aucun événement val/test ne contribue au fit.

[Notebook GPU final](B1_GPU_FINAL_20261111.ipynb) autonome : code et petites
métadonnées intégrés, aucun GeoTIFF ni secret intégré. Attacher les lots train
001/002 déjà publiés et le sample historique pour la validation 2020 disponible,
ou attacher le corpus compact après accord de transfert. Le sample offre deux
événements val : aucune validation des 201 n'est simulée si seuls ces datasets
sont montés. La disponibilité et les hashes des fichiers attachés sont vérifiés
à l'exécution. Le chemin raw du notebook passe localement sur le sample existant.

Le profil fait 25 mises à jour réelles, puis une validation technique limitée aux
64 premières tuiles selon les métadonnées pour rester court ; ce sous-ensemble
n'est pas un benchmark. Les trois époques complètes utilisent toute la validation
montée et une cellule d'activation séparée, désactivée par défaut jusqu'à accord
sur la campagne GPU. Checkpoints immuables par époque, normalisation, optimiseur,
RNG, compteurs, métriques et courbe sont persistés dans `/kaggle/working`.
Un test CPU du chemin de profil et du rechargement des checkpoints passe aussi.

**Aucun GPU exécuté ici.** Machine locale : PyTorch CPU, CUDA absent. Quota Kaggle
actuel inconnu : accès navigateur spécialisé indisponible ; contrôle natif arrêté
par l'outil car l'URL courante ne pouvait pas être vérifiée. Aucune action UI
poursuivie, aucun ancien rapport de quota présenté comme actuel. B2/B3 restent
les modèles réutilisables existants, sans nouveau score ni campagne revendiquée.

## 4. Spark et application

**Spark 4.0.2 local[2] exécuté**, 65,11 s démarrage compris, dont 21,79 s après
démarrage JVM. SQL effectifs annuels/splits, fenêtres, qualité, doublons, fuite
annuelle et jointure cibles réellement exécuté ; résultats Parquet et requêtes
conservés dans [spark/report.json](spark/report.json). Inventaire : 607 événements,
13 607 observations. Qualité train/val issue des comptages certifiés existants ;
2021 présent seulement dans l'inventaire de métadonnées. Aucun cluster distribué.

Le React existant fournit sélecteurs, cartes canvas, observation, prédiction et
vérité terrain. FastAPI conserve `/health`, `/events`, `/predict`, `/ground-truth`.
[Plan d'intégration](INTEGRATION_APPLICATION.md) pour Wildfire Explorer,
Prediction Lab, Model Benchmark et Big Data Monitor. Adaptateur FastAPI en lecture
seule préparé et vérifié par six requêtes ASGI, y compris montage frontend racine
et 404 sur carte absente. Il reste à brancher les panneaux React et l'inférence
du nouveau checkpoint B1 ; le service historique n'a pas été remplacé.

## Préservation et suite

Toutes les sorties sont nouvelles sous `reports/mission_20261111`. Aucun transfert,
re-upload, job GPU, commit, push, merge, ni modification de checkpoint historique
ou de l'orchestrateur. Les modifications déjà présentes dans le dépôt sont
conservées. Les diagnostics de premières tentatives restent disponibles.

Prochaine action à autoriser : transfert des **9 845 667 110 octets** listés,
avec budget réseau/retries défini, puis profil GPU court et vérification du quota.
Une campagne GPU prolongée nécessite un accord distinct sur un budget mesuré.
