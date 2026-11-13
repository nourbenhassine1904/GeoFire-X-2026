# Résultats exécutés — 11 novembre 2026

B0 a évalué **les 201 événements validation 2020 et 3 287 fenêtres** du protocole
commun en **561,68 s**, lecture progressive depuis D:. Les 263 630 731 pixels
du domaine commun T=5+cible comprennent 644 306 positifs. Masques explicites
GDAL/alpha et NoData fini appliqués ; NaN active-fire signifie absence de détection.

| Agrégation micro pixels valides | Dice | IoU | Précision | Rappel | Average Precision |
|---|---:|---:|---:|---:|---:|
| B0 Persistence, validation entière | 0,441082 | 0,282941 | 0,438351 | 0,443848 | 0,195920 |

L'AP est exacte, non interpolée, avec regroupement des deux scores binaires.
Le calcul a été confronté à sklearn. Les macros événement et fenêtre, TP/FP/FN/TN,
conventions des cas vides et effectifs définis sont dans `b0_v2/results.json`.
Dice macro événement : 0,215798 ; différence attendue avec le micro pondéré pixels.
Rappel sur les nouvelles détections hors feu observé à J : zéro pour Persistence.
Quatre cartes PNG et leurs tableaux/masques/provenances sont sous `b0_v2/map_*`.
Elles illustrent les premières tuiles contenant des erreurs, sans sélection pour
estimer la performance. Le compteur RAM interne du premier B0 était défectueux
(zéro, valeur inutilisable) ; ne pas le citer. Le lecteur est borné à 64 Mio,
GDAL à 32 Mio, et une seule tuile est scorée à la fois.

## Corpus compact : deux pilotes mesurés

| Mesure | NPZ indépendants | NPZ regroupés par événement |
|---|---:|---:|
| Événements train / val | 10 / 6 | 10 / 6 |
| GeoTIFF sources / fenêtres T=3 | 340 / 260 | 340 / 260 |
| Tuiles temporelles / fragments jour | 6 025 / 7 850 | 6 025 / 7 850 |
| Octets GeoTIFF sources | 1 246 381 470 | 1 246 381 470 |
| Octets fragments compressés | 340 719 827 | 341 642 559 |
| Métadonnées mesurées lors du rapport | 11 106 803 | 11 355 243 |
| Charge mesurée lors du rapport | 351 826 630 | 352 997 802 |
| Temps conversion avant vérification finale | 456,06 s | 132,28 s |
| Pic RAM processus | instrument défectueux | 469 581 824 octets |

Ces volumes excluent le rapport et les reçus écrits ensuite : le budget final
de transfert sera mesuré sur la liste exacte des fichiers à transférer.
La réduction à environ 28,3 % du brut résulte aussi du choix des 8 canaux
feu/terrain existants, et ne mesure pas une compression sans perte des 23 bandes.
Les huit canaux encodés sont conservés en float32, sans quantification.
Chaque jour/tuile est stocké une seule fois ; T=3, cible J+1 et domaine valide
T=5 sont référencés par manifeste. Dates, événements, CRS et transforms restent
disponibles. Les ZIP par événement contiennent des NPZ déjà compressés, sans
seconde compression. Ce sont des partitions, pas un fichier monolithique.

Les 7 850 fragments du second pilote sont identiques octet par octet à ceux du
premier. Les DataLoaders train/val et dix comparaisons exactes avec les originaux
(positifs, négatifs difficiles, négatifs et bordures) passent. Un bug d'extraction
de date dans le vérificateur d'archives a été corrigé ; son contrôle relancé est
dans `compact_archive_pilot/pytorch-verification-recovery.json`. Les fichiers de
première tentative sont conservés.

Décision : étendre localement aux **250 train / 201 val**, en trois partitions
de travail disjointes et archives par événement. Conserver toutes les tuiles ;
la validation reste exhaustive sans rééquilibrage. Au ratio pilote, environ
10 Go sont plausibles pour 35 Go bruts train/val : **estimation**, à remplacer
par les octets réellement obtenus. Aucune publication/transfert n'est lancé.

## B1 : optimisation réelle sur CPU

Deux époques complètes sur les dix événements train du pilote : 3 450 tuiles par
époque, **6 900 tuiles vues et 432 mises à jour**, 324,32 s. Architecture existante
`spatial_cnn_v2`, 8 canaux, T=3, loss existante BCE pondérée + Dice, seed42.
Loss train 0,666318 → 0,597948 ; loss validation 0,605275 → 0,583692.
Six événements val exhaustifs, 2 575 tuiles, 8 351 749 pixels valides.

| Même validation de six événements | Dice | AP |
|---|---:|---:|
| B0 | 0,430987 | 0,187116 |
| B1 CPU après deux époques | 0,316374 | 0,180658 |

B1 ne dépasse pas B0 sur ce pilote. Ces scores ne constituent pas le benchmark
neural des 201 événements. Checkpoints modèle/optimiseur/RNG et courbe loss sous
`b1_cpu_pilot/` ; rechargement vérifié avec prédictions identiques.

## Spark, notebook et application

Spark **4.0.2, local[2] réellement exécuté**, 65,11 s démarrage compris,
21,79 s de traitement après démarrage. SQL annuels, fenêtres, qualité, doublons,
assignations et jointure cibles exécutés et exportés dans `spark/`.
607 événements / 13 607 observations en inventaire ; qualité train/val issue des
comptages certifiés existants, aucun nouvel audit pixel général. Inventaire test
2021 uniquement en métadonnées ; aucun pixel test lu par cette mission.

Le notebook autonome embarque seulement code et petites métadonnées. Il utilise
les deux lots train déjà publiés et le sample historique pour disposer de vrais
événements val 2020, ou un corpus compact attaché après accord de transfert.
Le sample offre deux événements val : cette limite est explicite. Profil GPU
court de 25 mises à jour, validation, compteurs et checkpoints ; trois époques
complètes dans une cellule séparée avec activation explicite après accord GPU.
CUDA est absent localement. Aucune exécution GPU revendiquée. Le quota Kaggle
actuel demeure inconnu : accès navigateur indisponible, puis contrôle natif
arrêté car l'URL courante ne pouvait pas être vérifiée. Aucune action UI poursuivie.

`INTEGRATION_APPLICATION.md` relie les quatre sections demandées aux composants
React et endpoints existants. Un APIRouter de résultats réels en lecture seule
est préparé dans `src/geofirex/mission_results.py`. L'inférence historique et
le frontend restent conservés ; le nouveau checkpoint B1 n'est pas présenté
comme déjà intégré au service historique.

Aucun transfert, re-upload, job GPU, commit, push, merge ni écrasement historique.
