# Diagnostic scientifique initial et deux expériences conditionnelles

## Ce qui est établi avant la calibration

Les chiffres suivants sont ceux communiqués par l'utilisateur ; aucune nouvelle
mesure Kaggle n'a été exécutée pendant la préparation de ce livrable.

| Modèle, seuil 0,5 | Dice | AP | Précision | Rappel | Prédictions positives erronées |
|---|---:|---:|---:|---:|---:|
| B1 CNN | 0,2871 | 0,2079 | 0,1897 | 0,5898 | 81,03 % |
| B2 U-Net | 0,2117 | 0,2621 | 0,1225 | 0,7785 | 87,75 % |

La dernière colonne vaut `1 - précision` : c'est la proportion de faux positifs
parmi les prédictions positives, **pas** le taux FP/(FP+TN). Elle correspond à
environ 4,27 FP par TP pour B1 et 7,16 pour B2. On ne peut pas déduire les nombres
TP/FP/FN ni le taux de faux positifs sans les comptages. B2 classe mieux les pixels
selon l'AP fournie tout en segmentant moins bien à 0,5 : cela justifie d'abord la
recherche d'un seuil, sans garantir un gain significatif sur les 169 événements.

`scripts/train_compact_benchmark.py` utilise `SegmentationLoss(.5,100)` :
`0.5 * BCEWithLogits(pos_weight=100) + 0.5 * DiceLoss`. La BCE est moyennée sur
les pixels valides ; la Dice est calculée par image puis moyennée sur les images
non entièrement invalides. Une image de feu rare ne pèse donc pas de la même
façon dans les deux termes. Une pondération positive élevée favorise le rappel,
comme décrit dans la [documentation PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
Cela rend plausible un excès de prédictions positives, mais ne démontre pas que
100 soit excessif : il faut connaître la prévalence pixel et le comportement
des scores. Les checkpoints ne sauvegardent pas explicitement les paramètres
de loss : cette constatation vient du code d'entraînement disponible.

Le sampler `LocalitySampler` mélange des blocs pour réutiliser les jours voisins
et visite toutes les tuiles une fois par époque ; il ne suréchantillonne pas le feu.
Le rapport compact existant recense 6 136 tuiles positives / 93 603 train (6,56 %),
et 6 551 / 78 771 val (8,32 %). Ce sont des proportions de **tuiles**, différentes
de la prévalence des pixels et du sous-ensemble historique de 32 événements.

## Ce que le notebook mesurera

- **Loss et déséquilibre :** prévalence pixel train exhaustive et ratio négatifs /
  positifs, confrontés aux 32 et aux 169 ; aucune pondération issue de la validation.
- **Distribution des scores :** histogrammes feu/non-feu séparément normalisés,
  fiabilité par classes de probabilité, Brier et AP globale/par événement. Un bon
  classement AP peut coexister avec des probabilités mal calibrées sous loss pondérée.
  Le notebook ne transforme pas les probabilités et n'applique pas de correction
  analytique fondée sur la seule BCE à une loss mixte BCE+Dice.
- **Échantillonnage et domaine :** années, événements disjoints, absence de 2021,
  normaliseur train-only, mêmes tuiles et mêmes masques communs aux cinq jours
  historiques plus la cible. Aucun équilibrage des tuiles validation.
- **Nouveaux feux :** définir `observed = feu à J`, `target = feu à J+1`.
  Sur `valid & (observed == 0)`, compter séparément nouveaux TP/FN et FP ; sur
  `valid & (observed == 1)`, compter persistance et extinctions apparentes. Les
  deux domaines doivent sommer exactement aux comptages globaux. Un nouveau
  pixel détecté ne prouve pas une ignition physique : cette cible est une
  détection de feu actif issue du corpus.
- **Événements difficiles :** métriques micro, moyenne et quantiles macro par
  événement, distributions et fraction FP/pixels valides selon la prévalence.
  Les événements sans feu et les valeurs indéfinies sont explicitement conservés.

L'[AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
se calcule depuis les scores ; le
[Brier](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
depuis les probabilités et les labels. Changer seulement le seuil ne modifie
aucune de ces deux métriques. Le terme « calibration » désigne ici le choix d'un
seuil de segmentation, pas un recalibrage probabiliste.

## Deux expériences futures, sous autorisation seulement

Les deux propositions forment une ablation contrôlée depuis le **même checkpoint**.
Aucune nouvelle architecture ou modification temporelle n'est ajoutée maintenant.
Le notebook de calibration ne contient aucun appel d'entraînement.

1. **Contrôle de prolongation, loss inchangée.** Si les historiques epoch 1–4
   montrent encore une amélioration de l'AP et de la loss validation, prolonger
   le modèle prioritaire d'une époque, depuis son checkpoint, même AdamW et mêmes
   données. Conserver le contrat de reprise et les RNG. Fixer un budget GPU
   à partir du débit réellement mesuré ; aucun budget n'est autorisé ici.
   Cette branche mesure l'effet d'une optimisation supplémentaire seule.

2. **Ablation du poids positif, même prolongation.** Si les FP restent élevés
   après sélection du seuil et si la fiabilité montre une surestimation du feu,
   repartir du **même état initial** que l'expérience 1, même nombre d'updates,
   même ordre de batches, optimiser avec `dice_weight=0.5` et
   `pos_weight=min(100, sqrt(N_nonfeu_train / N_feu_train))`. Le ratio vient de
   `train_audit.json` uniquement. La racine est un choix expérimental modéré,
   pas une valeur optimale démontrée. Si ce calcul redonne 100, cette expérience
   n'est pas distincte du contrôle et ne doit pas être lancée telle quelle.
   Consigner ce changement dans un nouveau run ; ne pas éditer les anciens
   historiques ni modifier silencieusement leur critère.

Choisir les candidats et leurs seuils exclusivement sur les mêmes 32 événements.
Comparer AP, Dice calibrée, Brier, nouveaux-feux et dispersion par événement,
avec une attention aux pertes de rappel ; ne pas sélectionner seulement un gain
de précision. Après observation de leurs résultats de calibration, les 169
événements deviennent un panel de confirmation connu, pas un nouveau test aveugle.
Ne pas choisir la loss ou ajuster le seuil pour optimiser ce panel. Aucun test
2021 n'est prévu. Le suivi du 18 novembre peut déjà présenter le gain du seuil
et sa confirmation, même si aucun entraînement supplémentaire n'est autorisé.
