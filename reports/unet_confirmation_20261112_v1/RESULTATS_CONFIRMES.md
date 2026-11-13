# Confirmation U-Net poids 10 — résultat final

Une seule confirmation du candidat fixé sur les 32 a été réalisée : modèle B2,
poids10, seuil0,78, sur les 169 événements complémentaires 2020. Exécution CPU,
747,955 secondes (12 min 28 s), aucun GPU et aucun entraînement. Le programme ne
dispose d'aucun paramètre pour modifier le poids ou le seuil. Les fichiers de
sortie existants sont refusés à la relance ; aucune seconde confirmation lancée.

## Comparaison commune

Les trois lignes couvrent **217 506 967 pixels valides**, 562 953 pixels feu,
65 012 tuiles et les mêmes 169 IDs. Les résultats historiques sont lus depuis
leur ZIP immuable, pas recalculés. Les comptages de pixels valides, positifs,
tuiles et invalides sont également vérifiés événement par événement.

| Modèle | Seuil | Dice micro | IoU | Précision | Rappel | AP | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN B1 historique | 0,95 | **0,440879** | **0,282774** | 0,359088 | **0,570920** | 0,268377 | 0,00569349 |
| U-Net B2 historique | 0,95 | 0,406695 | 0,255252 | 0,323257 | 0,548191 | 0,300972 | 0,00951151 |
| **U-Net B2 poids10 retenu sur les 32** | **0,78** | 0,433182 | 0,276473 | **0,404808** | 0,465835 | **0,318618** | **0,00349779** |

| Modèle | TP | FP | FN |
|---|---:|---:|---:|
| CNN historique | 321 401 | 573 648 | 241 552 |
| U-Net historique | 308 606 | 646 071 | 254 347 |
| U-Net poids10 | 262 243 | 385 578 | 300 710 |

Face à l'U-Net historique, le poids10 gagne **0,026488 de Dice** et 0,017646 d'AP,
avec 40,32 % de FP en moins. La précision augmente, mais le rappel diminue de
0,082357 et les FN augmentent de 46 363. Face au CNN, la Dice reste inférieure
de **0,007696**, malgré une AP supérieure de 0,050241 et un Brier plus faible.
Le nouveau modèle ne domine donc pas les références sur tous les critères.

Le Dice macro par événement du candidat vaut 0,217320 (médiane 0,148710 ;
quantiles 10/90 : 0 / 0,501135). Il ne remplace pas le Dice micro annoncé comme
critère de sélection. Les distributions par événement sont affichées dans
l'application et les valeurs complètes sont conservées dans les CSV/JSON.

Les scores AP et Brier concernent les probabilités, pas un effet du seuil. Le
Brier de la constante égale à la prévalence train sur ce panel vaut environ
0,00258316 : le Brier du nouveau modèle reste supérieur. Une meilleure AP et
un meilleur Brier que les modèles historiques ne prouvent pas une calibration
probabiliste parfaite.

## Limite importante : les nouveaux pixels de feu

Sous-domaine `valid & (feu_J == 0)` : 216 936 449 pixels candidats, dont 308 168
positifs à J+1. Pour le poids10 à 0,78 :

| Mesure | Valeur |
|---|---:|
| TP nouveaux pixels | 45 245 |
| FP hors feu détecté à J | 143 828 |
| FN nouveaux pixels | 262 923 |
| Dice | 0,181984 |
| IoU | 0,100100 |
| Précision | 0,239299 |
| Rappel | **0,146819** |

Le rappel sur ce sous-domaine était 0,226733 pour le CNN et 0,262123 pour l'U-Net
historiques, à leurs seuils figés. **Le poids10 réduit les fausses alertes mais
détecte moins de nouveaux pixels de feu à ce point de fonctionnement.** Il ne
faut pas présenter le gain global de Dice sur l'ancien U-Net comme un progrès
de détection des nouvelles détections/propagation. Aucun seuil n'est modifié
après cette observation. Une nouvelle détection n'est pas nécessairement une
ignition physique.

Sur les pixels déjà détectés à J : 254 785 positifs, TP216 998, FP241 750,
FN37 787 ; Dice0,608235, précision0,473022, rappel0,851691. Les deux sous-domaines
somment exactement à la matrice de confusion globale. Les métriques de
segmentation des nouveaux pixels sont aussi conservées événement par événement ;
AP et Brier sont fournis sur le domaine global et par événement, pas revendiqués
séparément pour ce sous-domaine.

## Décision et prudence

Le poids10 reste le candidat retenu **selon la règle fixée sur les 32**, où son
Dice dépasse celui du poids30 de seulement 0,000865. Le poids30 a la meilleure
AP de réglage. Ce très petit écart et l'unique seed ne démontrent pas une
supériorité robuste du poids10. Les 169 confirment son point de fonctionnement,
sans servir à choisir un autre poids ou seuil. Ils sont désormais observés ;
aucune prétention de test final indépendant. Le test 2021 n'a pas été ouvert.

Les U-Net indépendants ont été entraînés sous PyTorch2.11.0+cu130. Cette
confirmation utilise PyTorch2.15.0+cpu, en float32. La plateforme est enregistrée
dans le verrou et les sorties ; aucun réajustement n'a compensé d'éventuelles
différences numériques avec CUDA. Les trois références partagent le domaine de
pixels et la définition des métriques, sans prétendre à une exécution matérielle
identique.

## Preuves directement consultables

- Checkpoint poids10 SHA256 :
  `7295eaa6f957e59a31b7155403881a9b3bf19933a35b6872768896bda1f1b14e`.
- Verrou avant confirmation :
  `9862113e2be1146de370d07a8d47b8570c0a20273cbbdf97bcf5a3f91af883a4`.
- Évaluation complète :
  `4944478ae4d5d2e017149992550d19d304370338c6406be84ffc3d76e50b02d5`.
- Manifeste des résultats/cartes :
  `40647f2434b6999860a1c30816525ae4f77ea7e2c59f2837c66f67fe9ce96d38`.

Voir `confirmation_cpu/status_complete.json`, `comparison.csv`,
`B2_w10_holdout169/per_event.csv`, `new_fire.json`, et `app_manifest.json`.
Trois vrais événements, dates médianes choisies par métadonnées :
`2020/fire_23756984` (2020-02-09), `2020/fire_24332880` (2020-09-04),
`2020/fire_24695963` (2020-11-22). Les PNG/NPZ contiennent les probabilités, la
cible, la segmentation et les erreurs sur le masque joint. Ils restent des
illustrations, pas un échantillon représentatif de performance.

L'application existante présente ces résultats dans « Ablation U-Net », conserve
« CNN / U-Net historiques » et les vrais résultats Big Data. Aucune publication
Kaggle, nouvelle ablation, modification des historiques, GPU, commit ou push.
