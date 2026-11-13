# Résultats réels, loss et prochaine expérience

Source : `reports/calibration_20261112_v1/GeoFireX_CALIBRATION_RESULTATS.zip`, SHA256
`bd1e2f7e70ad6b8d1ce516e72626b1dfc9cb7050d0e02ad120b7b5a38031d6dd`.
Les valeurs ci-dessous sont lues dans les sorties existantes. Aucun modèle n'a
été réévalué pour écrire ce rapport et aucun résultat historique n'a été modifié.

## Gain observé sur les 169 événements de confirmation

| Modèle | Dice 0,50 → 0,95 | Δ Dice | Précision 0,50 → 0,95 | Rappel 0,50 → 0,95 | FP 0,50 → 0,95 | AP | Brier |
|---|---|---:|---|---|---|---:|---:|
| CNN B1 | 0,331877 → 0,440879 | +0,109001 | 0,221553 → 0,359088 | 0,661057 → 0,570920 | 1 307 563 → 573 648 | 0,268377 | 0,00569349 |
| U-Net B2 | 0,250793 → 0,406695 | +0,155901 | 0,149142 → 0,323257 | 0,787613 → 0,548191 | 2 529 549 → 646 071 | 0,300972 | 0,00951151 |

Le gain relatif de Dice est de 32,84 % pour B1 et 62,16 % pour B2. Les FP baissent
de 56,13 % et 74,46 %, mais le rappel baisse de 9,01 et 23,94 points. Les FN montent
respectivement de 190 809 à 241 552 et de 119 564 à 254 347. Il faut présenter cet
échange, sans qualifier le seuil de gain gratuit. Aucun test statistique de
significativité n'est revendiqué. AP et Brier sont identiques avant/après seuil.

L'IoU à 0,95 vaut 0,282774 pour B1 et 0,255252 pour B2. Les scores micro agrègent
217 506 967 pixels valides, 562 953 positifs et 65 012 tuiles par modèle.
Le Dice macro événement est nettement plus bas : 0,213258 pour B1 et environ
0,202157 pour B2, car chaque événement y pèse
autant, indépendamment de sa surface. Ne pas confondre les deux agrégations.

Les nouveaux pixels de feu restent difficiles : sur le sous-domaine sans feu
détecté à J, rappel à 0,95 = 0,226733 pour B1 et 0,262123 pour B2 ; Dice = 0,215867
et 0,208299. Une détection nouvelle à J+1 ne prouve pas une ignition physique.

Le modèle de segmentation est choisi sur les **32 événements** : Dice à seuil
sélectionné 0,384148 (B1), 0,356238 (B2). Pour le classement probabiliste, l'AP sur
ces mêmes 32 est 0,207868 (B1), 0,262108 (B2). Les résultats des 169 sont une
confirmation désormais observée, pas un critère pour multiplier les essais.

## Loss actuelle : ce qui est effectivement implémenté

Dans `scripts/train_compact_benchmark.py` : `SegmentationLoss(.5,100)` ; dans
`src/geofirex/unet.py` :

```text
L = 0.5 × Σvalid[-100 y log(p) - (1-y) log(1-p)] / Σvalid
  + 0.5 × moyenne_images_valides[1 - (2 Σvalid p y + ε)/(Σvalid p + Σvalid y + ε)]
ε = 10^-6 ; p = sigmoid(logits)
```

Le poids ne multiplie que le terme positif de la BCE ; il n'est ni un poids
global du feu dans toute la loss, ni une correction des probabilités de sortie.
À logit nul, à masque identique, le gradient BCE d'un positif pèse 100 fois
celui d'un négatif en valeur absolue. La pondération peut favoriser le rappel,
comme décrit par [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

Le train contient 313 977 876 pixels valides et 409 488 positifs : prévalence
0,130419 %, ratio non-feu/feu 765,757. Le poids 100 est donc **inférieur** au ratio
de classes ; il n'est pas prouvé excessif parce qu'il semble grand. Le sampler
visite toutes les 93 603 tuiles sans suréchantillonnage positif. Catégories :
85 940 négatives, 6 136 positives, 1 527 négatives difficiles.

La Dice est moyennée **par image**, la BCE **par pixel valide**. Sur une image
sans feu cible, sa Dice loss vaut `1-ε/(Σp+ε)` ; pour une masse prédite non
négligeable, le gradient est très faible. Ce terme est alors presque constant
et la BCE fournit l'essentiel du signal de rejet du non-feu. Sur une petite
image positive, le terme Dice peut en revanche peser fortement. Changer seulement
`pos_weight` isole une cause plausible, sans attribuer tous les FP à ce paramètre.

Les paramètres de loss ne sont pas stockés explicitement dans les checkpoints
historiques ; leur valeur est établie à partir du code disponible et du
`train_audit.json` exporté. Les nouvelles variantes les enregistrent dans config.

## Probabilités : une surestimation mesurée sur les 32 événements

| Mesure sur les 32 | B1 | B2 |
|---|---:|---:|
| Fréquence réelle de feu | 0,176380 % | 0,176380 % |
| Moyenne des scores sigmoid | 1,428948 % | 2,409756 % |
| Pixels avec score ≥ 0,95 | 127 759 | 164 668 |
| Score moyen dans la classe [0,95 ; 1] | 0,982734 | 0,983529 |
| Fréquence réelle de feu dans cette classe | 0,314381 | 0,266117 |
| Brier | 0,00430631 | 0,00809031 |
| Brier de la constante = prévalence train | 0,00176090 | 0,00176090 |

Les diagrammes de fiabilité montrent donc un décalage important : un score moyen
de 0,98 correspond ici à environ 31 % ou 27 % de positifs, pas à 98 %. Une meilleure
AP du U-Net n'implique pas une meilleure calibration probabiliste. Le Brier seul
mélange fiabilité, résolution et incertitude ; la conclusion de surestimation
s'appuie aussi sur les fréquences par classe de probabilité. Voir la
[documentation de calibration scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).

Pour une BCE pondérée **seule**, l'optimum théorique sous hypothèses idéales est
`p_w = wη/(1-η+wη)`. À w=100, p_w=0,95 correspondrait à η≈0,1597, pas 0,95.
Cette relation est une dérivation de la BCE, pas un calibrateur valable pour
le modèle actuel : Dice, capacité, dépendance spatiale et optimisation changent
le problème. Aucune inversion analytique n'est appliquée dans l'application.

## Pourquoi 0,95 ne démontre pas l'optimum du seuil

La Dice sur les 32 continue à monter entre 0,90 et 0,95 : B1, 0,370243 → 0,384148 ;
B2, 0,330637 → 0,356238. L'optimum retenu touche
la borne de la grille : c'est le **meilleur seuil testé dans [0,10 ; 0,95]**.
On ne sait pas si un seuil supérieur serait meilleur, ni si le même maximum
se transférerait à d'autres événements. Le présent livrable conserve exactement
0,95 et ne relance pas la calibration terminée. Les anciennes métriques restent
associées à ce seuil et à leurs checkpoints SHA.

Pour les **nouveaux entraînements uniquement**, la grille étendue est définie
avant leurs résultats : 0,10–0,95 par 0,005, puis 0,96, 0,97, 0,98, 0,99, 0,995,
0,999. Elle est identique pour tous les poids et les deux architectures, évaluée
uniquement sur les 32. Une nouvelle borne atteinte sera encore signalée.

## Historique et décision d'expérience

À l'époque 4, les AP sur les 32 restent meilleures qu'à l'époque 1, mais les
courbes ne garantissent pas qu'un simple prolongement suffise. B1 : loss val
0,6579 → 0,6910 → 0,6477 → 0,6620 ; B2 : 0,6109 → 0,5945 → 0,5884 → 0,5829.
Les propositions anciennes de continuation restent dans les historiques ; la
nouvelle demande impose des variantes **indépendantes**, décrites dans
`../loss_ablation_20261112_v1/PLAN.md`. Elles ne reprennent pas les modèles entraînés.
Les valeurs des loss pondérées entre poids 100, 30 et 10 ne devront pas servir à
classer les variantes : leurs échelles changent mécaniquement avec le poids.
