# Priorité scientifique suivante : modalités disponibles à J

La Focal est une ablation ponctuelle dont la préparation logistique est achevée ; elle n'est pas la stratégie principale d'amélioration. **La prochaine implémentation prioritaire est un chemin multimodal séparé et comparable au U-Net actuel**, précédé des seules preuves de disponibilité temporelle qui manquent. Aucun entraînement, corpus ou normaliseur supplémentaire n'est produit maintenant.

Cette décision de priorité suit la demande du14novembre2026 et succède à l'ordre A→E du document préparatoire. La configuration de l'unique expérience Focal et ses critères restent figés ; on ne modifie pas le contrat existant pour réordonner les travaux futurs. L'ablation T=1 peut attendre et n'est plus un préalable à l'extension multimodale.

## Fondations déjà validées et conservées

Le partage2018–2019/2020/2021, les fenêtres J−2/J−1/J→J+1, le domaine de validité historique incluant les cinq jours antérieurs, les93 603 tuiles train et les32 événements de développement restent la base de comparaison. Ne pas refaire leurs audits, refitter le normaliseur actuel ou reconstruire les11 partitions compactes. Le test2021 reste fermé et les169 événements déjà observés restent descriptifs, jamais un jeu de sélection indépendant.

`real_constants.py` décrit23 bandes **brutes**, pas23 variables nécessairement causales disponibles à J. `feature_profiles.py` prévoit déjà trois schémas :

| Profil existant | Continues brutes, indices0-based | Encodage produit par `unet_data.Normalizer` | Situation |
|---|---|---|---|
| `conservative_fire_terrain` | 12 pente,14 altitude | 2z +2missing +sin/cos/missing aspect +feu =8 | Compact actuel, inchangé |
| `terrain_vegetation_v1` | 0M11,1I2,2I1,3NDVI_last,4EVI2_last,12pente,14altitude | 7z +7missing +3aspect +feu +17LC/unknown =36 | Profil prévu, disponibilité réelle non validée |
| `multimodal_observed_j_v1` | précédentes +5pluie,6vitesse vent,8Tmin,9Tmax,10ERC,11humidité spécifique,15PDSI | 14z +14missing +3aspect +feu +18LC =50 | Cible multimodale complète après preuves |

L'occupation du sol est la bande16, encodée en17 classes entières1–17 et un canal unknown ; aucune z-normalisation ni interpolation bilinéaire des classes. Aspect13 reste circulaire sin/cos avec indicateur manquant. Feu22 reste `~isnan` selon la convention historique, pas un seuil sur un code de détection. Direction observée7, direction prévue19 et les prévisions17–21 ne font partie d'aucun des deux profils étendus proposés.

## Seuls contrôles encore indispensables : causalité par asset

Réutiliser la matrice des23 bandes et les constats déjà établis dans [le bilan scientifique existant](../../docs/SCIENTIFIC_IMPROVEMENTS_V1.md). Les commentaires du code ne remplacent pas ces corrections documentaires : «8-day» ne suffit pas pour caractériser le composite NDVI/EVI2 et le défaut historique de quadrant documenté concerne la direction **GFS prévue**, pas un défaut démontré de GRIDMET observé.

Pour chaque clé exhaustive `(événement, J, date d'entrée, bande)`, fournir ID/version/empreinte du produit/asset et de sa preuve, bornes d'acquisition ou de composite, publication réellement accessible et coupure UTC fixée à finJ. Reprendre `experiment_readiness.validate_availability`, puis revue de la vérité des preuves externes. `reviewed=true` et un texte libre dans `feature_profiles.feature_profile` seuls sont insuffisants : le validateur renforcé doit être relié au futur constructeur et au manifeste de l'overlay.

| Groupe des23 bandes | Preuve manquante et décision avant inclusion |
|---|---|
| 0–2, réflectances | Version du produit, acquisitions et publication avant finJ ; géoréférencement/qualité, pas seulement date de fichier |
| 3–4, NDVI/EVI2 | Fin effective du composite et publication ≤J ; `last` et timestamp nominal n'établissent pas la causalité. Si nécessaire sélectionner une vintage plus ancienne réellement publiée, avec âge explicite et clés conservées |
| 5–6,8–11, météo observée/ERC | Fenêtre journalière entièrement achevée et vintage disponible avantJ ; distinguer early/provisional/permanent, documenter latence. Si indisponible, retarder explicitement l'entrée avec version de profil séparée, sans prétendre utiliser l'observationJ |
| 7, direction observée | Exclue ; disponibilité et encodage circulaire à traiter dans une proposition distincte |
| 12–14, relief/aspect | Preuves statiques existantes réutilisées ; pas de nouvel audit de cette base |
| 15, PDSI | Période couverte, borne finale, vintage et publication≤J. Un intervalle contenantJ ne prouve pas sa disponibilité ; pas de consolidation future |
| 16, occupation du sol | La classe annuelle de l'année deJ peut utiliser des observations futures. Exiger un millésime antérieur effectivement publié avantJ, avec correspondance pixels/produit et rééchantillonnage catégoriel documentés |
| 17–21, GFS | Exclues du premier modèle. Une éventuelle extension devra prouver issue-time, publication, valid-time et lead-time pertinents pourJ+1 ; la bande quotidienne n'est pas automatiquement une prévision deJ+1 |
| 22, feu | Hypothèse historique conservée pour comparaison ; distinguer date d'acquisition et disponibilité opérationnelle. Aucun gain prospectif n'est revendiqué sans cette preuve également |

Les versions provisoires de GRIDMET sont remplacées au fil du temps : une valeur réanalysée plus tard ne prouve pas ce qui était accessible le jourJ. La carte MCD12Q1 est annuelle ; la bande PDSI appartient à un produit de sécheresse multi-jours. Ces caractéristiques motivent les contrôles ciblés, sans refaire les audits des tuiles. Sources primaires : [GRIDMET](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET), [MCD12Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1), [GRIDMET DROUGHT](https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT).

Si les assets ne prouvent pas la disponibilité, le profil concerné reste **BLOCKED**. Ne pas remplir les bandes indisponibles par zéro pour déclarer le profil validé, ni déplacer le cutoff aprèsJ. Ne pas réduire les tuiles ou les masques selon la présence des modalités : conserver le domaine et encoder les vraies données manquantes seulement après validation de la causalité.

## Changements futurs du corpus et du normaliseur

Le compact8 contient uniquement les canaux prétraités relief/feu et ne permet pas de reconstruire NDVI, EVI2, LC, météo ou PDSI. Préparer **plus tard**, après preuve et autorisation, un overlay immuable séparé à partir des assets bruts autorisés2018–2020. Aucun téléchargement, overlay, normaliseur ou nouvelle publication n'est effectué ici.

L'overlay doit :

1. Référencer les SHA de la base compacte, du manifeste de fenêtres et des preuves temporelles ; employer les mêmes clés événement/date-cible/tile-row/tile-col et les trois dates d'entrée. Stocker les éventuelles vintages différentes explicitement, sans changer la cible.
2. Stocker seulement les modalités supplémentaires, avec masques de données manquantes, grille/CRS/résolution/nodata et rééchantillonnage documentés. Préserver les géométries, tuiles, domaine cinqjours et masques de la base ; pas de sélection de cas favorables. Vérifier exhaustivité/unicité des clés au moment de sa construction future.
3. Garder les fichiers et normaliseur8 certifiés. Écrire un **nouveau** normaliseur/manifest composite et son SHA ; réutiliser les statistiques pente/altitude historiques plutôt que normaliser deux fois les entrées compactes. Les nouvelles moyennes/std/population/counts sont fit uniquement sur les chemins d'entrée train uniques2018/2019, jamais cibles,2020 ou2021. Conserver liste/empreinte des chemins et ordre des canaux ; all-missing refusé, canal constant std1, imputation à la moyenne train +indicateur missing, clipping déclaré.
4. Assembler les canaux par nom selon `Normalizer.channel_names` ; le profil36 place le feu à l'indice17 et le profil50 à31. **`FastCompactTiles.__getitem__` et `compact_calibration.evaluate` codent actuellement le feu à l'indice7.** Le nouveau dataset doit réutiliser les y/valid/feu de la base8 et l'évaluateur multimodal doit recevoir explicitement la position/nature du feu observé. Ne pas lui transmettre aveuglément un tenseur36/50 avec le code actuel.
5. Isoler le nouveau loader/évaluateur et le nouveau modèle de Studio V2 et des runners historiques. Sur fixtures synthétiques puis métadonnées réelles autorisées, vérifier clés identiques, invariance y/valid/observed, équivalence du chemin8, ordre/noms des36/50 canaux et absence de fuite. Aucune modification de `read_raw` ne doit ouvrir2021 en passant par un manifeste global.

Le `fit_normalizer` existant fournit les conventions d'encodage et un fit sur entrées uniques ; son appel ne constitue pas à lui seul une preuve de disponibilité ni une protection contre un manifeste mal sélectionné. Le futur constructeur devra imposer explicitement les années autorisées et les clés causales avant toute ouverture de raster.

## Premier modèle puis candidats ultérieurs

**Premier modèle recommandé : le même Light U-Net B2, fusion précoce T=3, avec le profil `terrain_vegetation_v1` (36 canaux/jour,108 à l'entrée)**, après validation des réflectances/NDVI/EVI2/LC. Cela mesure d'abord l'apport végétation/combustible sans changer simultanément l'architecture. Conserver seed42, mêmes tuiles/masques, batch16, quatre époques/23 404 updates, AdamW et la **loss historique**0,5BCE(pos10)+0,5Dice. Ne pas combiner automatiquement Focal et nouveaux canaux : cela confondrait les effets. Aucun nouveau fit historique8 n'est requis ; comparer aux résultats8 existants lorsque le domaine est exactement identique.

La cible d'implémentation suivante est **le même U-Net au profil `multimodal_observed_j_v1`,50 canaux/jour,150 à l'entrée**, ajoutant pluie, vitesse du vent, températures, ERC, humidité et sécheresse aux36 canaux. L'activer seulement si les preuves météo/PDSI sont recevables. Une absence de preuve météo n'impose pas de lancer un modèle50 vide ; la branche36 peut rester la seule expérience multimodale autorisée. Les deux étapes devront faire l'objet de décisions distinctes et ne forment pas une recherche de paramètres automatisée.

Puis **U-Net–ResNet18**, à modalités, T et domaine constants : documenter adaptation de la première convolution108/150canaux, normalisation, nombre de paramètres et éventuel préentraînement ImageNet. Garder le budget d'updates comparable et rapporter en plus le temps/mémoire ; ne pas charger de poids WSTS/WSTS+ tant que les recouvrements d'années/événements/tuning avec2021 ne sont pas exclus.

Enfin **U-TAE**, pour tester la fusion temporelle à modalités constantes. Commencer par T=3 et les mêmes cibles/domaines ; changer ensuite T serait une autre expérience. Documenter positional/time encoding, masquage temporel, adaptation de la tête binaire, capacité et coût. L'implémentation officielle fournit une architecture pour séries satellitaires ; elle ne garantit aucun gain sur GeoFire-X. [Code officiel U-TAE](https://github.com/VSainteuf/utae-paps).

WSTS+ motive l'étude des architectures et séries temporelles, mais ses protocoles, plis et scores ne sont pas directement les scores pooled de notre développement32. On ne transpose ni ses AP comme cibles chiffrées, ni ses poids publics sans audit de recouvrement. [Article WSTS+, version3](https://arxiv.org/html/2502.12003v3).

## Ordre décidé, coûts et critères avant chaque future autorisation

| Ordre | Travail / expérience | Coût défendable maintenant | Critère de passage |
|---|---|---|---|
| 0 | Terminer la préparation Focal ; éventuellement UN run si autorisé | Réserve120min de session, fit scénario11,1–22,2min, plafond fit45min ;12GiB libres | Cinq critères Focal inchangés et archive/checkpoint vérifiés ; aucun candidat de secours |
| 1 | Preuves àJ, schéma overlay et normaliseur multimodal séparé | 0 entraînement GPU ; travail de provenance et futur I/O brut non chiffré sans inventaire causal | Toutes les clés couvertes et preuves recevables ; Studio/base/test intacts |
| 2 | U-Net36 végétation/LC comparable | 1 futur fit,23 404 updates ; entrées4,5×C8 àT constant ; temps GPU et compression non mesurés | Domaine identique, normalisation train-only, critères figés avant résultats |
| 3 | U-Net50 ajout météo/sécheresse | 1 futur fit si causalité validée ; entrées6,25×C8 ; coût réel à qualifier ultérieurement | Preuves supplémentaires et comparaison50vs36 isolant ce groupe |
| 4 | U-Net–ResNet18 sur le profil retenu | 1 futur fit proposé ; mémoire/temps inconnus, pas d'estimation à partir des seuls paramètres | Gain au même protocole suffisamment utile face au surcoût ; préentraînement documenté |
| 5 | U-TAE àT3 sur ce même profil | 1 futur fit proposé ; coût attention/activations à mesurer lors d'une mission distincte autorisée | Apport temporel sans changement de domaine ; métriques et ressources rapportées |

Ratios de volumes uniquement : batch16/T3/64²/float32 contient6MiB d'entréesC8,27MiBC36,37,5MiBC50, hors activations, cibles et workers. Même B2/base8/profondeur3 : l'élargissement de sa première convolution ajoute6 048 paramètres pourC36 et9 072 pourC50, soit130 025 et133 049 paramètres ; ces calculs d'architecture ne sont pas des mesures de temps. Les9,85GB compressés du compact actuel ne se multiplient pas mécaniquement par4,5 ou6,25 : l'overlay LC/missing se compresse différemment. Aucun budget global de toutes ces expériences n'est autorisé par ce plan.

Pour les futures comparaisons multimodales, proposition à figer **avant tout résultat** : AP exacte pooled principale, rapport systématique Dice/IoU micro, Dice macro par événement, rappel/FP nouveaux feux, Brier et ressources ; conserver au minimum les mêmes seuils de gain et non-régression que Focal comme garde d'ingénierie (+0,005AP,−0,005Dice,+0,02rappel nouveaux,≤1,15FP nouveaux,−0,01Dice macro), avec référence explicite pour chaque contraste36vs8 puis50vs36. Cela ne modifie pas les critères Focal ni ne constitue une campagne déjà autorisée.

Sélection32 uniquement selon une règle préenregistrée, checkpoint d'époque fixe et pas de sélection d'architecture sur2021/169. Rapporter limites de la seed unique et du développement réutilisé ; une différence de score n'est pas une preuve de généralisation indépendante. Une future campagne de réplication éventuelle serait décidée séparément, jamais déclenchée dans cette mission.
