# FIRE13 : une seule expérience d'entrées enrichies

Protocole fixé le22 novembre2026, avant tout entraînement FIRE13. Identité `B2_C13_FIRE_V2_seed42`. PROP-HN2 reste intact comme secours. Suivi exploratoire : les80 événements de validation ont déjà été observés. Aucun gain n'est annoncé avant mesure.

## Données et admission

Les6913 TIFF train/validation sous `D:\GeoFire-X\WildfireSpreadTS\rasters` sont réutilisés sur place : contrôle des chemins et tailles, réemploi du certificat C36 des6913 en-têtes, comparaison exacte des dimensions/CRS/affines de leurs métadonnées certifiées. Deux fenêtres64×64, une train et une validation, permettent de vérifier directement descriptions, NaN, unités, scales/offsets et tags des23 bandes. Les comptes de NaN ainsi obtenus sont des échantillons, pas une nouvelle estimation exhaustive du corpus. Aucun TIFF2021 n'est ouvert. Les données compactes existantes suffisent au modèle ; aucune copie massive ou nouvelle partition n'est nécessaire.

`inventory.json`, `scene_geometry.json` et `raw_band_decisions.json` contiennent les reçus et les23 décisions. C8 utilise les bandes brutes12/13/14/22 (indices0-based) : pente, aspect, altitude et feu, transformées en8 canaux. Les19 autres bandes brutes restent exclues. Pour VNP09GA manquent les assets/versions/acquisitions/publications effectives ; pour NDVI/EVI2 VNP13A1 manquent la fin du composite16jours, sa vintage et sa publication ; pour GRIDMET/PDSI manquent vintage, fin de période et livraison ; pour LC annuel manque un millésime antérieur réellement publié et lié aux pixels ; pour GFS manquent cycle/issue/lead/valid-time, publication et livraison avant le cutoff, plus provenance du vent et de la pluie sum/last. Aucune de ces preuves n'est créée par la présence des bandes, leurs NaN, leur alignement ou leur date nominale.

Cette décision réutilise `reports/c36_cpu_20261121_v1/asset_evidence_search.json`, la matrice des23 bandes et les corrections documentées dans `docs/SCIENTIFIC_IMPROVEMENTS_V1.md`. Les sources primaires décrivent les produits, sans certifier les assets locaux : [GFS création/validité](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25), [générateur officiel](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset/blob/master/DataPreparation/satellites/FirePred.py), [correctif de direction du vent](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset#updates). L'ancien commentaire «8-day» des constantes NDVI/EVI2 n'est pas réutilisé comme preuve.

Les ajouts admis sont donc **dérivés uniquement du feu C8 déjà observé jusqu'àJ**. Ils héritent de l'hypothèse historique de disponibilité du feu C8 ; ils ne constituent pas une nouvelle certification de livraison opérationnelle. Ils rendent explicite une structure spatiale/temporelle déjà contenue dans C8, sans apporter de nouvelle information météo ou combustible. Hypothèse : cette représentation pourrait améliorer le classement de la propagation et limiter les expansions non justifiées. Risques : redondance, surapprentissage, dégradation du rappel, de l'AP ou du Brier. Une hausse du rappel seule ne suffit pas.

## Les13 canaux par jour

Les8 canaux C8 gardent exactement leurs valeurs et leur ordre, dont `fire_J` à l'indice7. Pour chacun des trois jours d'entrée, ajouter :

| Ajout | Calcul fixé avant résultats |
|---|---|
| Distance au feu | Distance euclidienne en pixels de grille au feu observé de ce jour, plafonnée à8 ; vaut8 en absence de détection |
| Densité locale | Moyenne du feu binaire sur7×7 pixels |
| Apparition depuis l'observation la plus ancienne | `fire_t & ~fire_oldest` ; vaut0 au premier jour, par définition |
| Évolution signée depuis cette observation | `fire_t - fire_oldest`, dans−1/0/+1 ; inclut les disparitions observées |
| Disponibilité des dérivées | Voisinage de rayon8 entièrement couvert sur les trois observations ; extérieur de scène/empreinte inconnu |

Les cartes sont calculées sur la **scène entière**, reconstituée depuis les petits shards C8, puis découpées. Pas de distance ou densité artificiellement interrompue à chaque bord64×64. Les dérivées sont mises à0 quand leur disponibilité vaut0, sans modifier les cibles ou masques d'évaluation. La distance est en pixels, pas une distance physique déclarée en mètres. Le masque de disponibilité utilise seulement les empreintes des trois entrées, jamais celle de la cible.

Distance et densité sont standardisées avec les moments de population calculés en float64 sur les4803 fenêtres train uniques, trois entrées par fenêtre, pixels supportés non paddés, puis clip±8. Les canaux binaires/signés conservent leur encodage physique. Ni cible ni masque cible n'intervient dans ces statistiques. Fichier gelé `derived_normalization.json`, lié par SHA à l'index train, la géométrie et au normaliseur C8. Aucun calcul de statistiques sur validation/test.

Résultat :13 canaux/jour,39 empilés. Même `StackedUNet/LightUNet`, base8/profondeur3 ; seule la largeur d'entrée de la première convolution change mécaniquement. Paramètres :125057 au lieu de123977 (+1080). Initialisation neuve seed42, pas de reprise C8/NEWPOS. Même loss C8 `SegmentationLoss(0.5,10)`, LocalitySampler seed42/block256 chaque tuile une fois,318train/80val/53embargo/156test fermé, mêmes cibles/masques, batch16, AdamW0,001/weight_decay0,0001/clip1, float32/AMP-TF32OFF, sans augmentation,4époques/28976updates, checkpoint époque4 fixe et seuil0,78. Les changements de représentation et de normalisation des seuls ajouts appartiennent au facteur d'entrées ; aucune modification de loss ou sampling n'est combinée.

## Contrôles CPU et ressources

Avant qualification exhaustive, le pilote a lu256 présentations LocalitySampler en environ1,109s et vérifié un forward/backward CPU fini ; aucune mesure GPU n'a eu lieu. L'extrapolation loader quatre époques≈33,5min est une estimation locale, pas un débit Kaggle garanti. Reçus détaillés dans `cost_pilot.json`, `derived_normalization.json`, `parity_full.json` et `verification_finale.json`.

La qualification réelle compare tous les115900 train et30882 validation : C8 commun, cibles, masques et feu observé identiques bit à bit ; digest de domaine validation officiel inchangé. Tests spécifiques : scène/halo, bords et empreintes inconnues, apparitions antérieures àJ+1, cible/masque cible empoisonnés, refus2021 et statistiques validation, sampler identique, normalisation train-only, modèle/gradients, AP/égalités de scores, décisions de seuil verrouillées avant estimation, checkpoint/sauvegardes et notebook autonome désactivé.

| Budget conservateur | Réserve |
|---|---:|
| Précontrôles |15min |
| Fit quatre époques |90min maximum |
| Inférence et AP exactes |120min |
| Comparaisons et analyse FP appariée |15min |
| Archives/transfert |30min |
| Marge |30min |
| Session / quota jusqu'à deux cartes réservées |**300min / 10heures-carte** |
| Disque initial / plancher en cours |18Gio / 2Gio libres |
| RAM hôte totale / disponible |≥16Gio / ≥8Gio |

Corpus monté≈9,17Gio, sans recopie ; cache C8 + dérivées :64+64Mio par processus, deux workers ; une scène maximale109648pixels, environ1,2Mo de cartes journalières. Batch d'entrée float32≈9,75Mio au lieu de6Mio. Probabilités validation≈0,825Gio ; réserve de fusion AP≈11,31Gio ; checkpoints≈66Mio plus petits fichiers et ZIP. Pas de nouvel overlay disque. La déduction Kaggle réelle n'est pas déterminée par CUDA_VISIBLE_DEVICES=0 : relever quota et durée réels avant puis après session. Préflight sans installation ni mise à jour automatique ; SciPy doit être présent et passer le contrôle numérique embarqué.

## Lancement exact, après nouvelle autorisation

1. Importer `notebooks/GEOFIREX_C13_FIRE_DISABLED.ipynb` dans un notebook privé neuf, InternetOFF ; attacher les onze datasets p00–p10 existants, listés dans `kernel_metadata_NOT_PUBLISHED.json`. Tous les drapeaux àFalse. Exécuter les cellules d'extraction et précontrôle CPU une fois. Le notebook contient code, index, géométrie, normalisation, preuves de parité et références : aucun clone, téléchargement ou TIFF brut requis.
2. Après autorisation explicite de cet unique fit, copier les templates `ROOT/reports/fire13_20261122_v1/authorization_gpu-preflight_DISABLED.json` et `authorization_fire13-fit_DISABLED.json` vers `/kaggle/working/fire13_gpu_authorization.json` et `/kaggle/working/fire13_fit_authorization.json`. Compléter `authorized=true`, opérateur/dateUTC, date du contrôle quota, durée≥300min, quota et réserve≥10heures-carte, deux destinations de sauvegarde distinctes. Conserver action, identité, runs=1 et SHA du manifeste. Ne pas remplacer cette autorisation par un ancien grant.
3. Dans un noyau neuf avec T4, activer seulement `RUN_GPU_PREFLIGHT=True`. Exiger PASS : onze montages, ressources, torch2.11.0+cu130/NumPy2.1.3/CUDA13.0/cuDNN92700, une seule TeslaT4 visible et tests numériques SciPy. Puis mettre `RUN_FIT=True` une seule fois, laisser `RUN_RECOVERY=False`. Le runner crée un verrou exclusif, entraîne puis archive TRAINING-SAFE avant toute évaluation. Aucune reprise ou seconde tentative de fit après interruption ; un run incomplet restePARTIAL. Le verrou de session ne remplace pas le suivi humain de l'autorisation consommée entre sessions Kaggle indépendantes.
4. Télécharger chaque ZIP et sonSHA, vérifier deux copies distinctes via `scripts/run_fire13.py backup --archive CHEMIN --sidecar CHEMIN_SHA --local-root DOSSIER_LOCAL --external-root D:/GeoFire-X/GeoFire-X-V2`. Si seul le calcul d'évaluation échoue après un fit complet, utiliser TRAINING-SAFE et une autorisation `fire13-evaluate` distincte dans la section récupération ; aucun optimiseur n'est alors créé.

## Mesures et décision fixées

Comparaison C8/FIRE13 sur les mêmes80 événements, avec NEWPOS4/NEWPOS2 conservés comme références historiques. Mesures officielles à0,78 : rappel/Dice/précision/AP exacte nouveaux feux, FP, Dice/APglobaux, Brier, macroDice, comptages par événement. AP calculée exactement avec égalités globales ; aucun proxy d'AP. Bootstrap apparié2000 réplications, seed20261121, unité événement ; AP exacte rapportée séparément, sans bootstrap pixels ou APapproximée. Une seule seed ne mesure pas la variance d'initialisation.

Les critères scientifiques V2 restent **strictement ceux déjà fixés**, copiés dans `acceptance.json` : APnouveaux≥0,11512837301201887, APglobale≥0,34023390370637985, Diceglobal≥0,4339254871567759, rappel nouveaux≥0,08138351147320427, Dicenouveaux≥0,1106963425412671, FP≤29134, macroDice≥0,22778327086319602, Brier≤0,0031547185297208474 ; domaine et budget autorisé obligatoires. `FIRE13_acceptance.json` fait foi. Pas de promotion automatique.

Analyse supplémentaire préfixée du compromis FP : cinq groupes déterministes de16, calibration64/estimation16, taux de référence exact C8 `25334/103727130`. Sur chaque calibration, maximiser le rappel sous budget entier à ce taux, égalités indivisibles, seuil le plus haut à rappel égal. Les cinq décisions sont sauvegardées et hachées **avant** l'estimation. Comparer ensuite aux comptages C8 à0,78 sur les mêmes événements, avec bootstrap apparié ; rapporter les FP réellement obtenus et tous les dépassements. Le budget de calibration ne garantit pas un budget tenu hors événement : ne jamais annoncer des FP identiques si ce n'est pas observé. Ce diagnostic exploratoire ne choisit pas un nouveau seuil opérationnel, ne change ni AP/Brier ni les gardes de promotion.

Décision de préparation : GO uniquement après tests CPU et parité réellePASS. Lancement GPU actuellementNO-GO tant que l'autorisation nouvelle et le préflight réel ne sont pas acquis. Aucun pixel2021 n'est utilisé ; aucun ancien fichier scientifique, checkpoint, split ou archive n'est modifié. Aucun commit, push, Spark, Kafka ou développement BigData.
