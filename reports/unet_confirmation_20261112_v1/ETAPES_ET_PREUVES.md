# U-Net retenu — vérification, confirmation et démonstration du 18 novembre

## Entrées et décision

Le ZIP fourni a été trouvé dans
`C:/Users/nourb/Downloads/GeoFireX_UNet_ABLATION_3_MODELS.zip`. Une copie identique
est conservée dans `inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip` pour rendre le
livrable autonome. SHA256 des deux copies :
`a122dfa4ac419f3b8f8d32e1ae309033ef02ccfb1891fbaed3b5ed7c6a751e07`.
Ni les trois checkpoints, ni les modèles historiques n'ont été modifiés.

L'audit `ablation_verified.json` vérifie les SHA des trois checkpoints, la
cohérence config/checkpoint, époque suivante=5, curseur=0, quatre époques finies,
23 404 updates, 93 603 tuiles train, seed42, batch16, AdamW, architecture et grille
de seuils identiques. Les configs ne diffèrent que par `pos_weight`. L'état initial
enregistré a le même SHA pour les trois branches ; les trois états entraînés ont
des SHA différents. Cette vérification exploite les enregistrements et les poids
sauvegardés ; elle ne rejoue pas l'entraînement.

L'identité exacte des 32 événements, de l'index, de la normalisation et du
protocole est confrontée au ZIP de calibration historique. Les métriques sont
recalculées depuis les matrices de confusion enregistrées ; le maximum de Dice
sur la grille sauvegardée, les sommes par événement et le Brier pondéré sont
contrôlés. L'AP enregistrée est vérifiée comme invariante entre seuils ; elle n'est
pas recalculée sur les 32, car aucune nouvelle inférence de réglage n'est lancée.

| Poids | Seuil | Dice micro sur 32 | AP sur 32 | Brier sur 32 |
|---:|---:|---:|---:|---:|
| 100 | 0,96 | 0,382562 | 0,259647 | 0,00947959 |
| 30 | 0,88 | 0,386417 | 0,269986 | 0,00334782 |
| **10** | **0,78** | **0,387282** | 0,267884 | 0,00278906 |

Le poids 10 est retenu selon la règle annoncée, Dice micro. Son avance sur le
poids 30 est seulement **0,000865109**, soit 0,08651 point de pourcentage de Dice.
Le poids 30 a une AP supérieure de 0,002102166. Une seed et des différences si
faibles ne démontrent aucune supériorité statistique générale du poids 10.
La grille s'étend jusqu'à 0,999 pour ces nouveaux modèles : 0,78 est un maximum
intérieur testé, pas une borne supérieure atteinte comme dans l'ancien résultat.

Le fichier de protocole dans le ZIP historique a le SHA
`5983837d035a9cd536969bc3c601705d054e100e7f85aae279e4adc385d2630e`.
La copie Windows `protocol32.json` peut avoir une empreinte différente à cause
des fins de ligne ; l'audit se réfère aux octets du protocole réellement utilisé
dans Kaggle, et ne confond pas différences de transport et identité scientifique.

## Confirmation unique, sans GPU

L'environnement local comporte PyTorch `2.15.0+cpu`. La confirmation est exécutée
sur **CPU uniquement** ; aucun lancement GPU n'a été fait, donc aucune autorisation
GPU supplémentaire n'a été nécessaire. Les entraînements provenaient de
PyTorch `2.11.0+cu130` : cette différence de plateforme est consignée, sans
réajuster le seuil pour la compenser. Le calcul utilise les scores float32,
sans autocast, quantification ou recalibrage probabiliste.

Avant le premier pixel de confirmation, `confirmation_cpu/confirmation_locked.json`
fige le checkpoint poids10, le seuil 0,78, les SHA des sources et les 169 IDs.
Ces IDs sont disjoints des 32 et couvrent exactement 65 012 tuiles. La validité
reste l'intersection de la cible et des cinq jours historiques, padding exclu.
Ni les poids 30/100, ni le CNN ne sont réinférés sur ce panel : leurs références
historiques sont lues depuis les résultats existants sur les mêmes pixels.

Commande de vérification, **sans inférence** :

```powershell
.\.venv\Scripts\python.exe scripts\confirm_unet_ablation.py
```

Commande de confirmation déjà utilisée :

```powershell
.\.venv\Scripts\python.exe scripts\confirm_unet_ablation.py --confirm --device cpu --workers 0 --batch-size 32
```

Ne pas relancer cette dernière commande après la réussite : le dossier de sortie
existant est refusé pour éviter tout écrasement. Le code CPU refuse tout autre
device et n'offre aucun paramètre de seuil ou de choix du poids. Il ne contient
aucun optimiseur ni entraînement. Le modèle et le seuil ne sont jamais choisis
depuis les résultats des 169 événements. Le test 2021 reste fermé.

Les sorties complètes se trouvent dans `confirmation_cpu/` :

| Preuve | Contenu |
|---|---|
| `confirmation_locked.json` | Décision figée avant toute inférence de confirmation |
| `B2_w10_holdout169/evaluation.json` | AP exacte, Brier, confusion globale/par événement, probabilités |
| `B2_w10_holdout169/per_event.csv` | Dice, IoU, précision, rappel, AP et Brier pour les 169 événements |
| `B2_w10_holdout169/new_fire_per_event.csv` | TP/FP/FN/TN et métriques de segmentation hors feu détecté à J |
| `new_fire.json` | Partition nouveaux pixels / pixels déjà détectés à J |
| `comparison.csv`, `comparison.json` | CNN historique, U-Net historique, U-Net poids10, même domaine |
| `app_manifest.json` | Cartes réelles PNG/NPZ et seuil confirmé, identités des sources |
| `status_complete.json` | Fin effective, durée, CPU, SHA du manifeste et de l'évaluation |

AP et Brier sont calculés sur les probabilités originales, sur tout le domaine
valide et par événement. AP exacte : tri externe avec regroupement des scores
égaux, pas de quantification ni d'aire PR trapézoïdale. Les métriques nouveaux
feux décrivent `valid & (feu_J == 0)`, avec TP/FN des nouvelles détections à J+1 ;
elles ne démontrent pas une ignition physique. Les sous-domaines nouveaux feux /
déjà observés doivent sommer exactement aux comptages globaux.

La mosaïque PNG de synthèse conserve aussi un aperçu visuel à 0,50 du générateur
historique. Cet aperçu n'entre pas dans les métriques ni dans la sélection :
l'évaluation de confirmation ne contient qu'une ligne, au seuil 0,78. Les couches
affichées dans l'onglet de confirmation utilisent le seul seuil figé 0,78.

## Application React/FastAPI

Depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe scripts\serve_demo.py --results-only --port 8014
```

Ouvrir **http://127.0.0.1:8014**. Le nouveau parcours commence dans « Ablation
U-Net ». Le lanceur n'importe plus `geofirex.serving` ni PyTorch en mode résultats.
Il n'initialise donc pas le Predictor historique ni ses vérifications de snapshot.
Ces vérifications restent intégralement actives lorsqu'on demande réellement le
mode de prévision ; aucun manifeste ni contrôle n'a été désactivé ou réécrit.

Parcours de présentation :

1. Lire la table des trois poids sur les **32**. Expliquer le critère Dice et la
   faible différence 10/30 ; signaler l'AP légèrement meilleure du poids30.
2. Montrer la décision verrouillée poids10/seuil0,78 et les résultats de la
   confirmation **169**. Comparer aux deux références historiques, sans changer
   la règle parce qu'une métrique de confirmation serait meilleure ailleurs.
3. Consulter les distributions par événement, les nouveaux pixels de feu, puis
   trois vrais événements/date : vérité, score, segmentation, FP rouges et FN bleus.
   Les exemples sont choisis par métadonnées, pas par performance. Les cartes sont
   des grilles de pixels, sans géoréférencement inventé ; invalides transparents.
4. Ouvrir « CNN / U-Net historiques » pour retrouver les résultats antérieurs.
   La vue de prévision est désactivée en mode résultats, sans promettre un service
   de prédiction en ligne pour le nouveau checkpoint.

Nouvelles routes : `/science/unet-ablation`, `/science/retained-model`,
`/science/unet-assets/<chemin autorisé>`. La confirmation n'est servie comme
complète qu'après contrôle des SHA du verrou, du manifeste et de l'évaluation,
ainsi que des IDs et comptages du domaine commun. Aucune API de réglage du seuil.

## Démonstration Big Data — deux minutes

L'onglet conserve les vrais résultats Spark 4.0.2 en `local[2]`. Il s'agit d'une
exécution locale à deux threads, **pas d'un cluster**. Aucun Spark n'est relancé.

1. **Inventaire** : 607 événements, 13 607 observations. Train : 176 événements
   2018 / 3 773 observations et 74 événements 2019 / 1 425 observations.
   Validation 2020 : 201 événements / 4 292 observations. L'ancien inventaire
   2021 contient 156 événements / 4 117 observations, uniquement des métadonnées,
   sans ouverture de pixels ou calcul de score de test.
2. **Fenêtres** : 3 948 cibles train et 3 287 validation, alignées pour T=1/3/5.
   Expliquer la différence entre observation, fenêtre et tuile : les nombres ne
   représentent pas le même objet.
3. **Requêtes SQL** : groupements annuels et qualité, contrôle des doublons,
   contrôle annuel et jointure fenêtre/cible. Zéro doublon et zéro violation.
   Montrer le SQL enregistré et ses durées, plutôt que lancer un nouveau job.
4. **Parquet → API → React** : cinq petits agrégats existants sont relus avec
   PyArrow et comparés au rapport, empreintes SHA affichées. Par exemple qualité
   train : 429 682 pixels feu / 414 333 956 valides ; validation : 674 958 /
   345 155 486. Ces comptes d'observations ne sont pas ceux des masques joints
   des fenêtres de calibration.

Temps historiques : JVM 43,324 s ; traitement 21,787 s ; total 65,111 s.
Sources : `reports/mission_20261111/spark/report.json`, ses requêtes et Parquet.
Leurs empreintes sont conservées dans les preuves de livraison. Une interface
réactive relisant ces sorties ne constitue pas un nouveau benchmark Spark.
