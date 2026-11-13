# Application intégrée — suivi du 18 novembre

L'application React/FastAPI existante comporte maintenant trois vues : résultats
CNN/U-Net, Big Data, prévision historique. Les routes historiques de prédiction
ne changent pas de modèle ni de seuil. Les résultats des modèles compacts sont
rejoués depuis les sorties Kaggle ; aucun nouveau checkpoint n'est transféré.

## Démarrage

Depuis la racine du dépôt, avec les dépendances locales existantes :

```powershell
.\.venv\Scripts\python.exe scripts\serve_demo.py --results-only --port 8012
```

Ouvrir http://127.0.0.1:8012. Le frontend arrive par défaut sur les résultats
CNN/U-Net terminés. Aucun modèle n'est chargé ; les vraies cartes et métriques
proviennent du ZIP fourni. Le bouton d'inférence historique est désactivé dans
ce mode pour ne pas annoncer un service absent.

Avec un snapshot historique compatible avec ses sources certifiées, les routes
scientifiques s'ajoutent aussi au service de prévision existant :

```powershell
.\.venv\Scripts\python.exe scripts\serve_demo.py --port 8012
```

Le frontend a été compilé avec les dépendances locales, sans téléchargement.
Pour recompiler après une modification : `npm --prefix frontend run build`.

Le lancement réel de l'ancien snapshot configuré dans
`artifacts/demo-serving/gpu-pilot-local.json` a été essayé et **refusé** par son
contrôle « Model/preprocessing source differs from the trained snapshot ».
Les sources déjà présentes divergent de celles certifiées avec ce snapshot.
Cette protection est conservée ; ni le manifeste ni les checkpoints historiques
n'ont été modifiés pour la contourner. Le mode résultats ci-dessus est celui
validé et immédiatement fonctionnel pour cette livraison.

## Parcours

1. **Résultats CNN / U-Net** : commencer sur les 169 événements. Montrer les
   Dice avant/après, les FP et la perte de rappel ; rappeler que 0,95 est la
   borne de recherche. Consulter la table micro et les distributions par événement.
2. Passer aux **32 événements de réglage** pour expliquer la sélection. Les
   métriques AP/Brier sont invariantes au seuil ; aucune sélection sur les 169.
3. Dans **Cartes réelles**, choisir un événement/date. Les deux modèles montrent
   vérité J+1, probabilité, segmentation et erreurs sur la même grille. TP vert,
   FP rouge, FN bleu ; invalides transparents. NPZ disponibles pour analyse.
4. **Big Data** : inventaire annuel, fenêtres T=1/3/5, qualité déjà calculée,
   durées SQL, fichiers Parquet et SHA. Spark 4.0.2 en `local[2]`, pas un cluster.
   Les 156 événements 2021 ne sont qu'un inventaire de métadonnées ; test verrouillé.
5. **Prévision historique**, avec un snapshot compatible : parcours existant, explicitement distinct des
   nouveaux résultats compacts. Les nouveaux modèles ne sont pas présentés comme
   chargés par l'ancien Predictor.

## Routes ajoutées

| Route | Contenu |
|---|---|
| `/science/summary` | Comparaisons, quantiles macro, diagnostic, historique et SHA |
| `/science/events?phase=calibration32` | Métriques par événement, fiabilité et courbes sur les 32 |
| `/science/events?phase=holdout169` | Métriques aux seuls seuils fixés ; aucune courbe de recherche |
| `/science/maps?phase=holdout169` | Cas réels CNN/U-Net, chemins des couches et NPZ |
| `/science/assets/<chemin autorisé>` | PNG ou NPZ figurant dans le manifeste, liste fermée |
| `/science/big-data` | Agrégats Parquet existants et provenance, sans moteur Spark |

Ces routes sont insérées **avant** le montage statique racine du frontend.
Une requête de phase 2021 est refusée. Les fichiers internes du ZIP ne sont pas
exposés librement. Au chargement, le backend vérifie la complétude, le SHA du
verrou, la correspondance des tableaux, la somme des comptages par événement
et l'invariance AP/Brier. Le ZIP reste intact et n'est pas extrait sur disque.

Le backend relit cinq petits agrégats Parquet existants et vérifie qu'ils
correspondent au rapport Spark. Il ne réexécute pas leurs requêtes SQL et ne
relance aucun audit raster. Les comptes d'observations et de fenêtres restent
distincts. Dépendances : FastAPI/uvicorn existants, NumPy et PyArrow ; le frontend
utilise les dépendances React/esbuild déjà installées.

## Validation

15 tests ciblés réussis (`tests.xml`) : vrais scores et Parquet, contrats API,
préservation des routes de prédiction, masques, cartes, AP/Brier, précontrôle
sans pixels et indépendance des initialisations. Les requêtes HTTP locales
vérifient également index/JS/CSS, résumé, événements, cartes et Big Data.

La validation visuelle navigateur a été **bloquée** par l'outil computer-use :
impossible de déterminer l'URL courante avec assez de confiance pour appliquer
sa politique. Aucun parcours visuel n'est donc revendiqué. Les contrôles HTTP
et de fichiers ont été poursuivis ; aucune protection navigateur n'a été contournée.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_science_results.py tests\test_compact_calibration.py -q --basetemp reports\science_integration_20261112_v1\verification-nouvelle
```

Aucun transfert externe, entraînement réel, republication, recalibration terminée,
job Spark, commit ou push n'a été effectué pendant cette intégration.
