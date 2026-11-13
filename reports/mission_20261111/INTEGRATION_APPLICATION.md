# Intégration des résultats exécutés — 11 novembre 2026

Le frontend existant est `frontend/src/EventForecast.mjs`, monté par `main.mjs`.
Il propose déjà catalogue événement/date, canvas `FireMap`, observation J,
probabilité, prédiction seuillée et lecture séparée de la vérité terrain.
Les contrôles de géométrie et l'annulation des requêtes sont réutilisables.
FastAPI expose `/health`, `/events`, `/predict`, `/ground-truth` dans
`src/geofirex/serving.py`. Ces routes servent le run historique configuré,
pas automatiquement les expériences de cette mission.

| Section souhaitée | Réutilisation | Ajout préparé / manque réel |
|---|---|---|
| Wildfire Explorer | Sélecteurs et FireMap | `/mission/events` fournit les 201 événements val réellement évalués ; prévoir pagination et dates depuis per_window. Le catalogue historique ne devient pas le corpus complet implicitement. |
| Prediction Lab | Observation, cible, prédiction, contrôles CRS/transform | `/mission/map/{0..3}` fournit les cartes B0 réelles et leurs erreurs, avec origine de tuile. Inférence B1 du nouveau checkpoint et choix des modèles restent à connecter. |
| Model Benchmark | Interface React et états chargement/erreur | `/mission/benchmark` expose B0 complet, B1 CPU et comparaison sur les six mêmes événements. Ajouter tableau/courbes ; ne pas classer B1 pilote face à B0 complet. Aucun score nouveau B2/B3. |
| Big Data Monitor | Infrastructure React/FastAPI | `/mission/big-data` fournit SQL exécuté, effectifs, qualité et durées ; `/mission/compact` fournit les volumes mesurés. Ajouter panneaux et provenance ; afficher Spark local[2], pas cluster. |

Le nouveau module `src/geofirex/mission_results.py` fournit un APIRouter de
lecture seule. Dans l'application créée par le launcher, après `create_app(...)` :

```python
from geofirex.mission_results import attach_mission
attach_mission(app, 'reports/mission_20261111')
```

Cette activation reste explicite : aucune modification du service existant ni
réécriture du frontend. Les routes absentes retournent 404 ; aucune métrique de
remplacement. Les cartes sont des tuiles réelles sélectionnées pour illustrer des
erreurs, pas une estimation de performance. Le PNG et le JSON donnent événement,
date, ligne/colonne, CRS et transform de scène ; l'origine de tuile est à appliquer
avant superposition géographique. Le test 2021 reste fermé.
