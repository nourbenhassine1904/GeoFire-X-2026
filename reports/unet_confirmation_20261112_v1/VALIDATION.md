# Vérifications de livraison

- Audit des trois checkpoints réels et de leurs métriques sauvegardées : réussi,
  sans réentraînement ni nouvelle inférence sur les 32.
- Une confirmation réelle sur les 169 : **terminée sur CPU**, 747,955 s,
  65 012 tuiles, seuil0,78 fixé avant lecture. Aucun GPU.
- Comparaison aux deux références historiques : IDs et comptages du domaine
  commun vérifiés globalement et par événement. Aucun recalcul des références.
- **21 tests ciblés réussis** : contrats précédents, audit réel des trois runs,
  rejet d'un seuil sauvegardé altéré, confirmation figée, intégrité des fichiers,
  accès aux cartes, nouveaux feux, refus de 2021 et données Spark conservées.
- Le lanceur mode résultats est exécuté dans un sous-processus qui interdit les
  imports `torch` et `geofirex.serving`. Il réussit : l'ancien Predictor n'est pas
  initialisé et ses gardes restent réservées au mode de prévision historique.
- React compilé avec les dépendances locales existantes, sans téléchargement.
- API et fichiers du frontend vérifiés en HTTP local. Les images synthèse des
  nouveaux cas sont inspectées directement ; aucun parcours visuel navigateur
  n'est revendiqué. Le contrôle navigateur avait été indisponible dans le travail
  précédent ; les preuves de cette livraison sont les appels HTTP, les tests et
  les cartes réelles inspectées.

Commande des tests, sans inférer à nouveau le corpus réel :

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_unet_confirmation.py tests\test_science_results.py tests\test_compact_calibration.py -q --basetemp reports\unet_confirmation_20261112_v1\verification-nouvelle
```

Les tests de pipeline de calibration portent sur de petits exemples synthétiques,
pas sur un nouvel entraînement ou une seconde confirmation réelle.
Le rapport `tests.xml` et le relevé `delivery.json` donnent les preuves détaillées.

La copie du ZIP fourni, le ZIP historique et les cinq agrégats Parquet gardent
leurs SHA. Aucun transfert externe, publication Kaggle, nouvel entraînement,
nouvelle campagne d'ablation, Spark, commit ou push.
