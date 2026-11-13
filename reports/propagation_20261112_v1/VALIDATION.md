# Validation de la livraison du 12 novembre

Application démarrée sur http://127.0.0.1:8014, mode `scientific-results`. La vue « Propagation J+1 » distingue feu observé, nouveaux feux, probabilités et erreurs. Les vues de comparaison historique, ablation et Big Data sont conservées. Le module de prévision historique et ses contrôles de source/checkpoint ne sont pas contournés.

Contrôles réalisés :

- **29 tests CPU réussis**, comprenant les contrats existants et les nouveaux tests de forme, identité du backbone dans un même environnement, porte basée uniquement sur J, gradients des pixels nouveaux/négatifs/invalides, précontrôle sans pixels ni CUDA, intégrité des résultats/cartes, rejet de deux GPU visibles et plafond GPU après arrêt brutal.
- Build React réussi. Rendu du véritable composant chargé, avec ReactDOMServer et les diagnostics enregistrés : 169 lignes d'événements et cinq couches de carte, sans erreur. Ce contrôle n'est pas un test d'interactions de navigateur.
- **88 réponses HTTP200**, dont **75 couches PNG** historiques et nouvelles. Vérification des pages statiques, health, comparaisons, événements32/169, cartes, Big Data, modèle retenu et diagnostic de propagation.
- Les archives de calibration/ablation et les cinq Parquet sont identiques aux empreintes de la livraison de confirmation précédente. Aucun checkpoint historique modifié ; les nouveaux scripts n'ont chargé aucun optimiseur historique pour entraîner.
- Vérification ciblée des labels/masks : 70 tuiles des trois cartes existantes, sans inférence ; cible, canal observé J et intersection des six masques identiques. Les comptages nouveau/ancien partitionnent les comptages globaux et leurs sommes par événement concordent.
- Figure [propagation_diagnostic.png](propagation_diagnostic.png) inspectée visuellement : cadre de pixels, vérité nouvelle distincte de J, FP/FN et distribution par événement ; le blanc des probabilités/erreurs exclut les pixels déjà détectés à J et les invalides.

Limites explicites : l'hôte navigateur a répondu `The browser access host is unavailable`. Aucune validation visuelle des interactions React n'est donc revendiquée. L'identité d'initialisation du nouvel entraînement doit encore passer dans Torch2.11.0 sur Kaggle ; le code s'arrête avant tout update si son digest diffère de l'ablation. Le débit, les kernels et la mémoire GPU du nouveau modèle ne sont pas mesurés. Aucun gain scientifique nouveau n'est annoncé.

Les fichiers [delivery.json](delivery.json), [tests.xml](tests.xml), [react-component-check.json](react-component-check.json), [diagnosis.json](diagnosis.json) et [notebook_build.json](notebook_build.json) contiennent les preuves et empreintes. Aucun GPU lancé, aucun calcul historique répété, aucun Spark relancé, aucun dataset publié ou transféré, aucun pixel2021 ouvert, aucun commit/push/merge.
