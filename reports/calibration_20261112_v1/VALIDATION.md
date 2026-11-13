# Vérifications du livrable

Préparation locale du 12 novembre 2026 ; aucune évaluation des checkpoints Kaggle
n'a été exécutée. Ces checkpoints ne sont pas présents dans le workspace local.

- **7 tests réussis**, rapport `tests.xml` : comptages contre calcul direct à tous
  les seuils avec égalités float32 ; AP exacte avec égalités et fusions multipasses
  contre scikit-learn ; Brier en float64 contre scikit-learn ; sélection grossière
  puis affinée ; pipeline CPU synthétique complet avec B1/B2, masques et exports ;
  refus d'un split différent / de 2021 ; contrôle des 11 identités et des octets.
- **Autonomie du bundle vérifiée dans un sous-processus isolé** qui refuse les
  imports rasterio, pyarrow, scikit-learn et mlflow. Les cellules du notebook et
  tous les modules embarqués compilent.
- **Parité sur trois tuiles compactes réelles** de validation 2020 : négative,
  positive, négative difficile. Tenseurs, cibles et masques identiques au lecteur
  existant. Rapport `local-loader-verification.json` ; cette vérification ne
  constitue pas une évaluation des modèles.
- **Couverture de l'index réel vérifiée** : 250 événements / 93 603 tuiles train,
  201 événements / 78 771 tuiles validation. SHA index enregistré dans le rapport.
- Graphiques de diagnostic produits par le test synthétique inspectés visuellement.
  Ils restent dans les répertoires de tests et sont exclus de l'archive livrée.

Le seul adaptateur appliqué au lecteur embarqué remplace son import de deux
fonctions I/O (`save`, `sha`) par un module léger équivalent. Le chargement des
NPZ, la normalisation déjà appliquée, les masques et les modèles sont conservés.
`source-manifest.json` donne les SHA des sources et du bundle embarqué.

Les résultats scientifiques attendus — seuils, nouveaux scores, cartes réelles —
seront produits dans un nouveau dossier de sortie par l'exécution Kaggle décrite
dans `EXECUTION.md`. Aucun gain chiffré n'est affirmé avant cette exécution.
Les résultats historiques et fichiers existants n'ont pas été édités dans cette
mission. Aucun entraînement, reconstruction, upload, B0, Spark, commit ou push.
