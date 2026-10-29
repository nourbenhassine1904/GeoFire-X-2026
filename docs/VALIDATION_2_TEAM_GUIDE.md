# GeoFire-X — Guide de l'équipe pour la Validation 2

Discours réparti entre 4 intervenantes, ~12-15 minutes au total. Les slides
référencées correspondent à `docs/VALIDATION_2_PRESENTATION.md`.

**Important — distinction à ne jamais effacer devant le jury** : la
répartition ci-dessous organise *qui présente quoi à l'oral*. Elle ne décrit
pas qui a écrit quelle ligne de code. Si le jury demande qui a fait quoi
techniquement, répondez factuellement et collectivement — ne présentez pas la
répartition de présentation comme une répartition de contribution technique.

---

## Étudiante 1 — Contexte, objectifs, architecture et Git

**Slides** : 1, 2, 3 (~3 minutes)

**Discours (résumé à adapter avec vos mots) :**
> "Bonjour, nous sommes [équipe], et nous présentons GeoFire-X, notre projet
> de prédiction de la propagation des incendies de forêt au lendemain. Lors
> de la première validation, nous avons positionné le projet autour de trois
> questions : la fusion multimodale, la modélisation temporelle explicite, et
> la robustesse face aux changements de distribution. Aujourd'hui, nous
> montrons la première implémentation technique vérifiée de ce projet :
> un pipeline de données réelles fonctionnel, pas encore un modèle entraîné.
> Notre architecture logicielle sépare volontairement un pipeline
> synthétique, utilisé pour développer et tester le code, et un pipeline
> réel, qui lit directement les données officielles WildfireSpreadTS. Cette
> séparation nous permet de ne jamais confondre un test logiciel avec un
> résultat scientifique."

**Concepts à maîtriser** : RQ1/RQ2/RQ3, différence pipeline synthétique vs
réel, notion de branche Git et de Pull Request (comment le code est partagé
et relu en équipe).

**Démonstration éventuelle** : ouvrir `src/geofirex/` dans l'explorateur de
fichiers pour montrer l'organisation du code.

**3 questions probables du jury et réponses attendues :**
1. *"Pourquoi deux pipelines séparés ?"* → Pour développer et tester le code
   sans dépendre du téléchargement des données réelles, et pour ne jamais
   mélanger les deux formats (23 bandes vs 40 canaux) par erreur.
2. *"Quelle est la différence entre votre Phase 2 et votre Phase 3 ?"* →
   Phase 2 = acquisition et audit des données réelles ; Phase 3 = DataLoader,
   fenêtres temporelles, baseline et métriques construits sur ces données.
3. *"Le code est-il partagé et relu en équipe ?"* → Oui, chaque phase est
   développée sur une branche dédiée, relue avant fusion dans `main` via une
   Pull Request sur GitHub.

---

## Étudiante 2 — Dataset réel, audit et visualisations

**Slides** : 4, 5 (~2,5 minutes)

**Discours :**
> "WildfireSpreadTS est le dataset officiel que nous utilisons, publié sur
> Zenodo par l'équipe du KTH. Il couvre 607 événements d'incendie aux
> États-Unis entre 2018 et 2021, avec une observation quotidienne par
> événement. L'archive complète fait 48 gigaoctets, ce que nous n'avons pas
> téléchargé en entier : nous avons lu uniquement la table d'index du
> fichier compressé à distance pour vérifier sa structure, puis extrait
> seulement 5 événements réels de façon ciblée. Chaque observation contient
> 23 canaux : réflectance satellite, indices de végétation, météo,
> topographie, occupation du sol et feu actif. Nous avons découvert en
> auditant les fichiers réels que le format brut utilise 23 bandes, et non
> 40 comme une documentation annexe le suggérait initialement — nous l'avons
> vérifié et corrigé."

**Concepts à maîtriser** : notion de requête HTTP Range (lire une partie
d'un fichier distant sans le télécharger entier), les 23 canaux et leur
origine (satellite VIIRS, météo GRIDMET, prévisions NOAA GFS), la convention
d'encodage du canal "feu actif" (valeurs manquantes = pas de détection).

**Démonstration éventuelle** : montrer `reports/DATA_AUDIT.md` ou lancer
`python scripts/demo_validation2.py` (sections 1 à 4).

**3 questions probables du jury et réponses attendues :**
1. *"Pourquoi seulement 5 événements sur 607 ?"* → Choix délibéré pour
   respecter la contrainte de ne pas télécharger des dizaines de gigaoctets
   sans nécessité ; l'échantillon permet de valider tout le pipeline
   logiciel, l'extension viendra ensuite.
2. *"Comment savez-vous que vos chiffres (607 événements, 13607
   observations) sont corrects ?"* → Vérifiés directement depuis les
   métadonnées officielles de l'archive (table centrale du ZIP), pas
   recopiés d'un article.
3. *"Qu'est-ce que la convention d'encodage du feu actif ?"* → Les pixels
   sans détection de feu sont marqués comme valeur manquante, pas comme
   zéro ; nous binarisons donc par présence de valeur, jamais par un seuil
   numérique.

---

## Étudiante 3 — DataLoader, temporalité et séparation des événements

**Slides** : 6, 7 (~2,5 minutes)

**Discours :**
> "Pour prédire le feu à J+1, nous construisons des fenêtres temporelles :
> T jours d'observations consécutives en entrée, et le jour suivant comme
> cible. Nous avons testé T=1, 2 et 3 jours de contexte. Une fenêtre n'est
> acceptée que si toutes ses dates sont réellement consécutives et
> appartiennent au même événement — sinon elle est rejetée, et nous
> enregistrons pourquoi. Sur notre échantillon actuel, aucune fenêtre n'a
> été rejetée : il n'y a aucun trou de date. Nous avons aussi mesuré le
> déséquilibre des classes : le feu actif ne représente qu'une fraction
> infime des pixels, ce qui est une vraie difficulté scientifique que nous
> documentons plutôt que de la cacher."

**Concepts à maîtriser** : notion de fenêtre glissante, fuite d'information
(pourquoi la cible doit toujours être après le contexte, jamais dedans),
règle "un événement = un seul split" pour éviter la fuite entre
entraînement/test.

**Démonstration éventuelle** : `python scripts/demo_validation2.py` (section
5 à 7, construction d'une fenêtre en direct).

**3 questions probables du jury et réponses attendues :**
1. *"Qu'est-ce qu'une fuite d'information temporelle, concrètement ?"* →
   Utiliser une observation du futur (ou du jour cible lui-même) pour
   prédire ce même jour — nous l'empêchons en vérifiant que la cible est
   strictement le jour suivant le dernier jour de contexte.
2. *"Pourquoi tester T=1, 2 et 3 ?"* → Pour comparer plus tard l'effet de la
   longueur de l'historique temporel sur la prédiction — c'est directement
   lié à notre question de recherche RQ2.
3. *"Comment garantissez-vous qu'un incendie n'est pas à la fois en
   entraînement et en test ?"* → Le split est fait au niveau de l'événement
   entier, jamais au niveau du jour ; un contrôle automatisé
   (`check_no_leakage`) vérifie cette règle et est testé.

---

## Étudiante 4 — Persistence, métriques, tests, MLflow et feuille de route

**Slides** : 8, 9, 10, 11, 12 (~5 minutes)

**Discours :**
> "Avant d'entraîner un modèle de deep learning, nous avons implémenté une
> baseline simple : Persistence, qui prédit que le feu de demain sera
> identique à celui d'aujourd'hui. C'est notre point de comparaison
> minimal. Sur nos 67 fenêtres réelles, elle obtient un Dice de 0.558 en
> moyenne globale. Mais ce chiffre seul est trompeur : près de la moitié de
> nos fenêtres n'ont aucun feu ni observé ni prédit, ce qui gonfle
> artificiellement la moyenne. Si on se concentre uniquement sur les
> fenêtres où un feu est réellement présent, le Dice tombe à 0.247 — ce qui
> illustre concrètement la limite de Persistence : elle ne peut jamais
> prédire une nouvelle ignition. Nous insistons : avec seulement 5
> événements, ceci est une vérification technique de notre pipeline, pas un
> résultat de recherche généralisable. Tout notre travail est testé
> automatiquement — 56 tests passent actuellement — et chaque exécution est
> tracée dans MLflow avec sa configuration, sa seed et son commit Git. La
> suite logique est d'étendre l'échantillon réel, d'implémenter une vraie
> validation croisée par année, puis d'entraîner un premier modèle U-Net."

**Concepts à maîtriser** : baseline Persistence, Dice/IoU/Precision/Recall
(savoir les définir simplement), pourquoi une moyenne peut être trompeuse
avec des classes déséquilibrées, rôle de MLflow (traçabilité, pas
d'entraînement de modèle ici), notion de seuil minimal d'événements pour une
évaluation scientifique.

**Démonstration éventuelle** : lancer `pytest -q` en direct (quelques
secondes), ou montrer `mlflow ui --backend-store-uri sqlite:///mlflow.db`
si une connexion locale est possible.

**3 questions probables du jury et réponses attendues :**
1. *"Pourquoi le Dice global (0.558) est-il si différent du Dice sur les
   fenêtres avec feu réel (0.247) ?"* → Parce que la moyenne globale inclut
   des fenêtres sans aucun feu, où Persistence obtient un score parfait par
   convention mathématique (rien à prédire, rien prédit) — ça gonfle la
   moyenne sans refléter la vraie difficulté de la tâche.
2. *"Avez-vous entraîné un modèle de deep learning ?"* → Non, explicitement
   hors périmètre de cette validation ; nous avons seulement posé une
   baseline de référence sans réseau de neurones.
3. *"Que garantit MLflow ici ?"* → La traçabilité : chaque exécution
   enregistre sa configuration, la version du split, la seed, le commit Git
   et les métriques — reproductibilité, pas performance.

---

## Transition entre intervenantes (suggestions courtes)

- 1 → 2 : *"Je laisse [nom] vous présenter le dataset que nous avons
  effectivement utilisé."*
- 2 → 3 : *"Maintenant que les données sont posées, [nom] va expliquer
  comment nous en faisons des séquences temporelles exploitables."*
- 3 → 4 : *"Pour terminer, [nom] va présenter nos premiers résultats et la
  suite du projet."*
