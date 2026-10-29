# GeoFire-X — Plan de présentation, Validation 2

Durée cible : 12-15 minutes · 10-12 slides maximum · public : enseignants Big
Data / Deep Learning.

---

## Slide 1 — Titre

**Contenu** : GeoFire-X — Prédiction spatio-temporelle multimodale de la
propagation des incendies (J+1). Validation 2. Équipe (4 noms). TEK-UP,
10/2026.
**Figure** : aucune.
**Message scientifique** : positionnement du projet (rappel bref RQ1-RQ3).
**À expliquer oralement** : transition depuis la Validation 1 (état de
l'art) vers une première implémentation technique vérifiée.

## Slide 2 — Rappel : objectifs et questions de recherche

**Contenu** : RQ1 fusion multimodale, RQ2 modélisation temporelle explicite,
RQ3 robustesse géographique/temporelle. Objectif de cette validation : poser
les fondations de données vérifiées avant tout modèle.
**Figure** : aucune (texte).
**Message scientifique** : ce qui est montré aujourd'hui est une fondation,
pas un résultat de performance.
**À expliquer oralement** : pourquoi une fondation solide est un prérequis
scientifique avant d'entraîner un modèle.

## Slide 3 — Architecture logicielle

**Contenu** : vue d'ensemble du pipeline réel + pipeline synthétique
(indépendants).
**Figure** : `reports/figures/validation2/figure1_architecture.png`.
**Message scientifique** : séparation stricte synthétique/réel pour ne
jamais confondre un test logiciel avec un résultat scientifique.
**À expliquer oralement** : rôle de chaque bloc (manifeste, fenêtres,
DataLoader, métriques, MLflow).
**Fichier à montrer si besoin** : `src/geofirex/` (arborescence).

## Slide 4 — Le dataset réel : WildfireSpreadTS

**Contenu** : source officielle (Zenodo), 607 événements / 13 607
observations, 23 bandes, résolution 375 m. Stratégie d'acquisition sans
téléchargement complet (requêtes HTTP Range).
**Figure** : `reports/figures/data_audit/events_per_year_full_corpus.png`.
**Message scientifique** : les chiffres officiels ont été **vérifiés
empiriquement**, pas recopiés d'un article.
**À expliquer oralement** : comment on a indexé 607 événements sans
télécharger 48 Go (table centrale du ZIP).

## Slide 5 — Une observation multimodale réelle

**Contenu** : les 23 canaux, exemple visuel sur un événement réel.
**Figure** : `reports/figures/data_audit/sample_observation.png`.
**Message scientifique** : les données sont de vraies observations
satellite/météo/topographie géoréférencées, pas des images génériques.
**À expliquer oralement** : différence entre les 23 bandes brutes et les 40
canaux post-prétraitement (erreur initiale corrigée, documentée
honnêtement).

## Slide 6 — Construire une fenêtre J+1

**Contenu** : principe T jours de contexte → cible au jour T+1, règle de
consécutivité stricte.
**Figure** : `reports/figures/validation2/figure4_window_construction.png`.
**Message scientifique** : aucune fuite d'information future, fenêtres
rejetées explicitement si trou de date.
**À expliquer oralement** : pourquoi T=1,2,3 sont testés, et pourquoi 0 rejet
sur l'échantillon actuel.

## Slide 7 — Déséquilibre des classes

**Contenu** : proportion de fenêtres avec/sans feu au J+1, déséquilibre
pixel-level.
**Figure** : `reports/figures/validation2/figure5_label_distribution.png`.
**Message scientifique** : la rareté du feu est une vraie difficulté
scientifique, pas un artefact de code.
**À expliquer oralement** : impact direct sur l'interprétation des métriques
(slide suivante).

## Slide 8 — Baseline Persistence : exemple visuel

**Contenu** : feu au jour J, prédiction, vérité terrain J+1, carte d'erreur.
**Figure** : `reports/figures/real_baseline/sample_01_2018_fire_21997864_2018-08-04.png`.
**Message scientifique** : Persistence ne peut jamais prédire une nouvelle
ignition — limite structurelle visible directement sur la carte d'erreur.
**À expliquer oralement** : pourquoi c'est la baseline de référence minimale
avant tout modèle de deep learning.

## Slide 9 — Résultats chiffrés et leur interprétation honnête

**Contenu** : Dice 0.558 / IoU 0.520 / Precision 0.685 / Recall 0.673 (67
fenêtres, 3 événements) — ET le contraste avec le sous-ensemble à cible
réellement positive (Dice 0.247, Recall 0.269).
**Figure** : tableau simple (pas d'image).
**Message scientifique** : une moyenne globale peut être trompeuse si on ne
distingue pas les cas triviaux (pas de feu) des cas difficiles (vrai feu) —
rigueur méthodologique.
**À expliquer oralement** : la méthode d'agrégation (macro par fenêtre), et
pourquoi ce n'est PAS un benchmark scientifique (5 événements seulement).

## Slide 10 — Qualité logicielle et reproductibilité

**Contenu** : 56/56 tests automatisés, MLflow (seed, commit Git, config
tracés), `.gitignore` protège les données brutes.
**Figure** : aucune (ou capture d'écran MLflow UI si disponible en direct).
**Message scientifique** : chaque résultat est reproductible et tracé, pas
un chiffre isolé.
**À expliquer oralement** : démonstration rapide possible de
`pytest -q` ou `python scripts/demo_validation2.py` en direct.
**Fichier/commande à montrer** : `scripts/demo_validation2.py`.

## Slide 11 — Travaux restants et feuille de route

**Contenu** : extension du dataset, split multi-années réel, U-Net 2D,
ConvLSTM, fusion multimodale, évaluation OOD.
**Figure** : `reports/figures/validation2/figure7_roadmap.png`.
**Message scientifique** : la feuille de route découle directement des
limites identifiées aujourd'hui, pas d'une liste arbitraire.
**À expliquer oralement** : priorisation (pourquoi étendre le dataset avant
d'entraîner un modèle).

## Slide 12 — Conclusion

**Contenu** : résumé en 3 points (pipeline réel fonctionnel et vérifié,
baseline de référence posée, limites documentées honnêtement).
**Figure** : aucune.
**Message scientifique** : fondation solide prête pour la suite, sans
survendre les résultats actuels.
**À expliquer oralement** : ouverture aux questions du jury.
