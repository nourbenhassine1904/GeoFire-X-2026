# GeoFire-X — Proposition de stratégie de séparation (train/val/test)

Document de proposition, Phase 2 (données réelles). Ne constitue pas encore un split
définitif — voir "Statut actuel" ci-dessous.

## Règle non négociable

Un événement incendie appartient à exactement un split (`train`, `val` ou `test`).
Aucun jour d'un événement d'entraînement ne doit apparaître en validation ou en test
(règle déjà énoncée dans `docs/PROJECT_CONTEXT.md`).

## Ce qui est réellement disponible pour construire un split

Vérifié sur le corpus officiel complet (607 événements, via la table centrale de
l'archive — voir `reports/DATA_AUDIT.md`) :

- **Identifiant d'événement** : oui (`<année>/fire_<id_numérique>`).
- **Année** : oui, directement dans le chemin (2018, 2019, 2020, 2021). Répartition
  réelle : 2018=176, 2019=74, 2020=201, 2021=156 événements.
- **Nombre d'observations par événement** : oui, dérivable du manifeste.
- **Localisation géographique précise (coordonnées du centre de l'événement)** : **non
  disponible** dans le nom de fichier ou le chemin. Seul le CRS (zone UTM) de chaque
  fichier GeoTIFF est lisible — un indicateur *grossier* de longitude (une zone UTM
  couvre ~6° de longitude), pas une coordonnée précise. Confirmer via la documentation
  officielle PDF si des coordonnées plus précises existent ailleurs dans l'archive
  (non encore vérifié).

## Recommandation officielle des auteurs (source de vérité)

Le datasheet officiel (`WildfireSpreadTS_Documentation.pdf`, §"Are there recommended
data splits?") est explicite :

> "We recommend to evaluate models using a cross-validation across years, since the
> data distributions vary a lot between years."

Les auteurs eux-mêmes ne recommandent **pas** un split aléatoire classique, mais une
validation croisée **par année**. C'est la base la plus défendable scientifiquement
pour GeoFire-X, et elle confond naturellement notre besoin de Temporal OOD avec le
split principal plutôt que de les traiter comme deux choses séparées.

## Proposition de protocole

1. **Split principal, aligné sur la recommandation officielle** : validation croisée
   par année (2018/2019/2020/2021) plutôt qu'un split événementiel aléatoire classique.
   Ex. : plis successifs où une année sert de test, les trois autres d'entraînement/val.
2. **Temporal OOD** : découle directement du point 1 — entraîner sur un sous-ensemble
   d'années, évaluer sur une année non vue, est déjà la méthode recommandée par les
   auteurs, pas une extension séparée.
3. **Geographic OOD** : regrouper les événements par zone UTM (proxy de région
   géographique) et entraîner sur certaines zones, évaluer sur d'autres — réalisable
   en proxy grossier seulement. **Ne pas présenter ce proxy comme une évaluation
   géographique précise** sans vérifier d'abord si l'archive contient des coordonnées
   plus fines (bounding box exact par événement, lisible depuis le `transform` GDAL de
   chaque fichier — à calculer, pas encore fait sur l'échantillon réel).

## Statut actuel

**Aucun split réel n'a été créé — ni définitif, ni minimal — à ce stade.** L'échantillon
réel extrait (voir `reports/DATA_AUDIT.md`) contient 5 événements sur 607 — bien trop
peu pour qu'un split train/val/test ou une validation croisée par année aient le moindre
sens statistique (2020 et 2018 y sont représentés deux fois, 2019 et une partie de 2020
une fois, aucun événement de 2021). Créer un fichier de split sur un échantillon aussi
petit serait trompeur s'il était ensuite confondu avec un split définitif. Décision
délibérée de ne pas en produire un pour l'instant.

## Décisions qui restent à l'équipe

1. Combien d'événements réels supplémentaires extraire avant de figer un split
   définitif (dépend du temps/bande passante que l'équipe souhaite allouer) ?
2. Calculer les bounding box géographiques réelles (via le `transform` GDAL de chaque
   fichier) pour juger si un split géographique OOD plus fin que "zone UTM" est
   possible — tâche non réalisée à ce stade.
3. Valider ou ajuster les ratios 70/15/15 une fois le nombre réel d'événements
   disponibles connu.
