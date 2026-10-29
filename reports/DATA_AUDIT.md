# GeoFire-X — Audit du dataset réel WildfireSpreadTS

Commit Git de référence (HEAD au moment de la génération de ce rapport, pas
nécessairement le dernier commit permanent du dépôt) : `cdedd393d26fef828e6909cc052fc721e2226788`

## 1. Source et version du dataset

- Source officielle : Zenodo record 8006177 — https://zenodo.org/records/8006177 (DOI 10.5281/zenodo.8006177)
- Dépôt de code officiel : https://github.com/SebastianGer/WildfireSpreadTS
- Version : v1.0, publiée le 2023-10-25
- Licence données : CC-BY-4.0 · Licence code : MIT
- Taille de l'archive officielle : 48 359 369 821 octets (48,36 Go), checksum MD5 `dc1a04e63ccc70037b277d585b8fe761`
- **Stratégie d'accès** : l'archive complète (48,36 Go) n'a PAS été téléchargée. La table centrale du
  ZIP a été lue à distance via des requêtes HTTP Range (1,6 Mo transférés), ce qui a permis de vérifier
  la structure réelle de l'intégralité des 607 événements sans télécharger leur contenu. Un échantillon
  réel de 5 événement(s) a ensuite été extrait par requêtes Range ciblées (quelques dizaines
  de Mo), sans jamais télécharger le fichier complet.

## 2. Vérification des chiffres annoncés

| Chiffre | Annoncé (article/Zenodo) | Vérifié empiriquement sur l'archive officielle |
| --- | --- | --- |
| Événements | 607 | **607** (confirmé via la table centrale du ZIP) |
| Observations journalières | 13 607 | **13 607** (confirmé via la table centrale du ZIP) |
| Canaux multimodaux | 23 | **23** (confirmé en ouvrant les fichiers réels avec rasterio) |

Aucun écart constaté sur ces trois chiffres.

## 3. Écart majeur découvert et documenté

**Le fichier GeoTIFF brut réel contient 23 bandes, pas 40.** Une documentation antérieure du projet
(`docs/DATASET_VERSION.md`) mentionnait "40 canaux" en se basant sur la description du code officiel
(`FireSpreadDataset.py`, qui décrit le tenseur *après* prétraitement HDF5). Vérifié empiriquement sur
les fichiers réels : les 23 canaux de base sont bien présents, mais le land cover reste une **seule**
bande catégorielle (`LC_Type1`), pas encore éclaté en 17 canaux one-hot. L'expansion à 40 canaux
n'existe qu'après exécution du script officiel `CreateHDF5Dataset.py`, que nous n'avons pas (encore)
reproduit.

Noms des 23 bandes, lus directement dans les métadonnées GDAL du fichier (aucune supposition) :
`M11, I2, I1, NDVI_last, EVI2_last, total precipitation, wind speed, wind direction, minimum temperature, maximum temperature, energy release component, specific humidity, slope, aspect, elevation, pdsi, LC_Type1, total_precipitation_surface_sum, forecast wind speed, forecast wind direction, forecast temperature, forecast specific humidity, active fire`

La bande **"active fire" est la 23ᵉ et dernière bande**, et représente le feu actif observé **à la
date du fichier lui-même** — pas le lendemain. La cible J+1 doit donc être construite en lisant la
bande "active fire" du fichier du **jour suivant**, jamais une bande interne au même fichier. Ceci est
confirmé mot pour mot par le datasheet officiel : *"we binarize the next day's active fire detection
and use this as the prediction target."*

**Deuxième écart découvert — encodage non-binaire de la bande "active fire"** : vérifié empiriquement
sur l'échantillon réel, 99.96% des pixels de
cette bande sont `NaN` (datasheet officiel, agrégé sur tout le corpus : 99,83%), et les pixels restants
portent des valeurs numériques non-binaires (ex. `918`, `2036` observés) — pas des `1`. Le datasheet
officiel (Table 3, §5.3) confirme et explique : *"only
active fire detections are marked as known values. All pixels that are not detected as burning are
marked as missing values. In preprocessing, we replace these missing values with zeros."* Le masque
binaire "feu actif" doit donc être dérivé par **présence d'une valeur connue** (`~isnan`), jamais par un
seuil sur sa magnitude — c'est la méthode utilisée dans ce rapport et dans `src/geofirex/remote_archive.py`
en aval.

## 3bis. Provenance et résolution native des canaux (Table 1 du datasheet officiel)

| Canal(aux) | Produit source | Résolution native |
| --- | --- | --- |
| Active fire | VNP14IMG (Schroeder et al. 2014) | 375 m / 24h |
| VIIRS M11, I2, I1 | VNP09GA (Vermote et al. 2016) | I1/I2: 500 m · M11: 1 km / 24h |
| NDVI, EVI2 | VNP13A1 (Didan & Barreto 2018) | 500 m / **8 jours** |
| Slope, aspect, elevation | SRTMGL1 v003 | 30 m |
| Land cover | MCD12Q1.061 | 500 m |
| Précipitation, vent, température, humidité, PDSI, ERC | GRIDMET | 4638,3 m / 24h |
| 5 canaux "forecast ..." | NOAA_GFS0P25 (prévisions météo GFS) | 27 830 m / 1h |

Tout est réaligné à 375 m / 24h par Google Earth Engine (interpolation bilinéaire) pour produire les
fichiers distribués. **Les canaux "forecast" sont des prévisions météo du modèle GFS, indépendantes de
l'observation du feu** — l'ambiguïté précédemment notée sur leur nature est levée. Reste une question
plus fine (non vérifiée ici) : l'horizon exact de la prévision par rapport à la date du fichier.

NDVI/EVI2 ont une résolution native de 8 jours, pas journalière — leur valeur journalière dans le
fichier est donc une interpolation/répétition, pas une nouvelle mesure chaque jour. À garder en tête
pour toute analyse de variabilité journalière de ces deux canaux.

## 4. Organisation réelle des données

- Structure sur disque : `<année>/fire_<id_numérique>/<date_ISO>.tif` (ex. `2018/fire_21748813/2018-05-03.tif`)
  — l'identifiant d'événement est un **numéro**, pas un nom descriptif.
- Canaux (bandes) observés dans l'échantillon : ['23']
- Dimensions (H×W) observées : ['311x265', '314x231', '336x293', '338x297', '339x299']
- Type numérique : ['float32']
- Système de projection (CRS) observé : ['EPSG:32610']
- Résolution spatiale : 375 m/pixel (confirmé par la transform GDAL du fichier)

## 5. Résultats de l'audit de qualité

Échantillon réel : 5 événement(s), 14-28 observations/événement (moyenne 21.6).

Corpus officiel complet (607 événements, vérifié via la table centrale de l'archive, AUCUN téléchargement des .tif) : {'2018': 176, '2019': 74, '2020': 201, '2021': 156}.

Canaux les plus affectés par des valeurs manquantes, hors 'active fire' (normal à ~99,8%, voir section 3) : NDVI_last (5.55%), EVI2_last (5.55%), I2 (2.78%).

Observation réelle visualisée : 2019/fire_23300839 / 2019-09-25 (canaux : ['active fire', 'NDVI_last', 'LC_Type1', 'maximum temperature']).

Séquence réelle du label 'active fire' visualisée pour l'événement 2020/fire_23654679 (28 jours disponibles).

### Déséquilibre des pixels (feu actif)

- Pixels positifs (feu actif) sur l'échantillon réel : 3,959 / 9,818,898
  (0.0403% des pixels, si disponible)
- Événements avec au moins un pixel de feu actif observé : 5 / 5
- Proportion de pixels positifs par événement :
  - `2018/fire_21748813` : 0.0241%
  - `2018/fire_21997864` : 0.0958%
  - `2019/fire_23300839` : 0.0266%
  - `2020/fire_23654679` : 0.0037%
  - `2020/fire_24104674` : 0.0672%

**Limite reconnue** : le déséquilibre pixel-level ci-dessus porte sur un échantillon de
5 événement(s) seulement — bien trop peu pour généraliser au corpus complet (607
événements). Le ratio exact pour le corpus complet ne peut être affirmé sans traiter davantage
d'événements réels.

## 6. Définition de la tâche J+1 — faisabilité

La tâche J+1 (prédire le masque binaire "active fire" du jour J+1 à partir des observations des jours
≤ J) **est constructible** avec ces données, confirmé par le datasheet officiel lui-même, sous réserve de :

- Utiliser la bande "active fire" du fichier du jour **suivant** comme cible — jamais celle du fichier
  d'entrée. Binariser par **présence d'une valeur connue** (`~isnan`), jamais par un seuil sur sa
  magnitude (voir section 3).
- Les 5 canaux "forecast ..." sont des prévisions météo du modèle **NOAA GFS** (Table 1 du datasheet),
  indépendantes de l'observation du feu — **ambiguïté levée**, ils peuvent légitimement faire partie de
  l'entrée. Point encore ouvert : l'horizon exact de la prévision par rapport à la date du fichier n'a
  pas été vérifié (non critique pour la faisabilité, mais à documenter avant publication de résultats).
- Gérer les trous de dates réels (voir section 7) : une fenêtre de contexte ne doit jamais être
  construite à travers un trou (jours non consécutifs).
- `LC_Type1` est catégoriel (pas continu) — nécessite un encodage dédié (one-hot ou embedding), pas une
  simple normalisation comme les autres canaux.
- NDVI/EVI2 ont une résolution native de 8 jours (Table 1) — leur variation journalière dans le fichier
  n'est pas une nouvelle mesure indépendante chaque jour.

Aucune cible "zone brûlée cumulée" distincte n'a été trouvée dans les 23 canaux — seule la carte
journalière "active fire" est disponible. Toute notion de "nouvelle progression" devra être dérivée
(différence entre deux masques consécutifs), pas lue directement.

### Bruit documenté officiellement sur le label (datasheet §5.1)

Le produit VIIRS actif-feu est explicitement documenté comme bruité : faux positifs dus aux nuages, aux
toits à haute température, et à des sources de chaleur anthropiques (ex. centrales à charbon) ; feux de
propagation rapide qui s'éteignent entre deux passages satellite ; petits feux difficiles à détecter.
Les auteurs ont par ailleurs **déjà écarté** les événements sans aucune détection de feu active sur
toute leur série, ainsi que 9 événements hors période GlobFire et 3 pour anomalies de format — le
corpus de 607 événements est donc déjà filtré, pas un échantillonnage brut.

### Comparaison avec les statistiques officielles d'évolution des pixels (Table 2 du datasheet)

Sur l'ensemble du corpus officiel (~13 000 transitions de frames), les auteurs rapportent : ignition
(nouveaux pixels en feu) médiane 44, moyenne 123,1 · persistance (pixels qui restent en feu) médiane 23,
moyenne 110,3 · extinction médiane 42, moyenne 123,4. **La persistance est systématiquement plus faible
que l'ignition + l'extinction** — à garder en tête en interprétant une future baseline Persistence sur
données réelles : un recall imparfait est attendu structurellement, pas seulement un défaut du modèle.

## 7. Limites constatées

- Échantillon réel limité à 5 événement(s) sur 607 — toute statistique de cette section est
  préliminaire, pas représentative du corpus complet.
- Aucun trou de date détecté dans l'échantillon réel (observations strictement consécutives sur tous les événements extraits).
- Aucune coordonnée géographique précise (centre de l'événement) n'est documentée dans le manifeste —
  seul le CRS (zone UTM) par fichier est disponible, ce qui limite la capacité à construire un split
  géographique OOD fin (voir `docs/SPLIT_STRATEGY.md`).
- L'horizon temporel exact des 5 canaux "forecast" (prévision faite quand, pour quand) n'est pas
  confirmé au-delà de leur provenance (NOAA GFS).

## 8. Décisions qui restent à prendre (équipe)

1. Étendre l'échantillon réel (combien d'événements supplémentaires, selon le temps/bande passante
   disponibles) pour des statistiques de déséquilibre représentatives.
2. Décider si GeoFire-X reproduit lui-même l'étape de prétraitement officielle (one-hot land cover,
   HDF5) ou travaille directement sur les GeoTIFF à 23 canaux.
3. Adopter ou non la recommandation officielle de validation croisée par année comme split principal
   (voir `docs/SPLIT_STRATEGY.md`) plutôt qu'un split événementiel aléatoire classique.

## 9. Reproduire cet audit

```bash
python scripts/inspect_remote_archive.py --out data/raw/WildfireSpreadTS/archive_index.json
python scripts/download_real_sample.py --dest data/raw/WildfireSpreadTS/real_sample --num_events 5 --seed 42
python scripts/build_real_manifest.py --data_dir data/raw/WildfireSpreadTS/real_sample --out data/manifests/real/manifest_wildfirespreadts.csv
python scripts/generate_data_audit.py --manifest data/manifests/real/manifest_wildfirespreadts.csv --archive_index data/raw/WildfireSpreadTS/archive_index.json --figures_dir reports/figures/data_audit --out reports/DATA_AUDIT.md
```
