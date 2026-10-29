# GeoFire-X — Rapport technique : pipeline réel spatio-temporel J+1 (Phase 3)

Commit Git de référence : voir `git log -1` au moment de l'exécution ; les résultats
numériques ci-dessous proviennent de `reports/real_baseline_results.json`,
régénérable par les commandes de la section 8.

## 1. Portée et avertissement scientifique

Ce pipeline tourne sur l'échantillon réel actuel : **5 événements WildfireSpreadTS
réels sur 607** (voir `reports/DATA_AUDIT.md`). C'est très en dessous du seuil de
20 événements retenu comme plancher conservateur pour une évaluation scientifique
(`geofirex.real_splits.MIN_EVENTS_FOR_SCIENTIFIC_SPLIT`). En conséquence :

- Le split utilisé (`splits/real/smoke_test_split.csv`) est un **smoke-test**, pas
  un split scientifique. Il n'a jamais pour objectif de mesurer une généralisation.
- Les résultats de la baseline Persistence (section 6) sont qualifiés de
  **vérification technique sur échantillon**, jamais de benchmark.
- Aucun score, aucune performance OOD, aucun événement supplémentaire n'a été
  inventé pour combler ce manque de données.

## 2. Lot 1 — DataLoader réel (`src/geofirex/real_dataset.py`)

- Lit directement les 23 bandes brutes (pas 40 — voir `docs/DATASET_VERSION.md`).
- Ordre et signification des bandes documentés dans `src/geofirex/real_constants.py`
  (`RAW_BAND_NAMES`), vérifiés contre les descriptions GDAL réelles ET contre le
  datasheet officiel — les deux sources concordent exactement.
- Convention de binarisation "active fire" : `~isnan(valeur_brute)`, jamais un
  seuil sur la valeur (qui est un code de détection VIIRS non binaire, ex. 918,
  2036 observés) — appliquée à CHAQUE lecture du canal 23 (contexte ET cible),
  conforme au datasheet §5.3.
- NaN géré explicitement : canal "active fire" → binarisation par présence ;
  22 autres canaux → NaN remplacé par 0.0 (nuages/capteur, 1 à 8% des pixels
  selon le canal, voir `reports/DATA_AUDIT.md` §5).
- Tenseurs produits : `X: [T, 23, H, W]`, `Y: [1, H, W]` — vérifiés sans NaN
  (tests `tests/test_real_dataset.py`).
- Informations géographiques (CRS, dates, chemins) jamais jetées : accessibles
  via `RealGeoFireXDataset.crs_for(idx)`, hors du tenseur batché (qui doit rester
  homogène pour le `collate_fn` par défaut de PyTorch — les 5 événements réels
  n'ont pas tous la même résolution spatiale : 311×265 à 339×299 px).
- Aucune modification destructive des fichiers GeoTIFF bruts (lecture seule).
- Le pipeline synthétique (`src/geofirex/dataset.py`, 40 canaux) n'a été touché
  en aucune façon — `pytest tests/test_dataset.py tests/test_baseline.py`
  continue de passer sans changement (vérifié section 7).

## 3. Lot 2 — Fenêtres multi-jours (`src/geofirex/windows.py`)

Constructeur configurable pour T=1, 2, 3 (testé explicitement pour les trois).
Chaque fenêtre appartient à un seul événement, ses dates sont vérifiées
strictement consécutives, et sa forme spatiale est vérifiée homogène avant
acceptation.

Sur l'échantillon réel actuel (108 observations, 5 événements, 0 trou de date) :

| T (context_days) | Fenêtres créées | Rejets |
| --- | --- | --- |
| 1 | 103 | 0 |
| 2 | 98 | 0 |
| 3 | 93 | 0 |

Trois motifs de rejet sont possibles et testés unitairement
(`tests/test_windows.py`) : `event_too_short`, `date_gap`, `shape_mismatch`.
Aucun ne s'est produit sur l'échantillon réel actuel (confirmé : le dataset
réel ne contient aucun trou de date, déjà vérifié en Phase 2).

## 4. Lot 3 — Stratégie de split (`src/geofirex/real_splits.py`)

- **Protocole scientifique cible** (une fois assez d'événements disponibles) :
  validation croisée par année, recommandation **officielle** des auteurs du
  dataset (`split_events_by_year`) — jamais implémentée comme un split
  événementiel aléatoire classique, conformément au datasheet.
- **Protocole actuel** : `make_smoke_test_split` — assignation déterministe
  (seed=42), un événement = un seul split, contrôlée par
  `geofirex.splits.check_no_leakage`. Résultat sur les 5 événements réels :

```
2018/fire_21748813 -> test
2018/fire_21997864 -> train
2019/fire_23300839 -> train
2020/fire_23654679 -> train
2020/fire_24104674 -> val
```

Distribution par année : 2018 (test=1, train=1), 2019 (train=1),
2020 (train=1, val=1). **2021 n'est représenté dans aucun split** — raison
supplémentaire pour laquelle ce split ne peut pas servir d'évaluation OOD
temporelle représentative.

Fichier versionnable : `splits/real/smoke_test_split.csv`.

## 5. Lot 4 — Métriques (`src/geofirex/metrics.py`)

Dice, IoU, Precision, Recall (déjà présents en Phase 1, inchangés par défaut)
étendus avec :

- **Masque de validité optionnel** (`mask=`) — exclut des pixels du calcul sans
  changer la signature utilisée par les appels existants (`mask=None` = tous
  les pixels valides, comportement antérieur préservé).
- **PR-AUC** (`pr_auc_score`, via `sklearn.metrics.average_precision_score`) —
  retourne `None` (jamais `0.0` fabriqué) quand la cible n'a aucun pixel positif,
  car la métrique y est mathématiquement indéfinie.
- **Agrégation par événement** (`aggregate_metrics_by_event`) — moyenne les
  métriques par `event_id`, en excluant proprement les `None` plutôt que de les
  compter comme zéro.

9 tests dans `tests/test_metrics_real.py`, chacun avec un résultat calculable à
la main (ex. 4 pixels, masque sur 2, TP=1/FP=1/FN=0 → precision=0.5 exact).

## 6. Lot 5 — Baseline Persistence réelle

**Vérification technique sur échantillon (3 événements du split "train" du
smoke-test) — pas un benchmark.** Aucun réseau de neurones entraîné.

```
Fenêtres valides : 67 (context_days=1)
Temps d'exécution : 13.54 s
Métriques globales (moyenne sur 67 fenêtres) :
  dice:      0.5584
  iou:       0.5201
  precision: 0.6848
  recall:    0.6728
  pr_auc:    0.1247  (peu informatif pour un prédicteur binaire — voir limite ci-dessous)
```

Métriques par événement (fort écart, attendu sur un échantillon de 3) :

| Événement | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- |
| 2018/fire_21997864 | 0.413 | 0.273 | 0.453 | 0.453 |
| 2019/fire_23300839 | 0.437 | 0.436 | 0.612 | 0.654 |
| 2020/fire_23654679 | 0.753 | 0.748 | 0.893 | 0.828 |

3 figures générées sous `reports/figures/real_baseline/` (feu courant / cible
réelle / prédiction Persistence / différence). Elles illustrent visuellement
la limite structurelle de Persistence : elle ne peut jamais prédire une
nouvelle ignition (zone rouge sur la carte de différence), seulement reproduire
ce qui était déjà observé.

**Limites d'interprétation :**
- PR-AUC est dégénéré ici car Persistence produit une prédiction déjà binaire
  (0/1), pas un score continu — la métrique garde son sens mathématique mais
  perd une grande partie de son intérêt pratique pour ce prédicteur précis.
  Elle redeviendra pleinement informative pour un futur modèle produisant des
  probabilités continues.
- Comparaison directe avec les statistiques officielles du datasheet (Table 2 :
  persistance médiane 23 pixels, ignition médiane 44, extinction médiane 42,
  sur l'ensemble du corpus) cohérente avec le pattern observé : Persistence
  rate systématiquement les nouvelles ignitions, d'où un recall imparfait
  structurel, pas un défaut d'implémentation.
- Le chiffre Dice=0.558 ci-dessus est une moyenne sur **toutes** les fenêtres,
  dont près de la moitié n'ont aucun feu ni observé ni prédit (score parfait
  par convention mathématique). Sur les seules fenêtres à cible réellement
  positive, Dice tombe à 0.247 — analyse détaillée dans
  `reports/VALIDATION_2.md` §F.

## 7. Lot 6 — MLOps (MLflow)

Expérience `geofirex-real-persistence-baseline`, run `real-persistence-baseline`.
Enregistré : configuration (split_name, context_days, seed), version du
dataset (référence textuelle Zenodo + échantillon 5-événements), version du
split (`smoke_test`, explicitement qualifié non-scientifique), seed, commit
Git, métriques globales, temps d'exécution, et le rapport JSON complet en
artefact. `mlflow.db` / `mlruns/` / `mlartifacts/` restent hors Git
(`.gitignore` déjà en place depuis la Phase 1).

## 8. Lot 7 — Tests et reproduction

### Résultats des tests

```
python3 -m pytest -q
→ 56 passed
```

Composition : tests hérités Phase 1/2 (compatibilité synthétique + Zenodo
distant, inchangés), `tests/test_windows.py` (7), `tests/test_real_dataset.py`
(6, rasters factices 23 bandes), `tests/test_real_splits.py` (6),
`tests/test_metrics_real.py` (5), `tests/test_real_pipeline_smoke.py` (2,
auto-skip si l'échantillon réel est absent du disque — jamais présent sur un
clone frais, donné qu'il est `.gitignore`).

### Commandes exactes de reproduction

```bash
# Lot 2/3 — fenêtres et split (déjà versionné, regénérable à l'identique)
python -c "from geofirex.real_splits import make_smoke_test_split, save_split_csv; \
  import csv; rows = list(csv.DictReader(open('data/manifests/real/manifest_wildfirespreadts.csv'))); \
  save_split_csv(make_smoke_test_split(sorted({r['event_id'] for r in rows}), seed=42), 'splits/real/smoke_test_split.csv')"

# Lot 5/6 — baseline Persistence réelle + figures + MLflow
python scripts/run_real_persistence_baseline.py \
  --manifest data/manifests/real/manifest_wildfirespreadts.csv \
  --split splits/real/smoke_test_split.csv \
  --split_name train --context_days 1 \
  --figures_dir reports/figures/real_baseline \
  --out reports/real_baseline_results.json

# Lot 7 — tous les tests
python -m pytest -q
```

## 9. Limites globales et prochaines décisions

- Échantillon réel à étendre (au-delà du seuil de 20 événements) avant tout
  split scientifique ou toute mesure de généralisation.
- `LC_Type1` (canal catégoriel) toujours utilisé tel quel (valeur numérique de
  classe) dans `X` — pas encore encodé en one-hot ; acceptable pour une
  baseline Persistence (qui l'ignore totalement) mais à traiter avant tout
  modèle qui l'utiliserait réellement.
- PR-AUC peu informatif pour Persistence (prédicteur binaire) — sera plus
  pertinent pour un futur modèle à sortie continue.
- Aucun entraînement de réseau de neurones n'a été effectué dans cette phase,
  conformément à la contrainte de mission.
