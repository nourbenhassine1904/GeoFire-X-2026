# GeoFire-X — Runbook d'exécution, Validation 2

Toutes les commandes ci-dessous ont été vérifiées par exécution réelle le
10/10/2026 sur la branche `feature/phase3-real-pipeline`. Privilégiez la
démonstration hors-ligne (sections 2 à 5) le jour de la validation pour
éviter toute dépendance à une connexion Internet en salle.

## 1. Installer les dépendances

```bash
cd GeoFire-X          # racine du repository (contient pyproject.toml)
pip install -e .
```

Dépendances installées : `numpy`, `rasterio`, `torch`, `mlflow`, `pyyaml`,
`matplotlib`, `scikit-learn`. Aucune n'est spécifique à un OS ; testé sous
Windows avec Python 3.14.

## 2. Vérifier l'environnement

```bash
python -c "import geofirex; print('geofirex OK')"
python --version
```

Vérifiez aussi que les données réelles sont présentes (sinon, voir section
7) :

```bash
# Windows PowerShell :
Test-Path data\manifests\real\manifest_wildfirespreadts.csv
Test-Path data\raw\WildfireSpreadTS\real_sample
```

## 3. Lancer les tests automatisés

```bash
python -m pytest -q
```

**Résultat attendu (vérifié)** : `56 passed` en 15-20 secondes selon la
machine. Deux tests (`tests/test_real_pipeline_smoke.py`) s'auto-désactivent
si l'échantillon réel est absent — ce n'est pas un échec, juste une absence
de données locales.

Pour un détail test par test :

```bash
python -m pytest -v
```

## 4. Exécuter la démonstration

```bash
python scripts/demo_validation2.py
```

**Ne nécessite aucune connexion Internet.** Ne télécharge rien, ne lance
aucun entraînement, n'écrase aucun résultat existant. Si l'échantillon réel
est absent, le script l'explique et bascule automatiquement vers un mode
synthétique clairement distinct (jamais présenté comme des données réelles).

Durée : quelques secondes.

## 5. Afficher les figures

Toutes les figures sont déjà générées sur disque, aucune régénération n'est
nécessaire le jour J. Ouvrez-les directement avec la visionneuse d'images du
système, ou intégrez-les dans la présentation PowerPoint :

| Figure | Chemin |
| --- | --- |
| Architecture logicielle | `reports/figures/validation2/figure1_architecture.png` |
| Événements par année (corpus complet) | `reports/figures/data_audit/events_per_year_full_corpus.png` |
| Observation multimodale réelle | `reports/figures/data_audit/sample_observation.png` |
| Construction de fenêtre J+1 | `reports/figures/validation2/figure4_window_construction.png` |
| Répartition feu/non-feu | `reports/figures/validation2/figure5_label_distribution.png` |
| Exemple Persistence (4 panneaux) | `reports/figures/real_baseline/sample_01_2018_fire_21997864_2018-08-04.png` |
| Feuille de route | `reports/figures/validation2/figure7_roadmap.png` |

Pour les régénérer si besoin (optionnel, aucune donnée supplémentaire
requise) :

```bash
python scripts/generate_data_audit.py --manifest data/manifests/real/manifest_wildfirespreadts.csv --archive_index data/raw/WildfireSpreadTS/archive_index.json --figures_dir reports/figures/data_audit --out reports/DATA_AUDIT.md
python scripts/run_real_persistence_baseline.py --manifest data/manifests/real/manifest_wildfirespreadts.csv --split splits/real/smoke_test_split.csv --split_name train --context_days 1 --figures_dir reports/figures/real_baseline --out reports/real_baseline_results.json
python scripts/generate_validation2_figures.py --out_dir reports/figures/validation2
```

## 6. Consulter les expériences MLflow (si disponible)

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Puis ouvrir `http://127.0.0.1:5000` dans un navigateur. Expérience à
consulter : `geofirex-real-persistence-baseline`. **Ceci nécessite que
`mlflow.db` soit présent localement** (généré par une exécution précédente
de `run_real_persistence_baseline.py` — jamais versionné dans Git). Si
`mlflow.db` est absent, ce n'est pas bloquant pour la démonstration : les
résultats chiffrés sont déjà dans `reports/real_baseline_results.json`.

## 7. Données devant être présentes localement

| Donnée | Chemin | Versionnée dans Git ? |
| --- | --- | --- |
| Manifeste réel | `data/manifests/real/manifest_wildfirespreadts.csv` | Oui |
| Fichiers GeoTIFF réels (108 fichiers, ~440 Mo) | `data/raw/WildfireSpreadTS/real_sample/` | **Non** (`.gitignore`) |
| Index de l'archive officielle | `data/raw/WildfireSpreadTS/archive_index.json` | **Non** |
| Split smoke-test | `splits/real/smoke_test_split.csv` | Oui |
| Résultats de la baseline | `reports/real_baseline_results.json` | Oui |
| Base MLflow | `mlflow.db` | **Non** |

Les fichiers GeoTIFF bruts ne sont jamais dans Git : ils doivent être
récupérés une fois par machine (voir section 8). Le manifeste et les
résultats, eux, sont déjà versionnés — un clone frais du repository suffit
pour lire `reports/VALIDATION_2.md` et toutes les figures sans rien
télécharger.

## 8. Si les données réelles sont absentes sur la machine de démonstration

```bash
python scripts/inspect_remote_archive.py --out data/raw/WildfireSpreadTS/archive_index.json
python scripts/download_real_sample.py --dest data/raw/WildfireSpreadTS/real_sample --num_events 5 --seed 42
python scripts/build_real_manifest.py --data_dir data/raw/WildfireSpreadTS/real_sample --out data/manifests/real/manifest_wildfirespreadts.csv
```

**Nécessite une connexion Internet** (requêtes vers Zenodo, ~440 Mo
transférés, quelques minutes selon le réseau — le serveur Zenodo a parfois
des lenteurs). Déterministe (seed=42) : reproduit exactement le même
échantillon de 5 événements que celui déjà documenté dans ce projet.
**Ne télécharge jamais l'archive complète de 48,36 Go.**

## 9. Démonstration de secours (hors ligne, sans données réelles)

Si la connexion Internet est indisponible en salle ET que les données
réelles ne sont pas déjà sur la machine, la démonstration de secours
fonctionne entièrement hors ligne sur le pipeline synthétique :

```bash
python scripts/demo_validation2.py
```

Si même les fixtures synthétiques (`data/manifests/demo/`,
`splits/demo/`) sont absentes (cas extrême, improbable sur un clone normal
du repository puisqu'elles sont versionnées), les régénérer localement sans
connexion Internet :

```bash
python scripts/generate_synthetic.py --data_dir data/raw/synthetic --num_events 8 --seed 42
python scripts/run_baseline.py --config configs/baseline.yaml
```

Dans tous les cas, **présenter oralement que ce mode est une démonstration
logicielle sur données synthétiques, jamais un résultat WildfireSpreadTS.**
