# GeoFire-X

Multimodal Spatio-Temporal Deep Learning for Next-Day Wildfire Spread Forecasting under Geographic and Temporal Domain Shift.

## Current Phase

**Suivi 3 — GPU pilot completed (1 November 2026).** B1 CNN, B2 U-Net and B3 ConvLSTM
each completed 10 epochs / 1,280 updates on the certified 108-TIFF sample.
B1 is the demonstration candidate (validation AP 0.3535); Persistence retains
slightly higher Dice (0.4814 vs 0.4800). These are pilot results, one seed,
three training events and two validation events; test 2021 remains locked.

- [Demonstration, installation and restored MLflow](docs/DEMO_SUIVI3.md)
- [GPU results and verified backup](reports/GPU_PILOT_2026-11-01.md)
- [GitHub publication and integration plan](docs/GITHUB_SUIVI3.md)

The following preparation and CPU sections describe historical work. Do not
rerun their acquisition or training commands to launch the current demonstration.

Benchmark preparation now extends the project to all 607 official events:
[delivery report](reports/BENCHMARK_PREPARATION_REPORT.md),
[verified U-Net audit](reports/UNET_AUDIT_V2.md),
[acquisition and storage budget](docs/FULL_CORPUS_PLAN.md), and
[multi-model protocol](docs/BENCHMARK_PROTOCOL.md).
Annual splits and B0–B3 prototypes are prepared; the full corpus has not been
downloaded or trained. Large acquisition and long training require a reviewed budget.

Suivi 3 — First trained lightweight 2D U-Net for J+1 active-fire segmentation.
The five-event sample supports technical verification only. The first short
training performs substantially worse than Persistence; all measured results
are retained in [the training report](reports/UNET_BASELINE_REPORT.md).

The preceding real-data spatio-temporal pipeline remains documented in
`reports/REAL_PIPELINE_REPORT.md`.

### U-Net baseline (real data, CPU or CUDA)

Use the project environment on Windows (Python 3.11), not the global Python:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -u scripts/train_unet.py --config configs/unet_baseline.yaml --device cpu
```

See [the audited protocol and inference commands](docs/UNET_BASELINE.md).
Training creates a unique artifact directory and a new run in
`geofirex-unet2d-real-v1`. Raw rasters, historical manifests/splits and previous
experiments are preserved. Checkpoints and MLflow artifacts stay outside Git.
The conservative input policy uses current fire and terrain; annual land cover
has an explicit optional one-hot encoder and is excluded from this first run.

See [Project Context](docs/PROJECT_CONTEXT.md) for the scientific objective,
input and target contracts, scope, and experimental rules. See
[Dataset Version](docs/DATASET_VERSION.md) and
[Data Audit](reports/DATA_AUDIT.md) for the verified real-data facts (23 raw
bands, not 40; active-fire binarization convention; official split
recommendation), and [Split Strategy](docs/SPLIT_STRATEGY.md) for why the
current 5-event real sample only supports a smoke-test split.

### Reproducing the real pipeline (Phase 3)

```bash
pip install -e .

# Fetch a real, deterministic 5-event sample (no full 48.36 GB download)
python scripts/inspect_remote_archive.py --out data/raw/WildfireSpreadTS/archive_index.json
python scripts/download_real_sample.py --dest data/raw/WildfireSpreadTS/real_sample --num_events 5 --seed 42
python scripts/build_real_manifest.py --data_dir data/raw/WildfireSpreadTS/real_sample --out data/manifests/real/manifest_wildfirespreadts.csv

# Persistence baseline on the real DataLoader (no neural network trained)
python scripts/run_real_persistence_baseline.py \
  --manifest data/manifests/real/manifest_wildfirespreadts.csv \
  --split splits/real/smoke_test_split.csv --split_name train --context_days 1 \
  --figures_dir reports/figures/real_baseline --out reports/real_baseline_results.json

pytest -q
```

## Primary Dataset

WildfireSpreadTS

## Team

4 Data Science / AI engineering students

## Scientific Pipeline

Data Audit → Event-Level Split → DataLoader → Persistence Baseline → Metrics → ID/OOD Evaluation → Multimodal & Temporal Modeling → Robustness

ID/OOD refers to in-distribution and out-of-distribution evaluation.

## Repository Structure

| Directory | Purpose |
| --- | --- |
| `data/` | Local raw, interim, and processed datasets; tracked metadata in `manifests/`. Dataset contents are ignored by Git. |
| `notebooks/` | Exploratory analysis and data audit notebooks. |
| `src/geofirex/` | Shared Python package for reusable project code. |
| `scripts/` | Reproducible command-line workflows. |
| `tests/` | Checks for data contracts, leakage prevention, and project behavior. |
| `splits/` | Versioned event-level train/validation/test assignments. |
| `configs/` | Shared data and experiment configurations. |
| `reports/` | Evaluation reports and figures in `figures/`. |
| `docs/` | Scientific context and project documentation. |

Empty directories contain `.gitkeep` placeholders so Git preserves the structure.
Keep manifests and split definitions small and versioned; keep datasets and
model artifacts outside Git.

`data/manifests/demo/` and `splits/demo/` hold synthetic reference fixtures
(file names end in `_synthetic`) used to develop and test the pipeline before
real WildfireSpreadTS data is downloaded — see `docs/DATASET_VERSION.md`. Real
manifests and splits must use non-`demo` paths so the two are never confused.

## Collaboration Workflow

- Keep `main` stable and ready for the whole team to use.
- Develop each contribution on a focused feature branch created from `main`.
- Submit contributions through a Pull Request with a clear description and relevant validation.
- Have another student review the Pull Request before merging.
- Never commit large datasets, checkpoints, or experiment artifacts to Git.

Assign each wildfire event to exactly one split: no day from a training event
may appear in validation or test. Every experiment must record its dataset
version, split version, configuration, random seed, and Git commit.

## Historique et provenance

Cette version poursuit les travaux du projet GeoFire-X
initialement developpe dans :
https://github.com/nourbenhassine1904/GeoFire-X

Le code et les resultats des suivis precedents constituent
la base de cette nouvelle phase de developpement.
Les contributions historiques restent consultables dans
le depot source.

