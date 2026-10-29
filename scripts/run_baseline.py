"""End-to-end reproducible run: manifest -> event split -> DataLoader ->
Persistence baseline -> Dice/IoU/Precision/Recall -> MLflow.

Usage:
    python scripts/run_baseline.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mlflow
import yaml

from geofirex.baseline import evaluate_persistence
from geofirex.dataset import GeoFireXDataset
from geofirex.manifest import build_manifest, save_manifest
from geofirex.splits import check_no_leakage, save_split, split_events


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    data_dir = Path(config["data_dir"])
    manifest_path = Path(config["manifest_path"])
    split_path = Path(config["split_path"])

    rows = build_manifest(data_dir)
    if not rows:
        raise SystemExit(
            f"No GeoTIFF files found under {data_dir}. "
            "Run scripts/generate_synthetic.py first, or point data_dir at real data."
        )
    save_manifest(rows, manifest_path)

    event_ids = [r["event_id"] for r in rows]
    assignment = split_events(event_ids, seed=config["seed"], ratios=tuple(config["split_ratios"]))
    check_no_leakage(assignment)
    save_split(assignment, split_path)

    val_set = GeoFireXDataset(
        manifest_path, split_path, split_name="val", context_days=config["context_days"]
    )
    metrics = evaluate_persistence(val_set, batch_size=config["batch_size"])

    # mlflow>=3 puts the file-store backend ("./mlruns") in maintenance mode and
    # defaults to a local sqlite db instead; mlflow.db (+ sqlite sidecar files)
    # is excluded in .gitignore alongside mlruns/ so no run DB reaches Git.
    mlflow.set_experiment("geofirex-baseline")
    with mlflow.start_run(run_name="persistence-baseline"):
        mlflow.log_param("dataset_version", config["dataset_version"])
        mlflow.log_param("split_version", config["split_version"])
        mlflow.log_param("config_path", str(args.config))
        mlflow.log_param("seed", config["seed"])
        mlflow.log_param("context_days", config["context_days"])
        mlflow.log_param("git_commit", git_commit())
        mlflow.log_artifact(str(args.config))
        mlflow.log_artifact(str(split_path))
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    print("Persistence baseline (val split):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
