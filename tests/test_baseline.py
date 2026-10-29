from pathlib import Path

from geofirex.baseline import evaluate_persistence
from geofirex.dataset import GeoFireXDataset
from geofirex.manifest import build_manifest, save_manifest
from geofirex.splits import save_split, split_events
from geofirex.synthetic import generate_synthetic_dataset


def test_persistence_baseline_runs_and_returns_bounded_metrics(tmp_path: Path):
    data_dir = tmp_path / "data"
    generate_synthetic_dataset(data_dir, num_events=6, min_days=6, max_days=10, height=24, width=24, seed=5)

    rows = build_manifest(data_dir)
    manifest_path = tmp_path / "manifest.csv"
    save_manifest(rows, manifest_path)

    event_ids = [r["event_id"] for r in rows]
    assignment = split_events(event_ids, seed=5, ratios=(0.5, 0.25, 0.25))
    split_path = tmp_path / "split.csv"
    save_split(assignment, split_path)

    val_set = GeoFireXDataset(manifest_path, split_path, split_name="val", context_days=3)
    if len(val_set) == 0:
        val_set = GeoFireXDataset(manifest_path, split_path, split_name="train", context_days=3)

    metrics = evaluate_persistence(val_set, batch_size=2)
    for key in ("dice", "iou", "precision", "recall"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
