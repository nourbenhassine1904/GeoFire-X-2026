from pathlib import Path

import torch

from geofirex.constants import NUM_CHANNELS
from geofirex.dataset import GeoFireXDataset
from geofirex.manifest import build_manifest, save_manifest
from geofirex.splits import save_split, split_events
from geofirex.synthetic import generate_synthetic_dataset


def _build_small_dataset(tmp_path: Path, context_days: int = 3):
    data_dir = tmp_path / "data"
    generate_synthetic_dataset(data_dir, num_events=4, min_days=6, max_days=8, height=16, width=16, seed=3)

    rows = build_manifest(data_dir)
    manifest_path = tmp_path / "manifest.csv"
    save_manifest(rows, manifest_path)

    event_ids = [r["event_id"] for r in rows]
    assignment = split_events(event_ids, seed=3, ratios=(0.5, 0.25, 0.25))
    split_path = tmp_path / "split.csv"
    save_split(assignment, split_path)

    return manifest_path, split_path


def test_dataset_produces_contract_shapes(tmp_path: Path):
    manifest_path, split_path = _build_small_dataset(tmp_path, context_days=3)

    # At least one split must be non-empty given 4 events of 6-8 days each.
    found_non_empty = False
    for split_name in ("train", "val", "test"):
        ds = GeoFireXDataset(manifest_path, split_path, split_name=split_name, context_days=3)
        if len(ds) == 0:
            continue
        found_non_empty = True
        x, y = ds[0]
        assert x.shape == (3, NUM_CHANNELS, 16, 16)
        assert y.shape == (1, 16, 16)
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()
        assert set(y.unique().tolist()) <= {0.0, 1.0}  # binary active-fire mask

    assert found_non_empty, "no split produced any sample - check event lengths vs context_days"
