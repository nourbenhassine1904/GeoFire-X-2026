"""Integration tests for RealGeoFireXDataset on small artificial rasters
mimicking the REAL 23-band raw layout - no dependency on the full download.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from geofirex.real_constants import RAW_ACTIVE_FIRE_CHANNEL, RAW_BAND_NAMES, RAW_NUM_CHANNELS
from geofirex.real_dataset import RealGeoFireXDataset, active_fire_target, read_real_channels
from geofirex.real_manifest import build_real_manifest, save_real_manifest
from geofirex.real_splits import save_split_csv


def _write_fake_real_day(
    path: Path, fire_pixels: list[tuple[int, int]] | None = None, nan_other: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((RAW_NUM_CHANNELS, 6, 6), 1.0, dtype="float32")

    if nan_other:
        data[0, 0, 0] = np.nan  # simulate a cloud gap on a non-active-fire channel

    # Active fire: NaN everywhere except the given pixels (official convention).
    af = np.full((6, 6), np.nan, dtype="float32")
    for (r, c) in fire_pixels or []:
        af[r, c] = 2036.0  # a real-looking non-binary VIIRS detection code
    data[RAW_ACTIVE_FIRE_CHANNEL] = af

    with rasterio.open(
        path, "w", driver="GTiff", height=6, width=6, count=RAW_NUM_CHANNELS, dtype="float32",
        crs="EPSG:32610", transform=from_origin(0.0, 0.0, 375.0, 375.0),
    ) as dst:
        dst.write(data)
        dst.descriptions = list(RAW_BAND_NAMES)


def test_read_real_channels_binarizes_active_fire_and_zero_fills_others(tmp_path: Path):
    path = tmp_path / "day.tif"
    _write_fake_real_day(path, fire_pixels=[(1, 1), (2, 2)], nan_other=True)

    arr = read_real_channels(path)
    assert arr.shape == (RAW_NUM_CHANNELS, 6, 6)
    assert not np.isnan(arr).any()  # no NaN survives into the output
    assert arr[0, 0, 0] == 0.0  # cloud-gap NaN on a non-active-fire channel -> 0
    af = arr[RAW_ACTIVE_FIRE_CHANNEL]
    assert set(np.unique(af).tolist()) == {0.0, 1.0}  # binarized, never the raw code
    assert af[1, 1] == 1.0 and af[2, 2] == 1.0
    assert af.sum() == 2.0


def test_active_fire_target_matches_read_real_channels_binarization(tmp_path: Path):
    path = tmp_path / "day.tif"
    _write_fake_real_day(path, fire_pixels=[(0, 0)])

    target = active_fire_target(path)
    assert target.shape == (1, 6, 6)
    assert target[0, 0, 0] == 1.0
    assert target.sum() == 1.0


def test_read_real_channels_rejects_wrong_band_count(tmp_path: Path):
    path = tmp_path / "bad.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=5, dtype="float32",
    ) as dst:
        dst.write(np.zeros((5, 4, 4), dtype="float32"))

    import pytest

    with pytest.raises(ValueError, match="23"):
        read_real_channels(path)


def _build_tiny_real_dataset(tmp_path: Path, context_days: int = 1):
    data_dir = tmp_path / "real_sample"
    _write_fake_real_day(data_dir / "2020" / "fire_1" / "2020-01-01.tif", fire_pixels=[(1, 1)])
    _write_fake_real_day(data_dir / "2020" / "fire_1" / "2020-01-02.tif", fire_pixels=[(1, 1), (1, 2)])
    _write_fake_real_day(data_dir / "2020" / "fire_1" / "2020-01-03.tif", fire_pixels=[(1, 1), (1, 2), (2, 2)])

    # No explicit root: data_dir (under tmp_path) is outside the repo, so
    # build_real_manifest falls back to an absolute path - resolvable
    # regardless of the test runner's current working directory.
    rows = build_real_manifest(data_dir)
    manifest_path = tmp_path / "manifest.csv"
    save_real_manifest(rows, manifest_path)

    split_path = tmp_path / "split.csv"
    save_split_csv({"2020/fire_1": "train"}, split_path)

    return manifest_path, split_path


def test_real_dataset_produces_contract_shapes(tmp_path: Path):
    manifest_path, split_path = _build_tiny_real_dataset(tmp_path, context_days=1)

    ds = RealGeoFireXDataset(manifest_path, split_path, split_name="train", context_days=1)
    assert len(ds) == 2  # 3 days, context_days=1 -> 2 windows

    x, y = ds[0]
    assert x.shape == (1, RAW_NUM_CHANNELS, 6, 6)
    assert y.shape == (1, 6, 6)
    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()
    assert set(y.unique().tolist()) <= {0.0, 1.0}

    # Target of window 0 must be the fire mask of 2020-01-02 (2 pixels), never
    # a channel read from the context day itself (no future/self leakage).
    assert y.sum().item() == 2.0


def test_real_dataset_preserves_geographic_and_temporal_metadata(tmp_path: Path):
    manifest_path, split_path = _build_tiny_real_dataset(tmp_path, context_days=1)
    ds = RealGeoFireXDataset(manifest_path, split_path, split_name="train", context_days=1)

    w = ds.crs_for(0)
    assert w.event_id == "2020/fire_1"
    assert w.context_dates == ("2020-01-01",)
    assert w.target_date == "2020-01-02"


def test_real_dataset_empty_split_has_zero_length(tmp_path: Path):
    manifest_path, split_path = _build_tiny_real_dataset(tmp_path, context_days=1)
    ds = RealGeoFireXDataset(manifest_path, split_path, split_name="test", context_days=1)
    assert len(ds) == 0
    assert "created:  0" in ds.window_report()
