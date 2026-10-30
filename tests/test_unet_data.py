"""Raster fixtures are artificial contract tests, never MLflow experiments."""
import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.real_manifest import build_real_manifest, save_real_manifest
from geofirex.real_splits import save_split_csv
from geofirex.unet import LightUNet
from geofirex.unet_data import (Normalizer, PatchDataset, WindowStore, fit_normalizer,
                               load_event_windows, read_raw)
from geofirex.unet_evaluation import predict_tiled
from scripts.predict_unet import predict_file


def write_day(path, slope, fire=True, forecast_variant=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.ones((23, 20, 24), dtype=np.float32)
    raw[12], raw[14], raw[13] = slope, 100, 90
    raw[22] = np.nan
    if fire:
        raw[22, 3:5, 3:5] = 2036
    names = list(RAW_BAND_NAMES)
    if forecast_variant:
        names[17] = "total_precipitation_surface_last"
    with rasterio.open(path, "w", driver="GTiff", height=20, width=24, count=23,
                       dtype="float32", crs="EPSG:32610", transform=from_origin(0, 0, 375, 375)) as dst:
        dst.write(raw)
        dst.descriptions = names
    return raw


@pytest.fixture
def inputs(tmp_path):
    for event, values in (("fire_1", [1, 3, 1000000]), ("fire_2", [900, 900]), ("fire_3", [800, 800])):
        for day, value in enumerate(values, 1):
            write_day(tmp_path / "raw" / "2020" / event / f"2020-01-{day:02d}.tif", value)
    manifest, split = tmp_path / "manifest.csv", tmp_path / "split.csv"
    save_real_manifest(build_real_manifest(tmp_path / "raw"), manifest)
    save_split_csv({"2020/fire_1": "train", "2020/fire_2": "val", "2020/fire_3": "test"}, split)
    rows, mapping, windows, rejections = load_event_windows(manifest, split, tmp_path)
    return manifest, split, windows


def test_normalization_fits_only_train_inputs_not_targets_or_heldout(inputs):
    _, _, windows = inputs
    norm = fit_normalizer(windows["train"])
    assert norm.mean == [2., 100.]
    assert norm.std == [1., 1.]
    assert len(norm.fit_paths) == 2
    assert not set(norm.fit_paths) & {w.context_paths[0] for w in windows["val"] + windows["test"]}
    assert all("2020-01-03" not in p for p in norm.fit_paths)
    assert len(norm.channel_names) == 8


def test_nan_inf_angles_and_categorical_unknowns_have_explicit_encoding():
    norm = Normalizer([12, 14], [2., 100.], [1., 1.], [3, 3], [], True)
    raw = np.ones((23, 1, 6), dtype=np.float32)
    raw[12] = np.array([2, np.nan, np.inf, 4, 5, 6])
    raw[14], raw[13], raw[22] = 100, 90, np.nan
    raw[22, 0, 0] = 0  # Any finite detection code is present under the dataset contract.
    raw[16, 0] = [1, 17, np.nan, 1.5, 255, 0]
    raw[13, 0, 2] = np.nan
    x = norm.transform(raw)
    assert x.shape == (26, 1, 6) and np.isfinite(x).all()
    assert x[0, 0, 1] == x[0, 0, 2] == 0
    assert x[2, 0, 1] == x[2, 0, 2] == 1
    assert x[4, 0, 0] == pytest.approx(1)
    assert abs(x[5, 0, 0]) < 1e-6
    assert x[6, 0, 2] == 1 and x[7, 0, 0] == 1
    assert x[8, 0, 0] == 1 and x[24, 0, 1] == 1
    np.testing.assert_array_equal(x[25, 0], [0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(x[8:].sum(axis=0), np.ones((1, 6)))


def test_dataset_balanced_sampling_reproducible_and_padding_excluded(inputs):
    _, _, windows = inputs
    store = WindowStore(windows["train"], fit_normalizer(windows["train"]))
    ds = PatchDataset(store, size=16, samples=10, positive_fraction=.5, seed=42)
    labels = [ds[i][1].sum() > 0 for i in range(len(ds))]
    assert sum(labels) == 5
    a = ds[0]
    assert all(torch.equal(x, y) for x, y in zip(a, ds[0]))
    x, y, valid = next(iter(DataLoader(ds, batch_size=2)))
    assert x.shape == (2, 8, 16, 16) and y.shape == valid.shape == (2, 1, 16, 16)
    validation = PatchDataset(store, size=16)
    # 2 windows * 20*24 valid pixels; all four tiles cover each raster once.
    assert sum(float(validation[i][2].sum()) for i in range(len(validation))) == 2*20*24
    assert all(y.sum() <= valid.sum() for _, y, valid in validation)


def test_split_rejects_duplicate_event_assignment(inputs, tmp_path):
    manifest, split, _ = inputs
    with split.open("a") as f:
        f.write("2020/fire_1,val\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_event_windows(manifest, split, tmp_path)


def test_manifest_rejects_file_reuse_across_events(inputs, tmp_path):
    manifest, split, _ = inputs
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[-1]["relative_path"] = rows[0]["relative_path"]
    save_real_manifest(rows, manifest)
    with pytest.raises(ValueError, match="reused"):
        load_event_windows(manifest, split, tmp_path)


def test_spatial_misalignment_is_rejected(inputs):
    _, _, windows = inputs
    norm = fit_normalizer(windows["train"])
    with rasterio.open(windows["train"][0].target_path, "r+") as ds:
        ds.transform = from_origin(375, 0, 375, 375)
    with pytest.raises(ValueError, match="misalignment"):
        WindowStore(windows["train"], norm).get(0)


def test_layout_variant_allowed_but_reordering_rejected(tmp_path):
    path = tmp_path / "day.tif"
    write_day(path, 1, forecast_variant=True)
    assert read_raw(path).shape == (23, 20, 24)
    with rasterio.open(path, "r+") as ds:
        names = list(ds.descriptions)
        names[12], names[14] = names[14], names[12]
        ds.descriptions = names
    with pytest.raises(ValueError, match="layout"):
        read_raw(path)


def test_normalizer_rejects_unsafe_continuous_features(inputs):
    _, _, windows = inputs
    for index in (13, 16, 17, 22):
        with pytest.raises(ValueError, match="Unsafe"):
            fit_normalizer(windows["train"], continuous=[index])


def test_checkpoint_inference_roundtrip_without_target_file(inputs, tmp_path):
    _, _, windows = inputs
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        norm = fit_normalizer(windows["train"])
        model = LightUNet(8, 4, 2).eval()
        source = windows["test"][0].context_paths[0]
        expected = predict_tiled(model, norm.transform(read_raw(source)), size=16, batch_size=2)
        path = tmp_path / "model.pt"
        torch.save({"model_state": model.state_dict(), "normalizer": norm.to_dict(),
                    "in_channels": 8, "threshold": .4, "threshold_selected_on": "val",
                    "config": {"base_channels": 4, "depth": 2, "cpu_threads": 2,
                               "patch_size": 16, "batch_size": 2}}, path)
        # Deployment consumes J only, so absence of any J+1 file is harmless.
        Path(windows["test"][0].target_path).unlink()
        result = predict_file(path, source, tmp_path / "prediction")
        np.testing.assert_array_equal(result, expected)
        with rasterio.open(tmp_path / "prediction" / "probability.tif") as dst, rasterio.open(source) as src:
            assert dst.crs == src.crs and dst.transform == src.transform
            np.testing.assert_array_equal(dst.read(), result)
        with rasterio.open(tmp_path / "prediction" / "mask.tif") as dst:
            np.testing.assert_array_equal(dst.read(), (result >= .4).astype("uint8"))
        with pytest.raises(FileExistsError):
            predict_file(path, source, tmp_path / "prediction")
    finally:
        torch.set_num_threads(old_threads)
