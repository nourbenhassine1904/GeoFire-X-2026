"""Tiny synthetic GeoTIFFs: runner integrity tests, never scientific evidence."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin
from sklearn.metrics import average_precision_score

from geofirex.benchmark import event_assignments
from geofirex.benchmark_models import make_model
from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.runner import (export_run, final_test, load_config, lock_selection,
                            preflight, train)
from geofirex.runner_data import FoldData, RasterReader, TemporalStore, TrainTiles, fit_fold_normalizer, read_tile
from geofirex.runner_evaluation import evaluate
from geofirex.unet import SegmentationLoss


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def corpus(tmp_path):
    rows, entries = [], []
    for year in (2018, 2019, 2020, 2021):
        h, w = (9, 13) if year % 2 else (11, 15)
        for day in range(1, 9):
            path = tmp_path/f"{year}/fire_1/{year}-01-{day:02d}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = np.ones((23, h, w), np.float32)
            raw[12], raw[14], raw[13] = day if year < 2020 else 1000, 100, 90
            raw[12, 1, 1] = np.nan
            raw[22] = np.nan
            raw[22, 2:5, 1:day+1] = 2036
            with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=23,
                               dtype="float32", crs="EPSG:32610",
                               transform=from_origin(year*100, 0, 375, 375)) as ds:
                ds.write(raw)
                ds.descriptions = RAW_BAND_NAMES
                valid = np.full((h, w), 255, np.uint8)
                valid[0] = 0
                ds.write_mask(valid)
            row = {"event_id": f"{year}/fire_1", "date": f"{year}-01-{day:02d}",
                   "relative_path": path.relative_to(tmp_path).as_posix(), "height": h, "width": w}
            rows.append(row)
            entries.append({"event_id": row["event_id"], "date": row["date"]})
    manifest = tmp_path/"manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split = tmp_path/"split.csv"
    with split.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "split"])
        writer.writerows(event_assignments([f"{year}/fire_1" for year in (2018, 2019, 2020, 2021)], 0).items())
    index = tmp_path/"index.json"
    index.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    config = tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump({"data_root": str(tmp_path), "manifest": str(manifest),
                                     "split": str(split), "index": str(index), "output_root": str(tmp_path/"runs"),
                                     "epochs": 2, "samples_per_epoch": 4, "batch_size": 2, "eval_batch_size": 2,
                                     "patch_size": 8, "context": 3, "preflight_steps": 40,
                                     "mlflow_enabled": False}), encoding="utf-8")
    return load_config(config)


def store_for(c, role="train"):
    data = FoldData(c)
    data.audit({"train", "val"})
    norm = fit_fold_normalizer(data.groups["train"][5], size=8)
    return data, norm, TemporalStore(data.windows(role), norm, data.groups[role][5])


def test_train_only_maximum_context_normalization_and_validity(corpus):
    data, norm, store = store_for(corpus)
    # Aligned targets days 6..8 -> unique train inputs days 1..7; no heldout 1000.
    assert norm.mean[0] == pytest.approx(4)
    assert len(norm.fit_paths) == 14
    assert all("2020" not in p and "2021" not in p for p in norm.fit_paths)
    x, y, valid, observed = store.tile(0, 0, 0, 8)
    assert x.shape == (3, 8, 8, 8)
    assert np.isfinite(x).all() and np.isin(y, [0, 1]).all()
    assert valid[:, 0].sum() == 0 and valid.sum() == 56
    # Absence encoded as NaN stays a valid negative.
    assert valid[0, 7, 7] == 1 and y[0, 7, 7] == 0
    assert observed.sum() < y.sum()
    ws = data.windows("train")[0]
    assert ws.context_dates == ("2018-01-03", "2018-01-04", "2018-01-05")
    assert ws.target_date == "2018-01-06"


def test_rejects_wrong_fold_partial_scientific_and_path_reuse(corpus):
    with pytest.raises(ValueError, match="annual fold"):
        FoldData({**corpus, "fold": 1})
    with pytest.raises(ValueError, match="607"):
        FoldData({**corpus, "qualification": "scientific"})
    path = Path(corpus["manifest"])
    rows = list(csv.DictReader(path.open()))
    rows[-1]["relative_path"] = rows[0]["relative_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="reused"):
        FoldData(corpus)


def test_temporal_domain_and_normalization_are_shared_across_contexts(corpus):
    _, norm, store = store_for(corpus)
    first_day = Path(corpus["data_root"])/"2018/fire_1/2018-01-01.tif"
    with rasterio.open(first_day, "r+") as ds:
        mask = ds.dataset_mask()
        mask[2, 2] = 0
        ds.write_mask(mask)
    results = []
    for context in (1, 3, 5):
        data = FoldData({**corpus, "context": context})
        temporal = TemporalStore(data.windows("train"), norm, data.groups["train"][5])
        x, _, mask, _ = temporal.tile(0, 0, 0, 8)
        assert x.shape[0] == context
        assert mask[0, 2, 2] == 0
        results.append((x[-1], mask))
    for x, mask in results[1:]:
        np.testing.assert_array_equal(x, results[0][0])
        np.testing.assert_array_equal(mask, results[0][1])


@pytest.mark.parametrize("name", ["B1", "B2", "B3"])
def test_real_pipeline_forward_backward_and_mini_overfit(corpus, name):
    _, _, store = store_for(corpus)
    tiles = TrainTiles(store, 8, 4, 42)
    torch.manual_seed(42)
    model = make_model(name, context=3, **({"depth": 1, "base_channels": 2} if name == "B2" else {}))
    report = preflight(model, tiles, SegmentationLoss(.5), corpus, "cpu")
    assert report["passed"], report
    assert report["input_shape"] == [1, 3, 8, 8, 8]


def test_complete_scene_micro_ap_confusion_and_ignitions(corpus, tmp_path):
    _, _, store = store_for(corpus, "val")
    model = make_model("B0", context=3)
    result = evaluate(model, store, corpus, "cpu", tmp_path/"evaluation")
    ps, ys = [], []
    for i, shape in enumerate(store.shapes):
        # Read the scene independently, excluding its explicit invalid top row.
        w = store.windows[i]
        previous, pmask = read_tile(w.context_paths[-1], 0, 0, 100)
        target, mask = read_tile(w.target_path, 0, 0, 100)
        valid = (mask*pmask).astype(bool)[0]
        ps.extend(np.isfinite(previous[22])[valid])
        ys.extend(np.isfinite(target[22])[valid])
    assert result["micro"]["average_precision"] == pytest.approx(average_precision_score(ys, ps))
    assert result["micro"]["pixels"] == len(ys)
    assert result["new_ignitions"]["tp"] == 0
    assert result["new_ignitions"]["fn"] > 0
    assert result["threshold"] == max(corpus["thresholds"])
    assert result["macro_event"]["average_precision"]["defined"] == 1


def test_checkpoint_resume_is_identical_and_test_requires_lock(corpus, monkeypatch):
    original_open = rasterio.open
    def guard(path, *args, **kwargs):
        assert "2021/fire_1" not in str(path).replace("\\", "/"), "Test raster opened during training"
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as guard_context:
        guard_context.setattr(rasterio, "open", guard)
        first = train(corpus)
    summary = json.loads((first/"summary.json").read_text())
    assert all(not key.startswith("2021/") for key in json.loads((first/"data_audit.json").read_text())["raster_fingerprints"])
    resumed = train(corpus, first/"epoch_0001.pt")
    a = torch.load(first/"epoch_0002.pt", weights_only=False)
    b = torch.load(resumed/"epoch_0002.pt", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["best_ap"] == b["best_ap"]
    lock_path = first.parent/"selection.json"
    lock_selection([first], lock_path)
    result = final_test(first, lock_path)
    metrics = json.loads((result/"test_metrics.json").read_text())
    assert metrics["threshold"] == summary["threshold"]
    assert metrics["micro"]["pixels"] > 0
    with pytest.raises(ValueError, match="not selected"):
        final_test(resumed, lock_path)
    (first/"normalization.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        final_test(first, lock_path)


@pytest.mark.parametrize("name", ["B0", "B2", "B3"])
def test_other_architectures_end_to_end_smoke(corpus, name):
    c = {**corpus, "model": name, "epochs": 1,
         "model_kwargs": {"depth": 1, "base_channels": 2} if name == "B2" else {}}
    out = train(c)
    s = json.loads((out/"summary.json").read_text())
    assert 0 <= s["validation_average_precision"] <= 1
    assert (out/s["checkpoint"]).is_file() and (out/"example.png").is_file()
    assert s["parameters"] == 0 if name == "B0" else s["parameters"] > 0
    archive = export_run(out, out.parent/f"{out.name}.tar.gz")
    assert archive.is_file()
    with pytest.raises(FileExistsError):
        export_run(out, archive)


def test_mlflow_records_finished_run_metrics_and_artifacts(corpus, tmp_path):
    from mlflow.tracking import MlflowClient
    c = {**corpus, "model": "B0", "mlflow_enabled": True,
         "tracking_uri": "sqlite:///"+(tmp_path/"tracking.db").as_posix(),
         "artifact_root": (tmp_path/"mlflow_artifacts").as_uri(),
         "experiment": "synthetic-runner-integration"}
    out = train(c)
    summary = json.loads((out/"summary.json").read_text())
    client = MlflowClient(tracking_uri=c["tracking_uri"])
    run = client.get_run(summary["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["phase"] == "train"
    assert run.data.tags["qualification"] == "diagnostic"
    assert run.data.metrics["validation_average_precision"] == summary["validation_average_precision"]
    assert "normalization.json" in {f.path for f in client.list_artifacts(summary["run_id"])}
    lock_selection([out], tmp_path/"selection.json")


def test_validation_without_positives_fails_instead_of_selecting(corpus, tmp_path):
    _, _, store = store_for(corpus, "val")
    for w in store.windows:
        with rasterio.open(w.target_path, "r+") as ds:
            ds.write(np.full(ds.shape, np.nan, np.float32), 23)
    with pytest.raises(ValueError, match="Validation has no positives"):
        evaluate(make_model("B0", context=3), store, corpus, "cpu", tmp_path/"empty")


def test_resume_refuses_changed_training_budget(corpus):
    first = train({**corpus, "epochs": 1})
    with pytest.raises(ValueError, match="identity differs"):
        train({**corpus, "epochs": 2}, first/"epoch_0001.pt")


def test_bounded_physical_cache_matches_window_reader_and_evicts(corpus):
    import pickle
    paths = sorted(Path(corpus["data_root"]).glob("2018/fire_1/*.tif"))
    reader = RasterReader(cache_bytes=4000)  # One 11x15 raster, 3300 bytes.
    for path in paths[:2]:
        expected = read_tile(str(path), 2, 3, 8)
        actual = reader.tile(str(path), 2, 3, 8)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
        assert reader.bytes <= reader.budget
    assert len(reader.cache) == 1
    reader.tile(str(paths[1]), 0, 0, 8)
    assert reader.hits == 1
    worker_reader = pickle.loads(pickle.dumps(reader))
    assert not worker_reader.cache and worker_reader.bytes == 0
    no_cache = RasterReader(cache_bytes=0)
    for a, b in zip(no_cache.tile(str(paths[0]), 2, 3, 8), read_tile(str(paths[0]), 2, 3, 8)):
        np.testing.assert_array_equal(a, b)
