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


def test_positive_sampling_is_reproducible_and_train_only(corpus):
    _, _, store = store_for(corpus)
    tiles = TrainTiles(store, 8, 200, 42, positive_fraction=1.)
    prior = tiles.prior()
    assert prior["positive_tiles"] < prior["total_tiles"]
    choices = [tiles.coordinate(i) for i in range(len(tiles))]
    assert choices == [tiles.coordinate(i) for i in range(len(tiles))]
    assert all(t in tiles.positive_tiles for t in choices)
    tiles.epoch = 2
    assert choices != [tiles.coordinate(i) for i in range(len(tiles))]


def test_early_stopping_and_successful_optimizer_update_counts(corpus):
    c = {**corpus, "epochs": 5, "early_stopping_patience": 1,
         "early_stopping_min_delta": 1., "positive_tile_fraction": .5}
    out = train(c)
    summary = json.loads((out/"summary.json").read_text())
    assert summary["early_stopped"] and len(summary["history"]) == 2
    assert summary["optimizer_updates"] == 4
    assert not summary["complete_corpus"]
    assert summary["corpus_counts"]["train"]["manifest_observations"] == 16
    assert summary["history"][0]["observations_covered"] > 0


def test_resume_across_changed_readonly_mount_paths(corpus, tmp_path):
    import shutil
    first = train(corpus)
    mounted = tmp_path/"new-session-input"
    mounted.mkdir()
    roots = []
    for year in (2018, 2019, 2020, 2021):
        # Reverse lexical mount order relative to event/year order.
        root = mounted/str(10000-year)
        shutil.copytree(Path(corpus["data_root"])/str(year), root/str(year))
        roots.append(str(root))
    c = {**corpus, "data_root": str(tmp_path/"empty"), "data_roots": roots,
         "output_root": str(tmp_path/"new-session-runs")}
    resumed = train(c, first/"epoch_0001.pt")
    a = torch.load(first/"epoch_0002.pt", weights_only=False)
    b = torch.load(resumed/"epoch_0002.pt", weights_only=False)
    assert a["identity"] == b["identity"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_multimodal_requires_availability_and_retains_fire_channel(corpus, tmp_path):
    from geofirex.feature_profiles import PROFILES, feature_profile
    from geofirex.runner import runner_model
    policy = "multimodal_observed_j_v1"
    with pytest.raises(ValueError, match="availability"):
        feature_profile({**corpus, "feature_policy": policy})
    declaration = tmp_path/"availability.json"
    declaration.write_text(json.dumps({"policy": policy, "reviewed": True,
        "decision_time": "end_of_day_J", "available_by_J_bands": list(range(23)),
        "evidence": ["Synthetic fixture only; no real-world availability claim"]}))
    c = {**corpus, "model": "B0", "feature_policy": policy, "feature_availability": str(declaration)}
    profile = feature_profile(c)
    assert 7 not in profile["continuous"] and not set(range(17, 22)) & set(profile["continuous"])
    data = FoldData(c)
    norm = fit_fold_normalizer(data.groups["train"][5], size=8,
                               continuous=profile["continuous"], landcover=True)
    store = TemporalStore(data.windows("train"), norm, data.groups["train"][5], cache_bytes=0)
    x, y, valid, observed = store.tile(0, 0, 0, 8)
    assert x.shape[1] == 50 and np.isfinite(x).all()
    np.testing.assert_array_equal(runner_model(c, norm).predict_proba(torch.from_numpy(x)[None])[0], observed)
    assert norm.mean[profile["continuous"].index(12)] < 1000
    out = train(c)
    assert json.loads((out/"features.json").read_text())["channels"] == norm.channel_names


def test_prepare_mounted_real_headers_full_split_and_duplicate_guard(corpus, tmp_path):
    import shutil
    import zipfile
    from geofirex.cloud_corpus import batch_inventory, plan_batch, prepare_mounted
    from geofirex.corpus import acquire_members, index_zip, json_new
    zip_path = tmp_path/"raster-fixture.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for year in (2018, 2019, 2020, 2021):
            for path in (Path(corpus["data_root"])/str(year)).rglob("*.tif"):
                z.write(path, path.relative_to(corpus["data_root"]).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        index = index_zip(z, archive_size=zip_path.stat().st_size)
        roots = []
        for year in (2018, 2020):
            plan = plan_batch(index, years=[year], max_events=1, transfer_bytes=10**7,
                storage_bytes=2*1024**3, private_free_bytes=10**7, upload_bytes=10**7)
            root = tmp_path/f"mounted-{year}"
            acquire_members(z, plan["entries"], root)
            json_new(root/"inventory.json", batch_inventory(plan, root))
            roots.append(root)
    dest = tmp_path/"prepared"
    report = prepare_mounted(index, roots, dest)
    assert report["actual_events"] == 2 and not report["complete_corpus"]
    assert report["by_split"]["train"]["observations"] == 8
    assert len(list(csv.DictReader((dest/"split.csv").open()))) == 4
    json_new(dest/"index.json", index)
    c = {**corpus, "manifest": str(dest/"manifest.csv"), "index": str(dest/"index.json"),
         "split": str(dest/"split.csv"), "data_roots": [str(p) for p in roots], "qualification": "pilot"}
    data = FoldData(c)
    assert data.audit({"train", "val"})["corpus_counts"]["train"]["used_events"] == 1
    with pytest.raises(ValueError, match="Duplicate events"):
        prepare_mounted(index, [*roots, roots[0]], tmp_path/"duplicate")


def test_mlflow_training_snapshot_contains_portable_metric_history(corpus, tmp_path):
    from geofirex.kaggle_persistence import snapshot_training, verify_package
    c = {**corpus, "model": "B0", "mlflow_enabled": True,
         "tracking_uri": "sqlite:///"+(tmp_path/"snapshot-tracking.db").as_posix(),
         "artifact_root": (tmp_path/"snapshot-mlflow-artifacts").as_uri(),
         "experiment": "snapshot-integration"}
    out = train(c)
    package = tmp_path/"snapshot-package"
    inv = snapshot_training(out, package, tracking_db=tmp_path/"snapshot-tracking.db")
    assert verify_package(package, inv)["files"] > 10
    history = json.loads((package/"mlflow_metric_history.json").read_text())
    assert history["validation_average_precision"][0]["value"] >= 0


def test_epoch_checkpoints_publish_versions_and_final_summary(corpus, tmp_path, monkeypatch):
    from geofirex import kaggle_persistence as persistence
    from test_cloud_corpus import FakeKaggle
    backend = FakeKaggle(tmp_path/"fake-kaggle-datasets")
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: True if p.as_posix() == "/kaggle/input" else original_is_dir(p))
    monkeypatch.setattr(persistence, "KaggleBackend", lambda: backend)
    monkeypatch.setattr(persistence, "session_id", lambda: "synthetic-notebook-session")
    c = {**corpus, "checkpoint_dataset": "tester/geofirex-epoch-test",
         "checkpoint_upload_budget_bytes": 100*1024**2, "checkpoint_private_free_bytes": 100*1024**2}
    out = train(c)
    assert backend.uploads == 3  # each of two epochs + final closed-run snapshot
    final_receipt = json.loads((out/"backup_0002_final.json").read_text())
    assert final_receipt["status"] == "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
    assert (backend.versions[3]/"run/summary.json").is_file()
    state = torch.load(backend.versions[3]/"run/epoch_0002.pt", weights_only=False)
    assert state["epoch"] == 2 and state["optimizer"]


def test_gpu_campaign_generates_ten_comparable_configs_and_refuses_tiny_default(corpus, tmp_path):
    import shutil
    from scripts.kaggle_campaign import configure
    from geofirex.corpus import json_new
    prepared = tmp_path/"campaign-prepared"
    prepared.mkdir()
    for key in ("manifest", "split"):
        shutil.copy2(corpus[key], prepared/f"{key}.csv")
    json_new(prepared/"mounts.json", [corpus["data_root"]])
    json_new(prepared/"completeness.json", {"complete_corpus": False, "actual_events": 4,
        "actual_observations": 32, "by_split": {"train": {"events": 2}, "val": {"events": 1}}})
    with pytest.raises(ValueError, match="substantial"):
        configure(prepared, corpus["index"], tmp_path/"tiny-refused", "tester/runs", 10**8, 10**8)
    configs = configure(prepared, corpus["index"], tmp_path/"campaign-configs", "tester/runs", 10**8, 10**8,
                        min_train_events=2, min_val_events=1)
    assert len(configs) == 10
    assert sum(c["model"] == "B0" for c in configs) == 1
    for item in configs:
        config = load_config(item["config"])
        assert config["context"] == 3 and config["epochs"] == 30
        assert config["positive_tile_fraction"] == .5
        assert config["checkpoint_dataset"].startswith("tester/runs-")
