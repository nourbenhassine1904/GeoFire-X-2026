"""CPU contract checks with tiny generated rasters; no scientific GPU results."""
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pytest
import rasterio
import torch

from test_runner import corpus, threads
from geofirex.runner import train
from geofirex.kaggle_persistence import snapshot_training


def test_epoch_boundary_stop_resumes_with_unchanged_total_budget(corpus):
    config = {**corpus, "epochs": 3}
    full = train(config)
    partial = train(config, stop_after_epochs=1)
    summary = json.loads((partial/"summary.json").read_text())
    assert summary["status"] == "PAUSED_AT_EPOCH_BOUNDARY" and not summary["early_stopped"]
    assert summary["optimizer_updates"] == 2
    resumed = train(config, partial/"epoch_0001.pt")
    a = torch.load(full/"epoch_0003.pt", weights_only=False)
    b = torch.load(resumed/"epoch_0003.pt", weights_only=False)
    assert a["identity"] == b["identity"]
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])


def test_spatial_cnn_v2_is_trained_and_records_all_epoch_metrics(corpus):
    config = {**corpus, "model_kwargs": {"architecture": "spatial_cnn_v2", "width": 4}}
    out = train(config)
    summary = json.loads((out/"summary.json").read_text())
    assert summary["optimizer_updates"] == 4 and summary["train_patches"] == 8
    row = summary["history"][0]
    assert all("validation_"+key in row for key in ("average_precision", "dice", "iou", "precision", "recall", "threshold"))
    assert row["gpu_compute_seconds"] is None and row["gpu_peak_allocated_bytes"] is None
    assert row["train_patches_per_second"] > 0


def test_throughput_probe_reads_only_training_rasters(corpus, monkeypatch):
    from scripts.measure_gpu_throughput import measure
    original = rasterio.open
    def guard(path, *args, **kwargs):
        assert not any(year in str(path).replace("\\", "/") for year in ("2020/fire_", "2021/fire_"))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(rasterio, "open", guard)
    result = measure(corpus, steps=3, warmup=1)
    assert result["optimizer_updates"] == 3 and result["patches"] == 6
    assert result["device"] == "cpu" and result["gpu"] is None
    assert result["train_events"] == 2 and result["peak_allocated_bytes"] is None


async def request(app, path, query=None):
    sent = []
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
             "query_string": urlencode(query or {}).encode(), "headers": [],
             "client": ("127.0.0.1", 10000), "server": ("127.0.0.1", 8000), "root_path": ""}
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        sent.append(message)
    await app(scope, receive, send)
    return sent[0]["status"], json.loads(b"".join(m.get("body", b"") for m in sent))


def test_verified_serving_real_model_no_future_pixels_and_test_lock(corpus, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from geofirex.serving import create_app, Predictor
    out = train(corpus)
    package = tmp_path/"serve-package"
    inventory = snapshot_training(out, package)
    app = create_app(package, [corpus["data_root"]], inventory["identity"])
    status, catalog = asyncio.run(request(app, "/events"))
    assert status == 200 and all(not event["event_id"].startswith("2021/") for event in catalog)
    status, _ = asyncio.run(request(app, "/predict", {"event_id": "2021/fire_1", "as_of": "2021-01-05"}))
    assert status == 403
    original = rasterio.open
    def guard(path, *args, **kwargs):
        assert not str(path).endswith(("2018-01-06.tif", "2018-01-07.tif", "2018-01-08.tif")), "Future raster opened"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(rasterio, "open", guard)
    status, prediction = asyncio.run(request(app, "/predict", {"event_id": "2018/fire_1", "as_of": "2018-01-05"}))
    assert status == 200 and prediction["target_date"] == "2018-01-06"
    probabilities = np.array(prediction["probability"], dtype=float)
    assert np.isfinite(probabilities[1:]).all()
    assert ((probabilities[1:] > 0) & (probabilities[1:] < 1)).all()
    assert prediction["optimizer_updates"] == 4
    assert prediction["data_origin"] == "custom_or_test_data"
    assert prediction['split'] == 'train'
    monkeypatch.setattr(rasterio, "open", original)
    status, truth = asyncio.run(request(app, "/ground-truth", {"event_id": "2018/fire_1", "as_of": "2018-01-05"}))
    assert status == 200 and truth["target_date"] == prediction["target_date"]
    assert truth["preview_stride"] == prediction["preview_stride"]
    assert truth["transform"] == prediction["transform"]
    assert np.array(truth["ground_truth"], dtype=object)[0].tolist() == [None]*15
    status, _ = asyncio.run(request(app, "/ground-truth", {"event_id": "2021/fire_1", "as_of": "2021-01-05"}))
    assert status == 403
    with pytest.raises(ValueError, match="trusted"):
        Predictor(package, [corpus["data_root"]], "wrong-identity")
    checkpoint = package/"run"/json.loads((out/"summary.json").read_text())["checkpoint"]
    checkpoint.write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="SHA256"):
        Predictor(package, [corpus["data_root"]], inventory["identity"])


def test_pilot_plans_exclude_manual_batch_and_have_representative_scale(tmp_path):
    from scripts.prepare_kaggle_pilot import prepare
    index = json.loads((Path(__file__).resolve().parents[1]/"data/manifests/real/corpus_zip_index_v1.json").read_text())
    result = prepare(index, tmp_path/"pilot", private_free_bytes=12*1024**3, storage_bytes=8*1024**3)
    assert result["plans"][0]["eligible_events_by_dates"] >= 30
    assert result["plans"][1]["eligible_events_by_dates"] >= 20
    assert result["plans"][0]["staging_plus_reserve_bytes"] <= 8*1024**3
    first = json.loads((tmp_path/"pilot/train2018.json").read_text())
    assert "2018/fire_21458798" not in first["events"]


def test_operator_notebooks_compile_and_disable_expensive_cells_by_default():
    root = Path(__file__).resolve().parents[1]
    for name in ("kaggle_cpu_unblock_v3.ipynb", "kaggle_gpu_pilot_v3.ipynb"):
        notebook = json.loads((root/"notebooks"/name).read_text())
        assert notebook["nbformat"] == 4
        assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(cell["source"], f"{name}:cell{index}", "exec")
                assert cell["outputs"] == [] and cell["execution_count"] is None
        source = "\n".join(cell["source"] for cell in notebook["cells"])
        assert "RUN_" in source and "= False" in source


def test_application_rejects_untrained_persistence_snapshot(corpus, tmp_path):
    pytest.importorskip("fastapi")
    from geofirex.serving import Predictor
    out = train({**corpus, "model": "B0"})
    package = tmp_path/"b0-package"
    inventory = snapshot_training(out, package)
    with pytest.raises(ValueError, match="actually trained"):
        Predictor(package, [corpus["data_root"]], inventory["identity"])
