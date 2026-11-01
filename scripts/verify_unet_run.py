"""Verify real-run provenance, unchanged raw files, trained weights and MLflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import rasterio
import torch

from audit_unet_data import ROOT, sha256
from geofirex.unet import LightUNet
from geofirex.unet_data import load_event_windows
from geofirex.unet_evaluation import score


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(run_dir, output, inference_dir=None):
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    results = read_json(run_dir / "results.json")
    config = read_json(run_dir / "config.json")
    data = read_json(run_dir / "data_audit.json")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert sha256(run_dir / "best.pt") == results["checkpoint_sha256"]
    assert sha256(ROOT / config["manifest"]) == data["manifest_sha256"]
    assert sha256(ROOT / config["split"]) == data["split_sha256"]
    for file in data["files"]:
        assert sha256(ROOT / file["path"]) == file["sha256"], file["path"]
    source_hashes = read_json(run_dir / "source_hashes.json")
    for relative, digest in source_hashes.items():
        assert sha256(run_dir / "source" / relative) == digest, relative
    assert sha256(run_dir / "source_hashes.json") == results["provenance"]["source_hash"]
    assert checkpoint["source_hash"] == results["provenance"]["source_hash"]
    _, _, windows, _ = load_event_windows(ROOT / config["manifest"], ROOT / config["split"], ROOT)
    expected_train_paths = {p for w in windows["train"] for p in w.context_paths}
    assert set(checkpoint["normalizer"]["fit_paths"]) == expected_train_paths
    torch.manual_seed(config["seed"])
    initial = LightUNet(checkpoint["in_channels"], config["base_channels"], config["depth"]).state_dict()
    changed = [name for name, tensor in checkpoint["model_state"].items()
               if not torch.equal(initial[name], tensor)]
    for group in ("encoders", "bottleneck", "decoders", "head"):
        assert any(name.startswith(group) for name in changed), group
    assert checkpoint["threshold_selected_on"] == "val"
    assert checkpoint["threshold"] == results["threshold"]
    assert checkpoint["epoch"] == results["best_epoch"]
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    client = mlflow.MlflowClient()
    run = client.get_run(results["run_id"])
    assert run.info.status == "FINISHED"
    assert client.get_experiment(run.info.experiment_id).name == config["experiment"]
    for key in ("train_loss", "val_loss"):
        points = client.get_metric_history(run.info.run_id, key)
        assert [(p.step, p.value) for p in points] == [(r["epoch"], r[key]) for r in results["history"]]
    uri = urlparse(run.info.artifact_uri)
    assert uri.scheme == "file"
    artifact_path = unquote(uri.path)
    if os.name == "nt" and artifact_path.startswith("/"):
        artifact_path = artifact_path[1:]
    artifacts = Path(artifact_path)
    for file in ("best.pt", "config.json", "normalization.json", "results.json", "source_hashes.json"):
        assert sha256(artifacts / file) == sha256(run_dir / file), file
    inference_verified = False
    probability_range = None
    if inference_dir:
        with rasterio.open(Path(inference_dir) / "probability.tif") as ds:
            p, crs, transform, tags = ds.read(), ds.crs, ds.transform, ds.tags()
        with rasterio.open(Path(inference_dir) / "mask.tif") as ds:
            assert (ds.read() == (p >= results["threshold"])).all()
            assert ds.crs == crs and ds.transform == transform
        # The integration example used J=2018-05-08, whose next-day window is fixed.
        source_date = tags["source_date"]
        candidates = [w for w in windows["test"] if w.context_dates[0] == source_date]
        assert len(candidates) == 1
        w = candidates[0]
        with rasterio.open(w.context_paths[0]) as ds:
            assert ds.crs == crs and ds.transform == transform
        with rasterio.open(w.target_path) as ds:
            y = (~torch.isnan(torch.from_numpy(ds.read(23)))).numpy()[None]
        actual = score(p, y, results["threshold"])
        expected = next(r for r in results["evaluations"]["test"]["unet"]["per_window"]
                        if r["event_id"] == w.event_id and r["target_date"] == w.target_date)
        assert actual == {k: v for k, v in expected.items() if k not in {"event_id", "target_date"}}
        inference_verified = True
        probability_range = [float(p.min()), float(p.max())]
    report = {"run_id": run.info.run_id, "status": "verified", "raw_files_unchanged": len(data["files"]),
              "source_files_verified": len(source_hashes), "changed_model_tensors": len(changed),
              "all_model_stages_updated": True, "train_only_normalization_verified": True,
              "mlflow_checkpoint_and_metrics_verified": True, "real_inference_verified": inference_verified,
              "inference_probability_range": probability_range, "checkpoint_sha256": results["checkpoint_sha256"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inference-dir")
    args = parser.parse_args()
    verify(args.run_dir, args.out, args.inference_dir)
