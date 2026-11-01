"""Local FastAPI preparation: verified trained checkpoint -> real J+1 inference.

Only server-configured trusted snapshots are loaded; no checkpoint uploads.
Inference reads observations through J only. Test 2021 is excluded by default.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import threading
from pathlib import Path

import numpy as np
import rasterio
import torch

from geofirex.corpus import sha256_file, verify_member
from geofirex.kaggle_persistence import verify_package
from geofirex.runner import ROOT, choose_device, runner_model
from geofirex.runner_data import RasterReader
from geofirex.unet_data import Normalizer, padded_patch, tile_coordinates, validate_layout


class Predictor:
    def __init__(self, package, data_roots, expected_inventory_identity, *, device="cpu", allow_test=False):
        package = Path(package)
        inventory = json.loads((package/"inventory.json").read_text(encoding="utf-8"))
        if not expected_inventory_identity or inventory["identity"] != expected_inventory_identity:
            raise ValueError("Snapshot differs from the trusted publication inventory identity")
        verify_package(package, inventory)
        if inventory["kind"] != "training":
            raise ValueError("Expected a training snapshot")
        run = package/"run"
        self.config = json.loads((run/"config.json").read_text(encoding="utf-8"))
        self.summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
        if self.config["model"] == "B0" or self.summary["optimizer_updates"] < 1:
            raise ValueError("Application inference requires an actually trained B1/B2/B3 checkpoint")
        source = json.loads((run/"source_hashes.json").read_text(encoding="utf-8"))
        source_names = ["benchmark_models.py", "unet.py", "unet_data.py", "runner_data.py", "feature_profiles.py"]
        if self.config.get("model_kwargs", {}).get("upsampling") == "nearest_v1":
            source_names += ["runner.py", "deterministic_models.py"]
        for name in source_names:
            key = "src/geofirex/"+name
            if source.get(key) != sha256_file(ROOT/key):
                raise ValueError("Model/preprocessing source differs from the trained snapshot")
        checkpoint = run/self.summary["checkpoint"]
        if sha256_file(checkpoint) != self.summary["checkpoint_sha256"]:
            raise ValueError("Best checkpoint checksum mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["identity"] != self.summary["identity"] or state["identity"] != inventory.get("run_identity"):
            raise ValueError("Training checkpoint identity mismatch")
        self.normalizer = Normalizer(**json.loads((run/"normalization.json").read_text(encoding="utf-8")))
        self.device = choose_device(device)
        if self.device == "cpu":
            torch.set_num_threads(self.config["cpu_threads"])
        self.model = runner_model(self.config, self.normalizer).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.roots = [Path(root).resolve() for root in data_roots]
        self.allow_test = allow_test
        self.lock = threading.Lock()
        with (run/"manifest.csv").open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        with (run/'split.csv').open(encoding='utf-8', newline='') as f:
            self.splits = {r['event_id']: r['split'] for r in csv.DictReader(f)}
        self.audit = json.loads((run/"data_audit.json").read_text(encoding="utf-8"))["raster_fingerprints"]
        self.inventory_identity = inventory["identity"]
        from geofirex.cloud_corpus import validate_index
        index = json.loads((run/"index.json").read_text(encoding="utf-8"))
        try:
            validate_index(index, official=True)
            self.official_entries = {r["event_id"]+"|"+r["date"]: r for r in index["entries"]}
        except (KeyError, ValueError):
            self.official_entries = {}
        self.data_origin = "official_wildfirespreadts_verified_bytes" if self.official_entries else "custom_or_test_data"

    def events(self):
        groups = {}
        for row in self.rows:
            event = row["event_id"]
            if event.startswith("2021/") and not self.allow_test:
                continue
            groups.setdefault(event, []).append(row["date"])
        return [{"event_id": event, "split": self.splits.get(event), "dates": sorted(dates), "observations": len(dates),
                 "forecast_dates": [day for day in sorted(dates)
                    if all((dt.date.fromisoformat(day)+dt.timedelta(days=i)).isoformat() in dates
                           for i in (-4, -3, -2, -1, 1))]}
                for event, dates in sorted(groups.items())]

    def _path(self, row):
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Serving requires a portable prepared manifest")
        candidates = {(root/relative).resolve() for root in self.roots if (root/relative).is_file()
                      and (root/relative).resolve().is_relative_to(root)}
        if len(candidates) != 1:
            raise ValueError("Missing or ambiguous observation across dataset mounts")
        path = candidates.pop()
        expected = self.audit.get(row["event_id"]+"|"+row["date"])
        if not expected or sha256_file(path) != expected["sha256"]:
            raise ValueError("Input observation differs from training data audit")
        if self.official_entries and not verify_member(path, self.official_entries[row["event_id"]+"|"+row["date"]]):
            raise ValueError("Input differs from the official archive size/CRC")
        with rasterio.open(path) as ds:
            sidecars = {Path(file).name: sha256_file(file) for file in ds.files if Path(file).resolve() != path}
        if sidecars != expected.get("sidecars", {}):
            raise ValueError("Input GDAL sidecar/mask differs from training data audit")
        return str(path)

    def ground_truth(self, event_id, as_of, *, max_side=256):
        """Historical J+1 display only. Never called by predict or model selection."""
        if event_id.startswith("2021/") and not self.allow_test:
            raise PermissionError("Test 2021 is locked")
        if not 16 <= max_side <= 512:
            raise ValueError("Invalid preview size")
        target_date = (dt.date.fromisoformat(as_of)+dt.timedelta(days=1)).isoformat()
        row = next((r for r in self.rows if r["event_id"] == event_id and r["date"] == target_date), None)
        if row is None:
            raise KeyError("Historical J+1 observation unavailable")
        path = self._path(row)
        with rasterio.open(path) as ds:
            validate_layout(ds)
            if ds.height*ds.width > 20_000_000:
                raise ValueError("Event exceeds serving limits")
            # Same top-left sampling grid as prediction; respect GDAL validity.
            from geofirex.runner_data import read_tile
            stride = max(1, math.ceil(max(ds.height, ds.width)/max_side))
            full_fire = np.zeros((ds.height, ds.width), dtype=np.uint8)
            full_valid = np.zeros((ds.height, ds.width), dtype=bool)
            for top, left in tile_coordinates(ds.height, ds.width, 256):
                raw, valid = read_tile(path, top, left, 256)
                hh, ww = valid.shape[-2:]
                full_fire[top:top+hh, left:left+ww] = np.isfinite(raw[22])
                full_valid[top:top+hh, left:left+ww] = valid[0].astype(bool)
            mask = full_valid[::stride, ::stride]
            fire = full_fire[::stride, ::stride].astype(object)
            grid = np.where(mask, fire, None).tolist()
            return {"event_id": event_id, "target_date": target_date, "ground_truth": grid,
                    "full_shape": [ds.height, ds.width], "preview_stride": stride,
                    "crs": str(ds.crs), "transform": list(ds.transform),
                    "source_sha256": self.audit[event_id+"|"+target_date]["sha256"],
                    "purpose": "historical_display_only; excluded from inference inputs"}

    @torch.inference_mode()
    def predict(self, event_id, as_of, *, max_side=256):
        if event_id.startswith("2021/") and not self.allow_test:
            raise PermissionError("Test 2021 is locked")
        rows = {r["date"]: r for r in self.rows if r["event_id"] == event_id}
        if not rows:
            raise KeyError("Unknown event")
        day = dt.date.fromisoformat(as_of)
        dates = [(day-dt.timedelta(days=i)).isoformat() for i in range(4, -1, -1)]
        if any(date not in rows for date in dates):
            raise ValueError("Five consecutive observations through J are required for the shared validity domain")
        paths = [self._path(rows[date]) for date in dates]
        layout = None
        for path in paths:
            with rasterio.open(path) as ds:
                validate_layout(ds)
                current = (ds.shape, str(ds.crs), tuple(ds.transform))
                if layout is not None and current != layout:
                    raise ValueError("Input observation geometries differ")
                layout = current
        (height, width), crs, transform = layout
        if height*width > 20_000_000 or not 16 <= max_side <= 512:
            raise ValueError("Event/preview exceeds the local serving limits")
        size = self.config["patch_size"]
        reader = RasterReader(int(self.config["reader_cache_mb"]*1024**2),
                              full=self.normalizer.continuous != [12, 14] or self.normalizer.include_landcover)
        probability = np.empty((height, width), dtype=np.float32)
        observed = np.zeros((height, width), dtype=np.uint8)
        footprint = np.zeros((height, width), dtype=bool)
        # Serialize requests for the shared torch model/device.
        with self.lock:
            for row, col in tile_coordinates(height, width, size):
                data = [reader.tile(path, row, col, size) for path in paths]
                masks = np.logical_and.reduce([valid[0].astype(bool) for _, valid in data])
                encoded = [padded_patch(self.normalizer.transform(raw), 0, 0, size)
                           for raw, _ in data[-self.config["context"]:]]
                x = torch.from_numpy(np.stack(encoded)[None]).to(self.device)
                p = self.model.predict_proba(x)[0, 0].cpu().numpy()
                hh, ww = masks.shape
                probability[row:row+hh, col:col+ww] = p[:hh, :ww]
                footprint[row:row+hh, col:col+ww] = masks
                observed[row:row+hh, col:col+ww] = np.isfinite(data[-1][0][22])
        if not np.isfinite(probability).all():
            raise ValueError("Model produced nonfinite probabilities")
        stride = max(1, math.ceil(max(height, width)/max_side))
        mask = footprint[::stride, ::stride]
        preview = probability[::stride, ::stride]
        def nullable(values):
            return np.where(mask, values.astype(object), None).tolist()
        return {"event_id": event_id, "as_of": as_of, "target_date": (day+dt.timedelta(days=1)).isoformat(),
                "model": self.config["model"], "qualification": self.config["qualification"], "data_origin": self.data_origin,
                "split": self.splits.get(event_id),
                "training_status": self.summary["status"], "optimizer_updates": self.summary["optimizer_updates"],
                "checkpoint_sha256": self.summary["checkpoint_sha256"], "inventory_identity": self.inventory_identity,
                "threshold": self.summary["threshold"], "full_shape": [height, width], "preview_stride": stride,
                "crs": crs, "transform": list(transform), "input_dates": dates[-self.config["context"]:],
                "probability": nullable(preview), "observed_fire": nullable(observed[::stride, ::stride]),
                "predicted_fire": nullable((preview >= self.summary["threshold"]).astype(np.uint8))}


def create_app(package, data_roots, expected_inventory_identity, *, device="cpu", allow_test=False):
    from fastapi import FastAPI, HTTPException, Query
    predictor = Predictor(package, data_roots, expected_inventory_identity, device=device, allow_test=allow_test)
    app = FastAPI(title="GeoFire-X J+1", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ready": True, "model": predictor.config["model"], "inventory_identity": predictor.inventory_identity}

    @app.get("/events")
    def events():
        return predictor.events()

    @app.get("/predict")
    def predict(event_id: str, as_of: str, max_side: int = Query(256, ge=16, le=512)):
        try:
            return predictor.predict(event_id, as_of, max_side=max_side)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/ground-truth")
    def ground_truth(event_id: str, as_of: str, max_side: int = Query(256, ge=16, le=512)):
        try:
            return predictor.ground_truth(event_id, as_of, max_side=max_side)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    frontend = ROOT/"frontend/dist"
    if (frontend/"index.html").is_file():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


def app_from_env():
    import os
    return create_app(os.environ["GEOFIREX_SNAPSHOT"], json.loads(os.environ["GEOFIREX_DATA_ROOTS"]),
                      os.environ["GEOFIREX_INVENTORY_IDENTITY"], device=os.environ.get("GEOFIREX_DEVICE", "cpu"))
