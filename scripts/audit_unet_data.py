"""Audit all local rasters without changing them or the historical manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.unet_data import load_event_windows, read_raw

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(manifest, split, root=ROOT):
    rows, mapping, windows, rejections = load_event_windows(manifest, split, root)
    missing = np.zeros(23, dtype=np.int64)
    infinite = np.zeros(23, dtype=np.int64)
    fractional_lc = unknown_lc = pixels = 0
    lc_values, shapes, files, targets = Counter(), set(), [], {}
    forecast_rain_names = Counter()
    event_grids = {}
    for row in rows:
        path = row["relative_path"]
        raw = read_raw(path)
        with rasterio.open(path) as ds:
            forecast_rain_names[ds.descriptions[17]] += 1
            grid = (ds.shape, str(ds.crs), tuple(ds.transform))
            if row["event_id"] in event_grids and event_grids[row["event_id"]] != grid:
                raise ValueError("Event raster grids differ")
            event_grids[row["event_id"]] = grid
            if ds.shape != (int(row["height"]), int(row["width"])):
                raise ValueError("Manifest dimensions do not match raster")
        shapes.add(str(raw.shape))
        missing += np.isnan(raw).sum(axis=(1, 2))
        infinite += np.isinf(raw).sum(axis=(1, 2))
        pixels += raw.shape[1] * raw.shape[2]
        lc = raw[16]
        finite = np.isfinite(lc)
        fractional_lc += int((finite & (lc != np.floor(lc))).sum())
        unknown_lc += int((~finite | (lc < 1) | (lc > 17) | (lc != np.floor(lc))).sum())
        values, counts = np.unique(lc[finite], return_counts=True)
        lc_values.update({str(float(v)): int(n) for v, n in zip(values, counts)})
        targets[path] = int((~np.isnan(raw[22])).sum())
        files.append({"path": str(Path(path).relative_to(root)), "sha256": sha256(path),
                      "event_id": row["event_id"], "split": mapping[row["event_id"]],
                      "shape": list(raw.shape), "fire_pixels": targets[path]})
    split_report = {}
    for role, ws in windows.items():
        target_pixels = sum(int(r["height"]) * int(r["width"]) for r in rows
                            if r["relative_path"] in {w.target_path for w in ws})
        positive = sum(targets[w.target_path] for w in ws)
        split_report[role] = {"events": sorted(e for e in mapping if mapping[e] == role),
                              "windows": len(ws), "target_pixels": target_pixels,
                              "target_positive_pixels": positive,
                              "target_positive_fraction": positive / target_pixels,
                              "empty_target_windows": sum(targets[w.target_path] == 0 for w in ws)}
    return {"files": files, "num_files": len(files), "num_events": len(mapping),
            "shapes": sorted(shapes), "total_pixels": pixels,
            "per_band": [{"name": name, "nan_pixels": int(missing[i]),
                          "nan_fraction": float(missing[i] / pixels),
                          "inf_pixels": int(infinite[i])} for i, name in enumerate(RAW_BAND_NAMES)],
            "lc_finite_values": dict(lc_values), "lc_fractional_pixels": fractional_lc,
            "lc_unknown_pixels": unknown_lc, "splits": split_report,
            "forecast_rain_band_name_counts": dict(forecast_rain_names),
            "window_rejections": rejections, "manifest_sha256": sha256(manifest),
            "split_sha256": sha256(split), "qualification": "technical 5-event sample; no generalization claim"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifests/real/manifest_wildfirespreadts.csv")
    p.add_argument("--split", default="splits/real/smoke_test_split.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = audit(ROOT / args.manifest, ROOT / args.split)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("num_files", "num_events", "splits", "lc_fractional_pixels")}, indent=2))
