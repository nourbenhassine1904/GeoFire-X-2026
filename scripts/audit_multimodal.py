"""Bounded-memory audit of all 23 physical bands; no normalization or training."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from geofirex.corpus import json_new, sha256_file
from geofirex.feature_profiles import PROFILES
from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.unet_data import tile_coordinates, validate_layout
from geofirex.runner_data import digest


def audit(paths):
    records = []
    for path in paths:
        bands = [{"index": i, "name": name, "finite": 0, "nan": 0, "inf": 0,
                  "min": None, "max": None} for i, name in enumerate(RAW_BAND_NAMES)]
        with rasterio.open(path) as ds:
            validate_layout(ds)
            for row, col in tile_coordinates(ds.height, ds.width, 256):
                raw = ds.read(window=rasterio.windows.Window(col, row, min(256, ds.width-col), min(256, ds.height-row)))
                for values, record in zip(raw, bands):
                    finite = values[np.isfinite(values)]
                    record["finite"] += int(finite.size)
                    record["nan"] += int(np.isnan(values).sum())
                    record["inf"] += int(np.isinf(values).sum())
                    if finite.size:
                        record["min"] = min(float(finite.min()), record["min"]) if record["min"] is not None else float(finite.min())
                        record["max"] = max(float(finite.max()), record["max"]) if record["max"] is not None else float(finite.max())
        records.append({"path": str(path), "sha256": sha256_file(path), "bands": bands})
    return {"files": records, "profiles": PROFILES, "profiles_identity": digest(PROFILES),
            "availability_by_J": "NOT inferable from GeoTIFF names/values; review production time provenance",
            "excluded": {"7": "legacy wind direction quadrant ambiguity", "17-21": "forecast issue/valid times not established"},
            "active_fire": "finite detection code = fire; NaN = no detection; explicit GDAL mask handles invalid footprint"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        p.error("Audit output already exists")
    json_new(args.out, audit(args.files))
