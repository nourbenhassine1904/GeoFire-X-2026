"""Inspect metadata only (explicit --refresh-remote), or plan offline acquisition."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from geofirex.corpus import BudgetRangeFile, index_zip, json_new, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def plan(index_path, manifest, root):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    with Path(manifest).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    pixels = [int(r["height"]) * int(r["width"]) for r in rows]
    n = index["observations"]
    low, high, mean = min(pixels)*n, max(pixels)*n, sum(pixels)/len(pixels)*n
    return {"index_sha256": sha256_file(index_path),
            "events": index["events"], "observations": n,
            "events_per_year": dict(Counter(e["year"] for e in
                {e["event_id"]: e for e in index["entries"]}.values())),
            "observations_per_year": dict(Counter(e["year"] for e in index["entries"])),
            "archive_bytes_exact": index["archive_bytes"],
            "tif_payload_compressed_bytes_exact": index["compressed_tif_bytes"],
            "extracted_geotiff_bytes_exact": index["extracted_geotiff_bytes"],
            "disk_free_bytes_measured": shutil.disk_usage(root).free,
            "decoded_float32_23_bands_estimate": {"low": low*92, "sample_mean": mean*92, "high": high*92},
            "compact_memmap_3_float32_fire_lc_estimate": {"low": low*14, "sample_mean": mean*14, "high": high*14},
            "decoded_float32_40_channels_estimate": {"low": low*160, "sample_mean": mean*160, "high": high*160},
            "estimate_warning": "Pixel estimates extrapolate 108 local scenes, not full-corpus raster headers; not guaranteed bounds.",
            "sample_pixels_min": min(pixels), "sample_pixels_max": max(pixels),
            "source_sample_scenes": len(rows),
            "transfer_time_hours_at_Mbps": {str(rate): index["compressed_tif_bytes"]*8/(rate*1e6)/3600 for rate in (10, 50, 100)},
            "scope": "ZIP central directory and local sample only; no full-corpus pixels processed"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--refresh-remote", action="store_true")
    p.add_argument("--manifest", default="data/manifests/real/manifest_wildfirespreadts.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.refresh_remote:
        stream = BudgetRangeFile(8 * 1024 * 1024)
        with zipfile.ZipFile(stream) as zf:
            index = index_zip(zf, archive_size=stream.size, remote=True)
        index["metadata_transfer"] = {"successful_body_bytes": stream.bytes_fetched,
                                       "reserved_bytes_including_retries": stream.bytes_reserved,
                                       "requests": stream.requests, "session_limit_bytes": stream.budget_bytes}
        json_new(args.index, index)
        print("Metadata only:", index["metadata_transfer"], flush=True)
    result = plan(args.index, args.manifest, ROOT)
    json_new(args.out, result)
    print(json.dumps(result, indent=2))
