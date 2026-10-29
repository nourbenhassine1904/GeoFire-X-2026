"""Deep audit of the extracted REAL WildfireSpreadTS sample: actual channel
count (do NOT assume the 40-channel post-preprocessing figure applies to the
raw per-day GeoTIFFs), dtype, nodata, per-channel NaN/Inf, value ranges, and
basic temporal-consistency checks per event.

This is intentionally separate from geofirex.audit (which is generic and
format-agnostic) because here we specifically need to VERIFY assumptions
documented in docs/DATASET_VERSION.md against the real files, and report any
discrepancy instead of forcing the data to match the paper.

Usage:
    python scripts/audit_real_sample.py --data_dir data/raw/WildfireSpreadTS/real_sample \\
        --out reports/real_sample_audit.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import rasterio

from geofirex.io import list_event_days, list_events


def audit_file(path: Path) -> dict:
    with rasterio.open(path) as ds:
        arr = ds.read()
        tags = ds.tags()
        band_descriptions = ds.descriptions  # None per-band unless the file sets them
        dtype = str(ds.dtypes[0])
        nodata = ds.nodata
        crs = str(ds.crs) if ds.crs else None
        transform = tuple(ds.transform)[:6]
        height, width = ds.height, ds.width
        count = ds.count

    nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    per_band_stats = []
    for b in range(arr.shape[0]):
        band = arr[b]
        finite = band[np.isfinite(band)] if np.issubdtype(band.dtype, np.floating) else band
        per_band_stats.append(
            {
                "band": b + 1,
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "mean": float(finite.mean()) if finite.size else None,
                "unique_values_sample": sorted({float(x) for x in finite.flat[:5000]})[:10],
            }
        )

    return {
        "path": str(path),
        "band_count": count,
        "height": height,
        "width": width,
        "dtype": dtype,
        "nodata": nodata,
        "crs": crs,
        "transform": transform,
        "band_descriptions": list(band_descriptions) if band_descriptions else None,
        "tags": dict(tags) if tags else {},
        "nan_count": nan_count,
        "inf_count": inf_count,
        "per_band_stats": per_band_stats,
    }


def audit_event(event_dir: Path) -> dict:
    days = list_event_days(event_dir)
    file_reports = [audit_file(d.path) for d in days]
    dates = [d.date for d in days]

    gaps = []
    for prev_s, next_s in zip(dates, dates[1:]):
        try:
            prev_d = dt.date.fromisoformat(prev_s)
            next_d = dt.date.fromisoformat(next_s)
            if (next_d - prev_d).days != 1:
                gaps.append((prev_s, next_s, (next_d - prev_d).days))
        except ValueError:
            pass

    band_counts = {r["band_count"] for r in file_reports}
    shapes = {(r["height"], r["width"]) for r in file_reports}
    dtypes = {r["dtype"] for r in file_reports}

    return {
        "event_dir": str(event_dir),
        "num_days": len(days),
        "dates": dates,
        "date_gaps": gaps,
        "duplicate_dates": len(dates) != len(set(dates)),
        "band_counts_seen": sorted(band_counts),
        "shapes_seen": sorted(str(s) for s in shapes),
        "dtypes_seen": sorted(dtypes),
        "files": file_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    events = list_events(Path(args.data_dir))
    if not events:
        raise SystemExit(f"No events found under {args.data_dir} - nothing to audit.")

    reports = [audit_event(e) for e in events]

    all_band_counts = {bc for r in reports for bc in r["band_counts_seen"]}
    all_shapes = {s for r in reports for s in r["shapes_seen"]}
    all_dtypes = {dt_ for r in reports for dt_ in r["dtypes_seen"]}

    summary = {
        "num_events_audited": len(reports),
        "total_files_audited": sum(r["num_days"] for r in reports),
        "band_counts_seen_across_all_files": sorted(all_band_counts),
        "shapes_seen_across_all_files": sorted(all_shapes),
        "dtypes_seen_across_all_files": sorted(all_dtypes),
        "events": reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Audited {summary['num_events_audited']} real event(s), "
          f"{summary['total_files_audited']} real file(s)")
    print(f"Band counts actually observed in raw files: {summary['band_counts_seen_across_all_files']}")
    print(f"Shapes observed: {summary['shapes_seen_across_all_files']}")
    print(f"Dtypes observed: {summary['dtypes_seen_across_all_files']}")
    for r in reports:
        print(f"  {r['event_dir']}: {r['num_days']} days, gaps={r['date_gaps']}, "
              f"duplicate_dates={r['duplicate_dates']}")
    print(f"Full report saved to {out_path}")


if __name__ == "__main__":
    main()
