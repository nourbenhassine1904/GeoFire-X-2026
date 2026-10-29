"""Build an enriched manifest for the REAL WildfireSpreadTS sample.

Deliberately separate from geofirex.manifest (which stays fast and
format-agnostic for the synthetic pipeline): this module reads each GeoTIFF's
header with rasterio to record real channel count, shape and dtype, because
Phase F of the dataset audit requires that metadata and we must not assume it
matches the synthetic fixtures or the paper's post-preprocessing figures.

Fields we cannot determine from the files themselves (e.g. a human-readable
fire name, precise ignition coordinates) are left empty rather than guessed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import rasterio

from geofirex.io import list_event_days, list_events


def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` looking for pyproject.toml (the project root
    marker). Falls back to `start` itself (resolved) if none is found, so
    behavior stays predictable even when called outside this repository."""
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def build_real_manifest(data_dir: Path, root: Path | None = None) -> list[dict]:
    """One row per real day file, with metadata read directly from the GeoTIFF.

    `relative_path` is stored relative to `root` - an explicit, documented base
    (default: this repository's root, located by walking up from the current
    working directory until a pyproject.toml is found) - never implicitly
    relative to whatever the current working directory happens to be when the
    function is called. If `data_dir` lies outside `root`, an absolute POSIX
    path is stored instead of a misleading relative one.
    """
    resolved_root = Path(root).resolve() if root is not None else find_repo_root(Path.cwd())
    rows = []
    for event_dir in list_events(data_dir):
        year = event_dir.parent.name
        fire_id = event_dir.name  # e.g. "fire_21748813" - official numeric id, not a descriptive name
        event_id = f"{year}/{fire_id}"
        days = list_event_days(event_dir)
        dates_sorted = [d.date for d in days]

        prev_date = None
        for day in days:
            gap_days = None
            try:
                cur = dt.date.fromisoformat(day.date)
                if prev_date is not None:
                    gap_days = (cur - prev_date).days
                prev_date = cur
            except ValueError:
                pass  # filename not a valid ISO date - leave gap_days unset

            with rasterio.open(day.path) as ds:
                num_channels = ds.count
                height, width = ds.height, ds.width
                dtype = ds.dtypes[0]
                crs = str(ds.crs) if ds.crs else ""
                band_names = list(ds.descriptions) if ds.descriptions else []

            abs_path = day.path.resolve()
            try:
                stored_path = abs_path.relative_to(resolved_root).as_posix()
            except ValueError:
                # data_dir is outside root - an absolute path is more honest
                # than a relative one computed against the wrong base.
                stored_path = abs_path.as_posix()

            rows.append(
                {
                    "event_id": event_id,
                    "year": year,
                    "fire_id": fire_id,
                    "date": day.date,
                    "relative_path": stored_path,
                    "observation_type": "daily_geotiff",
                    "num_channels": num_channels,
                    "height": height,
                    "width": width,
                    "dtype": dtype,
                    "crs": crs,
                    "band_names": ";".join(band_names) if band_names else "",
                    "gap_days_since_previous_observation": gap_days if gap_days is not None else "",
                    # Not determinable from the files themselves - documented as absent,
                    # not guessed (Phase F instruction: "ne remplis pas artificiellement").
                    "fire_display_name": "",
                    "ignition_coordinates": "",
                }
            )
    return rows


FIELDNAMES = [
    "event_id", "year", "fire_id", "date", "relative_path", "observation_type",
    "num_channels", "height", "width", "dtype", "crs", "band_names",
    "gap_days_since_previous_observation", "fire_display_name", "ignition_coordinates",
]


def save_real_manifest(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Base directory that relative_path values are resolved against. "
            "Defaults to this repository's root (located by walking up from "
            "the current working directory until a pyproject.toml is found)."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root) if args.root else find_repo_root(Path.cwd())
    print(f"Paths will be stored relative to: {root}")

    rows = build_real_manifest(Path(args.data_dir), root=root)
    if not rows:
        raise SystemExit(f"No real GeoTIFF files found under {args.data_dir}.")
    save_real_manifest(rows, Path(args.out))
    n_events = len({r["event_id"] for r in rows})
    print(f"Real manifest: {len(rows)} day-rows across {n_events} event(s) -> {args.out}")


if __name__ == "__main__":
    main()
