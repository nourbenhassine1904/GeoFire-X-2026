"""Generate synthetic GeoTIFF events matching the WildfireSpreadTS on-disk layout.

Layout mirrors the official repository: <data_dir>/<year>/<fire_name>/<date>.tif,
one GeoTIFF per day, NUM_CHANNELS bands per file. Lets the rest of the pipeline
(io, audit, manifest, splits, dataset, metrics, baseline) be built and tested
before the real 48.4 GB archive is downloaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geofirex.constants import ACTIVE_FIRE_CHANNEL, NUM_CHANNELS


def _write_day(path: Path, channels: np.ndarray) -> None:
    """Write one day's [C, H, W] array to a GeoTIFF with a dummy georeference."""
    c, h, w = channels.shape
    transform = from_origin(0.0, 0.0, 375.0, 375.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(channels.astype("float32"))


def generate_event(
    event_dir: Path,
    start_date: dt.date,
    num_days: int,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> list[Path]:
    """Create `num_days` daily GeoTIFFs for one fire event with a growing fire mask."""
    fire_mask = np.zeros((height, width), dtype="float32")
    cy, cx = rng.integers(height // 4, 3 * height // 4), rng.integers(width // 4, 3 * width // 4)
    written = []
    for t in range(num_days):
        channels = rng.normal(loc=0.0, scale=1.0, size=(NUM_CHANNELS, height, width)).astype(
            "float32"
        )
        # Grow a small fire blob over time so persistence is a plausible baseline.
        radius = 1.0 + t * 0.6
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        fire_mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2).astype("float32")
        channels[ACTIVE_FIRE_CHANNEL] = fire_mask

        day = start_date + dt.timedelta(days=t)
        path = event_dir / f"{day.isoformat()}.tif"
        _write_day(path, channels)
        written.append(path)
    return written


def generate_synthetic_dataset(
    data_dir: Path,
    num_events: int = 6,
    min_days: int = 6,
    max_days: int = 10,
    height: int = 32,
    width: int = 32,
    seed: int = 42,
) -> list[Path]:
    """Create a small synthetic dataset with the official folder/channel layout."""
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    written: list[Path] = []
    for i in range(num_events):
        year = int(rng.integers(2018, 2022))
        fire_name = f"synthetic_fire_{i:02d}"
        event_dir = data_dir / str(year) / fire_name
        num_days = int(rng.integers(min_days, max_days + 1))
        start_date = dt.date(year, 1, 1) + dt.timedelta(days=int(rng.integers(0, 300)))
        written += generate_event(event_dir, start_date, num_days, height, width, rng)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_events", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    written = generate_synthetic_dataset(
        Path(args.data_dir), num_events=args.num_events, seed=args.seed
    )
    print(f"Wrote {len(written)} synthetic daily GeoTIFFs under {args.data_dir}")


if __name__ == "__main__":
    main()
