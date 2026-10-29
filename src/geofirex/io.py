"""Read WildfireSpreadTS-layout GeoTIFF events: <data_dir>/<year>/<fire_name>/<date>.tif."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


@dataclass(frozen=True)
class EventDay:
    date: str  # ISO date, taken from the filename stem
    path: Path


def list_events(data_dir: Path) -> list[Path]:
    """Return every event directory (<data_dir>/<year>/<fire_name>) that holds .tif files."""
    data_dir = Path(data_dir)
    events = []
    for year_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for event_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            if any(event_dir.glob("*.tif")):
                events.append(event_dir)
    return events


def list_event_days(event_dir: Path) -> list[EventDay]:
    """Return the day files of one event, sorted chronologically by filename."""
    paths = sorted(Path(event_dir).glob("*.tif"))
    return [EventDay(date=p.stem, path=p) for p in paths]


def read_geotiff(path: Path) -> np.ndarray:
    """Read one day's GeoTIFF as a [C, H, W] float32 array."""
    with rasterio.open(path) as ds:
        return ds.read().astype("float32")


def read_event(event_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Read an entire event as a [T, C, H, W] array plus the list of ISO dates."""
    days = list_event_days(event_dir)
    if not days:
        raise FileNotFoundError(f"No .tif files found under {event_dir}")
    arrays = [read_geotiff(d.path) for d in days]
    return np.stack(arrays, axis=0), [d.date for d in days]
