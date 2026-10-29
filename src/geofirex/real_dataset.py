"""PyTorch Dataset for the REAL WildfireSpreadTS sample (23 raw bands).

Contract (matches docs/PROJECT_CONTEXT.md): X: [T, 23, H, W] context days,
Y: [1, H, W] binary active-fire mask for the day AFTER the context window
(never a day inside it - see windows.build_windows for the consecutive-date
guarantee that makes this safe).

NaN handling (explicit, per channel):
  - "active fire" (band 22, every context day AND the target): the raw value
    is a non-binary VIIRS detection code where NaN means "no detection"
    (datasheet §5.3). Every reading of this channel is binarized via
    `~isnan`, never a magnitude threshold - this never introduces NaN into
    the output tensors for this channel.
  - All other 22 channels: real clouds/sensor gaps leave some pixels NaN
    (typically 1-8%, see reports/DATA_AUDIT.md). These are replaced with 0.0,
    a documented, simple default - not fabricated data.

Geographic metadata (CRS, per-day paths, dates) is never discarded: it is not
part of the batched tensors (which must stay homogeneous for the default
PyTorch collate_fn), but every sample's provenance is preserved and queryable
via `dataset.windows[idx]` and `dataset.crs_for(idx)`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from geofirex.real_constants import RAW_ACTIVE_FIRE_CHANNEL, RAW_NUM_CHANNELS
from geofirex.windows import Window, build_windows, format_window_report


def _load_manifest(manifest_path: Path) -> list[dict]:
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_split(split_path: Path) -> dict[str, str]:
    with Path(split_path).open("r", newline="", encoding="utf-8") as f:
        return {row["event_id"]: row["split"] for row in csv.DictReader(f)}


def read_real_channels(path: Path) -> np.ndarray:
    """Read one day's raw GeoTIFF as [23, H, W] float32, with the active-fire
    channel (last) already binarized to {0.0, 1.0} via presence-of-value, and
    every other channel's NaNs replaced with 0.0."""
    with rasterio.open(path) as ds:
        if ds.count != RAW_NUM_CHANNELS:
            raise ValueError(
                f"{path}: expected {RAW_NUM_CHANNELS} raw bands, found {ds.count} - "
                "refusing to silently misinterpret the channel layout."
            )
        arr = ds.read().astype("float32")

    active_fire = arr[RAW_ACTIVE_FIRE_CHANNEL]
    binarized = (~np.isnan(active_fire)).astype("float32")

    other = arr.copy()
    other[np.isnan(other)] = 0.0
    other[RAW_ACTIVE_FIRE_CHANNEL] = binarized
    return other


def active_fire_target(path: Path) -> np.ndarray:
    """Read only the binarized active-fire mask [1, H, W] of one day's file."""
    with rasterio.open(path) as ds:
        names = list(ds.descriptions) if ds.descriptions else []
        if "active fire" not in names:
            raise ValueError(f"{path}: no 'active fire' band found in descriptions")
        band = ds.read(names.index("active fire") + 1).astype("float32")
    mask = (~np.isnan(band)).astype("float32")
    return mask[None, :, :]


class RealGeoFireXDataset(Dataset):
    """One sample = `context_days` consecutive real days (X: [T,23,H,W]) and
    the immediately following day's binarized active-fire mask (Y: [1,H,W]),
    restricted to events assigned to `split_name`.
    """

    def __init__(
        self,
        manifest_path: Path,
        split_path: Path,
        split_name: str,
        context_days: int = 1,
    ) -> None:
        manifest_rows = _load_manifest(manifest_path)
        event_split = _load_split(split_path)

        selected_rows = [r for r in manifest_rows if event_split.get(r["event_id"]) == split_name]
        self.windows, self.rejections = build_windows(selected_rows, context_days)
        self.context_days = context_days

    def __len__(self) -> int:
        return len(self.windows)

    def window_report(self) -> str:
        return format_window_report(self.windows, self.rejections, self.context_days)

    def crs_for(self, idx: int) -> Window:
        """Return the Window metadata (event_id, dates, paths) for a sample -
        geographic/provenance info that does not fit in the batched tensors."""
        return self.windows[idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.windows[idx]
        x = np.stack([read_real_channels(p) for p in w.context_paths], axis=0)  # [T,23,H,W]
        y = active_fire_target(w.target_path)  # [1,H,W]
        return torch.from_numpy(x), torch.from_numpy(y)
