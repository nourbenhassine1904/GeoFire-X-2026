"""PyTorch Dataset reading manifest + event split, producing the contract tensors
from docs/PROJECT_CONTEXT.md: X: [T, C, H, W] context days, Y: [1, H, W] next-day
binary active-fire mask.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from geofirex.constants import ACTIVE_FIRE_CHANNEL
from geofirex.io import read_geotiff
from geofirex.manifest import load_manifest
from geofirex.splits import load_split


class GeoFireXDataset(Dataset):
    """One sample = `context_days` consecutive days (X) + the following day's
    active-fire mask (Y), drawn only from events assigned to `split_name`.
    """

    def __init__(
        self,
        manifest_path: Path,
        split_path: Path,
        split_name: str,
        context_days: int = 3,
    ) -> None:
        if context_days < 1:
            raise ValueError("context_days must be >= 1")
        self.context_days = context_days

        manifest = load_manifest(manifest_path)
        event_split = load_split(split_path)

        by_event: dict[str, list[dict]] = defaultdict(list)
        for row in manifest:
            by_event[row["event_id"]].append(row)

        self.samples: list[tuple[list[str], str]] = []
        for event_id, rows in by_event.items():
            if event_split.get(event_id) != split_name:
                continue
            rows_sorted = sorted(rows, key=lambda r: r["date"])
            paths = [r["path"] for r in rows_sorted]
            n = len(paths)
            for i in range(n - context_days):
                x_paths = paths[i : i + context_days]
                y_path = paths[i + context_days]
                self.samples.append((x_paths, y_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_paths, y_path = self.samples[idx]
        x = np.stack([read_geotiff(p) for p in x_paths], axis=0)  # [T, C, H, W]
        y_full = read_geotiff(y_path)  # [C, H, W]
        y = y_full[ACTIVE_FIRE_CHANNEL : ACTIVE_FIRE_CHANNEL + 1]  # [1, H, W]
        return torch.from_numpy(x), torch.from_numpy(y)
