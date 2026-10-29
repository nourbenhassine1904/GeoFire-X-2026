"""Persistence baseline: predict tomorrow's active-fire mask as today's observed
mask. The simplest non-trivial baseline every later model must beat.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from geofirex.constants import ACTIVE_FIRE_CHANNEL
from geofirex.dataset import GeoFireXDataset
from geofirex.metrics import all_metrics


def persistence_predict(x: torch.Tensor) -> torch.Tensor:
    """x: [B, T, C, H, W] -> prediction: [B, 1, H, W], the last context day's
    active-fire channel."""
    return x[:, -1, ACTIVE_FIRE_CHANNEL : ACTIVE_FIRE_CHANNEL + 1]


def evaluate_persistence(dataset: GeoFireXDataset, batch_size: int = 4) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    n_batches = 0
    for x, y in loader:
        pred = persistence_predict(x)
        batch_metrics = all_metrics(pred, y)
        for k, v in batch_metrics.items():
            totals[k] += v
        n_batches += 1
    if n_batches == 0:
        raise ValueError("Dataset produced zero samples - check split/context_days.")
    return {k: v / n_batches for k, v in totals.items()}
