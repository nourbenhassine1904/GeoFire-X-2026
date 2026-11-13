"""Read-only real T=1 data, train-only statistics, categorical encoding and tiles.

Separate from the historical RealGeoFireXDataset: missingness must be preserved
until normalization, rather than zero-filled in physical units.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.windows import build_windows


def validate_layout(ds):
    names = list(ds.descriptions)
    # Real 2020 files use "last" instead of "sum" at the excluded GFS rain band.
    # This is an observed semantic variant, not permission to reorder channels.
    if len(names) == 23 and names[17] == "total_precipitation_surface_last":
        names[17] = RAW_BAND_NAMES[17]
    if ds.count != 23 or tuple(names) != RAW_BAND_NAMES:
        raise ValueError(f"Unexpected 23-band layout: {ds.name}")


def read_raw(path):
    with rasterio.open(path) as ds:
        validate_layout(ds)
        raw = ds.read().astype(np.float32)
    if np.isinf(raw[22]).any():
        raise ValueError(f"Infinite active-fire detection code: {path}")
    return raw


def fire_mask(raw):
    return (~np.isnan(raw[22:23])).astype(np.float32)


def load_event_windows(manifest, split, root):
    """Reject duplicate assignments, missing events and cross-split path reuse."""
    with Path(manifest).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with Path(split).open(encoding="utf-8", newline="") as f:
        assignments = list(csv.DictReader(f))
    mapping = {}
    for row in assignments:
        event, role = row["event_id"], row["split"]
        if event in mapping or role not in {"train", "val", "test"}:
            raise ValueError("Duplicate event or invalid split assignment")
        mapping[event] = role
    if set(mapping) != {r["event_id"] for r in rows}:
        raise ValueError("Split must assign every manifest event exactly once")
    seen_dates, seen_paths = set(), set()
    for row in rows:
        key = (row["event_id"], row["date"])
        path = (Path(root) / row["relative_path"]).resolve()
        if key in seen_dates or path in seen_paths:
            raise ValueError("Duplicate date or reused raster path in manifest")
        seen_dates.add(key)
        seen_paths.add(path)
        row["relative_path"] = str(path)
    windows, rejections = {}, {}
    for role in ("train", "val", "test"):
        selected = [r for r in rows if mapping[r["event_id"]] == role]
        windows[role], rejections[role] = build_windows(selected, context_days=1)
        if not windows[role]:
            raise ValueError(f"No consecutive J+1 windows in {role}")
    return rows, mapping, windows, rejections


@dataclass
class Normalizer:
    continuous: list[int]
    mean: list[float]
    std: list[float]
    finite_count: list[int]
    fit_paths: list[str]
    include_landcover: bool = False
    clip: float = 8.0

    @property
    def channel_names(self):
        names = [RAW_BAND_NAMES[i] + "_z" for i in self.continuous]
        names += [RAW_BAND_NAMES[i] + "_missing" for i in self.continuous]
        names += ["aspect_sin", "aspect_cos", "aspect_missing", "fire_J"]
        if self.include_landcover:
            names += [f"LC_Type1_{i}" for i in range(1, 18)] + ["LC_Type1_unknown"]
        return names

    def transform(self, raw):
        features = raw[self.continuous]
        valid = np.isfinite(features)
        mean = np.array(self.mean, dtype=np.float32)[:, None, None]
        std = np.array(self.std, dtype=np.float32)[:, None, None]
        z = (np.where(valid, features, mean) - mean) / std
        z = np.clip(z, -self.clip, self.clip)
        aspect = raw[13]
        aspect_valid = np.isfinite(aspect) & (aspect >= 0) & (aspect <= 360)
        angle = np.deg2rad(np.where(aspect_valid, aspect, 0))
        channels = [z, (~valid).astype(np.float32),
                    (np.sin(angle) * aspect_valid)[None],
                    (np.cos(angle) * aspect_valid)[None],
                    (~aspect_valid)[None].astype(np.float32), fire_mask(raw)]
        if self.include_landcover:
            lc = raw[16]
            known = np.isfinite(lc) & (lc >= 1) & (lc <= 17) & (lc == np.floor(lc))
            # Fractional resampled classes are unknown; never round them into a class.
            classes = [((lc == i) & known)[None] for i in range(1, 18)]
            channels.extend(classes + [(~known)[None]])
        result = np.concatenate(channels).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite preprocessed input")
        return result

    def to_dict(self):
        return asdict(self)


def fit_normalizer(train_windows, *, continuous=(12, 14), include_landcover=False, clip=8.0):
    """Only unique input paths of the caller's validated training split are read.

    No targets or val/test inputs contribute. Population std over finite pixels;
    all-missing channels fail loudly, constant channels use std=1.
    """
    continuous = list(continuous)
    if not continuous or len(set(continuous)) != len(continuous):
        raise ValueError("Unique continuous features required")
    # Forecast channels and LC can never silently enter as continuous inputs.
    if any(i not in set(range(13)) | {14, 15} for i in continuous):
        raise ValueError("Unsafe or noncontinuous feature requested")
    if clip <= 0:
        raise ValueError("clip must be positive")
    paths = sorted({str(p) for w in train_windows for p in w.context_paths})
    count = np.zeros(len(continuous), dtype=np.int64)
    total = np.zeros(len(continuous), dtype=np.float64)
    squares = np.zeros(len(continuous), dtype=np.float64)
    for path in paths:
        raw = read_raw(path)[continuous].astype(np.float64)
        valid = np.isfinite(raw)
        clean = np.where(valid, raw, 0)
        count += valid.sum(axis=(1, 2))
        total += clean.sum(axis=(1, 2))
        squares += np.square(clean).sum(axis=(1, 2))
    if (count == 0).any():
        raise ValueError("Training channel has no finite values")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    std[std < 1e-6] = 1
    return Normalizer(continuous, mean.tolist(), std.tolist(), count.tolist(), paths,
                      include_landcover, clip)


def tile_coordinates(height, width, size):
    if size < 1:
        raise ValueError("Positive tile size required")
    return [(r, c) for r in range(0, height, size) for c in range(0, width, size)]


def padded_patch(arr, row, col, size):
    crop = arr[..., row:row + size, col:col + size]
    h, w = crop.shape[-2:]
    return np.pad(crop, [(0, 0)] * (crop.ndim - 2) + [(0, size - h), (0, size - w)])


class WindowStore:
    def __init__(self, windows, normalizer):
        self.windows, self.normalizer = windows, normalizer

    def __len__(self):
        return len(self.windows)

    @lru_cache(maxsize=8)
    def get(self, index):
        w = self.windows[index]
        raw = read_raw(w.context_paths[0])
        # Target reads ONLY the next day's fire band (other target bands unused).
        with rasterio.open(w.context_paths[0]) as ds, rasterio.open(w.target_path) as target:
            if (ds.shape, ds.crs, ds.transform) != (target.shape, target.crs, target.transform):
                raise ValueError("Spatial misalignment within J/J+1 window")
            validate_layout(target)
            target_fire = target.read(23)
        if np.isinf(target_fire).any():
            raise ValueError("Infinite target fire code")
        return self.normalizer.transform(raw), (~np.isnan(target_fire))[None].astype(np.float32), fire_mask(raw)


class PatchDataset(Dataset):
    """Training sampling with replacement, or exhaustive deterministic validation.

    Stratifies grid tiles by target fire presence, only in training. Padding has
    its own validity mask. Evaluation always covers the full raster once.
    """
    def __init__(self, store, size=64, samples=None, positive_fraction=0.5, seed=42):
        if samples is not None and samples < 1:
            raise ValueError("samples must be positive")
        if not 0 <= positive_fraction <= 1:
            raise ValueError("positive_fraction in [0,1] required")
        self.store, self.size, self.samples = store, size, samples
        self.fraction, self.seed, self.epoch = positive_fraction, seed, 0
        self.positive, self.negative, self.tiles = [], [], []
        for i in range(len(store)):
            _, y, _ = store.get(i)
            for r, c in tile_coordinates(*y.shape[-2:], size):
                tile = (i, r, c)
                self.tiles.append(tile)
                pool = self.positive if y[:, r:r+size, c:c+size].any() else self.negative
                pool.append(tile)
        if samples is not None and (not self.positive or not self.negative):
            raise ValueError("Training needs both fire and negative patches")

    def __len__(self):
        return self.samples if self.samples is not None else len(self.tiles)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.samples is None:
            i, r, c = self.tiles[index]
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            # Exact per-epoch quota, independent of DataLoader worker ordering.
            pool = self.positive if index < round(self.samples * self.fraction) else self.negative
            i, r, c = pool[int(rng.integers(len(pool)))]
        x, y, _ = self.store.get(i)
        valid = padded_patch(np.ones_like(y), r, c, self.size)
        return tuple(torch.from_numpy(a.copy()) for a in (
            padded_patch(x, r, c, self.size), padded_patch(y, r, c, self.size), valid))
