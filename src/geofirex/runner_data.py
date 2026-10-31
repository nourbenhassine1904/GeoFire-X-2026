"""Annual, aligned temporal data. Read raster tiles, never a dense corpus tensor.

WildfireSpreadTS active-fire NaN means *no detection*, not an invalid label.
Only explicit GDAL dataset/alpha masks and finite nodata codes mask the footprint.
Missing continuous inputs are encoded by the historical audited Normalizer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import MaskFlags
from rasterio.windows import Window as RasterWindow
from torch.utils.data import Dataset

from geofirex.benchmark import aligned_windows, event_assignments
from geofirex.corpus import sha256_file
from geofirex.unet_data import Normalizer, padded_patch, tile_coordinates, validate_layout


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FoldData:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["data_root"]).resolve()
        self.roots = [Path(p).resolve() for p in config.get("data_roots", [])] or [self.root]
        with Path(config["manifest"]).open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        with Path(config["split"]).open(encoding="utf-8", newline="") as f:
            split = list(csv.DictReader(f))
        self.assignment = {r["event_id"]: r["split"] for r in split}
        if len(self.assignment) != len(split):
            raise ValueError("Duplicate split event")
        if self.assignment != event_assignments(list(self.assignment), config["fold"]):
            raise ValueError("Split differs from the requested annual fold")
        self.index = json.loads(Path(config["index"]).read_text(encoding="utf-8"))
        entries = self.index["entries"]
        self.expected = {(e["event_id"], e["date"]): e for e in entries}
        if len(self.expected) != len(entries):
            raise ValueError("Duplicate indexed observation")
        if set(self.assignment) != {e[0] for e in self.expected}:
            raise ValueError("Split must assign every indexed event exactly once")
        present = {(r["event_id"], r["date"]) for r in self.rows}
        if len(present) != len(self.rows):
            raise ValueError("Duplicate manifest observation")
        if not present <= self.expected.keys():
            raise ValueError("Manifest observations absent from index")
        if config["qualification"] == "scientific":
            if len(self.assignment) != 607 or len(self.expected) != 13607 or present != self.expected.keys():
                raise ValueError("Scientific runs require the complete 607-event / 13607-observation manifest")
        if config["qualification"] in {"pilot", "scientific"}:
            selected_events = {r["event_id"] for r in self.rows}
            if present != {key for key in self.expected if key[0] in selected_events}:
                raise ValueError("Pilot/scientific manifests require complete events")
        # Canonicalize before checking reuse, including symlinks and absolute paths.
        resolved = [{**r, "relative_path": str(self.resolve_path(r["relative_path"]))}
                    for r in self.rows]
        self.path_tokens = {r["relative_path"]: f"{r['event_id']}/{r['date']}.tif" for r in resolved}
        if config["qualification"] == "scientific":
            # Inventory only for test: never open its pixels before the lock.
            for r in resolved:
                path = Path(r["relative_path"])
                expected = self.expected[(r["event_id"], r["date"])]
                if not path.is_file() or path.stat().st_size != expected["geotiff_bytes"]:
                    raise ValueError(f"Incomplete scientific raster inventory: {path}")
        self.groups, self.rejections = {}, {}
        for role in ("train", "val", "test"):
            self.groups[role], self.rejections[role] = aligned_windows(
                resolved, self.assignment, role, contexts=(1, 3, 5))
        for role in ("train", "val"):
            if not self.groups[role][5]:
                raise ValueError(f"No aligned windows in {role}")
        self.hashes = {key: sha256_file(config[key]) for key in ("manifest", "split", "index")}
        self.hashes["targets"] = digest({role: [(w.event_id, w.target_date) for w in groups[5]]
                                         for role, groups in self.groups.items()})
        self.fingerprints = {}
        self.geometries = {}

    def resolve_path(self, name):
        if Path(name).is_absolute():
            return Path(name).resolve()
        paths = {(root/name).resolve() for root in self.roots if (root/name).is_file()}
        if len(paths) > 1:
            raise ValueError(f"Ambiguous raster across mounts: {name}")
        return next(iter(paths)) if paths else (self.roots[0]/name).resolve()

    def audit(self, roles):
        """Hash/QC the requested roles only; train never opens test rasters."""
        seen = {v["sha256"]: self.assignment[k.split("|")[0]]
                for k, v in self.fingerprints.items()}
        for r in self.rows:
            role = self.assignment[r["event_id"]]
            if role not in roles:
                continue
            path = self.resolve_path(r["relative_path"])
            key = (r["event_id"], r["date"])
            expected = self.expected[key]
            checksum = sha256_file(path)
            if checksum in seen and seen[checksum] != role:
                raise ValueError("Identical raster content crosses split roles")
            seen[checksum] = role
            if self.config["qualification"] == "scientific":
                crc = 0
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if (path.stat().st_size != expected["geotiff_bytes"]
                        or f"{crc & 0xffffffff:08x}" != expected["crc32"]):
                    raise ValueError(f"Indexed size/CRC mismatch: {path}")
            with rasterio.open(path) as ds:
                validate_layout(ds)
                geometry = (ds.shape, str(ds.crs), tuple(ds.transform))
                if ds.shape != (int(r["height"]), int(r["width"])) or ds.crs is None:
                    raise ValueError(f"Invalid raster geometry: {path}")
                previous = self.geometries.setdefault(r["event_id"], geometry)
                if previous != geometry:
                    raise ValueError("Event rasters differ in shape/CRS/transform")
                # GDAL masks can live in sidecars; pin them as well as TIFF bytes.
                sidecars = {Path(p).name: sha256_file(p) for p in ds.files if Path(p).resolve() != path}
            self.fingerprints["|".join(key)] = {"sha256": checksum, "bytes": path.stat().st_size,
                                                "sidecars": sidecars}
        counts = {}
        for role, groups in self.groups.items():
            windows = groups[5]
            used = {self.path_tokens[p] for w in windows for p in (*w.context_paths, w.target_path)}
            counts[role] = {"manifest_events": len({r["event_id"] for r in self.rows if self.assignment[r["event_id"]] == role}),
                            "manifest_observations": sum(self.assignment[r["event_id"]] == role for r in self.rows),
                            "used_events": len({w.event_id for w in windows}),
                            "used_observations": len(used), "aligned_targets": len(windows)}
        return {"hashes": self.hashes, "raster_fingerprints": self.fingerprints,
                "event_geometries": self.geometries, "rejections": self.rejections,
                "corpus_counts": counts, "complete_corpus": len(self.assignment) == 607 and len(self.expected) == 13607 and len(self.rows) == 13607,
                "windows": {r: len(g[5]) for r, g in self.groups.items()}}

    def windows(self, role):
        return self.groups[role][self.config["context"]]


def _read_selected(path, row=0, col=0, size=None, *, full=False):
    with rasterio.open(path) as ds:
        window = None if size is None else RasterWindow(col, row, min(size, ds.width-col), min(size, ds.height-row))
        selected = ds.read(list(range(1, 24)) if full else [13, 14, 15, 23], window=window).astype(np.float32)
        fire_index = 22 if full else 3
        if np.isinf(selected[fire_index]).any():
            raise ValueError("Infinite active fire code")
        valid = np.ones((1, *selected.shape[-2:]), np.float32)
        if any(MaskFlags.per_dataset in flags or MaskFlags.alpha in flags for flags in ds.mask_flag_enums):
            valid *= (ds.dataset_mask(window=window) > 0)[None]
        nodata = ds.nodatavals[22]
        if nodata is not None and np.isfinite(nodata):
            valid *= (selected[fire_index:fire_index+1] != nodata)
            selected[fire_index][selected[fire_index] == nodata] = np.nan
    return selected, valid


def _raw(selected):
    if selected.shape[0] == 23:
        return selected
    raw = np.full((23, *selected.shape[-2:]), np.nan, np.float32)
    raw[[12, 13, 14, 22]] = selected
    return raw


def read_tile(path, row, col, size, *, full=False):
    selected, valid = _read_selected(path, row, col, size, full=full)
    raw = _raw(selected)
    return raw, valid


class RasterReader:
    """Per-process byte-bounded LRU of physical bands + footprint, never z-scores.

    Large rasters that exceed the budget fall back to window reads. No open
    GDAL handles or warm arrays are serialized into spawned DataLoader workers.
    """
    def __init__(self, cache_bytes=64*1024**2, full=False):
        self.budget = cache_bytes
        self.full = full
        self.cache = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def __getstate__(self):
        return {**self.__dict__, "cache": OrderedDict(), "bytes": 0, "hits": 0, "misses": 0}

    def tile(self, path, row, col, size):
        if path in self.cache:
            self.hits += 1
            self.cache.move_to_end(path)
        else:
            self.misses += 1
            with rasterio.open(path) as ds:
                estimated_bytes = ds.height*ds.width*(24 if self.full else 5)*4
            if estimated_bytes > self.budget:
                return read_tile(path, row, col, size, full=self.full)
            selected, valid = _read_selected(path, full=self.full)
            n = selected.nbytes + valid.nbytes
            while self.cache and self.bytes+n > self.budget:
                _, (old_selected, old_valid) = self.cache.popitem(last=False)
                self.bytes -= old_selected.nbytes + old_valid.nbytes
            self.cache[path] = (selected, valid)
            self.bytes += n
        selected, valid = self.cache[path]
        return _raw(selected[:, row:row+size, col:col+size]), valid[:, row:row+size, col:col+size]


def fit_fold_normalizer(windows, size=256, clip=8., *, continuous=(12, 14), landcover=False):
    # Floating-point reductions must keep the same order after mount slugs move.
    logical = {p: (w.event_id, date) for w in windows for p, date in zip(w.context_paths, w.context_dates)}
    paths = sorted(logical, key=logical.__getitem__)
    count = np.zeros(len(continuous), np.int64)
    mean, m2 = np.zeros(len(continuous)), np.zeros(len(continuous))
    for path in paths:
        with rasterio.open(path) as ds:
            coords = tile_coordinates(ds.height, ds.width, size)
        for r, c in coords:
            raw, valid = read_tile(path, r, c, size, full=tuple(continuous) != (12, 14))
            for i, band in enumerate(continuous):
                values = raw[band][np.isfinite(raw[band]) & valid[0].astype(bool)].astype(np.float64)
                n = len(values)
                if n:
                    delta = values.mean() - mean[i]
                    total = count[i] + n
                    m2[i] += ((values-values.mean())**2).sum() + delta**2 * count[i]*n/total
                    mean[i] += delta*n/total
                    count[i] = total
    if (count == 0).any():
        raise ValueError("No finite train input values for normalization")
    std = np.sqrt(m2/count)
    std[std < 1e-6] = 1
    return Normalizer(list(continuous), mean.tolist(), std.tolist(), count.tolist(), paths, landcover, clip)


class TemporalStore:
    def __init__(self, windows, normalizer, domain_windows=None, cache_bytes=64*1024**2):
        self.windows, self.normalizer = windows, normalizer
        self.reader = RasterReader(cache_bytes, full=normalizer.continuous != [12, 14] or normalizer.include_landcover)
        self.domain_windows = domain_windows if domain_windows is not None else windows
        if [(w.event_id, w.target_date) for w in windows] != [(w.event_id, w.target_date) for w in self.domain_windows]:
            raise ValueError("Validity domain target mismatch")
        self.shapes = []
        for w in windows:
            with rasterio.open(w.target_path) as ds:
                self.shapes.append(ds.shape)

    def tile(self, index, row, col, size):
        w = self.windows[index]
        inputs, masks = [], []
        for path in w.context_paths:
            raw, valid = self.reader.tile(path, row, col, size)
            inputs.append(self.normalizer.transform(raw))
            masks.append(valid)
        target, valid = self.reader.tile(w.target_path, row, col, size)
        # Same domain for B0/B1/B2/B3 and T=1/3/5: footprint of maximum context.
        for path in self.domain_windows[index].context_paths:
            if path not in w.context_paths:
                _, mask = self.reader.tile(path, row, col, size)
                masks.append(mask)
        valid = valid * np.logical_and.reduce(masks)
        x = np.stack(inputs)
        y = np.isfinite(target[22:23]).astype(np.float32)
        fire_index = self.normalizer.channel_names.index("fire_J")
        observed = inputs[-1][fire_index:fire_index+1]
        return tuple(padded_patch(a, 0, 0, size) for a in (x, y, valid, observed))


class TrainTiles(Dataset):
    """Reproducible mixture of positive train tiles and the uniform grid.

    Validation/test always use exhaustive tiles in runner_evaluation. Label-based
    pools are built only by prior(), on the training store.
    """
    def __init__(self, store, size, samples, seed, positive_fraction=0.):
        self.store, self.size, self.samples, self.seed = store, size, samples, seed
        if not 0 <= positive_fraction <= 1:
            raise ValueError("positive_fraction must be in [0,1]")
        self.positive_fraction = positive_fraction
        self.positive_tiles = []
        self.epoch = 0
        self.tiles = [(i, r, c) for i, shape in enumerate(store.shapes)
                      for r, c in tile_coordinates(*shape, size)]
        if not self.tiles:
            raise ValueError("Empty training grid")

    def __len__(self):
        return self.samples

    def coordinate(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        if self.positive_fraction:
            if not self.positive_tiles:
                raise ValueError("Call train prior() before positive sampling")
            if rng.random() < self.positive_fraction:
                return self.positive_tiles[int(rng.integers(len(self.positive_tiles)))]
        return self.tiles[int(rng.integers(len(self.tiles)))]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        x, y, valid, _ = self.store.tile(*self.coordinate(index), self.size)
        return tuple(torch.from_numpy(a.copy()) for a in (x, y, valid))

    def prior(self):
        positive = pixels = 0
        self.positive_tiles = []
        for i, r, c in self.tiles:
            _, y, valid, _ = self.store.tile(i, r, c, self.size)
            n = int((y*valid).sum())
            if n:
                self.positive_tiles.append((i, r, c))
            positive += n
            pixels += int(valid.sum())
        if not pixels or not positive or positive == pixels:
            raise ValueError("Training requires valid positive and negative pixels")
        return {"positive_pixels": positive, "pixels": pixels,
                "positive_tiles": len(self.positive_tiles), "total_tiles": len(self.tiles),
                "positive_tile_fraction": self.positive_fraction,
                "prevalence": positive/pixels, "negative_positive_ratio": (pixels-positive)/positive}
