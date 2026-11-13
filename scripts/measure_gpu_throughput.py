"""Discarded training-only throughput probe; produces no benchmark scores."""
import argparse
import time

import torch
from torch.utils.data import DataLoader

from geofirex.corpus import json_new
from geofirex.feature_profiles import feature_profile
from geofirex.runner import choose_device, load_config, runner_model, seed_all
from geofirex.runner_data import FoldData, TemporalStore, TrainTiles, fit_fold_normalizer
from geofirex.unet import SegmentationLoss


def measure(config, *, steps=100, warmup=10):
    if config["model"] == "B0" or steps < 1 or warmup < 0:
        raise ValueError("Use a trainable model, positive steps and nonnegative warmup")
    device = choose_device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    seed_all(config["seed"])
    data = FoldData(config)
    profile = feature_profile(config)
    norm = fit_fold_normalizer(data.groups["train"][5], continuous=profile["continuous"], landcover=profile["landcover"])
    store = TemporalStore(data.windows("train"), norm, data.groups["train"][5], int(config["reader_cache_mb"]*1024**2))
    tiles = TrainTiles(store, config["patch_size"], config["samples_per_epoch"], config["seed"], config["positive_tile_fraction"])
    prior = tiles.prior()
    loader = DataLoader(tiles, batch_size=config["batch_size"], num_workers=config["workers"], shuffle=False)
    model = runner_model(config, norm).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    weight = min(prior["negative_positive_ratio"], config["max_pos_weight"]) if config["pos_weight"] == "auto" else config["pos_weight"]
    criterion = SegmentationLoss(config["dice_weight"], weight).to(device)
    amp = bool(config["amp"] and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    iterator = iter(loader)
    patches = updates = 0
    started = None
    for step in range(warmup+steps):
        if step == warmup:
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x, y, valid = [a.to(device) for a in batch]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp, dtype=torch.float16):
            loss = criterion(model(x).float(), y, valid)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite throughput-probe loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if step >= warmup:
            patches += x.shape[0]
            updates += int(scaler.get_scale() >= scale)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    return {"purpose": "training throughput only; model discarded; no validation/test/quality claim",
            "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "steps": steps, "warmup_steps": warmup, "patches": patches, "optimizer_updates": updates,
            "seconds": seconds, "patches_per_second": patches/seconds,
            "estimated_training_seconds_per_epoch": config["samples_per_epoch"]/(patches/seconds),
            "estimate_excludes": "normalization, validation, MLflow sync and dataset publication",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "train_events": len({w.event_id for w in store.windows}), "data_hashes": data.hashes}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        p.error("Output already exists")
    json_new(args.out, measure(load_config(args.config), steps=args.steps, warmup=args.warmup))
