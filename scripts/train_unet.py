"""Bounded, reproducible real-data U-Net baseline with a new MLflow run.

Run from any working directory with the project's Python environment.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from audit_unet_data import ROOT, audit, sha256
from geofirex.unet import LightUNet, SegmentationLoss
from geofirex.unet_data import (PatchDataset, WindowStore, fit_normalizer,
                               load_event_windows, padded_patch)
from geofirex.unet_evaluation import predict_tiled, select_threshold, summarize


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch build/device is unavailable")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
        text=True, encoding="utf-8").strip()


def snapshot_code(out):
    files = [ROOT / "pyproject.toml", ROOT / ".gitignore"]
    for directory in ("src", "scripts", "tests", "configs"):
        files += [p for p in (ROOT / directory).rglob("*")
                  if p.is_file() and p.suffix in {".py", ".yaml"}]
    hashes = {}
    for file in sorted(files):
        relative = file.relative_to(ROOT)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
        hashes[relative.as_posix()] = sha256(file)
    save_json(out / "source_hashes.json", hashes)
    return sha256(out / "source_hashes.json")


def preflight(train_patches, config, channels, device):
    """A positive REAL training tile: forward, finite backward, then mini-overfit.

    This diagnostic model is discarded before a separately seeded main training.
    """
    # A deliberately informative training tile makes this a learnability check,
    # not a test of optimizing a seven-positive / 4096-pixel corner case.
    i, r, c = max(train_patches.positive,
                  key=lambda tile: train_patches.store.get(tile[0])[1][
                      :, tile[1]:tile[1]+train_patches.size,
                      tile[2]:tile[2]+train_patches.size].sum())
    x, y, _ = train_patches.store.get(i)
    size = config["patch_size"]
    x, y, valid = [torch.from_numpy(a.copy())[None].to(device) for a in (
        padded_patch(x, r, c, size), padded_patch(y, r, c, size),
        padded_patch(np.ones_like(y), r, c, size))]
    model = LightUNet(channels, config["base_channels"], config["depth"]).to(device)
    criterion = SegmentationLoss(config["dice_weight"], config["pos_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["mini_overfit_lr"])
    history = []
    start = time.perf_counter()
    for step in range(config["mini_overfit_steps"]):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y, valid)
        if logits.shape != y.shape or not torch.isfinite(loss):
            raise RuntimeError("Forward/loss preflight failed")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Backward preflight failed")
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        logits = model(x)
        final_loss = float(criterion(logits, y, valid).cpu())
        prediction = (logits.sigmoid() >= 0.5).float() * valid
        dice = float((2 * (prediction * y).sum() / (prediction.sum() + y.sum())).cpu())
    result = {"input_shape": list(x.shape), "logit_shape": list(logits.shape),
              "positive_pixels": int(y.sum()), "initial_loss": history[0],
              "final_loss": final_loss, "final_dice_at_0_5": dice,
              "steps": len(history), "loss_history": history,
              "seconds": time.perf_counter() - start,
              "event_id": train_patches.store.windows[i].event_id,
              "target_date": train_patches.store.windows[i].target_date,
              "tile_row": r, "tile_col": c}
    result["passed"] = final_loss < history[0] * 0.8 and dice > 0.5
    return result


def epoch_loss(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, valid in loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y, valid)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training/validation loss")
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradient")
                optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
    return total / count


def predict_store(model, store, config, device):
    records = []
    for i, w in enumerate(store.windows):
        x, target, persistence = store.get(i)
        probability = predict_tiled(model, x, config["patch_size"], config["batch_size"], device)
        records.append({"event_id": w.event_id, "context_date": w.context_dates[0],
                        "target_date": w.target_date, "probability": probability,
                        "target": target, "persistence": persistence})
    return records


def comparison_figure(record, threshold, path):
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.3))
    arrays = [record["persistence"], record["target"], record["persistence"],
              record["probability"], record["probability"] >= threshold]
    titles = ["Observed fire J", "Ground truth J+1", "Persistence J+1",
              "U-Net probability", f"U-Net mask (t={threshold:g})"]
    for ax, data, title in zip(axes, arrays, titles):
        im = ax.imshow(data[0], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.colorbar(im, ax=list(axes), fraction=0.015, pad=0.01, label="Fire / probability")
    fig.suptitle(f"{record['event_id']} | {record['context_date']} -> {record['target_date']} | technical sample")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(config):
    if config["feature_policy"] != "conservative_fire_terrain":
        raise ValueError("Only audited conservative_fire_terrain policy is enabled")
    if config["include_landcover"] and config["landcover_availability"] != "retrospective_annual_assumption":
        raise ValueError("LC requires explicit retrospective_annual_assumption acknowledgement")
    if config["patch_size"] < 2**config["depth"]:
        raise ValueError("Patch size is too small for model depth")
    if min(config["epochs"], config["mini_overfit_steps"], config["cpu_threads"], config["batch_size"]) < 1:
        raise ValueError("Epochs, preflight steps, threads, batch size must be positive")
    torch.set_num_threads(config["cpu_threads"])
    device = choose_device(config["device"])
    seed_all(config["seed"])
    start = time.perf_counter()
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out = ROOT / config["output_root"] / label
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "config.json", config)
    manifest, split = ROOT / config["manifest"], ROOT / config["split"]
    shutil.copy2(manifest, out / "manifest.csv")
    shutil.copy2(split, out / "split.csv")
    print(f"Output: {out}\nDevice: {device}; CPU threads: {torch.get_num_threads()}", flush=True)
    audited = audit(manifest, split)
    save_json(out / "data_audit.json", audited)
    print("Audit complete: " + str(audited["splits"]), flush=True)
    _, _, windows, _ = load_event_windows(manifest, split, ROOT)
    normalizer = fit_normalizer(windows["train"], include_landcover=config["include_landcover"],
                                clip=config["normalization_clip"])
    save_json(out / "normalization.json", normalizer.to_dict())
    stores = {role: WindowStore(ws, normalizer) for role, ws in windows.items()}
    train_patches = PatchDataset(stores["train"], config["patch_size"], config["samples_per_epoch"],
                                 config["positive_fraction"], config["seed"])
    val_patches = PatchDataset(stores["val"], config["patch_size"])
    channels = len(normalizer.channel_names)
    source_hash = snapshot_code(out)
    provenance = {"git_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                  "git_status": git("status", "--short"), "source_hash": source_hash,
                  "python": sys.version, "platform": platform.platform(),
                  "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "device": device,
                  "channel_names": normalizer.channel_names,
                  "num_parameters": sum(p.numel() for p in LightUNet(channels, config["base_channels"], config["depth"]).parameters()),
                  "train_grid_positive_tiles": len(train_patches.positive),
                  "train_grid_negative_tiles": len(train_patches.negative),
                  "val_grid_tiles": len(val_patches),
                  "training_positive_quota": round(len(train_patches) * config["positive_fraction"])}
    save_json(out / "provenance.json", provenance)
    (out / "requirements_freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    # Explicit local URI: append runs to the existing store, without overwriting history.
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment(config["experiment"])
    with mlflow.start_run(run_name=f"unet2d-{label}") as active:
        run_id = active.info.run_id
        mlflow.set_tags({"qualification": "technical_sample_not_scientific_benchmark",
                         "source_hash": source_hash, "git_commit": provenance["git_commit"]})
        mlflow.log_params({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in config.items()})
        mlflow.log_params({"actual_device": device, "in_channels": channels,
                           "parameters": provenance["num_parameters"],
                           "manifest_sha256": audited["manifest_sha256"], "split_sha256": audited["split_sha256"]})
        try:
            print("Real-patch forward/backward/mini-overfit...", flush=True)
            diagnostic = preflight(train_patches, config, channels, device)
            save_json(out / "preflight.json", diagnostic)
            if not diagnostic["passed"]:
                raise RuntimeError(f"Real-patch mini-overfit failed: loss={diagnostic['final_loss']:.6f}, Dice={diagnostic['final_dice_at_0_5']:.6f}; see preflight.json")
            print(f"Preflight passed: loss {diagnostic['initial_loss']:.5f} -> {diagnostic['final_loss']:.5f}; Dice {diagnostic['final_dice_at_0_5']:.4f}", flush=True)
            mlflow.log_metrics({"preflight_final_loss": diagnostic["final_loss"],
                                "preflight_dice": diagnostic["final_dice_at_0_5"]})
            # Fresh model: no diagnostic gradient/state enters the reported training.
            seed_all(config["seed"])
            model = LightUNet(channels, config["base_channels"], config["depth"]).to(device)
            criterion = SegmentationLoss(config["dice_weight"], config["pos_weight"]).to(device)
            optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}
            optimizer = optimizers[config["optimizer"]](model.parameters(), lr=config["learning_rate"],
                                                        weight_decay=config["weight_decay"])
            generator = torch.Generator().manual_seed(config["seed"])
            train_loader = DataLoader(train_patches, batch_size=config["batch_size"], shuffle=True,
                                      generator=generator, num_workers=0)
            val_loader = DataLoader(val_patches, batch_size=config["batch_size"], num_workers=0)
            history, best, best_epoch = [], float("inf"), 0
            for epoch in range(config["epochs"]):
                train_patches.epoch = epoch
                tick = time.perf_counter()
                train_loss = epoch_loss(model, train_loader, criterion, device, optimizer)
                val_loss = epoch_loss(model, val_loader, criterion, device)
                record = {"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss,
                          "seconds": time.perf_counter()-tick}
                history.append(record)
                save_json(out / "history.json", history)
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch+1)
                if val_loss < best:
                    best, best_epoch = val_loss, epoch+1
                    # Portable CPU tensors; optimizer saved for traceability, not automatic resume.
                    torch.save({"format_version": 1, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "optimizer_state": optimizer.state_dict(), "epoch": best_epoch,
                                "val_loss": best, "config": config, "normalizer": normalizer.to_dict(),
                                "in_channels": channels, "source_hash": source_hash,
                                "manifest_sha256": audited["manifest_sha256"],
                                "split_sha256": audited["split_sha256"]}, out / "best.pt")
                print(f"Epoch {epoch+1}/{config['epochs']}: train={train_loss:.6f}, val={val_loss:.6f}, {record['seconds']:.1f}s", flush=True)
            checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            print(f"Evaluate best epoch {best_epoch} on complete validation rasters...", flush=True)
            validation = predict_store(model, stores["val"], config, device)
            threshold, curve = select_threshold(validation, config["thresholds"], split_name="val")
            save_json(out / "threshold_selection.json", {"split": "val", "objective": "micro_pixel_dice", "threshold": threshold, "curve": curve})
            checkpoint["threshold"] = threshold
            checkpoint["threshold_selected_on"] = "val"
            torch.save(checkpoint, out / "best.pt")
            # Test is first inferred AFTER best checkpoint and threshold are frozen.
            test = predict_store(model, stores["test"], config, device)
            evaluations = {}
            for role, records in (("val", validation), ("test", test)):
                evaluations[role] = {name: summarize(records, threshold, name) for name in ("unet", "persistence")}
                figure_dir = out / "figures" / role
                figure_dir.mkdir(parents=True)
                # Deterministic first, most-positive, and empty example (where available).
                chosen = list(dict.fromkeys([0, max(range(len(records)), key=lambda i: records[i]["target"].sum()),
                                              next((i for i, r in enumerate(records) if not r["target"].any()), len(records)-1)]))
                for i in chosen[:config["max_figures_per_split"]]:
                    comparison_figure(records[i], threshold, figure_dir / f"window_{i:03d}.png")
                for name in ("unet", "persistence"):
                    for aggregation in ("micro_pixel", "macro_window", "macro_positive_window", "macro_event"):
                        metrics = evaluations[role][name][aggregation]
                        mlflow.log_metrics({f"{role}_{name}_{aggregation}_{k}": v for k, v in metrics.items()
                                            if k in {"dice", "iou", "precision", "recall", "pr_auc"} and v is not None})
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], "o-", label="train: balanced sampled tiles")
            ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], "o-", label="val: all tiles")
            ax.set(xlabel="Epoch", ylabel="BCE + Dice loss", title="First U-Net technical training")
            ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / "loss_curve.png", dpi=160); plt.close(fig)
            results = {"qualification": "real training on 5 events; technical baseline, no generalization claim",
                       "run_id": run_id, "experiment": config["experiment"], "output_dir": str(out),
                       "best_epoch": best_epoch, "best_val_loss": best, "threshold": threshold,
                       "checkpoint_sha256": sha256(out / "best.pt"),
                       "history": history, "preflight": {k: v for k, v in diagnostic.items() if k != "loss_history"},
                       "elapsed_seconds": time.perf_counter()-start, "provenance": provenance,
                       "splits": audited["splits"], "evaluations": evaluations}
            save_json(out / "results.json", results)
            mlflow.log_metrics({"best_epoch": best_epoch, "threshold": threshold,
                                "elapsed_seconds": results["elapsed_seconds"]})
        finally:
            mlflow.log_artifacts(str(out))
    print(json.dumps({k: results[k] for k in ("run_id", "output_dir", "best_epoch", "threshold", "elapsed_seconds")}, indent=2), flush=True)
    for role in ("val", "test"):
        for name in ("unet", "persistence"):
            print(role, name, results["evaluations"][role][name]["micro_pixel"], flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unet_baseline.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    run(config)
