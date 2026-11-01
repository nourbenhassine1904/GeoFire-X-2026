"""Read-only empirical diagnostics of the frozen first U-Net, validation only.

No optimizer, no new MLflow run and no modification to the historical outputs.
Ablations test sensitivity of this fixed model, not causes of training failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from geofirex.corpus import json_new
from geofirex.unet import SegmentationLoss
from geofirex.unet_data import PatchDataset, WindowStore, load_event_windows, padded_patch
from geofirex.unet_evaluation import predict_tiled, score, summarize
from predict_unet import load_predictor

ROOT = Path(__file__).resolve().parents[1]


def distribution(values):
    return {str(q): float(v) for q, v in zip((0, .01, .1, .5, .9, .99, 1),
                                             np.quantile(values, (0, .01, .1, .5, .9, .99, 1)))}


def diagnose(checkpoint_path, output):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, norm, checkpoint = load_predictor(checkpoint_path)
    cfg = checkpoint["config"]
    torch.set_num_threads(cfg["cpu_threads"])
    _, _, windows, _ = load_event_windows(ROOT / cfg["manifest"], ROOT / cfg["split"], ROOT)
    threshold = checkpoint["threshold"]
    records, variants = [], {"fire_zero": [], "fire_shift_64px": [], "full_scene_forward": [], "tile_origin_shift_32px": []}
    seams, interiors, differences, fire_effects, bce_losses, dice_losses = [], [], [], [], [], []
    inference_start = time.perf_counter()
    for i, w in enumerate(windows["val"]):
        store = WindowStore([w], norm)
        x, target, persistence = store.get(0)
        p = predict_tiled(model, x, cfg["patch_size"], cfg["batch_size"])
        common = dict(event_id=w.event_id, context_date=w.context_dates[0], target_date=w.target_date,
                      target=target, persistence=persistence)
        records.append(dict(**common, probability=p))
        for name in ("fire_zero", "fire_shift_64px"):
            altered = x.copy()
            altered[7] = 0 if name == "fire_zero" else np.roll(x[7], 64, axis=1)
            prob = predict_tiled(model, altered, cfg["patch_size"], cfg["batch_size"])
            variants[name].append(dict(**common, probability=prob))
            if name == "fire_zero":
                fire_effects.append(np.abs(prob-p).ravel())
        with torch.inference_mode():
            full = model(torch.from_numpy(x)[None]).sigmoid().numpy()[0]
        variants["full_scene_forward"].append(dict(**common, probability=full))
        differences.append(np.abs(full-p).ravel())
        # Change tiling origin, without changing the scored target footprint.
        shifted = np.pad(x, ((0,0), (32,0), (32,0)))
        shifted_p = predict_tiled(model, shifted, 64, cfg["batch_size"])[:, 32:, 32:]
        variants["tile_origin_shift_32px"].append(dict(**common, probability=shifted_p))
        h, width = target.shape[-2:]
        rr, cc = np.indices((h, width))
        seam = (rr % 64 <= 1) | (rr % 64 >= 62) | (cc % 64 <= 1) | (cc % 64 >= 62)
        seams.append(p[0][seam]); interiors.append(p[0][~seam])
        # Exact sum over tiles for diagnostic loss composition, not a new checkpoint criterion.
        for r in range(0, h, 64):
            for c in range(0, width, 64):
                xx, yy, mask = [torch.from_numpy(a.copy())[None] for a in (
                    padded_patch(x,r,c,64), padded_patch(target,r,c,64),
                    padded_patch(np.ones_like(target),r,c,64))]
                with torch.inference_mode():
                    logits = model(xx)
                    bce_losses.append(float(SegmentationLoss(0)(logits,yy,mask)))
                    dice_losses.append(float(SegmentationLoss(1)(logits,yy,mask)))
    inference_seconds = time.perf_counter()-inference_start
    p = np.concatenate([r["probability"].ravel() for r in records])
    target = np.concatenate([r["target"].ravel() for r in records]).astype(bool)
    observed = np.concatenate([r["persistence"].ravel() for r in records]).astype(bool)
    prediction = p >= threshold
    novel = target & ~observed
    bins = []
    edges = (0, .001, .005, .01, .02, .05, .1, 1.000001)
    for a, b in zip(edges, edges[1:]):
        keep = (p >= a) & (p < b)
        bins.append({"lower": a, "upper": b, "pixels": int(keep.sum()),
                     "mean_score": float(p[keep].mean()) if keep.any() else None,
                     "observed_fraction": float(target[keep].mean()) if keep.any() else None})
    train_store = WindowStore(windows["train"], norm)
    patches = PatchDataset(train_store, 64, cfg["samples_per_epoch"], cfg["positive_fraction"], cfg["seed"])
    summaries = {}
    for tile in patches.tiles:
        i,r,c = tile
        _, y, _ = train_store.get(i)
        crop = y[:,r:r+64,c:c+64]
        summaries[tile] = (int(crop.sum()), crop.size)
    counts, event_counts = Counter(), Counter()
    for epoch in range(cfg["epochs"]):
        for index in range(cfg["samples_per_epoch"]):
            rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"],epoch,index]))
            pool = patches.positive if index < round(len(patches)*cfg["positive_fraction"]) else patches.negative
            tile = pool[int(rng.integers(len(pool)))]
            counts[tile] += 1
            event_counts[windows["train"][tile[0]].event_id] += 1
    sampled_positive = sum(summaries[t][0]*n for t,n in counts.items())
    sampled_pixels = sum(summaries[t][1]*n for t,n in counts.items())
    gradients = {}
    for kind, tile in (("positive", max(patches.positive,key=lambda t:summaries[t][0])),
                        ("empty", patches.negative[0])):
        i,r,c = tile
        x,y,_ = train_store.get(i)
        xx,yy = [torch.from_numpy(padded_patch(a,r,c,64).copy())[None] for a in (x,y)]
        valid = torch.from_numpy(padded_patch(np.ones_like(y),r,c,64).copy())[None]
        logits = model(xx).detach().requires_grad_()
        bce = SegmentationLoss(0)(logits,yy,valid)
        dice = SegmentationLoss(1)(logits,yy,valid)
        gb = torch.autograd.grad(bce,logits,retain_graph=True)[0]
        gd = torch.autograd.grad(dice,logits)[0]
        gradients[kind] = {"positive_pixels": int(yy.sum()), "bce": float(bce.detach()),
                           "soft_dice_loss": float(dice.detach()),
                           "bce_logit_gradient_l2": float(gb.norm()),
                           "dice_logit_gradient_l2": float(gd.norm())}
    comparisons = {"original_tiled": summarize(records,threshold),
                   "persistence": summarize(records,threshold,"persistence")}
    comparisons.update({name: summarize(rs,threshold) for name,rs in variants.items()})
    result = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
              "scope": "frozen model, validation only; no training or test selection",
              "validation_windows": len(records), "threshold_frozen": threshold,
              "probability_quantiles": distribution(p), "positive_score_quantiles": distribution(p[target]),
              "negative_score_quantiles": distribution(p[~target]),
              "predicted_positive_fraction": float(prediction.mean()), "target_prevalence": float(target.mean()),
              "brier_score": float(np.mean((p-target)**2)),
              "constant_prevalence_brier": float(target.mean()*(1-target.mean())),
              "reliability_bins": bins,
              "new_fire_pixels": int(novel.sum()),
              "new_fire_recalled_unet": int((prediction & novel).sum()),
              "new_fire_recalled_persistence": int((observed & novel).sum()),
              "mean_probability_tile_seams": float(np.concatenate(seams).mean()),
              "mean_probability_tile_interiors": float(np.concatenate(interiors).mean()),
              "full_scene_vs_tiled_absolute_difference": distribution(np.concatenate(differences)),
              "zero_fire_vs_original_absolute_difference": distribution(np.concatenate(fire_effects)),
              "val_mean_tile_bce": float(np.mean(bce_losses)),
              "val_mean_tile_dice_loss": float(np.mean(dice_losses)),
              "loss_output_gradients": gradients,
              "training_exposure": {"patch_draws": sum(counts.values()), "unique_tiles": len(counts),
                                    "available_tiles": len(patches.tiles),
                                    "unique_positive_tiles": sum(t in patches.positive for t in counts),
                                    "available_positive_tiles": len(patches.positive),
                                    "sampled_positive_pixels": sampled_positive,
                                    "sampled_valid_pixels": sampled_pixels,
                                    "sampled_pixel_positive_fraction": sampled_positive/sampled_pixels,
                                    "patch_draws_by_event": dict(event_counts)},
              "comparisons": comparisons, "diagnostic_inference_seconds": inference_seconds,
              "total_seconds": time.perf_counter()-start}
    json_new(output / "diagnostics.json",result)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(p[~target],bins=60,density=True,histtype="step",label="negative pixels")
    ax.hist(p[target],bins=30,density=True,histtype="step",label="positive pixels")
    ax.axvline(threshold,color="black",linestyle="--",label="frozen val threshold")
    ax.set(xlabel="U-Net sigmoid score",ylabel="Conditional density",title="Validation: class-conditional scores")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "probability_distribution.png",dpi=160); plt.close(fig)
    print({k:result[k] for k in ("probability_quantiles","target_prevalence","training_exposure","loss_output_gradients","total_seconds")})
    print({k:v["micro_pixel"] for k,v in comparisons.items()})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    diagnose(args.checkpoint,args.out_dir)
