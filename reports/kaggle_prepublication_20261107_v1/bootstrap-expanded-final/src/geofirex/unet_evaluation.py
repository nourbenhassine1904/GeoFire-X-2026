"""Full-raster inference and paired evaluation, without balanced sampling."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from geofirex.unet_data import padded_patch, tile_coordinates


@torch.inference_mode()
def predict_tiled(model, image, size=64, batch_size=8, device="cpu"):
    if image.ndim != 3 or not np.isfinite(image).all() or batch_size < 1:
        raise ValueError("Finite [C,H,W] image and positive batch_size required")
    was_training = model.training
    model.eval()
    h, w = image.shape[-2:]
    output = np.zeros((1, h, w), dtype=np.float32)
    coordinates = tile_coordinates(h, w, size)
    try:
        for offset in range(0, len(coordinates), batch_size):
            coords = coordinates[offset:offset + batch_size]
            batch = np.stack([padded_patch(image, r, c, size) for r, c in coords])
            p = model(torch.from_numpy(batch).to(device)).sigmoid().cpu().numpy()
            for prob, (r, c) in zip(p, coords):
                hh, ww = min(size, h-r), min(size, w-c)
                output[:, r:r+hh, c:c+ww] = prob[:, :hh, :ww]
    finally:
        model.train(was_training)
    return output


def confusion_metrics(tp, fp, fn):
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def score(probability, target, threshold):
    probability, target = np.asarray(probability), np.asarray(target)
    if (probability.shape != target.shape or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or not np.isin(target, [0, 1]).all() or not 0 <= threshold <= 1):
        raise ValueError("Matching probabilities in [0,1], binary targets and valid threshold required")
    p, t = probability.ravel(), target.ravel().astype(bool)
    b = p >= threshold
    tp, fp, fn = int((b & t).sum()), int((b & ~t).sum()), int((~b & t).sum())
    return {**confusion_metrics(tp, fp, fn),
            "pr_auc": float(average_precision_score(t, p)) if t.any() else None,
            "tp": tp, "fp": fp, "fn": fn, "positive_pixels": int(t.sum()),
            "predicted_pixels": int(b.sum()), "pixels": int(t.size)}


def select_threshold(records, thresholds, *, split_name):
    if split_name != "val":
        raise ValueError("Decision threshold may only be selected on validation")
    if not thresholds or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("Thresholds must be inside (0,1)")
    if not sum(int(r["target"].sum()) for r in records):
        raise ValueError("Cannot select threshold without positive validation labels")
    curve = []
    for threshold in sorted(set(thresholds)):
        tp = fp = fn = 0
        for r in records:
            p, t = r["probability"] >= threshold, r["target"].astype(bool)
            tp += int((p & t).sum())
            fp += int((p & ~t).sum())
            fn += int((~p & t).sum())
        curve.append({"threshold": threshold, **confusion_metrics(tp, fp, fn)})
    # Deterministic tie-break: highest threshold among equally good pooled Dice.
    best = max(curve, key=lambda r: (r["dice"], r["threshold"]))
    return best["threshold"], curve


def _mean_metrics(rows):
    keys = ("dice", "iou", "precision", "recall", "pr_auc")
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows) else None for k in keys}


def summarize(records, threshold, predictor="unet"):
    if predictor not in {"unet", "persistence"}:
        raise ValueError("Unknown predictor; choose unet or persistence")
    key = "probability" if predictor == "unet" else "persistence"
    use_threshold = threshold if predictor == "unet" else 0.5
    rows = [{"event_id": r["event_id"], "target_date": r["target_date"],
             **score(r[key], r["target"], use_threshold)} for r in records]
    p = np.concatenate([r[key].ravel() for r in records])
    y = np.concatenate([r["target"].ravel() for r in records])
    by_event = {}
    for event in sorted({r["event_id"] for r in rows}):
        subset = [r for r in rows if r["event_id"] == event]
        by_event[event] = {"windows": len(subset), "macro_window": _mean_metrics(subset)}
    return {
        "threshold": use_threshold, "windows": len(rows),
        "empty_target_windows": sum(r["positive_pixels"] == 0 for r in rows),
        "empty_both_windows": sum(r["positive_pixels"] == r["predicted_pixels"] == 0 for r in rows),
        "macro_window": _mean_metrics(rows),
        "macro_positive_window": _mean_metrics([r for r in rows if r["positive_pixels"] > 0]),
        "metric_defined_windows": {k: sum(r[k] is not None for r in rows)
                                   for k in ("dice", "iou", "precision", "recall", "pr_auc")},
        "micro_pixel": score(p, y, use_threshold),
        "macro_event": _mean_metrics([v["macro_window"] for v in by_event.values()]),
        "per_event": by_event, "per_window": rows,
    }
