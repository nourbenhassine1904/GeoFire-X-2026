"""Exact pooled AP and threshold curves on all nonoverlapping scene tiles.

No truth-guided sampling. Each valid scene pixel occurs exactly once; pooling
tiles is mathematically identical to pooling reconstructed complete scenes.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from geofirex.benchmark_metrics import ExternalAveragePrecision
from geofirex.unet_data import tile_coordinates
from geofirex.unet_evaluation import confusion_metrics


class Counts:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.counts = np.zeros((len(thresholds), 3), np.int64)
        self.pixels = self.positive = 0
        self.brier_sum = 0.

    def update(self, p, y):
        t = y.astype(bool)
        self.pixels += len(y)
        self.positive += int(t.sum())
        self.brier_sum += float(np.square(p.astype(np.float64)-y).sum())
        for i, threshold in enumerate(self.thresholds):
            b = p >= threshold
            self.counts[i] += [int((b&t).sum()), int((b&~t).sum()), int((~b&t).sum())]

    def at(self, i):
        tp, fp, fn = map(int, self.counts[i])
        return {**confusion_metrics(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
                "pixels": self.pixels, "positive_pixels": self.positive,
                "prevalence": self.positive/self.pixels if self.pixels else None,
                "brier": self.brier_sum/self.pixels if self.pixels else None}


def _macro(rows):
    keys = ("average_precision", "dice", "iou", "precision", "recall")
    return {k: {"mean": float(np.mean([r[k] for r in rows if r[k] is not None]))
                if any(r[k] is not None for r in rows) else None,
                "defined": sum(r[k] is not None for r in rows)} for k in keys}


@torch.inference_mode()
def evaluate(model, store, config, device, directory, *, threshold=None, criterion=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    thresholds = sorted(set(config["thresholds"])) if threshold is None else [threshold]
    global_ap = ExternalAveragePrecision(directory/"global_ap")
    ignition_ap = ExternalAveragePrecision(directory/"ignition_ap")
    pooled, ignition = Counts(thresholds), Counts(thresholds)
    events, windows = {}, []
    model.eval()
    timings = {"read_seconds": 0., "forward_seconds": 0., "scoring_seconds": 0.}
    loss_sum = loss_pixels = 0
    for i, w in enumerate(store.windows):
        event_id = w.event_id
        if event_id not in events:
            events[event_id] = (Counts(thresholds), ExternalAveragePrecision(directory/f"event_{len(events):04d}"))
        event_counts, event_ap = events[event_id]
        window_counts = Counts(thresholds)
        window_ap = ExternalAveragePrecision(directory/f"window_{i:06d}")
        coordinates = tile_coordinates(*store.shapes[i], config["patch_size"])
        for offset in range(0, len(coordinates), config["eval_batch_size"]):
            t = time.perf_counter()
            coords = coordinates[offset:offset+config["eval_batch_size"]]
            samples = [store.tile(i, r, c, config["patch_size"]) for r, c in coords]
            x, y, valid, observed = [np.stack(a) for a in zip(*samples)]
            timings["read_seconds"] += time.perf_counter()-t
            t = time.perf_counter()
            inputs = torch.from_numpy(x).to(device)
            # FP32 evaluation is fixed across train AMP settings.
            probability = model.predict_proba(inputs).float().cpu().numpy()
            if criterion is not None:
                logits = model(inputs)
                n = float(valid.sum())
                loss_sum += float(criterion(logits, torch.from_numpy(y).to(device),
                                            torch.from_numpy(valid).to(device)).cpu())*n
                loss_pixels += n
            timings["forward_seconds"] += time.perf_counter()-t
            t = time.perf_counter()
            mask = valid.astype(bool)
            p, target = probability[mask], y[mask]
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise RuntimeError("Nonfinite/out-of-range evaluation probability")
            global_ap.update(p, target)
            event_ap.update(p, target)
            window_ap.update(p, target)
            for counts in (pooled, event_counts, window_counts):
                counts.update(p, target)
            candidate = mask & (observed == 0)
            ignition_ap.update(probability[candidate], y[candidate])
            ignition.update(probability[candidate], y[candidate])
            if i == 0 and offset == 0:
                np.savez_compressed(directory/"example_tile.npz", probability=probability[0],
                                    target=y[0], valid=valid[0], observed=observed[0])
            timings["scoring_seconds"] += time.perf_counter()-t
        t = time.perf_counter()
        windows.append({"event_id": event_id, "target_date": w.target_date,
                        "average_precision": window_ap.compute(), "counts": window_counts})
        timings["scoring_seconds"] += time.perf_counter()-t
    if not pooled.pixels:
        raise ValueError("No valid evaluation pixels")
    if threshold is None and not pooled.positive:
        raise ValueError("Validation has no positives; checkpoint/threshold cannot be selected")
    # Highest threshold wins a Dice tie, fixed before looking at test.
    best = max(range(len(thresholds)), key=lambda j: (pooled.at(j)["dice"], thresholds[j]))
    curve = [{"threshold": value, **pooled.at(j)} for j, value in enumerate(thresholds)]
    t = time.perf_counter()
    micro = {**pooled.at(best), "average_precision": global_ap.compute()}
    ignition_metrics = {**ignition.at(best), "average_precision": ignition_ap.compute()}
    per_event = {event: {**counts.at(best), "average_precision": ap.compute()}
                 for event, (counts, ap) in events.items()}
    per_window = [{k: v for k, v in w.items() if k != "counts"} | w["counts"].at(best) for w in windows]
    timings["scoring_seconds"] += time.perf_counter()-t
    return {"threshold": thresholds[best], "threshold_at_grid_boundary": best in (0, len(thresholds)-1),
            "threshold_curve": curve, "micro": micro, "new_ignitions": ignition_metrics,
            "per_event": per_event, "per_window": per_window,
            "macro_event": _macro(list(per_event.values())),
            "macro_positive_event": _macro([r for r in per_event.values() if r["positive_pixels"]]),
            "macro_window": _macro(per_window),
            "macro_positive_window": _macro([r for r in per_window if r["positive_pixels"]]),
            "validation_loss": loss_sum/loss_pixels if loss_pixels else None,
            "timings": timings}
