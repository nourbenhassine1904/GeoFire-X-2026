"""Persistence baseline on the REAL WildfireSpreadTS sample: predict tomorrow's
active-fire mask as today's observed mask. No neural network is trained here -
this is a reproducible technical verification of the real DataLoader + metrics
pipeline end to end, not a scientific benchmark (the 5-event sample is far too
small for that - see docs/SPLIT_STRATEGY.md).

Usage:
    python scripts/run_real_persistence_baseline.py \\
        --manifest data/manifests/real/manifest_wildfirespreadts.csv \\
        --split splits/real/smoke_test_split.csv \\
        --split_name train --context_days 1 \\
        --figures_dir reports/figures/real_baseline \\
        --out reports/real_baseline_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from geofirex.metrics import aggregate_metrics_by_event, all_metrics
from geofirex.real_constants import RAW_ACTIVE_FIRE_CHANNEL
from geofirex.real_dataset import RealGeoFireXDataset
from geofirex.real_splits import is_sample_too_small_for_science


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def make_figure(x, y, pred, out_path: Path, title: str) -> None:
    current_fire = x[-1, RAW_ACTIVE_FIRE_CHANNEL].numpy()
    target = y[0].numpy()
    prediction = pred[0].numpy()
    diff = target - prediction  # +1 = missed ignition, -1 = false persistence

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, data, name, cmap, vrange in [
        (axes[0], current_fire, "Current fire (last context day)", "hot", (0, 1)),
        (axes[1], target, "Target (J+1, real)", "hot", (0, 1)),
        (axes[2], prediction, "Persistence prediction", "hot", (0, 1)),
        (axes[3], diff, "Diff (target - prediction)", "coolwarm", (-1, 1)),
    ]:
        im = ax.imshow(data, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(name, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--context_days", type=int, default=1)
    parser.add_argument("--figures_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_figures", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = RealGeoFireXDataset(
        args.manifest, args.split, split_name=args.split_name, context_days=args.context_days
    )
    print(ds.window_report())

    event_ids = sorted({w.event_id for w in ds.windows})
    too_small = is_sample_too_small_for_science(event_ids)
    if too_small:
        print(
            "WARNING: fewer than the conservative floor of events for a scientific "
            "evaluation - the results below are a TECHNICAL VERIFICATION ON SAMPLE, "
            "not a benchmark. Do not report them as generalization performance."
        )

    if len(ds) == 0:
        raise SystemExit(
            f"No valid windows for split_name={args.split_name!r} - nothing to evaluate."
        )

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    start = time.perf_counter()
    for i in range(len(ds)):
        x, y = ds[i]
        pred = x[-1:, RAW_ACTIVE_FIRE_CHANNEL]  # [1,H,W], last context day's fire mask
        metrics = all_metrics(pred, y, include_pr_auc=True)
        w = ds.crs_for(i)
        per_sample.append((w.event_id, metrics))

        if i < args.max_figures:
            make_figure(
                x, y, pred,
                figures_dir / f"sample_{i:02d}_{w.event_id.replace('/', '_')}_{w.target_date}.png",
                f"{w.event_id} -> {w.target_date}",
            )
    elapsed = time.perf_counter() - start

    by_event = aggregate_metrics_by_event(per_sample)
    overall = {}
    for key in ("dice", "iou", "precision", "recall"):
        values = [m[key] for _, m in per_sample]
        overall[key] = sum(values) / len(values)
    pr_auc_values = [m["pr_auc"] for _, m in per_sample if m["pr_auc"] is not None]
    overall["pr_auc"] = (sum(pr_auc_values) / len(pr_auc_values)) if pr_auc_values else None

    report = {
        "qualification": (
            "TECHNICAL VERIFICATION ON SAMPLE - not a scientific benchmark "
            f"({len(event_ids)} event(s))"
            if too_small
            else "evaluation"
        ),
        "split_name": args.split_name,
        "context_days": args.context_days,
        "num_valid_windows": len(ds),
        "num_events": len(event_ids),
        "window_rejections": ds.rejections,
        "elapsed_seconds": elapsed,
        "overall_metrics": overall,
        "per_event_metrics": by_event,
        "git_commit": git_commit(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.set_experiment("geofirex-real-persistence-baseline")
    with mlflow.start_run(run_name="real-persistence-baseline"):
        mlflow.log_param("dataset_version", "WildfireSpreadTS v1.0 (zenodo.8006177), real 5-event sample")
        mlflow.log_param("split_version", "smoke_test (not scientific - see docs/SPLIT_STRATEGY.md)")
        mlflow.log_param("split_name", args.split_name)
        mlflow.log_param("context_days", args.context_days)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("git_commit", report["git_commit"])
        mlflow.log_param("num_events", len(event_ids))
        mlflow.log_param("too_small_for_science", too_small)
        mlflow.log_metric("num_valid_windows", len(ds))
        mlflow.log_metric("elapsed_seconds", elapsed)
        for key, value in overall.items():
            if value is not None:
                mlflow.log_metric(f"overall_{key}", value)
        mlflow.log_artifact(args.out)

    print(f"\n{report['qualification']}")
    print(f"Valid windows: {len(ds)} across {len(event_ids)} event(s)")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Overall metrics:")
    for key, value in overall.items():
        print(f"  {key}: {value if value is None else f'{value:.4f}'}")
    print(f"Per-event metrics written to {args.out}")
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
