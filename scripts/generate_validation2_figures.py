"""Generate the 4 NEW figures needed for Validation 2 that are not already
produced by Phase 2 (reports/figures/data_audit/) or Phase 3
(reports/figures/real_baseline/): architecture diagram, window construction
diagram, label distribution (computed from the real sample), and the future
roadmap diagram. Pure diagrams/computed stats - no new data download, no
training.

Usage:
    python scripts/generate_validation2_figures.py --out_dir reports/figures/validation2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def fig1_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color="#5b8def", fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, color="white", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4))

    # Real-data pipeline (top row)
    box(0.3, 6.0, 2.0, 1.0, "Zenodo archive\n(Range requests)", "#5b8def")
    box(2.7, 6.0, 1.8, 1.0, "real_manifest.py\n(23-band manifest)", "#5b8def")
    box(4.9, 6.0, 1.8, 1.0, "windows.py\n(J+1 windows)", "#5b8def")
    box(7.1, 6.0, 1.8, 1.0, "real_dataset.py\nX:[T,23,H,W]", "#5b8def")
    box(9.3, 6.0, 2.2, 1.0, "run_real_persistence\n_baseline.py", "#2e7d32")
    arrow(2.3, 6.5, 2.7, 6.5)
    arrow(4.5, 6.5, 4.9, 6.5)
    arrow(6.7, 6.5, 7.1, 6.5)
    arrow(8.9, 6.5, 9.3, 6.5)

    box(9.3, 4.4, 2.2, 1.0, "metrics.py\nDice/IoU/PR/Recall", "#2e7d32")
    box(9.3, 2.8, 2.2, 1.0, "MLflow\n(mlflow.db)", "#8e44ad")
    box(5.8, 2.8, 2.2, 1.0, "reports/figures/\n*.png, *.md", "#8e44ad")
    arrow(10.4, 6.0, 10.4, 5.4)
    arrow(10.4, 4.4, 10.4, 3.8)
    arrow(9.3, 3.3, 8.0, 3.3)

    # Synthetic pipeline (bottom row, independent)
    box(0.3, 0.3, 2.0, 1.0, "synthetic.py\n(generated GeoTIFF)", "#e67e22")
    box(2.7, 0.3, 1.8, 1.0, "manifest.py /\nsplits.py", "#e67e22")
    box(4.9, 0.3, 1.8, 1.0, "dataset.py\nX:[T,40,H,W]", "#e67e22")
    box(7.1, 0.3, 1.8, 1.0, "baseline.py\n(Persistence)", "#e67e22")
    arrow(2.3, 0.8, 2.7, 0.8)
    arrow(4.5, 0.8, 4.9, 0.8)
    arrow(6.7, 0.8, 7.1, 0.8)

    ax.text(0.3, 7.4, "Real data pipeline (23 raw bands, Phase 2-3)", fontsize=11, fontweight="bold")
    ax.text(0.3, 1.6, "Synthetic pipeline (40 channels, Phase 1 — kept independent, used for CI/tests)",
            fontsize=11, fontweight="bold", color="#8a4b17")

    fig.suptitle("Figure 1 — GeoFire-X Software Architecture (implemented components)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_window_construction(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    days = ["J-2", "J-1", "J"]
    colors = ["#5b8def", "#5b8def", "#5b8def"]
    for i, (d, c) in enumerate(zip(days, colors)):
        x = 0.5 + i * 1.8
        rect = mpatches.FancyBboxPatch((x, 1.6), 1.4, 1.4, boxstyle="round,pad=0.06",
                                        edgecolor="#333", facecolor=c)
        ax.add_patch(rect)
        ax.text(x + 0.7, 2.3, f"{d}\n23 bands", ha="center", va="center", color="white", fontsize=9)

    brace_x0, brace_x1 = 0.5, 0.5 + 3 * 1.8 - 0.4
    ax.annotate("", xy=(brace_x1, 1.3), xytext=(brace_x0, 1.3),
                arrowprops=dict(arrowstyle="-", color="#333"))
    ax.text((brace_x0 + brace_x1) / 2, 1.0, "X : [T=3, 23, H, W]  (context_days)", ha="center", fontsize=10)

    arrow_x = 0.5 + 3 * 1.8 + 0.3
    ax.annotate("", xy=(arrow_x + 1.3, 2.3), xytext=(brace_x1 + 0.4, 2.3),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.6))

    rect = mpatches.FancyBboxPatch((arrow_x + 1.3, 1.6), 1.4, 1.4, boxstyle="round,pad=0.06",
                                    edgecolor="#333", facecolor="#c0392b")
    ax.add_patch(rect)
    ax.text(arrow_x + 1.3 + 0.7, 2.3, "J+1\nactive fire", ha="center", va="center", color="white", fontsize=9)
    ax.text(arrow_x + 1.3 + 0.7, 1.0, "Y : [1, H, W]\n(binarized via ~isnan)", ha="center", fontsize=10)

    ax.text(5, 3.6, "Window accepted only if all dates are strictly consecutive\nand belong to the same event",
            ha="center", fontsize=9, style="italic", color="#555")

    fig.suptitle("Figure 4 — J+1 window construction (example: context_days=3)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_label_distribution(out_path: Path) -> None:
    import csv

    from geofirex.real_constants import RAW_ACTIVE_FIRE_CHANNEL
    from geofirex.real_dataset import RealGeoFireXDataset
    from geofirex.real_splits import make_smoke_test_split, save_split_csv

    manifest = Path("data/manifests/real/manifest_wildfirespreadts.csv")
    if not manifest.exists():
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "Real sample not available locally - figure not generated.",
                ha="center", va="center")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    event_ids = sorted({r["event_id"] for r in rows})
    split_path = out_path.parent / "_tmp_split_for_fig5.csv"
    save_split_csv(make_smoke_test_split(event_ids, seed=42), split_path)

    totals = {"empty_target": 0, "positive_target": 0}
    pixel_pos, pixel_total = 0, 0
    for split_name in ("train", "val", "test"):
        ds = RealGeoFireXDataset(manifest, split_path, split_name=split_name, context_days=1)
        for i in range(len(ds)):
            _, y = ds[i]
            if y.sum().item() > 0:
                totals["positive_target"] += 1
            else:
                totals["empty_target"] += 1
            pixel_pos += int(y.sum().item())
            pixel_total += y.numel()
    split_path.unlink(missing_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Empty target\n(no fire J+1)", "Positive target\n(fire present J+1)"],
                [totals["empty_target"], totals["positive_target"]], color=["#95a5a6", "#c0392b"])
    axes[0].set_ylabel("Number of windows")
    axes[0].set_title("Windows by target content (all 5 events, context_days=1)")
    for i, v in enumerate([totals["empty_target"], totals["positive_target"]]):
        axes[0].text(i, v + 0.5, str(v), ha="center")

    axes[1].bar(["Negative\npixels"], [pixel_total - pixel_pos], color="#95a5a6")
    axes[1].bar(["Positive\npixels"], [pixel_pos], color="#c0392b")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Pixel count (log scale)")
    axes[1].set_title(f"Pixel-level imbalance ({100*pixel_pos/pixel_total:.3f}% positive)")

    fig.suptitle("Figure 5 — Fire / no-fire label distribution (real sample, all splits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_roadmap(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_xlim(0, 15.3)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("Data audit\n& real sample", True),
        ("Real DataLoader\n& windows", True),
        ("Persistence\nbaseline", True),
        ("Extend sample\n(>=20 events)", False),
        ("Year-based\ncross-validation", False),
        ("U-Net 2D\nbaseline", False),
        ("ConvLSTM /\ntemporal model", False),
        ("Multimodal\nfusion", False),
        ("ID / OOD\nevaluation", False),
    ]
    for i, (label, done) in enumerate(steps):
        x = 0.3 + i * 1.65
        color = "#2e7d32" if done else "#bdbdbd"
        rect = mpatches.FancyBboxPatch((x, 1.0), 1.15, 1.0, boxstyle="round,pad=0.05",
                                        edgecolor="#333", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.57, 1.5, label, ha="center", va="center", fontsize=8,
                 color="white" if done else "#333")
        if i > 0:
            ax.annotate("", xy=(x, 1.5), xytext=(x - 0.5, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#333"))

    ax.legend(handles=[mpatches.Patch(color="#2e7d32", label="Implemented (this validation)"),
                        mpatches.Patch(color="#bdbdbd", label="Planned (future work)")],
              loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Figure 7 — Roadmap to final defense: implemented vs. planned", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_architecture(out_dir / "figure1_architecture.png")
    print("Figure 1 (architecture) written")
    fig4_window_construction(out_dir / "figure4_window_construction.png")
    print("Figure 4 (window construction) written")
    fig5_label_distribution(out_dir / "figure5_label_distribution.png")
    print("Figure 5 (label distribution) written")
    fig7_roadmap(out_dir / "figure7_roadmap.png")
    print("Figure 7 (roadmap) written")


if __name__ == "__main__":
    main()
