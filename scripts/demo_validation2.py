"""GeoFire-X — Validation 2 live demo.

Walks through the real pipeline end to end for an oral presentation: dataset
version, manifest stats, one real event, a multi-channel observation, a
temporal window, X/Y tensor shapes, the Persistence baseline, metrics, and
where to find the figures/report. Read-only: it never downloads data, never
overwrites previous results, and never trains anything.

Usage (from the repository root):
    python scripts/demo_validation2.py

No internet connection is required - everything here reads files already on
disk. If the real 5-event sample is not present (it is gitignored, so a fresh
clone never has it), the script explains exactly how to fetch it and falls
back to a clearly-labeled synthetic-data walkthrough instead of crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def run_real_demo() -> None:
    import csv

    from geofirex.real_constants import RAW_ACTIVE_FIRE_CHANNEL, RAW_BAND_NAMES
    from geofirex.real_dataset import RealGeoFireXDataset
    from geofirex.real_splits import is_sample_too_small_for_science, make_smoke_test_split
    from geofirex.metrics import all_metrics

    MANIFEST = Path("data/manifests/real/manifest_wildfirespreadts.csv")
    SPLIT = Path("splits/real/smoke_test_split.csv")

    section("1. Dataset version (real WildfireSpreadTS)")
    print("Source  : Zenodo record 8006177 (DOI 10.5281/zenodo.8006177)")
    print("Version : v1.0, published 2023-10-25")
    print("Official size: 48,359,369,821 bytes (48.36 GB) - NOT downloaded in this project")
    print("See docs/DATASET_VERSION.md and reports/DATA_AUDIT.md for full verification details.")

    section("2. Manifest statistics")
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    event_ids = sorted({r["event_id"] for r in rows})
    print(f"Manifest: {MANIFEST}")
    print(f"  {len(rows)} daily observations across {len(event_ids)} real event(s)")
    for eid in event_ids:
        n = sum(1 for r in rows if r["event_id"] == eid)
        print(f"    - {eid}: {n} days")

    section("3. One real event in detail")
    example_event = event_ids[0]
    example_rows = sorted((r for r in rows if r["event_id"] == example_event), key=lambda r: r["date"])
    print(f"Event: {example_event}")
    print(f"  Dates: {example_rows[0]['date']} -> {example_rows[-1]['date']} ({len(example_rows)} days)")
    print(f"  Image size: {example_rows[0]['height']}x{example_rows[0]['width']} px")
    print(f"  CRS: {example_rows[0]['crs']}")

    section("4. One multi-channel observation")
    print(f"Each observation has {len(RAW_BAND_NAMES)} raw bands (official order):")
    for i, name in enumerate(RAW_BAND_NAMES):
        marker = "  <- target source (binarized via presence, not magnitude)" if i == RAW_ACTIVE_FIRE_CHANNEL else ""
        print(f"  [{i:2d}] {name}{marker}")

    section("5. Building one temporal window (context_days=1)")
    print("Smoke-test split (seed=42, NOT a scientific split - see docs/SPLIT_STRATEGY.md):")
    if SPLIT.exists():
        split_path = SPLIT  # use the already-committed split as-is - never overwritten
        import csv as _csv

        with SPLIT.open(encoding="utf-8") as f:
            assignment = {row["event_id"]: row["split"] for row in _csv.DictReader(f)}
    else:
        # Split file absent (e.g. a fresh checkout of this branch before the
        # file is committed): compute it in memory and write it to a SCRATCH
        # path outside any Git-tracked directory, never into splits/real/.
        from geofirex.real_splits import save_split_csv

        assignment = make_smoke_test_split(event_ids, seed=42)
        scratch_dir = Path(".demo_scratch")
        scratch_dir.mkdir(exist_ok=True)
        split_path = scratch_dir / "smoke_test_split.csv"
        save_split_csv(assignment, split_path)
        print(f"  (committed split not found - wrote a scratch copy to {split_path}, "
              f"not tracked by Git)")
    for eid in event_ids:
        print(f"  {eid}: {assignment[eid]}")

    ds = RealGeoFireXDataset(MANIFEST, split_path, split_name="train", context_days=1)
    print(f"\n{ds.window_report()}")

    if len(ds) == 0:
        print("No window available in the 'train' split - stopping the live walkthrough here.")
        return

    x, y = ds[0]
    w = ds.crs_for(0)
    section("6. X tensor shape and meaning")
    print(f"X shape: {tuple(x.shape)}  =  [context_days, channels, height, width]")
    print(f"  context date(s): {w.context_dates}")
    print(f"  event: {w.event_id}")

    section("7. Y target: dimension and meaning")
    print(f"Y shape: {tuple(y.shape)}  =  [1, height, width]")
    print(f"  target date: {w.target_date}  (strictly the day AFTER the last context day)")
    print(f"  binary active-fire mask, {int(y.sum().item())} positive pixel(s) out of {y.numel()}")

    section("8. Persistence baseline (no neural network)")
    pred = x[-1:, RAW_ACTIVE_FIRE_CHANNEL]
    print("Prediction = last context day's binarized active-fire channel (unchanged).")

    section("9. Metrics on this single window")
    m = all_metrics(pred, y, include_pr_auc=True)
    for k, v in m.items():
        print(f"  {k}: {v if v is None else f'{v:.4f}'}")
    print("\n(For the full-sample aggregate - 67 windows, 3 events - see "
          "reports/real_baseline_results.json and reports/VALIDATION_2.md §F, "
          "which also explains the aggregation method and its limits.)")

    too_small = is_sample_too_small_for_science(event_ids)
    if too_small:
        print(f"\nNOTE: {len(event_ids)} event(s) available - below the conservative floor "
              "for a scientific evaluation. This demo is a technical walkthrough, not a benchmark.")

    section("10. Where to find the full results")
    print("Report          : reports/VALIDATION_2.md")
    print("Technical report: reports/REAL_PIPELINE_REPORT.md")
    print("Data audit      : reports/DATA_AUDIT.md")
    print("Figures         : reports/figures/data_audit/, reports/figures/real_baseline/, "
          "reports/figures/validation2/")
    print("MLflow (if run) : mlflow ui --backend-store-uri sqlite:///mlflow.db")


def run_synthetic_fallback_demo() -> None:
    from geofirex.constants import NUM_CHANNELS
    from geofirex.dataset import GeoFireXDataset
    from geofirex.baseline import persistence_predict
    from geofirex.metrics import all_metrics

    MANIFEST = Path("data/manifests/demo/manifest_synthetic.csv")
    SPLIT = Path("splits/demo/event_split_synthetic.csv")

    section("SYNTHETIC-DATA MODE (real sample not found on disk)")
    print("This is the Phase 1 synthetic pipeline (40 post-preprocessing channels),")
    print("used ONLY to demonstrate that the software works end to end.")
    print("These numbers are NOT WildfireSpreadTS data and must never be presented as such.\n")

    if not (MANIFEST.exists() and SPLIT.exists()):
        print("Synthetic demo fixtures are also missing - nothing to demonstrate.")
        print("Run: python scripts/generate_synthetic.py --data_dir data/raw/synthetic "
              "&& python scripts/run_baseline.py --config configs/baseline.yaml")
        return

    ds = GeoFireXDataset(MANIFEST, SPLIT, split_name="train", context_days=1)
    if len(ds) == 0:
        print("Synthetic 'train' split has zero windows - nothing to demonstrate.")
        return
    x, y = ds[0]
    print(f"X shape: {tuple(x.shape)} (synthetic, {NUM_CHANNELS} channels)")
    print(f"Y shape: {tuple(y.shape)}")
    pred = persistence_predict(x.unsqueeze(0))[0]
    m = all_metrics(pred, y)
    print("Persistence metrics on this synthetic sample:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")


def main() -> None:
    real_manifest = Path("data/manifests/real/manifest_wildfirespreadts.csv")
    real_sample_dir = Path("data/raw/WildfireSpreadTS/real_sample")

    print("GeoFire-X — Validation 2 demonstration")
    print("Read-only - no download, no training, no overwriting of existing results.")

    if real_manifest.exists() and real_sample_dir.exists() and any(real_sample_dir.rglob("*.tif")):
        try:
            run_real_demo()
        except Exception as e:  # noqa: BLE001 - never show a raw traceback in a live demo
            print(f"\nERROR while running the real-data demo: {e}")
            print("Falling back to the synthetic-data walkthrough below.\n")
            run_synthetic_fallback_demo()
            sys.exit(1)
    else:
        section("Real WildfireSpreadTS sample not found locally")
        print("Expected manifest : data/manifests/real/manifest_wildfirespreadts.csv")
        print("Expected raw files: data/raw/WildfireSpreadTS/real_sample/ (gitignored, ~440 MB)")
        print("\nTo fetch it (deterministic, seed=42, no full 48.36 GB download):")
        print("  python scripts/inspect_remote_archive.py --out data/raw/WildfireSpreadTS/archive_index.json")
        print("  python scripts/download_real_sample.py --dest data/raw/WildfireSpreadTS/real_sample --num_events 5 --seed 42")
        print("  python scripts/build_real_manifest.py --data_dir data/raw/WildfireSpreadTS/real_sample --out data/manifests/real/manifest_wildfirespreadts.csv")
        print("\nRunning the clearly-separate SYNTHETIC-DATA demo instead:")
        run_synthetic_fallback_demo()


if __name__ == "__main__":
    main()
