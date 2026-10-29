"""Event-level split strategies for the REAL WildfireSpreadTS data.

Two distinct things live here, and they must never be confused:

1. `split_events_by_year` - the officially recommended protocol (datasheet:
   "we recommend to evaluate models using a cross-validation across years,
   since the data distributions vary a lot between years"). This is the
   scientifically defensible strategy once enough events are available.

2. `make_smoke_test_split` - for the current 5-event real sample, which is
   far too small for any statistically meaningful train/val/test split or
   year-based cross-validation (2021 is not even represented). This produces
   a clearly-labeled smoke-test split whose only purpose is to exercise the
   pipeline code end-to-end - it must never be reported as a generalization
   or OOD evaluation. See docs/SPLIT_STRATEGY.md.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from geofirex.splits import check_no_leakage

MIN_EVENTS_FOR_SCIENTIFIC_SPLIT = 20  # a conservative floor, not a guarantee of
# sufficiency - documented judgment call, see docs/SPLIT_STRATEGY.md.


def split_events_by_year(event_years: dict[str, str]) -> dict[str, dict[str, str]]:
    """Build one leave-one-year-out fold per year present in `event_years`
    (event_id -> year as string). Returns {held_out_year: {event_id: "train"|"test"}}.
    All events from the held-out year are "test", everything else is "train".
    No "val" fold is carved out here - callers needing one should split the
    remaining years further themselves.
    """
    years = sorted(set(event_years.values()))
    folds: dict[str, dict[str, str]] = {}
    for held_out in years:
        assignment = {
            eid: ("test" if y == held_out else "train") for eid, y in event_years.items()
        }
        check_no_leakage(assignment)
        folds[held_out] = assignment
    return folds


def make_smoke_test_split(event_ids: list[str], seed: int = 42) -> dict[str, str]:
    """A deterministic, leak-free assignment over too few events to be a real
    evaluation. With <3 events every role still gets at least an attempt; with
    the current 5-event sample this yields a 3/1/1 train/val/test split that
    exists ONLY to exercise the pipeline, not to measure generalization.
    """
    import random

    ordered = sorted(set(event_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    assignment: dict[str, str] = {}
    n = len(ordered)
    if n == 0:
        return assignment
    if n == 1:
        assignment[ordered[0]] = "train"
        return assignment
    if n == 2:
        assignment[ordered[0]] = "train"
        assignment[ordered[1]] = "test"
        return assignment

    n_test = 1
    n_val = 1 if n >= 4 else 0
    n_train = n - n_test - n_val
    for i, eid in enumerate(ordered):
        if i < n_train:
            assignment[eid] = "train"
        elif i < n_train + n_val:
            assignment[eid] = "val"
        else:
            assignment[eid] = "test"
    check_no_leakage(assignment)
    return assignment


def is_sample_too_small_for_science(event_ids: list[str]) -> bool:
    return len(set(event_ids)) < MIN_EVENTS_FOR_SCIENTIFIC_SPLIT


def save_split_csv(assignment: dict[str, str], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "split"])
        for eid, split in sorted(assignment.items()):
            writer.writerow([eid, split])


def distribution_report(event_years: dict[str, str], assignment: dict[str, str]) -> str:
    by_year_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for eid, split in assignment.items():
        year = event_years.get(eid, "unknown")
        by_year_split[year][split] += 1

    lines = ["Distribution by year and split:"]
    for year in sorted(by_year_split):
        parts = ", ".join(f"{s}={n}" for s, n in sorted(by_year_split[year].items()))
        lines.append(f"  {year}: {parts}")
    lines.append(f"Total events: {len(assignment)}")
    return "\n".join(lines)
