"""Event-level train/val/test split — the Critical Experimental Rule from
docs/PROJECT_CONTEXT.md: a wildfire event belongs to exactly one split, so no day
from a training event may appear in validation or test.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


def split_events(
    event_ids: list[str],
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, str]:
    """Assign each event to exactly one of train/val/test."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    ordered = sorted(set(event_ids))  # sorted first so shuffling is reproducible across platforms
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])

    assignment = {}
    for i, eid in enumerate(ordered):
        if i < n_train:
            assignment[eid] = "train"
        elif i < n_train + n_val:
            assignment[eid] = "val"
        else:
            assignment[eid] = "test"
    return assignment


def save_split(assignment: dict[str, str], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "split"])
        for eid, split in sorted(assignment.items()):
            writer.writerow([eid, split])


def load_split(path: Path) -> dict[str, str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return {row["event_id"]: row["split"] for row in csv.DictReader(f)}


def check_no_leakage(assignment: dict[str, str]) -> None:
    """Each event_id key is unique by construction; this guards against callers
    merging two assignments for the same event into conflicting splits."""
    seen: dict[str, str] = {}
    for eid, split in assignment.items():
        if eid in seen and seen[eid] != split:
            raise AssertionError(f"Leakage: event {eid} assigned to both {seen[eid]} and {split}")
        seen[eid] = split
