"""Preparation contracts: authors' 12 annual folds and aligned T=1/3/5 targets."""
from __future__ import annotations

import datetime as dt
import itertools
from pathlib import Path

from geofirex.windows import build_windows

YEARS = (2018, 2019, 2020, 2021)


def annual_folds():
    # Same ordering as the official FireSpreadDataModule.split_fires.
    folds = []
    for train in itertools.combinations(YEARS, 2):
        remaining = [y for y in YEARS if y not in train]
        for val, test in (remaining, reversed(remaining)):
            folds.append({"fold_id": len(folds), "train_years": list(train),
                          "val_years": [val], "test_years": [test]})
    return folds


def event_assignments(event_ids, fold_id):
    if not 0 <= fold_id < 12:
        raise ValueError("fold_id must be in [0,11]")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Duplicate event identifier")
    fold = annual_folds()[fold_id]
    assignment = {}
    for event in sorted(event_ids):
        year = int(event.split("/")[0])
        roles = [role for role in ("train", "val", "test") if year in fold[f"{role}_years"]]
        if len(roles) != 1:
            raise ValueError(f"Unexpected event year: {event}")
        assignment[event] = roles[0]
    return assignment


def aligned_windows(rows, assignment, role, contexts=(1, 3, 5)):
    """Intersect actual (event,target_date) keys, so gaps/shapes are respected.

    Only pixel-independent manifest metadata is used. No label/fire filtering.
    Use the same target set for all architectures on validation and test.
    """
    if role not in {"train", "val", "test"} or not contexts or min(contexts) < 1:
        raise ValueError("Valid role and positive context lengths required")
    seen_dates, seen_paths = set(), set()
    selected = []
    for row in rows:
        if row["event_id"] not in assignment:
            raise ValueError("Manifest event not assigned to a fold")
        key = (row["event_id"], row["date"])
        canonical = str(Path(row["relative_path"]).resolve())
        if key in seen_dates or canonical in seen_paths:
            raise ValueError("Duplicate observation or reused file path")
        seen_dates.add(key); seen_paths.add(canonical)
        dt.date.fromisoformat(row["date"])
        if assignment[row["event_id"]] == role:
            selected.append(row)
    groups, rejected = {}, {}
    for context in sorted(set(contexts)):
        ws, rejected[context] = build_windows(selected, context)
        groups[context] = {(w.event_id, w.target_date): w for w in ws}
    common = set.intersection(*(set(g) for g in groups.values()))
    keys = sorted(common)
    return {t: [groups[t][key] for key in keys] for t in groups}, rejected
