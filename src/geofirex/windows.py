"""Build J+1 training windows from a real-data manifest (geofirex.real_manifest).

A window is `context_days` consecutive daily observations (X) plus the very
next day's observation (used to derive the target Y). Windows are built
per-event (never across two different fires) and are rejected - with a
recorded reason - whenever the temporal or spatial conditions required for a
valid J+1 example are not met. Nothing here reads pixel data; it only reasons
about the manifest rows, so it is cheap to call even on large manifests.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    event_id: str
    context_dates: tuple[str, ...]  # context_days dates, chronological
    context_paths: tuple[str, ...]
    target_date: str
    target_path: str


REASON_EVENT_TOO_SHORT = "event_too_short"
REASON_DATE_GAP = "date_gap"
REASON_SHAPE_MISMATCH = "shape_mismatch"


def build_windows(
    manifest_rows: list[dict], context_days: int
) -> tuple[list[Window], dict[str, int]]:
    """Return (windows, rejection_counts). rejection_counts maps a reason
    string to the number of candidate windows rejected for that reason.

    A candidate window of `context_days` input dates + 1 target date is
    accepted only if:
      - all context_days + 1 dates belong to the same event;
      - the dates are real, strictly consecutive calendar days (no gap);
      - every file in the window has the same (height, width) - defends
        against the rare malformed/mismatched file, since different events
        can legitimately have different shapes but a single event must not.
    """
    if context_days < 1:
        raise ValueError("context_days must be >= 1")

    by_event: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        by_event[row["event_id"]].append(row)

    windows: list[Window] = []
    rejections: Counter[str] = Counter()

    for event_id, rows in by_event.items():
        rows_sorted = sorted(rows, key=lambda r: r["date"])
        n = len(rows_sorted)
        span = context_days + 1  # context days + 1 target day

        if n < span:
            rejections[REASON_EVENT_TOO_SHORT] += 1
            continue

        for i in range(n - span + 1):
            group = rows_sorted[i : i + span]
            dates = [r["date"] for r in group]

            try:
                parsed = [dt.date.fromisoformat(d) for d in dates]
            except ValueError:
                rejections[REASON_DATE_GAP] += 1  # unparseable date treated as a gap
                continue

            consecutive = all(
                (b - a).days == 1 for a, b in zip(parsed, parsed[1:])
            )
            if not consecutive:
                rejections[REASON_DATE_GAP] += 1
                continue

            shapes = {(r["height"], r["width"]) for r in group}
            if len(shapes) != 1:
                rejections[REASON_SHAPE_MISMATCH] += 1
                continue

            windows.append(
                Window(
                    event_id=event_id,
                    context_dates=tuple(dates[:-1]),
                    context_paths=tuple(r["relative_path"] for r in group[:-1]),
                    target_date=dates[-1],
                    target_path=group[-1]["relative_path"],
                )
            )

    return windows, dict(rejections)


def format_window_report(
    windows: list[Window], rejections: dict[str, int], context_days: int
) -> str:
    total_rejected = sum(rejections.values())
    lines = [
        f"Window report (context_days={context_days}):",
        f"  created:  {len(windows)}",
        f"  rejected: {total_rejected}",
    ]
    for reason, count in sorted(rejections.items()):
        lines.append(f"    - {reason}: {count}")
    return "\n".join(lines)
