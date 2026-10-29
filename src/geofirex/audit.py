"""Audit a WildfireSpreadTS-layout dataset: shape, channel count, NaNs, date gaps."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from geofirex.constants import NUM_CHANNELS
from geofirex.io import list_event_days, list_events, read_geotiff


@dataclass
class EventAuditReport:
    event_dir: Path
    num_days: int
    height: int
    width: int
    channel_count_ok: bool
    nan_count: int
    value_min: float
    value_max: float
    missing_date_gaps: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.channel_count_ok and self.nan_count == 0 and not self.missing_date_gaps


def _find_date_gaps(dates: list[str]) -> list[tuple[str, str]]:
    """Return consecutive (prev, next) date pairs that are not exactly 1 day apart."""
    gaps = []
    for prev_s, next_s in zip(dates, dates[1:]):
        try:
            prev_d = dt.date.fromisoformat(prev_s)
            next_d = dt.date.fromisoformat(next_s)
        except ValueError:
            continue  # filename isn't a date; skip gap detection for this pair
        if (next_d - prev_d).days != 1:
            gaps.append((prev_s, next_s))
    return gaps


def audit_event(event_dir: Path, expected_channels: int = NUM_CHANNELS) -> EventAuditReport:
    days = list_event_days(event_dir)
    if not days:
        raise FileNotFoundError(f"No .tif files found under {event_dir}")

    arrays = [read_geotiff(d.path) for d in days]
    channel_counts = {a.shape[0] for a in arrays}
    shapes = {a.shape[1:] for a in arrays}
    channel_count_ok = channel_counts == {expected_channels}

    nan_count = sum(int((a != a).sum()) for a in arrays)  # NaN != NaN
    value_min = min(float(a.min()) for a in arrays)
    value_max = max(float(a.max()) for a in arrays)

    height, width = next(iter(shapes)) if len(shapes) == 1 else (arrays[0].shape[1], arrays[0].shape[2])

    return EventAuditReport(
        event_dir=event_dir,
        num_days=len(days),
        height=height,
        width=width,
        channel_count_ok=channel_count_ok,
        nan_count=nan_count,
        value_min=value_min,
        value_max=value_max,
        missing_date_gaps=_find_date_gaps([d.date for d in days]),
    )


def audit_dataset(data_dir: Path, expected_channels: int = NUM_CHANNELS) -> list[EventAuditReport]:
    return [audit_event(e, expected_channels) for e in list_events(data_dir)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    reports = audit_dataset(Path(args.data_dir))
    for r in reports:
        status = "OK" if r.ok else "ISSUES"
        print(
            f"[{status}] {r.event_dir.name}: {r.num_days} days, "
            f"{r.height}x{r.width}, channels_ok={r.channel_count_ok}, "
            f"nan={r.nan_count}, range=[{r.value_min:.3f}, {r.value_max:.3f}], "
            f"gaps={r.missing_date_gaps}"
        )
    n_bad = sum(1 for r in reports if not r.ok)
    print(f"\n{len(reports)} event(s) audited, {n_bad} with issues.")


if __name__ == "__main__":
    main()
