"""Build a manifest (event -> day files) from a WildfireSpreadTS-layout dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from geofirex.io import list_event_days, list_events


def event_id_for(event_dir: Path) -> str:
    """Stable event identifier: <year>/<fire_name>, matching the on-disk layout."""
    return f"{event_dir.parent.name}/{event_dir.name}"


def build_manifest(data_dir: Path) -> list[dict]:
    """Return one row per day file: event_id, year, fire_name, date, path."""
    rows = []
    for event_dir in list_events(data_dir):
        eid = event_id_for(event_dir)
        for day in list_event_days(event_dir):
            rows.append(
                {
                    "event_id": eid,
                    "year": event_dir.parent.name,
                    "fire_name": event_dir.name,
                    "date": day.date,
                    "path": day.path.as_posix(),  # forward slashes, portable across OSes
                }
            )
    return rows


def save_manifest(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "year", "fire_name", "date", "path"])
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = build_manifest(Path(args.data_dir))
    save_manifest(rows, Path(args.out))
    n_events = len({r["event_id"] for r in rows})
    print(f"Manifest: {len(rows)} day-rows across {n_events} event(s) -> {args.out}")


if __name__ == "__main__":
    main()
