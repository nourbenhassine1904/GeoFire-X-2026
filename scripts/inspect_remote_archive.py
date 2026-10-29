"""List the real structure of the official WildfireSpreadTS.zip WITHOUT
downloading it, by reading its central directory over HTTP Range requests.

Usage:
    python scripts/inspect_remote_archive.py --out data/raw/WildfireSpreadTS/archive_index.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from geofirex.remote_archive import list_real_events, open_remote_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="where to save the JSON index")
    args = parser.parse_args()

    zf, f = open_remote_zip()
    all_names = zf.namelist()
    events = list_real_events(zf)

    years = Counter(eid.split("/")[0] for eid in events)
    days_per_event = {eid: len(days) for eid, days in events.items()}
    total_days = sum(days_per_event.values())

    index = {
        "source_size_bytes": f.size,
        "total_zip_entries": len(all_names),
        "total_events": len(events),
        "total_daily_tif_files": total_days,
        "events_per_year": dict(sorted(years.items())),
        "days_per_event": days_per_event,
        "bytes_fetched_for_this_inspection": f.bytes_fetched,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"Remote archive size: {f.size:,} bytes ({f.size / 1e9:.2f} GB)")
    print(f"Total zip entries (incl. folders): {len(all_names):,}")
    print(f"Real events found: {len(events)}")
    print(f"Real daily .tif files found: {total_days:,}")
    print(f"Events per year: {dict(sorted(years.items()))}")
    print(f"Bytes actually fetched for this inspection: {f.bytes_fetched:,} "
          f"({f.bytes_fetched / 1e6:.2f} MB) - central directory only, no .tif content")
    print(f"Index saved to {out_path}")


if __name__ == "__main__":
    main()
