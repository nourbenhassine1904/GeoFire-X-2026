"""Extract a small, deterministic sample of REAL WildfireSpreadTS events from
the official Zenodo archive, without downloading the full 48.36 GB file.

Only the compressed bytes of the chosen events' .tif files are fetched (a real
event is roughly 20-90 MB for a handful of events total - see the printed
report). This is a deliberate choice to get genuine, verified data for the
Phase 2 audit while respecting the project rule against downloading tens of
GB without explicit team agreement.

Usage:
    python scripts/download_real_sample.py \\
        --dest data/raw/WildfireSpreadTS/real_sample \\
        --num_events 5 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from geofirex.remote_archive import (
    extract_events,
    list_real_events,
    missing_files_for_event,
    open_remote_zip,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--num_events", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zf, f = open_remote_zip()
    events = list_real_events(zf)
    print(f"Real events available in the official archive: {len(events)}")

    ordered_ids = sorted(events.keys())  # sort first so sampling is reproducible
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(ordered_ids, k=min(args.num_events, len(ordered_ids))))
    print(f"Selected {len(chosen)} event(s) with seed={args.seed}: {chosen}")

    # An event is only "complete" when every expected filename is present on
    # disk (non-empty) - a single leftover .tif from a prior interrupted run
    # must never be mistaken for a fully extracted event.
    missing_by_event = {c: missing_files_for_event(dest_dir, c, events[c]) for c in chosen}
    complete = [c for c in chosen if not missing_by_event[c]]
    incomplete_or_missing = [c for c in chosen if missing_by_event[c]]
    if complete:
        print(f"Already fully extracted, skipping: {complete}")
    for c in incomplete_or_missing:
        n_missing = len(missing_by_event[c])
        if n_missing < len(events[c]):
            print(f"Resuming incomplete event {c}: {n_missing}/{len(events[c])} file(s) still missing")

    written = extract_events(zf, incomplete_or_missing, dest_dir) if incomplete_or_missing else []

    print(f"Wrote {len(written)} real .tif file(s) to {dest_dir}")
    print(f"Bytes actually fetched from Zenodo this run: {f.bytes_fetched:,} "
          f"({f.bytes_fetched / 1e6:.2f} MB)")
    print("No part of the full 48.36 GB archive was downloaded.")


if __name__ == "__main__":
    main()
