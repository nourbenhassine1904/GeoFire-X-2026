"""Progressively cache explicitly selected, complete local events; no network."""
import argparse
import csv
import json
from pathlib import Path

from geofirex.event_cache import build_event_cache

ROOT=Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--index",required=True)
    p.add_argument("--events",nargs="+",required=True)
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    with Path(args.manifest).open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    index=json.loads(Path(args.index).read_text(encoding="utf-8"))
    for event in args.events:
        selected=[r for r in rows if r["event_id"]==event]
        dates=[e["date"] for e in index["entries"] if e["event_id"]==event]
        if not dates:
            raise ValueError(f"Unknown event in pinned index: {event}")
        metadata=build_event_cache(selected,ROOT,Path(args.out_dir)/event,expected_dates=dates)
        print(event,metadata["shape"],metadata["array_bytes"],flush=True)
