"""Offline pilot plans: 2018 train, 2020 validation, then 2019 extension."""
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from geofirex.cloud_corpus import plan_batch, validate_index, verify_corpus_package
from geofirex.corpus import json_new


def eligible_events(entries):
    groups = {}
    for e in entries:
        groups.setdefault(e["event_id"], []).append(dt.date.fromisoformat(e["date"]).toordinal())
    eligible = []
    for event, days in groups.items():
        run, previous = 0, None
        for day in sorted(days):
            run = run+1 if previous == day-1 else 1
            previous = day
            if run >= 6:
                eligible.append(event)
                break
    return eligible


def prepare(index, destination, *, private_free_bytes, storage_bytes, roots=(), transfer_bytes=3*1024**3):
    validate_index(index, official=True)
    excluded = {"2018/fire_21458798"}  # user-confirmed manual batch001; planning assertion only
    for root in map(Path, roots):
        inventory = json.loads((root/"inventory.json").read_text(encoding="utf-8"))
        verify_corpus_package(index, root, inventory)
        excluded.update(inventory["events"])
    if storage_bytes < 1024**3 or private_free_bytes < 1:
        raise ValueError("Provide measured available storage/private quota")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    reports, remaining = [], private_free_bytes
    for label, year, maximum, minimum in (("train2018", 2018, 40, 30), ("val2020", 2020, 30, 20),
                                          ("extension2019", 2019, 30, 20)):
        plan = plan_batch(index, years=[year], max_events=maximum, transfer_bytes=transfer_bytes,
                          storage_bytes=storage_bytes, private_free_bytes=remaining,
                          upload_bytes=transfer_bytes, exclude_events=excluded)
        usable = eligible_events(plan["entries"])
        if len(usable) < minimum:
            raise ValueError(f"{label}: only {len(usable)} eligible complete events fit; increase approved quotas")
        json_new(destination/f"{label}.json", plan)
        reports.append({"label": label, "year": year, "events": len(plan["events"]),
                        "eligible_events_by_dates": len(usable), "observations": plan["observations"],
                        "geotiff_bytes": plan["geotiff_bytes"], "transfer_cap": transfer_bytes,
                        "staging_plus_reserve_bytes": 2*plan["geotiff_bytes"]+1024**3})
        excluded.update(plan["events"])
        remaining -= plan["geotiff_bytes"]+1024**2
    result = {"status": "PLANNED_NOT_ACQUIRED", "known_manual_exclusion": "2018/fire_21458798",
              "private_quota_declared_bytes": private_free_bytes, "plans": reports}
    json_new(destination/"pilot.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="data/manifests/real/corpus_zip_index_v1.json")
    p.add_argument("--out", required=True)
    p.add_argument("--private-free-bytes", type=int, required=True)
    p.add_argument("--storage-bytes", type=int, help="Available allocation; capped by measured free working disk")
    p.add_argument("--transfer-bytes", type=int, default=3*1024**3)
    p.add_argument("--roots", nargs="*", default=[])
    args = p.parse_args()
    free = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").is_dir() else Path.cwd()).free
    storage = min(free, args.storage_bytes) if args.storage_bytes else free
    print(json.dumps(prepare(json.loads(Path(args.index).read_text()), args.out,
          private_free_bytes=args.private_free_bytes, storage_bytes=storage, roots=args.roots,
          transfer_bytes=args.transfer_bytes), indent=2))
