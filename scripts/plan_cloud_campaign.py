"""Offline complete-corpus shard plans + expected observation/split tables.

These are quota scenarios, never evidence that an account has those quotas.
First verify the single-event private publication before executing larger plans.
"""
import argparse
import csv
import json
from pathlib import Path

from geofirex.cloud_corpus import plan_batch, validate_index
from geofirex.corpus import json_new
from geofirex.benchmark import event_assignments


def plan_campaign(index, destination, *, transfer_bytes, storage_bytes, upload_bytes, private_free_bytes):
    validate_index(index, official=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    assignments = event_assignments(sorted({e["event_id"] for e in index["entries"]}), 0)
    with (destination/"expected_observations.csv").open("x", newline="", encoding="utf-8") as f:
        fields = ["path", "event_id", "date", "year", "split", "compressed_bytes", "geotiff_bytes", "crc32"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in index["entries"]:
            writer.writerow({**{k: e[k] for k in fields if k != "split"}, "split": assignments[e["event_id"]]})
    with (destination/"split.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "split"])
        writer.writerows(sorted(assignments.items()))
    acquired, plans, remaining = set(), [], private_free_bytes
    for year in (2018, 2019, 2020, 2021):
        while any(e["year"] == year and e["event_id"] not in acquired for e in index["entries"]):
            plan = plan_batch(index, years=[year], max_events=607, transfer_bytes=transfer_bytes,
                              storage_bytes=storage_bytes, upload_bytes=upload_bytes,
                              private_free_bytes=remaining, exclude_events=acquired)
            name = f"batch_{len(plans)+1:03d}_{year}.json"
            json_new(destination/name, plan)
            acquired.update(plan["events"])
            remaining -= plan["geotiff_bytes"]+1024**2
            plans.append({"plan": name, "year": year, "events": len(plan["events"]),
                          "observations": plan["observations"], "geotiff_bytes": plan["geotiff_bytes"]})
    roles = {role: {"events": sum(v == role for v in assignments.values()),
                    "observations": sum(assignments[e["event_id"]] == role for e in index["entries"])}
             for role in ("train", "val", "test")}
    report = {"status": "PLANNED_ONLY_NOT_ACQUIRED", "by_split": roles, "batches": plans,
              "events": len(acquired), "observations": sum(p["observations"] for p in plans),
              "private_storage_scenario_bytes": private_free_bytes, "remaining_scenario_bytes": remaining}
    json_new(destination/"campaign.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    for key in ("transfer-bytes", "storage-bytes", "upload-bytes", "private-free-bytes"):
        p.add_argument("--"+key, type=int, required=True)
    args = p.parse_args()
    report = plan_campaign(json.loads(Path(args.index).read_text(encoding="utf-8")), args.out,
                transfer_bytes=args.transfer_bytes, storage_bytes=args.storage_bytes,
                upload_bytes=args.upload_bytes, private_free_bytes=args.private_free_bytes)
    print(json.dumps(report, indent=2))
