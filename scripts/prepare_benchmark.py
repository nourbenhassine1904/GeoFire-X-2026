"""Generate all annual splits from remote metadata and a local availability check."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from geofirex.benchmark import aligned_windows, annual_folds, event_assignments
from geofirex.corpus import json_new, sha256_file
from geofirex.real_splits import save_split_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    events = sorted({e["event_id"] for e in index["entries"]})
    with Path(args.manifest).open(encoding="utf-8", newline="") as f:
        local = list(csv.DictReader(f))
    reports = []
    indexed_keys={(e["event_id"],e["date"]) for e in index["entries"]}
    local_keys={(r["event_id"],r["date"]) for r in local}
    if local_keys-indexed_keys:
        raise ValueError("Local manifest has observations absent from the pinned corpus index")
    for fold in annual_folds():
        assignment = event_assignments(events, fold["fold_id"])
        split_path = output / f"fold_{fold['fold_id']:02d}.csv"
        save_split_csv(assignment, split_path)
        availability = {}
        for role in ("train", "val", "test"):
            groups, rejections = aligned_windows(local, assignment, role)
            availability[role] = {"available_local_events": sorted({r["event_id"] for r in local if assignment[r["event_id"]] == role}),
                                  "aligned_target_count": len(groups[5]),
                                  "rejections_by_context": rejections}
            expected={key for key in indexed_keys if assignment[key[0]]==role}
            availability[role]["expected_observations"]=len(expected)
            availability[role]["local_observations"]=len(expected & local_keys)
            availability[role]["missing_observations"]=len(expected-local_keys)
            with (output / f"fold_{fold['fold_id']:02d}_{role}_aligned_targets.csv").open("x",newline="",encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["event_id","target_date","context_1_paths","context_3_paths","context_5_paths","target_path"])
                for i,w in enumerate(groups[5]):
                    writer.writerow([w.event_id,w.target_date,*[";".join(groups[t][i].context_paths) for t in (1,3,5)],w.target_path])
        reports.append({**fold,"events_per_split":dict(Counter(assignment.values())),
                        "split_sha256":sha256_file(split_path),"local_availability":availability,
                        "ready_for_local_training":all(availability[r]["aligned_target_count"] > 0 for r in availability)})
        reports[-1]["complete_fold_metadata"]=local_keys==indexed_keys
        reports[-1]["scientific_execution_enabled"]=False  # Preparation only; raw QC/runner gates remain.
    json_new(output / "folds.json", {"index_sha256":sha256_file(args.index),
                                     "local_manifest_sha256":sha256_file(args.manifest),
                                     "qualification":"607-event metadata splits; only local sample availability checked",
                                     "folds":reports})
    print(json.dumps(reports[0],indent=2))
