"""Generate comparable GPU configs after mounted corpus validation; never trains."""
import argparse
import json
from pathlib import Path

import yaml

from geofirex.corpus import json_new
from geofirex.runner import load_config
from geofirex.runner_data import FoldData


def configure(prepared, index, destination, checkpoint_prefix, upload_budget, private_free,
              *, scientific=False, min_train_events=30, min_val_events=20):
    prepared, destination = Path(prepared), Path(destination)
    report = json.loads((prepared/"completeness.json").read_text(encoding="utf-8"))
    if (report["by_split"]["train"]["events"] < min_train_events
            or report["by_split"]["val"]["events"] < min_val_events):
        raise ValueError("Acquire a substantial complete-event train/validation subset first")
    if scientific and not (report["complete_corpus"] and report["actual_events"] == 607
                            and report["actual_observations"] == 13607):
        raise ValueError("Cannot label a partial corpus scientific/complete")
    if not checkpoint_prefix or min(upload_budget, private_free) < 10:
        raise ValueError("Private checkpoint handle prefix and total approved byte budgets required")
    base = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/runner_kaggle_scale.yaml").read_text())
    base.update(data_roots=json.loads((prepared/"mounts.json").read_text(encoding="utf-8")),
                manifest=str((prepared/"manifest.csv").resolve()), split=str((prepared/"split.csv").resolve()),
                index=str(Path(index).resolve()), qualification="scientific" if scientific else "pilot",
                experiment="geofirex-kaggle-scale-annual-v1" if scientific else "geofirex-kaggle-scale-pilot-v1",
                checkpoint_upload_budget_bytes=upload_budget//10,
                checkpoint_private_free_bytes=private_free//10)
    destination.mkdir(parents=True, exist_ok=False)
    configs = []
    for model in ("B0", "B1", "B2", "B3"):
        for seed in ([42] if model == "B0" else [42, 43, 44]):
            c = {**base, "model": model, "seed": seed,
                 "checkpoint_dataset": f"{checkpoint_prefix}-{model.lower()}-s{seed}"}
            path = destination/f"{model}_s{seed}.yaml"
            path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
            checked = load_config(path)
            data = FoldData(checked)
            counts = {role: len({w.event_id for w in data.windows(role)}) for role in ("train", "val")}
            if counts["train"] < min_train_events or counts["val"] < min_val_events:
                raise ValueError("Insufficient events with aligned usable windows")
            configs.append({"config": str(path), "model": model, "seed": seed, "usable_events": counts,
                            "planned_optimizer_updates": 0 if model == "B0" else 30*512})
    json_new(destination/"campaign.json", {"configs": configs, "corpus": report,
             "total_approved_upload_budget": upload_budget, "private_free_budget": private_free,
             "test": "2021 locked until validation-only selection; no test command is generated"})
    return configs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint-prefix", required=True, help="owner/dataset-prefix; ten private handles")
    p.add_argument("--upload-budget-bytes", type=int, required=True)
    p.add_argument("--private-free-bytes", type=int, required=True)
    p.add_argument("--scientific", action="store_true")
    args = p.parse_args()
    print(json.dumps(configure(args.prepared, args.index, args.out, args.checkpoint_prefix,
                               args.upload_budget_bytes, args.private_free_bytes,
                               scientific=args.scientific), indent=2))
