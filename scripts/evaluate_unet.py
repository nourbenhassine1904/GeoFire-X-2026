"""Replay paired metrics from a frozen checkpoint; never re-select the threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_unet_data import ROOT, sha256
from predict_unet import load_predictor
from train_unet import predict_store, save_json
from geofirex.unet_data import WindowStore, load_event_windows
from geofirex.unet_evaluation import summarize


def evaluate(checkpoint_path, manifest, split, output, role="test", device="cpu", expected=None):
    if Path(output).exists():
        raise FileExistsError(output)
    model, norm, checkpoint = load_predictor(checkpoint_path, device)
    if sha256(manifest) != checkpoint["manifest_sha256"] or sha256(split) != checkpoint["split_sha256"]:
        raise ValueError("Manifest/split differ from the checkpoint's training provenance")
    torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    _, _, windows, _ = load_event_windows(manifest, split, ROOT)
    records = predict_store(model, WindowStore(windows[role], norm), checkpoint["config"], device)
    results = {name: summarize(records, checkpoint["threshold"], name) for name in ("unet", "persistence")}
    if expected:
        original = json.loads(Path(expected).read_text(encoding="utf-8"))["evaluations"][role]
        if results != original:
            raise RuntimeError("Replay differs from original metrics")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {"split": role, "threshold_selected_on": "val", "evaluations": results,
                       "exact_replay_verified": bool(expected), "checkpoint_sha256": sha256(checkpoint_path)})
    print(json.dumps({name: r["micro_pixel"] for name, r in results.items()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="data/manifests/real/manifest_wildfirespreadts.csv")
    p.add_argument("--split", default="splits/real/smoke_test_split.csv")
    p.add_argument("--split-name", choices=("val", "test"), default="test")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--expected-results", help="Assert exact equality with the original run results.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    evaluate(args.checkpoint, args.manifest, args.split, args.out, args.split_name,
             args.device, args.expected_results)
