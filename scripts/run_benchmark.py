"""Common CLI: inspect, train, lock, final-test, export (no implicit training)."""
import argparse
import json
from pathlib import Path

from geofirex.runner import (choose_device, export_run, final_test, hardware, load_config,
                            lock_selection, train)
from geofirex.runner_data import FoldData


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="phase", required=True)
    for phase in ("inspect", "train"):
        parser = commands.add_parser(phase)
        parser.add_argument("--config", required=True)
        parser.add_argument("--model", choices=["B0", "B1", "B2", "B3"])
        parser.add_argument("--fold", type=int)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--context", type=int, choices=[1, 3, 5])
        parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
        parser.add_argument("--data-root")
        parser.add_argument("--data-roots", nargs="+", help="Read-only corpus batch roots in /kaggle/input")
        parser.add_argument("--manifest")
        parser.add_argument("--split")
        parser.add_argument("--index")
        parser.add_argument("--output-root")
        if phase == "train":
            parser.add_argument("--resume", help="Trusted checkpoint from the same immutable configuration")
    parser = commands.add_parser("lock")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser = commands.add_parser("final-test")
    parser.add_argument("--run", required=True)
    parser.add_argument("--lock", required=True)
    parser = commands.add_parser("export")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = p.parse_args()
    if args.phase in {"inspect", "train"}:
        overrides = {k: getattr(args, k) for k in ("model", "fold", "seed", "context", "device", "data_root",
                                                  "data_roots", "manifest", "split", "index", "output_root") if getattr(args, k) is not None}
        if args.fold is not None and args.split is None:
            overrides["split"] = f"splits/benchmark/annual_v1/fold_{args.fold:02d}.csv"
        config = load_config(args.config, overrides)
        if args.phase == "train":
            print(train(config, args.resume))
        else:
            data = FoldData(config)
            info = hardware(choose_device(config["device"]))
            info.pop("packages")
            print(json.dumps({"config": config, "hardware": info, "hashes": data.hashes,
                              "aligned_windows": {r: len(g[5]) for r, g in data.groups.items()}}, indent=2))
    elif args.phase == "lock":
        lock_selection(args.runs, args.out, tuple(args.seeds))
        print(Path(args.out).resolve())
    elif args.phase == "final-test":
        print(final_test(args.run, args.lock))
    else:
        print(export_run(args.run, args.out))


if __name__ == "__main__":
    main()
