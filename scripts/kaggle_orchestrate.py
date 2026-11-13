"""Local private Kaggle job lifecycle, without notebook manipulation."""
import argparse
import json
import time
from pathlib import Path

from geofirex.orchestration import Orchestrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("kind", choices=["session-b", "verify-batch", "publication-probe", "bigdata", "acquire", "recover-batch", "throughput"])
    prepare.add_argument("--owner", default="nnnnbhss")
    prepare.add_argument("--proof")
    prepare.add_argument("--resume-receipt")
    prepare.add_argument("--plan")
    prepare.add_argument("--bootstrap")
    prepare.add_argument("--dependency-budget-bytes", type=int, default=600*1024**2)
    prepare.add_argument("--certificates", nargs="*", default=[])
    for name in ("submit", "refresh", "fetch", "watch"):
        p = sub.add_parser(name)
        p.add_argument("job_id")
        if name == "submit":
            p.add_argument("--authorization")
        if name == "watch":
            p.add_argument("--seconds", type=int, default=50)
    sub.add_parser("status")
    args = parser.parse_args()
    runner = Orchestrator(Path(__file__).resolve().parents[1], state=args.state)
    if args.command == "prepare":
        result = {"job": str(runner.prepare(args.kind, args.owner, proof=args.proof,
            resume_receipt=args.resume_receipt, plan=args.plan, bootstrap=args.bootstrap,
            dependency_budget_bytes=args.dependency_budget_bytes, certificates=args.certificates))}
    elif args.command == "status":
        result = runner.status()
    elif args.command == "watch":
        deadline = time.monotonic()+min(max(args.seconds, 0), 55)
        while True:
            result = runner.refresh(args.job_id)
            if result["remote_status"] in {"COMPLETE", "ERROR", "CANCELACKNOWLEDGED"}:
                result["retrieval"] = runner.fetch(args.job_id)
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(min(10, max(0, deadline-time.monotonic())))
    elif args.command == "submit":
        result = runner.submit(args.job_id, authorization=args.authorization)
    else:
        result = getattr(runner, args.command)(args.job_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
