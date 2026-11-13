"""Start the local React/FastAPI demo from an immutable trusted launch record."""
import argparse
import json
from pathlib import Path

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="artifacts/demo-serving/gpu-pilot-local.json")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--results-only", action="store_true", help="Existing React app + completed science, no historical model snapshot required")
    args = p.parse_args(argv)
    if not args.results_only and not Path(args.config).is_file():
        p.error("Launch record missing: run scripts/prepare_gpu_demo.py with the existing B1 snapshot and local TIFF root")
    if args.results_only:
        from geofirex.science_results import create_results_app
        app = create_results_app()
    else:
        from geofirex.cloud_corpus import check_seal
        from geofirex.serving import create_app
        config = json.loads(Path(args.config).read_text())
        check_seal(config)
        app = create_app(config["snapshot"], config["data_roots"], config["inventory_identity"])
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":main()
