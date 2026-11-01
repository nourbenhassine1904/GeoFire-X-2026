"""Start the local React/FastAPI demo from an immutable trusted launch record."""
import argparse
import json
from pathlib import Path

from geofirex.cloud_corpus import check_seal
from geofirex.serving import create_app

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="artifacts/demo-serving/gpu-pilot-local.json")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if not Path(args.config).is_file():
        p.error("Launch record missing: run scripts/prepare_gpu_demo.py with the existing B1 snapshot and local TIFF root")
    config = json.loads(Path(args.config).read_text())
    check_seal(config)
    app = create_app(config["snapshot"], config["data_roots"], config["inventory_identity"])
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
