"""Kaggle/Runpod setup checks. Never installs or upgrades PyTorch.

Optional --install-missing pins the already-installed torch distribution with a
pip constraint. A dependency conflict fails resolution instead of replacing it.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--install-missing", action="store_true", help="Explicitly allow dependency installation (network)")
    p.add_argument("--tests", action="store_true")
    p.add_argument("--out", default="gpu_preflight.json")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    import torch
    before = {"torch": str(torch.__version__), "distribution": importlib.metadata.version("torch"),
              "cuda_build": torch.version.cuda}
    requirements = {"numpy": "numpy", "rasterio": "rasterio", "mlflow": "mlflow", "yaml": "pyyaml",
                    "matplotlib": "matplotlib", "sklearn": "scikit-learn", "pytest": "pytest"}
    missing = []
    for module, package in requirements.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        if not args.install_missing:
            raise RuntimeError(f"Missing/broken dependencies: {missing}; use --install-missing or provision a compatible environment")
        with tempfile.TemporaryDirectory(prefix="geofirex-deps-") as tmp:
            constraint = Path(tmp)/"constraints.txt"
            constraint.write_text("torch=="+before["distribution"]+"\n", encoding="utf-8")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
                            "-c", str(constraint), *missing], check=True)
    # Inspect in a FRESH interpreter so a cached import cannot hide replacement.
    code = "import json,torch;print(json.dumps({'torch':str(torch.__version__),'cuda_build':torch.version.cuda,'available':torch.cuda.is_available()}))"
    after = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    if after["torch"] != before["torch"] or after["cuda_build"] != before["cuda_build"]:
        raise RuntimeError("PyTorch build changed; stop and restore the original GPU environment")
    if args.require_cuda and not after["available"]:
        raise RuntimeError("CUDA unavailable: enable the accelerator and use the host's CUDA PyTorch")
    if after["available"]:
        x = torch.ones((16, 16), device="cuda", requires_grad=True)
        (x@x).sum().backward()
        torch.cuda.synchronize()
        after["gpu"] = torch.cuda.get_device_name(0)
        after["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    env = {**os.environ, "PYTHONPATH": str(root/"src")+os.pathsep+os.environ.get("PYTHONPATH", "")}
    # Editable project registration without dependency resolution/downloads.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(root)], check=True)
    for module in requirements:
        subprocess.run([sys.executable, "-c", f"import {module}"], check=True)
    if args.tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, env=env, check=True)
    report = {"before": before, "after": after, "tests_requested": args.tests,
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(after, indent=2))


if __name__ == "__main__":
    main()
