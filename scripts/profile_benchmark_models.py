"""Seconds-long CPU forward/backward timing on real train patches; no optimizer.

Uses the historical train event/normalizer only for hardware profiling. These
untrained model timings are not benchmark experiments or predictive results.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from geofirex.benchmark_models import make_model
from geofirex.corpus import json_new
from geofirex.unet import SegmentationLoss
from geofirex.unet_data import Normalizer, read_raw
from geofirex.windows import build_windows


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--normalizer",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(42)
    norm=Normalizer(**json.loads(Path(args.normalizer).read_text(encoding="utf-8")))
    with Path(args.manifest).open(encoding="utf-8",newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["event_id"]=="2018/fire_21997864"]
    window=build_windows(rows,5)[0][0]
    images=np.stack([norm.transform(read_raw(path))[:,128:192,128:192] for path in window.context_paths])
    target=(~np.isnan(read_raw(window.target_path)[22:23,128:192,128:192])).astype(np.float32)
    y=torch.from_numpy(target)[None].repeat(2,1,1,1)
    records=[]
    for context in (1,3,5):
        x=torch.from_numpy(images[-context:].copy())[None].repeat(2,1,1,1,1)
        for name in ("B0","B1","B2","B3"):
            model=make_model(name,channels=8,context=context)
            seconds=[]
            for step in range(4):
                start=time.perf_counter()
                if name=="B0":
                    logits=model.predict_proba(x)
                else:
                    model.zero_grad(set_to_none=True)
                    logits=model(x)
                    SegmentationLoss()(logits,y).backward()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite prototype output")
                if step:
                    seconds.append(time.perf_counter()-start)
            records.append({"model":name,"context_days":context,"input_shape":list(x.shape),
                            "parameters":sum(p.numel() for p in model.parameters()),
                            "median_batch_seconds":float(np.median(seconds)),
                            "operation":"inference only" if name=="B0" else "forward + loss + backward, no optimizer",
                            "torch_threads":2})
    json_new(args.out,{"qualification":"hardware microbenchmark on real patches, untrained models; no accuracy results",
                       "torch":str(torch.__version__),"device":"cpu","batch_size":2,"patch_size":64,
                       "historical_normalization":"used for profiling only, not proposed fold training statistics",
                       "records":records})
    print(json.dumps(records,indent=2))
