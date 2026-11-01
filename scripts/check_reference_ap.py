"""Validate disk-backed AP against the real historical validation scores only."""
import argparse
import json
import time
from pathlib import Path

import torch

from geofirex.benchmark_metrics import ExternalAveragePrecision,new_fire_metrics
from geofirex.corpus import json_new
from geofirex.unet_data import WindowStore,load_event_windows
from geofirex.unet_evaluation import predict_tiled
from predict_unet import load_predictor

ROOT=Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    start=time.perf_counter()
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=False)
    run=Path(args.run_dir)
    result=json.loads((run/"results.json").read_text(encoding="utf-8"))
    model,norm,checkpoint=load_predictor(run/"best.pt")
    cfg=checkpoint["config"]
    torch.set_num_threads(cfg["cpu_threads"])
    _,_,windows,_=load_event_windows(ROOT/cfg["manifest"],ROOT/cfg["split"],ROOT)
    store=WindowStore(windows["val"],norm)
    aps={name:ExternalAveragePrecision(output/name) for name in ("unet","persistence","unet_new_fire","persistence_new_fire")}
    novel=[]
    for i in range(len(store)):
        x,y,observed=store.get(i)
        prob=predict_tiled(model,x,cfg["patch_size"],cfg["batch_size"])
        aps["unet"].update(prob,y); aps["persistence"].update(observed,y)
        candidate=~observed.astype(bool)
        aps["unet_new_fire"].update(prob[candidate],y[candidate])
        aps["persistence_new_fire"].update(observed[candidate],y[candidate])
        novel.append(new_fire_metrics(prob,y,observed,checkpoint["threshold"]))
    values={name:ap.compute() for name,ap in aps.items()}
    for name in ("unet","persistence"):
        expected=result["evaluations"]["val"][name]["micro_pixel"]["pr_auc"]
        if abs(values[name]-expected)>1e-12:
            raise RuntimeError("External AP differs from the historical sklearn reference")
    report={"qualification":"validation-only metric implementation verification; not new model training",
            "average_precision":values,"sklearn_reference_agrees_at_1e_12":True,
            "new_fire_positive_pixels":sum(r["new_positive_pixels"] for r in novel),
            "new_fire_tp_unet":sum(r["new_tp"] for r in novel),
            "new_fire_fn_unet":sum(r["new_fn"] for r in novel),
            "seconds":time.perf_counter()-start,
            "temporary_sort_bytes_retained":sum(p.stat().st_size for p in output.rglob("*.bin"))}
    json_new(output/"verification.json",report)
    print(json.dumps(report,indent=2))
