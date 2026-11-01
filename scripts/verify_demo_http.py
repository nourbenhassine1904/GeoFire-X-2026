"""Exercise the running real-data demo and archive actual HTTP results."""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from geofirex.corpus import json_new


def read(base, path):
    started = time.perf_counter()
    with urllib.request.urlopen(base+path, timeout=180) as response:
        value = json.load(response)
    return value, time.perf_counter()-started


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    health, _ = read(args.base, "/health")
    events, _ = read(args.base, "/events")
    selected = next(e for e in events if e['split'] == 'val' and e["forecast_dates"])
    query = urllib.parse.urlencode({"event_id": selected["event_id"], "as_of": selected["forecast_dates"][0], "max_side": 256})
    prediction, predict_seconds = read(args.base, "/predict?"+query)
    truth, truth_seconds = read(args.base, "/ground-truth?"+query)
    for key in ("target_date", "full_shape", "preview_stride", "crs", "transform"):
        assert prediction[key] == truth[key], key
    assert prediction["optimizer_updates"] > 0
    assert prediction["data_origin"] == "official_wildfirespreadts_verified_bytes"
    assert all(day <= prediction["as_of"] for day in prediction["input_dates"])
    locked = {}
    for endpoint in ("predict", "ground-truth"):
        try:
            read(args.base, "/"+endpoint+"?event_id=2021/fire_1&as_of=2021-01-05")
            raise AssertionError("Test set unexpectedly unlocked")
        except urllib.error.HTTPError as error:
            assert error.code == 403
            locked[endpoint] = error.code
    json_new(out/"prediction.json", prediction)
    json_new(out/"ground-truth.json", truth)
    report = {"scope": "real-data diagnostic HTTP integration; browser visual QA unavailable",
        "health": health, "event_id": selected["event_id"], "as_of": prediction["as_of"],
        "model": prediction["model"], "qualification": prediction["qualification"],
        "split": prediction['split'],
        "optimizer_updates": prediction["optimizer_updates"], "checkpoint_sha256": prediction["checkpoint_sha256"],
        "predict_seconds": predict_seconds, "ground_truth_seconds": truth_seconds,
        "test_2021_locked": locked, "geometry_matches": True}
    json_new(out/"verification.json", report)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, label, values, cmap in zip(axes,
        ["Observed J", "Predicted probability J+1", "Prediction at validation threshold", "Observed J+1"],
        [prediction["observed_fire"], prediction["probability"], prediction["predicted_fire"], truth["ground_truth"]],
        ["inferno", "magma", "inferno", "inferno"]):
        image = np.array(values, dtype=float)
        ax.imshow(np.ma.masked_invalid(image), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle("GeoFire-X · Real data, diagnostic CNN · no scientific performance claim")
    fig.tight_layout()
    fig.savefig(out/"real-inference.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
