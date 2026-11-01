"""Prepare a machine-local launch record from the existing B1 GPU snapshot."""
import argparse
import json
from pathlib import Path

from geofirex.cloud_corpus import seal
from geofirex.serving import Predictor


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', default='artifacts/sample-launch-20261101/verified-output/snapshots/B1')
    p.add_argument('--data-root', default='data/raw/WildfireSpreadTS/real_sample')
    p.add_argument('--out', default='artifacts/demo-serving/gpu-pilot-local.json')
    args = p.parse_args()
    snapshot, data = Path(args.snapshot).resolve(), Path(args.data_root).resolve()
    inventory = json.loads((snapshot/'inventory.json').read_text())
    predictor = Predictor(snapshot, [data], inventory['identity'])
    if predictor.config['model'] != 'B1':
        raise ValueError('The demonstration candidate is the certified B1 CNN')
    event = next(e for e in predictor.events() if e['split'] == 'val' and e['forecast_dates'])
    result = predictor.predict(event['event_id'], event['forecast_dates'][0], max_side=64)
    if result['optimizer_updates'] != 1280:
        raise ValueError('Expected the completed GPU pilot B1 checkpoint')
    config = seal({'snapshot': str(snapshot), 'data_roots': [str(data)],
                   'inventory_identity': inventory['identity'], 'qualification': 'pilot_gpu_sample',
                   'test_2021': 'locked'})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: json.dump(config, f, indent=2)
    print(out.resolve())
