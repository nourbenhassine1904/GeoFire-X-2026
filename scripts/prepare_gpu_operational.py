"""Prepare portable metadata and execute bounded, real CPU integration only."""
import argparse
import json
from pathlib import Path

from geofirex.gpu_operational import cpu_integration, prepare
from geofirex.runner import load_config

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--prepared', default='reports/full_corpus_v1/prepared_spark_v1')
    a.add_argument('--index', default='data/manifests/real/corpus_zip_index_v1.json')
    a.add_argument('--raw-root', default='D:/GeoFire-X/WildfireSpreadTS/rasters')
    a.add_argument('--out', default='reports/gpu_operational_v1/deployment')
    a.add_argument('--owner', required=True)
    a = sub.add_parser('cpu-check')
    a.add_argument('--config', required=True)
    a.add_argument('--raw-root', required=True)
    a.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        result = prepare(args.prepared, args.index, args.raw_root, args.out, owner=args.owner)
        print(json.dumps({k: v for k, v in result.items() if k != 'lots'}, indent=2))
    else:
        if Path(args.out).exists():
            p.error('Output already exists; choose a new receipt')
        c = load_config(args.config, dict(device='cpu', data_root=args.raw_root, data_roots=[args.raw_root]))
        result = cpu_integration(c, args.out)
        print(json.dumps(result, indent=2))
