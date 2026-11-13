"""Independent local full-corpus integration. No acquisition or GPU training."""
import argparse
import json
from pathlib import Path
import signal
import threading

from geofirex.cloud_corpus import validate_index
from geofirex.full_corpus import audit_corpus, prepare_corpus, validate_loaders, delivery_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['audit', 'prepare', 'validate', 'report'])
    p.add_argument('--root', default='D:/GeoFire-X/WildfireSpreadTS/rasters')
    p.add_argument('--index', default='data/manifests/real/corpus_zip_index_v1.json')
    p.add_argument('--output', default='reports/full_corpus_v1')
    p.add_argument('--limit', type=int)
    p.add_argument('--prepared', default='reports/full_corpus_v1/prepared_v1')
    p.add_argument('--engine', choices=['spark', 'python'], default='spark')
    p.add_argument('--reuse-prepared', help='Reuse certified train normalization after exact input-table comparison')
    args = p.parse_args()
    index = json.loads(Path(args.index).read_text(encoding='utf-8'))
    validate_index(index, official=True)
    if args.mode == 'report':
        print(json.dumps(delivery_report(args.prepared, args.output), indent=2))
        return 0
    if args.mode == 'validate':
        print(json.dumps(validate_loaders(args.prepared, args.root, args.index, args.output,
                                         reuse_from=args.reuse_prepared), indent=2))
        return 0
    if args.mode == 'prepare':
        print(json.dumps(prepare_corpus(index, args.root, args.output, args.prepared, engine=args.engine), indent=2))
        return 0
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *unused: stopped.set())
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, lambda *unused: stopped.set())
    result = audit_corpus(index, args.root, args.output, limit=args.limit, stop_requested=stopped.is_set)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'CERTIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
