"""Metadata-only V2 build. Intentionally has no training switch."""
import argparse
from geofirex.protocol_v2 import build_v2

if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--prepared', default='reports/full_corpus_v1/prepared_v1')
    p.add_argument('--old-index', default='reports/gpu_acceleration_20261111_v1/index')
    p.add_argument('--corpus', default='reports/mission_20261111/compact_full')
    p.add_argument('--publication', default='reports/gpu_acceleration_20261111_v1/publication')
    p.add_argument('--out', default='reports/protocol_v2_20261118')
    a = p.parse_args()
    meta = build_v2(a.prepared, a.old_index, a.corpus, a.publication, a.out)
    print(meta['cutoff'], meta['roles'])
