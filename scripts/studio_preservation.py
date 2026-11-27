"""Hash-only preservation audit. No writes to scientific artefacts or test locks."""
import argparse
import json
from pathlib import Path
from geofirex.calibration_io import sha
from geofirex.studio_v2 import ROOT,sources
from geofirex.studio_final_results import DEFAULT_ARCHIVE


def paths():
    manifest=json.loads((ROOT/'reports/protocol_v2_execution/execution_manifest.json').read_text())
    p={ROOT/n for n in manifest['files']}
    p.update(Path(s['archive']) for s in sources().values())
    for directory in ('report','artifacts/final_test_v2_ledger','artifacts/final_test_v2_backups',
                      'reports/final_test_v2_20261125_v1','reports/final_test_v2_authorization_20261125_v1'):
        p.update(f for f in (ROOT/directory).rglob('*') if f.is_file())
    p.add(DEFAULT_ARCHIVE)
    return sorted(p,key=str)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['snapshot','verify'])
    parser.add_argument('receipt',type=Path);args=parser.parse_args()
    if args.action=='snapshot':
        data={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths()}
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        with args.receipt.open('x',encoding='utf-8') as out:json.dump(data,out,indent=2)
        print('Protected files recorded:',len(data))
    else:
        data=json.loads(args.receipt.read_text());changed=[]
        for name,expected in data.items():
            p=Path(name)
            if not p.is_file() or p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:changed.append(name)
        print(json.dumps(dict(status='FAIL' if changed else 'PASS',verified_files=len(data),changed=changed)))
        if changed:raise SystemExit(1)


if __name__=='__main__':main()
