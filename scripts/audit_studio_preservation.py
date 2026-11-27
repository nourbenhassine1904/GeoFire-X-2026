"""Record/verify accessible protected files; report unavailable external archives."""
import argparse
import json
from pathlib import Path
from geofirex.calibration_io import sha
from geofirex.studio_v2 import ROOT
from studio_preservation import paths

parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['snapshot','verify'])
parser.add_argument('receipt',type=Path)
args=parser.parse_args()
if args.action=='snapshot':
    protected=set(paths())|{ROOT/'feu.png'}
    for directory in ('configs','splits','reports/jury_20261202_v1','reports/protocol_v2_20261118/index'):
        protected.update(p for p in (ROOT/directory).rglob('*') if p.is_file())
    files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(protected) if p.is_file()}
    unavailable=[str(p) for p in sorted(protected) if not p.is_file()]
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x',encoding='utf8') as stream:
        json.dump(dict(files=files,unavailable=unavailable),stream,indent=2)
    print(json.dumps(dict(recorded=len(files),unavailable=unavailable)))
else:
    before=json.loads(args.receipt.read_text(encoding='utf8'))
    changed=[name for name,row in before['files'].items() if not Path(name).is_file() or sha(name)!=row['sha256']]
    print(json.dumps(dict(status='FAIL' if changed else 'PASS_ACCESSIBLE_FILES',verified=len(before['files']),changed=changed,unavailable=before['unavailable'])))
    if changed:raise SystemExit(1)
