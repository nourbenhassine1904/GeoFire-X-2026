"""CPU-only resource-qualified final evaluation; no default action or GPU mode."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geofirex import final_test_v2_local as local

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile',type=Path,default=ROOT/'reports/final_test_v2_local_20261125_v1')
s=p.add_subparsers(dest='command',required=True)
s.add_parser('seal',help='exclusive operational extension, no 2021 pixel access')
check=s.add_parser('preflight',help='metadata/resources and tiny new write probes only')
check.add_argument('--out',type=Path,required=True)
run=s.add_parser('evaluate',help='future real 2021; separate independent CPU authorization mandatory')
for name in ('authorization','preflight','out'):run.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
if a.command=='seal':print(json.dumps(local.seal_profile(ROOT,a.profile),indent=2))
elif a.command=='preflight':
    if a.out.exists():raise FileExistsError(a.out)
    result=local.preflight(ROOT,a.profile,a.out);print(json.dumps(result,indent=2))
    sys.exit(0 if result['technical_status']=='PASS' else 2)
else:print(local.evaluate(ROOT,a.profile,a.authorization,a.preflight,a.out))
