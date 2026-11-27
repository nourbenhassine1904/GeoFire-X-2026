"""Default CPU identity preflight; fit needs a separately approved exact contract."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
from pathlib import Path
import json
from geofirex import v2_execution as v2

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',nargs='?',default='preflight',choices=['preflight','fit','evaluate-only','package','backup'])
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--maps',nargs='+'); p.add_argument('--out'); p.add_argument('--training-out')
    p.add_argument('--authorization'); p.add_argument('--gpu',action='store_true')
    p.add_argument('--archive'); p.add_argument('--sidecar'); p.add_argument('--local-root'); p.add_argument('--external-root')
    a=p.parse_args()
    if a.action=='preflight':
        print(json.dumps(v2.preflight(a.root,a.maps,'/kaggle/working' if a.gpu else None,'/tmp',a.gpu),indent=2))
    elif a.action=='fit':
        if not all((a.maps,a.out,a.authorization)): p.error('fit requires maps/out/authorization')
        v2.fit(a.root,a.maps,a.out,a.authorization)
    elif a.action=='evaluate-only':
        if not all((a.maps,a.out,a.training_out,a.authorization)): p.error('recovery requires maps/out/training-out/authorization')
        v2.authorize(a.root,a.authorization,'evaluate-only')
        out=Path(a.out); out.mkdir(exist_ok=False)
        try:
            _,lock,_=v2.identity(a.root); v2.preserve(a.root,out,lock)
            # Include the final checkpoint and receipt in the recovery archive.
            _,checkpoint=v2.final_state(a.training_out,a.root)
            import shutil
            for source in (checkpoint,checkpoint.with_suffix('.json'),Path(a.training_out)/'training_complete.json'):
                shutil.copyfile(source,out/source.name)
            policy=v2.read(Path(a.root)/v2.REPORT/'budget.json')
            with v2.evaluation_deadline(policy['evaluation_seconds']):
                v2.evaluate(a.root,a.maps,a.training_out,out/'evaluation',authorization=a.authorization)
        finally:
            v2.package(out,'EVALUATION-RECOVERY')
    elif a.action=='package': v2.package(a.out,'MANUAL-RESCUE')
    else: print(v2.backup(a.archive,a.sidecar,a.local_root,a.external_root))

if __name__=='__main__': main()
