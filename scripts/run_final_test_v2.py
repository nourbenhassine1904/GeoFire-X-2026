"""Explicit final-test commands. No default action, training or implicit GPU use."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geofirex import final_test_v2 as ft


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    sub=p.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('freeze',help='metadata and checkpoint checks; exclusive new freeze; no TIFF open')
    for name in ('archive','rasters','out'): freeze.add_argument('--'+name,type=Path,required=True)
    qualify=sub.add_parser('qualify-2020',help='2020 software parity ONLY; CPU by default')
    for name in ('frozen','archive','rasters','corpus','out'): qualify.add_argument('--'+name,type=Path,required=True)
    qualify.add_argument('--device',default='cpu',choices=['cpu','cuda:0'])
    check=sub.add_parser('preflight',help='stat-only mount/resource check; never opens 2021 TIFFs or initializes CUDA')
    for name in ('frozen','archive','rasters','work','scratch','out'): check.add_argument('--'+name,type=Path,required=True)
    check.add_argument('--qualification',type=Path)
    run=sub.add_parser('evaluate',help='REAL 2021: forbidden without separate independent authorization and durable one-use ledger')
    for name in ('frozen','archive','rasters','out','scratch','authorization','preflight','qualification'):
        run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--device',required=True,choices=['cpu','cuda:0'])
    recovery=sub.add_parser('recover-metrics',help='after an authorized partial run ONLY: complete saved predictions, CPU, no TIFF reads')
    for name in ('frozen','partial','scratch','out'): recovery.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze': result=ft.freeze(a.root,a.archive,a.rasters,a.out); print(json.dumps(result['inventory'],indent=2))
    elif a.command=='qualify-2020':
        result=ft.qualify(a.root,a.frozen,a.archive,a.rasters,a.corpus,a.out,a.device)
        print(json.dumps({k:v for k,v in result.items() if k!='validation_parity'},indent=2))
    elif a.command=='preflight':
        if a.out.exists(): raise FileExistsError(a.out)
        result=ft.preflight(a.root,a.frozen,a.archive,a.rasters,a.work,a.scratch,a.qualification)
        ft.save(a.out,result); print(json.dumps(result,indent=2))
    elif a.command=='evaluate':
        print(ft.evaluate(a.root,a.frozen,a.archive,a.rasters,a.out,a.scratch,a.authorization,a.preflight,a.qualification,a.device))
    else: print(ft.recover_metrics(a.root,a.frozen,a.partial,a.scratch,a.out))


if __name__=='__main__': main()
