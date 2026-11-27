"""Recompute only deterministic VALIDATION 2020 scenes and compare archived floats.

Does not evaluate a split, select a model/threshold, or import any 2021 runner.
Writes a new timestamped receipt and a current qualification pointer.
"""
from datetime import datetime, timezone, date, timedelta
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from geofirex.studio_v2 import V2Predictor, ROOT, QUALIFICATION, implementation_hashes
from geofirex.jury_app import evidence
from geofirex.calibration_io import sha


def compare(actual, reference):
    a=np.asarray(actual,dtype=np.float32); b=np.asarray(reference,dtype=np.float32)
    mask=np.isfinite(b)
    if not mask.any() or not np.isfinite(a[mask]).all(): raise ValueError('Empty/nonfinite parity domain')
    error=float(np.max(np.abs(a[mask]-b[mask])))
    classes=bool(np.array_equal(a[mask]>=np.float32(.78),b[mask]>=np.float32(.78)))
    passed=bool(np.allclose(a[mask],b[mask],atol=1e-6,rtol=1e-6)) and classes
    return dict(passed=passed,valid_pixels=int(mask.sum()),max_abs_error=error,threshold_classification_equal=classes)


def main():
    from importlib.metadata import version
    started=time.perf_counter(); diagnostic=json.loads(evidence('diagnostic_tile.json'))
    receipt=dict(schema='studio-v2-cpu-qualification-1',created_utc=datetime.now(timezone.utc).isoformat(),
                 implementation_sha256=implementation_hashes(),runtime={p:version(p) for p in ('torch','numpy','scipy')},models={},test_pixels_opened=0,
                 training_executed=False,scope='2020 deterministic software parity; no new model selection',
                 limitations='C8 archive contains no full probability arrays: archived diagnostic tile only; NEWPOS full scenes.')
    for model_id in ('C8','NEWPOS2','NEWPOS4','FIRE13'):
        print('Qualifying',model_id,flush=True)
        predictor=V2Predictor(model_id,require_qualification=False)
        try:
            key=diagnostic['key']; as_of=(date.fromisoformat(key[1])-timedelta(days=1)).isoformat()
            # Full scene through the same serving path, not an alternative forward loop.
            result=predictor.predict(key[0],as_of,512)
            assert result['preview_stride']==1
            case=next(c for c in diagnostic['cases'] if c['model']==model_id)
            p=np.asarray(result['probability'],dtype=np.float32)
            tile=p[key[2]:key[2]+64,key[3]:key[3]+64]
            checks=[dict(event_id=key[0],target_date=key[1],reference='sealed diagnostic_tile.json',
                         reference_sha256=sha(ROOT/'reports/jury_20261202_v1/diagnostic_tile.json'),
                         **compare(tile,case['result']['probability']))]
            if model_id!='C8':
                from geofirex.studio_spatial import SavedScenes
                saved=SavedScenes(); manifest=saved.manifest(model_id)
                events=sorted(r['event'] for r in manifest['files'])
                # Outcome-independent first, middle, last events, middle available day.
                for event in (events[0],events[len(events)//2],events[-1]):
                    row=next(r for r in manifest['files'] if r['event']==event)
                    days=sorted({k[1] for k in row['keys']}); target=days[len(days)//2]
                    as_of=(date.fromisoformat(target)-timedelta(days=1)).isoformat()
                    r=predictor.predict(event,as_of,512)
                    with zipfile.ZipFile(predictor.path) as z: raw=z.read('evaluation/predictions/'+row['path'])
                    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Archived scene SHA differs')
                    a=np.load(io.BytesIO(raw),allow_pickle=False)
                    actual=np.asarray(r['probability'],dtype=np.float32); stride=r['preview_stride']
                    h,w=r['full_shape']; ref=np.full((h,w),np.nan,np.float32)
                    for i,k in enumerate(row['keys']):
                        if k[1]==target:
                            rr,cc=k[2:]; ref[rr:rr+64,cc:cc+64]=np.where(a[i]['v'],a[i]['p'],np.nan)
                    checks.append(dict(event_id=event,target_date=target,reference_member=row['path'],
                                       reference_sha256=row['sha256'],preview_stride=stride,
                                       **compare(actual,ref[::stride,::stride])))
            row=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
                     checkpoint_sha256=predictor.source['checkpoint_sha256'],
                     archive_sha256=predictor.source['archive_sha256'],checks=checks,
                     torch_version=predictor.torch_version,device='cpu',channels=len(predictor.channels),parameters=predictor.parameters,context_days=3,
                     normalizer_sha256=predictor.contract['normalization_sha256'])
            receipt['models'][model_id]=row
            print(model_id,row['status'],[c['max_abs_error'] for c in checks],flush=True)
        finally: predictor.dataset.close()
    receipt['elapsed_seconds']=time.perf_counter()-started
    required=all(receipt['models'][m]['status']=='PASS' for m in ('C8','NEWPOS2','NEWPOS4'))
    receipt['status']='FAIL' if not required else 'PASS' if receipt['models']['FIRE13']['status']=='PASS' else 'PASS_WITH_ARCHIVED_FIRE13'
    QUALIFICATION.parent.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=QUALIFICATION.parent/('qualification-'+stamp+'.json')
    text=json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with path.open('x',encoding='utf-8') as stream:stream.write(text)
    QUALIFICATION.write_text(text,encoding='utf-8')
    print(path,flush=True)
    if not required:raise SystemExit(1)


if __name__=='__main__': main()
