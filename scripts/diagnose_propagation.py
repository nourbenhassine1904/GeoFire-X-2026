"""Describe completed predictions; no model loading, fitting or new inference."""
import json
from datetime import date, timedelta
from pathlib import Path
import zipfile

import numpy as np

from geofirex.calibration_io import save, sha
from geofirex.fast_compact import FastCompactTiles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/propagation_20261112_v1'
CONF = ROOT / 'reports/unet_confirmation_20261112_v1/confirmation_cpu'
HIST = ROOT / 'reports/calibration_20261112_v1/GeoFireX_CALIBRATION_RESULTATS.zip'
ABL = ROOT / 'reports/unet_confirmation_20261112_v1/inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip'


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                q10=float(np.quantile(values, .1)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                q90=float(np.quantile(values, .9)) if len(values) else None)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'diagnosis.json'
    if output.exists():
        raise FileExistsError('Existing diagnosis is immutable; choose a new report version')
    folder = CONF / 'B2_w10_holdout169'
    evaluation = json.loads((folder / 'evaluation.json').read_text())
    complete = json.loads((CONF / 'status_complete.json').read_text())
    assert sha(folder / 'evaluation.json') == complete['evaluation_sha256']
    with zipfile.ZipFile(HIST) as z:
        train = json.loads(z.read('train_audit.json'))
        protocol = json.loads(z.read('protocol.json'))
    with zipfile.ZipFile(ABL) as z:
        baseline32 = json.loads(z.read('B2_posweight_10/calibration32/evaluation.json'))
    full, new, old = [evaluation[k][0] for k in ('curves', 'new_fire_candidates', 'observed_fire_candidates')]
    for key in ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'positive_pixels'):
        assert new[key] + old[key] == full[key]
        assert sum(r[key] for r in evaluation['per_event_new_fire']) == new[key]
    rows = evaluation['per_event_new_fire']
    positive = [r for r in rows if r['positive_pixels']]
    ranked = sorted(positive, key=lambda r: r['fn'], reverse=True)
    summary = dict(new_positive_share=new['positive_pixels']/full['positive_pixels'],
        new_false_negative_share=new['fn']/full['fn'],
        observed_true_positive_share=old['tp']/full['tp'],
        events_with_new_fire=len(positive), events_without_new_fire=len(rows)-len(positive),
        events_zero_new_true_positives=sum(r['tp']==0 for r in positive),
        new_recall_event=distribution([r['recall'] for r in positive]),
        top10_event_share_of_new_false_negatives=sum(r['fn'] for r in ranked[:10])/new['fn'])
    ds = FastCompactTiles(ROOT/'reports/mission_20261111/compact_full',
                          ROOT/'reports/gpu_acceleration_20261111_v1/index', 'val')
    # Only the three already-saved map days: compare labels/masks, never re-infer.
    from matplotlib import use
    use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cases, checks = [], []
    for case in evaluation['maps']:
        raw = folder / case['raw']
        with np.load(raw, allow_pickle=False) as z:
            p,y,v,o = [z[k].copy() for k in ('probability','target','valid','observed')]
        day = date.fromisoformat(case['target_date']).toordinal()
        indices = [i for i,r in enumerate(ds.records) if int(r['day'])==day and ds.event(i)==case['event_id']]
        assert len(indices)==case['tile_count']
        for i in indices:
            x, target, valid = ds[i]
            rec=ds.records[i];a,d,r,c=[int(rec[k]) for k in ('archive','day','row','col')]
            h,w=target.shape[-2:];region=np.s_[r:r+h,c:c+w]
            expected=ds._day(a,d,r,c)[1].copy()
            for delta in (-5,-4,-3,-2,-1): expected &= ds._day(a,d+delta,r,c)[1]
            assert np.array_equal(expected.astype(bool),valid.numpy().astype(bool))
            assert np.array_equal(y[region],target[0].numpy())
            assert np.array_equal(o[region],x[-1,7].numpy())
            assert np.array_equal(v[region],valid[0].numpy().astype(bool))
            assert np.array_equal(x[-1,7].numpy(),ds._day(a,d-1,r,c)[0][7])
        checks.append(dict(event_id=case['event_id'], target_date=case['target_date'],tiles=len(indices),
            context_dates=[(date.fromordinal(day)+timedelta(days=d)).isoformat() for d in (-3,-2,-1)],
            target_and_observed_and_six_day_joint_mask_equal=True))
        candidate=v & (o==0);truth=candidate & (y==1);prediction=candidate & (p>=np.float32(.78))
        error=np.where(prediction,np.where(truth,1,2),np.where(truth,3,0)).astype(np.uint8)
        error[~candidate]=255
        groups={name: distribution(p[mask]) for name,mask in {
            'new_positive':truth,'new_false_negative':truth & ~prediction,
            'new_true_positive':truth & prediction,'observed_positive':v & (o==1) & (y==1),
            'new_false_positive':prediction & ~truth}.items()}
        layers={};stem=case['raw'].removesuffix('_raw.npz')
        palette=ListedColormap(['#16212b','#30bb80','#f25252','#3f8efc'])
        for name,array,mask,cmap,maximum in [
            ('observed',o,v,plt.get_cmap('inferno'),1),
            ('new_target',truth,v,plt.get_cmap('inferno'),1),
            ('new_prediction',prediction,v,plt.get_cmap('inferno'),1),
            ('new_probability',p,candidate,plt.get_cmap('inferno'),1),
            ('new_errors',error,candidate,palette,3)]:
            rgba=cmap(np.nan_to_num(array.astype(float),nan=0)/maximum);rgba[...,3]=mask
            name=stem+'_'+name+'.png';plt.imsave(OUT/name,rgba);layers[name.removeprefix(stem+'_').removesuffix('.png')]=name
        cases.append(dict(event_id=case['event_id'],target_date=case['target_date'],layers=layers,
            source_raw_sha256=sha(raw),candidate_pixels=int(candidate.sum()),new_positive_pixels=int(truth.sum()),
            new_tp=int((truth & prediction).sum()),new_fn=int((truth & ~prediction).sum()),
            probability_groups=groups,scope='one pre-existing median-day map; not all event days or all 169 events'))
    ds.close()
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (HIST,ABL,folder/'evaluation.json',
        CONF/'confirmation_locked.json',ROOT/'reports/gpu_acceleration_20261111_v1/index/normalization.json')}
    baseline_at=lambda key:next(r for r in baseline32[key] if r['threshold']==.78)
    report=dict(schema='geofirex-propagation-diagnosis-v1',status='SAVED_OUTPUTS_ONLY',
        definition='N = valid_joint & (fire_J == 0); new target = N & (fire_Jplus1 == 1)',
        label_caveat='New active-fire detection, not cumulative burned area or proof of physical spread/ignition; re-detection and isolated detections included.',
        temporal_contract=dict(context='J-2,J-1,J',target='J+1',fire_channel=7,
            validity='intersection of J-4,J-3,J-2,J-1,J,J+1; unchanged historical comparison domain'),
        features=['slope_z','elevation_z','slope_missing','elevation_missing','aspect_sin','aspect_cos','aspect_missing','fire'],
        feature_limitation='Only active-fire channel is time varying in the selected compact features. No wind, humidity, fuel or forecast weather available in these 8 channels.',
        summary=summary,full=full,new_fire=new,observed_fire=old,per_event=rows,
        top10_new_false_negatives=ranked[:10],train=train,
        baseline32=dict(full=baseline_at('curves'),new_fire=baseline_at('new_fire_candidates')),
        cases=cases,label_checks=checks,sources=sources,
        probability_scope='Full-169 probability histograms by target exist; conditional new-fire probabilities retained only for these three map days. No claim of full-domain new-fire AP or Brier.',
        training_executed=False,inference_executed=False,test_2021_pixels_opened=0)
    save(output,report)
    save(OUT/'assets_manifest.json',dict(diagnosis_sha256=sha(output),assets={n:sha(OUT/n) for c in cases for n in c['layers'].values()}))
    print(json.dumps(summary,indent=2));print(output)


if __name__=='__main__':build()
