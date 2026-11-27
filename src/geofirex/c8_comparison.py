"""Paired event differences and predeclared gates; no checkpoint/threshold search."""
import csv
from pathlib import Path
import numpy as np
from geofirex.protocol_v2 import save
from geofirex.compact_calibration import metrics


def rows_from_result(result):
    rows={}
    for row in result['per_event']:
        e=row['event_id']
        if e in rows:raise ValueError('Duplicate evaluation event')
        rows[e]=dict(pooled=np.array([row[k] for k in ('tp','fp','fn','tn')],np.int64),
            new_fire=np.array([row['new_fire'][k] for k in ('tp','fp','fn','tn')],np.int64),
            squared_error=row['brier']*row['valid_pixels'] if row['valid_pixels'] else 0.)
    return rows


def scalar_summary(rows,events):
    p=sum((rows[e]['pooled'] for e in events),np.zeros(4,np.int64))
    n=sum((rows[e]['new_fire'] for e in events),np.zeros(4,np.int64))
    a,b=metrics(p),metrics(n);dice=[metrics(rows[e]['pooled'])['dice'] for e in events]
    dice=[d for d in dice if d is not None]
    result={'pooled_dice':a['dice'],'macro_event_dice':float(np.mean(dice)) if dice else None,
            'brier':sum(rows[e]['squared_error'] for e in events)/int(p.sum()) if p.sum() else None}
    for name in ('recall','dice','precision','iou','fp'):result['new_'+name]=b[name]
    return result


def bootstrap_events(baseline,candidate,repetitions=2000,seed=20261121):
    if set(baseline)!=set(candidate) or not baseline:raise ValueError('Identical nonempty event coverage required')
    events=sorted(baseline);rng=np.random.default_rng(seed);samples={}
    for _ in range(repetitions):
        selected=[events[i] for i in rng.integers(0,len(events),len(events))]
        a,b=scalar_summary(baseline,selected),scalar_summary(candidate,selected)
        for name in a:
            if a[name] is not None and b[name] is not None:samples.setdefault(name,[]).append(b[name]-a[name])
    return dict(repetitions=repetitions,seed=seed,unit='paired event, not pixel',
        intervals={k:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),defined=len(v)) for k,v in samples.items()},
        scope='confusion-count metrics, macro Dice and Brier; exact pooled AP reported separately, not approximated or bootstrapped')


def compare(baseline,candidate,destination,expected_events=None):
    if (baseline['status']!='COMPLETE' or candidate['status']!='COMPLETE'
            or baseline['threshold']!=.78 or candidate['threshold']!=.78
            or baseline['evaluation_domain_sha256']!=candidate['evaluation_domain_sha256']
            or baseline['tiles']!=candidate['tiles']):raise ValueError('Incomplete or incomparable evaluation domains')
    a,b=rows_from_result(baseline),rows_from_result(candidate)
    if set(a)!=set(b) or (expected_events is not None and set(a)!=set(expected_events)):
        raise ValueError('Event coverage differs')
    for e in a:
        for domain in ('pooled','new_fire'):
            x,y=a[e][domain],b[e][domain]
            if (x.sum()!=y.sum() or x[0]+x[2]!=y[0]+y[2]):raise ValueError('Event target/mask counts differ')
    required=[baseline['pooled']['average_precision'],candidate['pooled']['average_precision'],
              baseline['propagation_pr']['average_precision'],candidate['propagation_pr']['average_precision']]
    if any(v is None or not np.isfinite(v) for v in required):raise ValueError('Exact AP required; no approximation fallback')
    deltas=dict(AP_pooled=required[1]-required[0],AP_new=required[3]-required[2],
        Dice_pooled=candidate['pooled']['dice']-baseline['pooled']['dice'],
        recall_new=candidate['new_fire']['recall']-baseline['new_fire']['recall'],
        Dice_new=candidate['new_fire']['dice']-baseline['new_fire']['dice'],
        macro_Dice=candidate['macro_event_dice']-baseline['macro_event_dice'],
        Brier=candidate['pooled']['brier']-baseline['pooled']['brier'])
    fp0=baseline['new_fire']['fp'];fp1=candidate['new_fire']['fp']
    gates=dict(AP_new=deltas['AP_new']>=.005,AP_pooled=deltas['AP_pooled']>=.005,
        Dice_pooled=deltas['Dice_pooled']>=-.005,recall_new=deltas['recall_new']>=.02,
        Dice_new=deltas['Dice_new']>=.01,FP_new=(fp1<=1.15*fp0 if fp0 else fp1==0),
        macro_Dice=deltas['macro_Dice']>=-.01,Brier=deltas['Brier']<=0.)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    rows=[]
    for e in sorted(a):
        x,y=scalar_summary(a,[e]),scalar_summary(b,[e]);r={'event_id':e}
        for name in x:r[name+'_baseline']=x[name];r[name+'_candidate']=y[name];r[name+'_delta']=y[name]-x[name] if x[name] is not None and y[name] is not None else None
        for domain in ('pooled','new_fire'):
            for i,name in enumerate(('tp','fp','fn','tn')):
                r[domain+'_'+name+'_baseline']=int(a[e][domain][i]);r[domain+'_'+name+'_candidate']=int(b[e][domain][i])
                r[domain+'_'+name+'_delta']=int(b[e][domain][i]-a[e][domain][i])
        rows.append(r)
    with (destination/'per_event_differences.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    boot=bootstrap_events(a,b);save(destination/'bootstrap_events.json',boot)
    result=dict(status='COMPLETE',deltas=deltas,scientific_gates=gates,all_scientific_gates_pass=all(gates.values()),
        runtime_budget_gate='Must additionally pass authorized runtime budget receipt',
        promotion_authorized=False,exploratory_single_seed=True,test_pixels_opened=0)
    save(destination/'comparison.json',result);return result
