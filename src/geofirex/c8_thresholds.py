"""Exact event crossfit. Calibration never opens its held-out event shards.

The (budget+1)th largest negative score is the forbidden inclusive tie group.
All admissible positives are strictly above it. Their minimum score is the
highest threshold achieving maximum recall; no approximate threshold grid.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
from geofirex.protocol_v2 import save,sha256
from geofirex.compact_calibration import metrics


def folds(events):
    events=list(events)
    if len(events)!=len(set(events)) or len(events)%5 or len(events)<5:raise ValueError('Five nonempty equal event folds required')
    ordered=sorted(events,key=lambda e:hashlib.sha256(('geofirex-threshold-v1:'+e).encode()).hexdigest())
    return {e:i%5 for i,e in enumerate(ordered)}


def choose_threshold(store,events,rate=Fraction(7,25000)):
    if rate!=Fraction(7,25000):raise ValueError('Frozen FPR budget must equal 0.00028')
    negatives=sum(store.files[e]['new_negative'] for e in events)
    k=negatives*rate.numerator//rate.denominator
    top=np.empty(0,np.float32);scanned=0
    for _,a in store.chunks(events):
        mask=(a['v']==1)&(a['o']==0)&(a['y']==0);scores=np.asarray(a['p'][mask]);scanned+=len(scores)
        joined=np.concatenate((top,scores));keep=min(k+1,len(joined))
        top=np.partition(joined,len(joined)-keep)[-keep:] if keep else joined
    if scanned!=negatives:raise ValueError('Declared negative population differs from saved arrays')
    boundary=float(top.min()) if negatives>k else None
    selected=None;tp=0;positives=0
    for _,a in store.chunks(events):
        pos=a['p'][(a['v']==1)&(a['o']==0)&(a['y']==1)]
        positives+=len(pos)
        eligible=pos if boundary is None else pos[pos>np.float32(boundary)]
        if len(eligible):
            value=float(eligible.min());selected=value if selected is None else min(selected,value);tp+=len(eligible)
    fp=int((top>=np.float32(selected)).sum()) if selected is not None else 0
    if fp>k:raise ValueError('Exact integer FP budget exceeded')
    return dict(threshold=selected,fp_budget=k,negative_count=negatives,calibration_tp=tp,
                calibration_fp=fp,forbidden_negative_tie_score=boundary,
                calibration_positive_count=positives,calibration_recall=tp/positives if positives else None,
                calibration_fpr=fp/negatives if negatives else None,
                rule='max recall, highest threshold on recall ties; None=reject-all',fpr_budget=.00028)


def counts(p,y,mask,threshold):
    pred=np.zeros(p.shape,bool) if threshold is None else p>=np.float32(threshold)
    truth=y.astype(bool);mask=mask.astype(bool)
    return np.array([int((pred&truth&mask).sum()),int((pred&~truth&mask).sum()),
        int((~pred&truth&mask).sum()),int((~pred&~truth&mask).sum())],np.int64)


def event_counts(store,events,threshold):
    result={e:dict(pooled=np.zeros(4,np.int64),new_fire=np.zeros(4,np.int64),squared_error=0.) for e in events}
    for e,a in store.chunks(events):
        p,y,v,o=(a[k] for k in ('p','y','v','o'));r=result[e]
        r['pooled']+=counts(p,y,v,threshold);r['new_fire']+=counts(p,y,(v==1)&(o==0),threshold)
        r['squared_error']+=float(np.square(p[v==1].astype(np.float64)-y[v==1]).sum())
    return result


def summarize(rows):
    pooled=sum((r['pooled'] for r in rows.values()),np.zeros(4,np.int64))
    new=sum((r['new_fire'] for r in rows.values()),np.zeros(4,np.int64))
    p=metrics(pooled);n=metrics(new);den=n['fp']+n['tn']
    n['false_positive_rate']=n['fp']/den if den else None
    p['brier']=sum(r['squared_error'] for r in rows.values())/p['valid_pixels'] if p['valid_pixels'] else None
    ds=[metrics(r['pooled'])['dice'] for r in rows.values()];ds=[d for d in ds if d is not None]
    return dict(pooled=p,new_fire=n,macro_event_dice=float(np.mean(ds)) if ds else None)


def serial_rows(rows):
    return {e:dict(pooled=r['pooled'].tolist(),new_fire=r['new_fire'].tolist(),squared_error=r['squared_error']) for e,r in rows.items()}


def crossfit(store,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    mapping=folds(store.events);selections=[]
    # All five threshold decisions are immutable on disk BEFORE assessment.
    for fold in range(5):
        calibration=[e for e in store.events if mapping[e]!=fold]
        assessment=[e for e in store.events if mapping[e]==fold]
        choice=choose_threshold(store,calibration)
        selections.append(dict(fold=fold,calibration_events=calibration,assessment_events=assessment,**choice))
    path=destination/'thresholds_locked.json'
    save(path,dict(predictions_manifest_sha256=store.manifest_sha256,fold_mapping=mapping,selections=selections,
                  policy='Exploratory on previously observed validation; no post-hoc adjustment',test_pixels_opened=0))
    lock_sha=sha256(path);chosen={};baseline={}
    for s in selections:
        if sha256(path)!=lock_sha:raise ValueError('Threshold lock changed before assessment')
        chosen.update(event_counts(store,s['assessment_events'],s['threshold']))
        baseline.update(event_counts(store,s['assessment_events'],.78))
    a,b=summarize(chosen),summarize(baseline)
    def gain(x,y,minimum):return x is not None and y is not None and x-y>=minimum
    gates=dict(new_recall=gain(a['new_fire']['recall'],b['new_fire']['recall'],.02),
        new_dice=gain(a['new_fire']['dice'],b['new_fire']['dice'],.01),
        new_fp=a['new_fire']['fp']<=29134,pooled_dice=gain(a['pooled']['dice'],b['pooled']['dice'],-.005))
    result=dict(status='COMPLETE',interpretation='exploratory event-disjoint validation crossfit; not a blind test',
        thresholds_sha256=lock_sha,crossfit=a,baseline_078_same_probabilities=b,
        fold_results=[dict(fold=s['fold'],threshold=s['threshold'],calibration_recall=s['calibration_recall'],
            calibration_fpr=s['calibration_fpr'],assessment=summarize({e:chosen[e] for e in s['assessment_events']})) for s in selections],
        per_event_crossfit=serial_rows(chosen),per_event_baseline=serial_rows(baseline),
        acceptance=gates,acceptance_all=all(gates.values()),
        AP_and_Brier_statement='Same probabilities: changing thresholds cannot improve AP or Brier',test_pixels_opened=0)
    save(destination/'crossfit.json',result);return result


def plots(exact_result,crossfit_result,directory,title='Exploratory C8 validation'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);directory.mkdir(exist_ok=False)
    curve=exact_result['propagation_pr']['curve'];rows=[r for r in curve if r['recall'] is not None]
    base=crossfit_result['baseline_078_same_probabilities']['new_fire'];cf=crossfit_result['crossfit']['new_fire']
    for name,xkey,ykey in [('precision_recall','recall','precision'),('fpr_recall','fpr','recall'),('threshold_tradeoff','threshold','recall')]:
        fig,ax=plt.subplots(figsize=(7,4))
        points=[r for r in rows if r[xkey] is not None and r[ykey] is not None]
        ax.plot([r[xkey] for r in points],[r[ykey] for r in points],label='Exact scan, display downsampled')
        for label,r in [('0.78',base),('Event crossfit',cf)]:
            x=r.get('false_positive_rate') if xkey=='fpr' else .78 if xkey=='threshold' and label=='0.78' else r.get(xkey)
            y=r.get(ykey)
            if x is not None and y is not None:ax.scatter([x],[y],label=label)
        if xkey=='threshold':
            pts=[(r['threshold'],r['assessment']['new_fire']['recall']) for r in crossfit_result['fold_results']
                 if r['threshold'] is not None and r['assessment']['new_fire']['recall'] is not None]
            if pts:ax.scatter([x for x,y in pts],[y for x,y in pts],label='Held-out recall per fold')
        ax.set(xlabel=xkey,ylabel=ykey,title=title);ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(directory/(name+'.png'),dpi=160);plt.close(fig)
