"""Fixed-threshold paired FIRE13 comparison, using historical exact AP code."""
import csv
from pathlib import Path
from geofirex import c8_comparison as paired
from geofirex.protocol_v2 import save
import numpy as np
from geofirex.c8_thresholds import folds,event_counts,summarize,serial_rows


def exact_fp_choice(store,events,budget):
    """Exact ties; fixed integer budget supplied before looking at scores."""
    if type(budget) is not int or budget<0:raise ValueError('Nonnegative fixed integer FP budget')
    top=np.empty(0,np.float32);negatives=0
    for _,a in store.chunks(events):
        scores=a['p'][(a['v']==1)&(a['o']==0)&(a['y']==0)]
        negatives+=len(scores);joined=np.concatenate((top,scores));keep=min(budget+1,len(joined))
        top=np.partition(joined,len(joined)-keep)[-keep:] if keep else joined
    boundary=top.min() if negatives>budget else None;selected=None;tp=0
    for _,a in store.chunks(events):
        p=a['p'][(a['v']==1)&(a['o']==0)&(a['y']==1)]
        eligible=p if boundary is None else p[p>boundary]
        if len(eligible):
            selected=float(eligible.min()) if selected is None else min(selected,float(eligible.min()));tp+=len(eligible)
    fp=int((top>=np.float32(selected)).sum()) if selected is not None else 0
    if fp>budget:raise ValueError('Indivisible tie exceeded budget')
    return dict(threshold=selected,calibration_fp=fp,calibration_tp=tp,fp_budget=budget,negatives=negatives)


def comparable_fp(root,store,destination):
    """64/16 event-disjoint calibration. No held-out correction or promotion."""
    from geofirex.fire13_experiment import read,REPORT,sha256
    root=Path(root);destination=Path(destination);destination.mkdir(exist_ok=False)
    baseline=read(root/REPORT/'baseline_reference.json');policy=read(root/REPORT/'comparable_fp_policy.json')
    num,den=policy['FPR_rational'];mapping=folds(store.events);selections=[]
    for fold in range(5):
        calibration=[e for e in store.events if mapping[e]!=fold]
        assessment=[e for e in store.events if mapping[e]==fold]
        negatives=sum(store.files[e]['new_negative'] for e in calibration)
        choice=exact_fp_choice(store,calibration,int(negatives*num//den))
        selections.append(dict(fold=fold,calibration_events=calibration,assessment_events=assessment,**choice))
    locked=destination/'thresholds_locked.json'
    save(locked,dict(policy=policy,selections=selections,fold_mapping=mapping,
         prediction_manifest_sha256=store.manifest_sha256,operational_threshold_selected=False))
    digest=sha256(locked);rows={};fold_results=[]
    for s in selections:
        if sha256(locked)!=digest:raise ValueError('Threshold decisions changed')
        assessed=event_counts(store,s['assessment_events'],s['threshold']);rows.update(assessed)
        summary=summarize(assessed);actual=summary['new_fire']['fp']
        n=summary['new_fire']['tn']+actual;budget=int(n*num//den)
        fold_results.append(dict(fold=s['fold'],actual_fp=actual,reference_rate_budget=budget,
            exceeds_reference_rate=actual>budget,metrics=summary))
    base=paired.rows_from_result(baseline)
    if set(base)!=set(rows):raise ValueError('Event coverage differs')
    save(destination/'bootstrap_events.json',paired.bootstrap_events(base,rows))
    save(destination/'result.json',dict(status='COMPLETE',candidate_crossfit=summarize(rows),
        baseline_fixed_078=summarize(base),fold_results=fold_results,per_event=serial_rows(rows),
        thresholds_sha256=digest,assessment_events_per_fold=16,calibration_events_per_fold=64,
        AP_and_Brier_unchanged=True,acceptance_criteria_unchanged=True,automatic_promotion=False,
        interpretation='Exploratory paired event crossfit on known validation; calibration FP cap does not guarantee assessment FP. Report actual FP and overruns; never retune.'))


def summary(result):
    return dict(recall_new=result['new_fire']['recall'], dice_new=result['new_fire']['dice'],
        precision_new=result['new_fire']['precision'], AP_new=result['propagation_pr']['average_precision'],
        FP_new=result['new_fire']['fp'], dice_global=result['pooled']['dice'],
        AP_global=result['pooled']['average_precision'], Brier=result['pooled']['brier'],
        macro_event_dice=result['macro_event_dice'], IoU_new=result['new_fire']['iou'],
        FPR_new=result['new_fire']['false_positive_rate'])


def gates(result, policy):
    s=summary(result)
    return dict(AP_new=s['AP_new']>=policy['AP_new_min'],
        AP_pooled=s['AP_global']>=policy['AP_pooled_min'],
        Dice_pooled=s['dice_global']>=policy['Dice_pooled_min'],
        recall_new=s['recall_new']>=policy['recall_new_min'],
        Dice_new=s['dice_new']>=policy['Dice_new_min'], FP_new=s['FP_new']<=policy['FP_new_max'],
        macro_Dice=s['macro_event_dice']>=policy['macro_Dice_min'], Brier=s['Brier']<=policy['Brier_max'])


def compare(root, candidate, destination, expected_events):
    from geofirex.fire13_experiment import REPORT, read, DOMAIN
    root=Path(root);destination=Path(destination)
    baseline=read(root/REPORT/'baseline_reference.json')
    historical=read(root/REPORT/'newpos4_reference.json')
    newpos2=read(root/REPORT/'newpos2_reference.json')
    policy=read(root/REPORT/'acceptance.json')
    if len(set(expected_events))!=80 or candidate['evaluation_domain_sha256']!=DOMAIN:
        raise ValueError('Exact frozen 80-event domain required')
    # Shared validator checks completion, threshold, domain, event targets/masks and exact AP.
    result=paired.compare(baseline,candidate,destination,expected_events)
    if historical['evaluation_domain_sha256']!=candidate['evaluation_domain_sha256']:
        raise ValueError('Historical NEWPOS4 domain differs')
    h,c=paired.rows_from_result(historical),paired.rows_from_result(candidate)
    if set(h)!=set(c):raise ValueError('Historical event set differs')
    for e in h:
        for domain in ('pooled','new_fire'):
            a,b=h[e][domain],c[e][domain]
            if a.sum()!=b.sum() or a[0]+a[2]!=b[0]+b[2]:
                raise ValueError('Historical target/mask counts differ')
    save(destination/'bootstrap_vs_newpos4.json',paired.bootstrap_events(h,c))
    if newpos2['evaluation_domain_sha256']!=candidate['evaluation_domain_sha256']:raise ValueError('NEWPOS2 domain differs')
    n2=paired.rows_from_result(newpos2)
    if set(n2)!=set(c):raise ValueError('NEWPOS2 event set differs')
    save(destination/'bootstrap_vs_newpos2.json',paired.bootstrap_events(n2,c))
    rows=[dict(experiment=n,**summary(r)) for n,r in
          [('C8_official',baseline),('NEWPOS4_078_historical',historical),('NEWPOS2_078_historical',newpos2),('FIRE13_078',candidate)]]
    with (destination/'metrics.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    locked=gates(candidate,policy)
    report=dict(status='COMPLETE',fixed_threshold=.78,scientific_gates=locked,
        all_scientific_gates_pass=all(locked.values()),paired_vs_C8=result['deltas'],
        exact_AP_definition='Average precision from exact global score ties; not trapezoidal PR area',
        budget_gate='runtime_budget.json and actual Kaggle quota receipt required before promotion',
        calibration='Brier and unchanged historical descriptive PR/calibration conventions',
        exploratory_single_seed=True,validation_already_observed=True,promotion_authorized=False,
        AP_bootstrap=False,test_pixels_opened=0)
    save(destination/'FIRE13_acceptance.json',report)
    return report
