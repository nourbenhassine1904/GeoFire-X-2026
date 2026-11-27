"""Fixed-threshold paired PROP_HN2 comparison, using historical exact AP code."""
import csv
from pathlib import Path
from geofirex import c8_comparison as paired
from geofirex.protocol_v2 import save


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
    from geofirex.prop_hn2_experiment import REPORT, read, DOMAIN
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
          [('C8_official',baseline),('NEWPOS4_078_historical',historical),('NEWPOS2_078_historical',newpos2),('PROP_HN2_078',candidate)]]
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
    save(destination/'PROP_HN2_acceptance.json',report)
    return report
