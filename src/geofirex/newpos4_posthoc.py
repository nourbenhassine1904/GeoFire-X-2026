"""Saved validation predictions only. No model, optimizer, raw TIFF or test reader."""
from fractions import Fraction
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zipfile
import numpy as np
from geofirex.c8_predictions import PIXEL, validate_keys, domain_tile
from geofirex.c8_thresholds import folds, event_counts, summarize, serial_rows
from geofirex.protocol_v2 import save, sha256

PREFIX='evaluation/predictions/'
DOMAIN='efc995be596d7d520bac472ceb1b5122fe2111fe9d5a53c5edbec02851c1accc'
RUN='B2_C8_V2_NEWPOS4_seed42'
CHECKPOINT='5e139fa2dde831e21a05ce10d73a88c62bad7069c23cdde687cb16b1d04c9819'


def load_json(z,name):return json.loads(z.read(name))


def verify_archive(path,sidecar):
    from geofirex.v2_execution import verify_zip
    expected=Path(sidecar).read_text(encoding='ascii').split()[0]
    if len(expected)!=64:raise ValueError('Invalid SHA sidecar')
    digest=verify_zip(path,expected)
    return dict(path=str(Path(path).resolve()),sha256=digest,bytes=Path(path).stat().st_size,
                mtime_ns=Path(path).stat().st_mtime_ns,all_members_verified=True)


class ArchivedPredictions:
    """Bounded streaming adapter; zero extraction and no checkpoint deserialization.

    Construct only after verify_archive. The manifest/member hashes are linked to
    its verified artifact manifest; a later mutation is checked before analysis.
    """
    def __init__(self,archive,expected_keys=None,expected_domain=DOMAIN,expected_identity=None):
        self.archive=Path(archive);self.deadline=None;self.guard=None
        with zipfile.ZipFile(self.archive) as z:
            raw=z.read(PREFIX+'predictions_manifest.json')
            self.manifest=json.loads(raw);self.manifest_sha256=hashlib.sha256(raw).hexdigest()
            if load_json(z,PREFIX+'complete.json')['manifest_sha256']!=self.manifest_sha256:raise ValueError('Manifest incomplete')
            artifact=load_json(z,'artifact_manifest.json')
            members={r['path']:r for r in artifact['files']}
            if artifact['stage']!='COMPLETE':raise ValueError('Only a complete run is admissible')
            self.files={};keys=[]
            for row in self.manifest['files']:
                name=row['path'];e=row['event'];ks=[tuple(k) for k in row['keys']]
                if (PurePosixPath(name).name!=name or not name.endswith('.npy') or '\\' in name or ':' in name or e in self.files
                        or members[PREFIX+name]['sha256']!=row['sha256']
                        or members[PREFIX+name]['bytes']!=row['bytes']):raise ValueError('Unsafe or inconsistent prediction member')
                validate_keys(ks)
                if any(k[0]!=e for k in ks):raise ValueError('Wrong event keys')
                self.files[e]=row;keys.extend(ks)
            validate_keys(keys)
            if self.manifest['schema']!='c8-validation-predictions-1' or self.manifest['tiles']!=len(keys):raise ValueError('Coverage differs')
            if expected_keys is not None and set(keys)!=set(expected_keys):raise ValueError('Frozen validation keys differ')
            if expected_domain is not None and self.manifest['evaluation_domain_sha256']!=expected_domain:raise ValueError('Domain differs')
            if expected_identity is not None and self.manifest['identity']!=expected_identity:raise ValueError('Identity differs')

    @property
    def events(self):return sorted(self.files)

    def header(self,stream,event):
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError('Unsupported NPY version')
        if fortran or dtype!=PIXEL or shape!=(len(self.files[event]['keys']),*self.manifest['shape']):
            raise ValueError('Prediction NPY schema differs')
        return int(np.prod(shape))

    def chunks(self,events,chunk_pixels=65536):
        if not 1<=chunk_pixels<=65536:raise ValueError('Bounded streaming block required')
        with zipfile.ZipFile(self.archive) as z:
            for e in events:
                if self.guard is not None:self.guard()
                with z.open(PREFIX+self.files[e]['path']) as stream:
                    remaining=self.header(stream,e)
                    while remaining:
                        if self.deadline is not None and time.monotonic()>self.deadline:raise TimeoutError('CPU analysis deadline')
                        n=min(remaining,chunk_pixels);raw=stream.read(n*PIXEL.itemsize)
                        if len(raw)!=n*PIXEL.itemsize:raise ValueError('Truncated predictions')
                        yield e,np.frombuffer(raw,dtype=PIXEL);remaining-=n
                    if stream.read(1):raise ValueError('Trailing prediction data')

    def tiles(self,events=None):
        pixels=int(np.prod(self.manifest['shape']))
        with zipfile.ZipFile(self.archive) as z:
            for e in self.events if events is None else events:
                with z.open(PREFIX+self.files[e]['path']) as stream:
                    self.header(stream,e)
                    for k in self.files[e]['keys']:
                        if self.deadline is not None and time.monotonic()>self.deadline:raise TimeoutError('CPU analysis deadline')
                        raw=stream.read(pixels*PIXEL.itemsize)
                        if len(raw)!=pixels*PIXEL.itemsize:raise ValueError('Truncated tile')
                        yield tuple(k),np.frombuffer(raw,dtype=PIXEL).reshape(self.manifest['shape'])
                    if stream.read(1):raise ValueError('Trailing prediction data')


def select_threshold(store,events,rate):
    """Same exact tie algorithm as c8_thresholds, extended to frozen sensitivity budgets."""
    rate=Fraction(*rate)
    if rate not in (Fraction(7,50000),Fraction(7,25000),Fraction(7,12500)):
        raise ValueError('Unregistered FPR sensitivity budget')
    negatives=sum(store.files[e]['new_negative'] for e in events);k=negatives*rate.numerator//rate.denominator
    top=np.empty(0,np.float32);seen=0
    for _,a in store.chunks(events):
        scores=a['p'][(a['v']==1)&(a['o']==0)&(a['y']==0)];seen+=len(scores)
        joined=np.concatenate((top,scores));keep=min(k+1,len(joined))
        top=np.partition(joined,len(joined)-keep)[-keep:] if keep else joined
    if seen!=negatives:raise ValueError('Negative population differs')
    boundary=float(top.min()) if negatives>k else None;threshold=None;tp=0;positive=0
    for _,a in store.chunks(events):
        p=a['p'][(a['v']==1)&(a['o']==0)&(a['y']==1)];positive+=len(p)
        eligible=p if boundary is None else p[p>np.float32(boundary)]
        if len(eligible):
            threshold=float(eligible.min()) if threshold is None else min(threshold,float(eligible.min()));tp+=len(eligible)
    fp=int((top>=np.float32(threshold)).sum()) if threshold is not None else 0
    if fp>k:raise ValueError('FP budget exceeded')
    return dict(threshold=threshold,fp_budget=k,negative_count=negatives,calibration_tp=tp,
        calibration_fp=fp,calibration_recall=tp/positive if positive else None,
        calibration_fpr=fp/negatives if negatives else None,fpr_budget=float(rate))


def crossfit_costs(store,policy,out,reference):
    """Lock ALL policies/folds before ANY held-out scoring; never select a winner."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);mapping=folds(store.events);choices=[]
    for scenario in policy['scenarios']:
        for fold in range(5):
            print('Calibrating',scenario['id'],'fold',fold,flush=True)
            cal=[e for e in store.events if mapping[e]!=fold];held=[e for e in store.events if mapping[e]==fold]
            choices.append(dict(scenario=scenario['id'],fold=fold,calibration_events=cal,assessment_events=held,
                **select_threshold(store,cal,scenario['fpr_rational'])))
    lock=out/'thresholds_locked.json'
    save(lock,dict(policy=policy,predictions_manifest_sha256=store.manifest_sha256,fold_mapping=mapping,selections=choices))
    locked_sha=sha256(lock);results=[]
    for scenario in policy['scenarios']:
        print('Assessing locked thresholds',scenario['id'],flush=True)
        rows={};fold_results=[]
        for choice in [c for c in choices if c['scenario']==scenario['id']]:
            if sha256(lock)!=locked_sha:raise ValueError('Threshold lock altered')
            r=event_counts(store,choice['assessment_events'],choice['threshold']);rows.update(r);summary=summarize(r)
            rate=summary['new_fire']['false_positive_rate']
            fold_results.append(dict(fold=choice['fold'],threshold=choice['threshold'],assessment=summary,
                assessment_FPR_exceeds_calibration_budget=rate is not None and rate>choice['fpr_budget']))
        summary=summarize(rows);n=summary['new_fire'];b=reference['new_fire']
        def gain(a,b,delta):return a is not None and b is not None and a-b>=delta
        gates=dict(recall=gain(n['recall'],b['recall'],.02),dice=gain(n['dice'],b['dice'],.01),
            FP=n['fp']<=29134,pooled_dice=gain(summary['pooled']['dice'],reference['pooled']['dice'],-.005))
        results.append(dict(scenario=scenario['id'],summary=summary,folds=fold_results,per_event=serial_rows(rows),
            threshold_operating_gates=gates,all_operating_gates_pass=all(gates.values()),
            costs={str(c):n['fp']+c*n['fn'] for c in policy['miss_to_false_alarm_cost_ratios']}))
    result=dict(status='COMPLETE',exploratory_validation=True,thresholds_sha256=locked_sha,scenarios=results,
        selected_scenario=None,promotion_authorized=False,
        scope='no new inference, fit, probability calibration, final threshold fit or test evaluation')
    save(out/'crossfit_costs.json',result);return result


def reliability(store,bins=20):
    """Fixed bins are descriptive calibration diagnostics, never fitted transforms."""
    totals={k:dict(n=np.zeros(bins,np.int64),positive=np.zeros(bins,np.int64),
        probability=np.zeros(bins),squared_error=0.) for k in ('pooled','new_fire')}
    for _,a in store.chunks(store.events):
        for name in totals:
            mask=a['v']==1
            if name=='new_fire':mask&=a['o']==0
            p=a['p'][mask].astype(np.float64);y=a['y'][mask];i=np.minimum((p*bins).astype(int),bins-1);r=totals[name]
            r['n']+=np.bincount(i,minlength=bins);r['positive']+=np.bincount(i[y==1],minlength=bins)
            r['probability']+=np.bincount(i,weights=p,minlength=bins);r['squared_error']+=float(np.square(p-y).sum())
    result={}
    for name,r in totals.items():
        N=int(r['n'].sum());rows=[];ece=0.
        for i,n in enumerate(r['n']):
            mp=float(r['probability'][i]/n) if n else None;freq=float(r['positive'][i]/n) if n else None
            if n:ece+=int(n)*abs(mp-freq)/N
            rows.append(dict(low=i/bins,high=(i+1)/bins,n=int(n),mean_probability=mp,observed_frequency=freq))
        result[name]=dict(brier=r['squared_error']/N if N else None,ECE_fixed_20=ece,bins=rows,
            caveat='Pixel-weighted descriptive bins; strong imbalance can hide rare-event miscalibration')
    return result


def cost_policy():
    return dict(schema='newpos4-exploratory-thresholds-1',primary='fpr_0.00028',
        registered_before_new_threshold_assessment=True,exploratory_after_observed_078_results=True,
        scenarios=[dict(id='fpr_0.00014',fpr_rational=[7,50000]),dict(id='fpr_0.00028',fpr_rational=[7,25000]),
            dict(id='fpr_0.00056',fpr_rational=[7,12500])],miss_to_false_alarm_cost_ratios=[1,5,20],
        folds=5,calibration_events=64,assessment_events=16,threshold_ties='indivisible, highest threshold at max recall',
        select_scenario_from_heldout=False,final_all80_threshold_fit=False)
