"""Verify completed U-Net ablation; confirm ONLY w10 at fixed .78, on CPU.

No optimizer, threshold search, training, Kaggle API or 2021 inference.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from geofirex.benchmark_models import make_model
from geofirex.calibration_io import save,sha
from geofirex.compact_calibration import (evaluate,event_ids,metrics,macro,render_maps,write_csv)
from geofirex.science_results import DEFAULT_ZIP,ROOT

BASE=ROOT/'reports/unet_confirmation_20261112_v1'
ABLATION=BASE/'inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip'
WEIGHTS=(100,30,10)
FIXED_THRESHOLD=.78


def digest(data):return hashlib.sha256(data).hexdigest()


def audit(archive=ABLATION,historical=DEFAULT_ZIP):
    """Check hashes, state, selection math and every complete-event count."""
    with zipfile.ZipFile(historical) as z:
        protocol_bytes=z.read('protocol.json');protocol=json.loads(protocol_bytes)
    selected=[];initial=set();configs=[];event_tables={}
    with zipfile.ZipFile(archive) as z:
        members=z.namelist()
        for weight in WEIGHTS:
            prefixes=[n.removesuffix('selection32.json') for n in members if n.endswith('selection32.json')
                      and json.loads(z.read(n)).get('pos_weight')==weight]
            if len(prefixes)!=1:raise ValueError('Expected exactly one completed run per weight')
            prefix=prefixes[0]
            row=json.loads(z.read(prefix+'selection32.json'))
            config=json.loads(z.read(prefix+'config.json'))
            checkpoint_bytes=z.read(prefix+'checkpoint_latest.pt')
            if digest(checkpoint_bytes)!=row['checkpoint_sha256']:raise ValueError('Checkpoint digest differs')
            state=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=False)
            if state['config']!=config:raise ValueError('Checkpoint/config mismatch')
            if (state['epoch'],state['cursor'],state['updates'],len(state['history']))!=(5,0,23404,4):
                raise ValueError('Expected completed epoch 4')
            if (config['model'],config['pos_weight'],config['seed'],config['epochs'],config['batch_size'],config['train_tiles'])!=('B2',weight,42,4,16,93603):
                raise ValueError('Unexpected training contract')
            if any(h['epoch']!=i or h['updates']!=5851*i for i,h in enumerate(state['history'],1)):
                raise ValueError('Epoch/update history differs')
            if config['calibration_events']!=protocol['calibration_events'] or config['calibration_tiles']!=13759:
                raise ValueError('32 event identities differ')
            for key in ('index_sha256','normalization_sha256'):
                if row[key]!=config[key] or config[key]!=protocol[key]:raise ValueError('Protocol digest mismatch: '+key)
            if row['protocol_sha256']!=digest(protocol_bytes) or config['protocol_sha256']!=digest(protocol_bytes):
                raise ValueError('Source protocol digest differs')
            if (row['status'],row['updates'],row['selection_scope'],row['holdout_pixels_opened'],row['test_2021_pixels_opened'])!=('COMPLETE',23404,'32 only',0,0):
                raise ValueError('Incomplete or contaminated selection')
            if row['initial_state_sha256']!=config['initial_state_sha256']:raise ValueError('Initialization record differs')
            initial.add(row['initial_state_sha256']);configs.append(config)
            model=make_model('B2',8,3);model.load_state_dict(state['model'],strict=True)
            if any(not torch.isfinite(t).all() for t in state['model'].values()):raise ValueError('Nonfinite checkpoint')
            evaluation=json.loads(z.read(prefix+'calibration32/evaluation.json'))
            if evaluation['status']!='COMPLETE' or evaluation['events']!=protocol['calibration_events'] or evaluation['tiles']!=13759:
                raise ValueError('Incomplete 32-event evaluation')
            curves=evaluation['curves']
            if [r['threshold'] for r in curves]!=config['threshold_candidates']:raise ValueError('Threshold grid differs')
            best=max(curves,key=lambda r:(r['dice'],r['precision'] if r['precision'] is not None else -1.,r['threshold']))
            if best!=row['metrics'] or row['threshold']!=best['threshold']:raise ValueError('Selected maximum not reproduced from saved counts')
            for c in curves:
                recalculated=metrics([c[k] for k in ('tp','fp','fn','tn')])
                for key in ('dice','iou','precision','recall','valid_pixels','positive_pixels'):
                    if recalculated[key]!=c[key]:raise ValueError('Confusion metric differs: '+key)
                if c['average_precision']!=best['average_precision'] or c['brier_score']!=best['brier_score']:
                    raise ValueError('Probability metric changes with threshold')
            per=[r for r in evaluation['per_event'] if r['threshold']==best['threshold']]
            if len(per)!=32 or {r['event_id'] for r in per}!=set(protocol['calibration_events']):raise ValueError('Event table incomplete')
            for key in ('tp','fp','fn','tn','valid_pixels','positive_pixels'):
                if sum(r[key] for r in per)!=best[key]:raise ValueError('Per-event/global counts differ')
            weighted_brier=sum(r['brier_score']*r['valid_pixels'] for r in per)/best['valid_pixels']
            if abs(weighted_brier-best['brier_score'])>1e-12:raise ValueError('Brier aggregation differs')
            event_tables[str(weight)]=per
            selected.append(dict(**row,checkpoint_member=prefix+'checkpoint_latest.pt',
                                 prefix=prefix,history=state['history']))
    if len(initial)!=1 or len({r['checkpoint_sha256'] for r in selected})!=3:
        raise ValueError('Initialization/trained checkpoint identities differ')
    reference=configs[0]
    for c in configs[1:]:
        if {k:v for k,v in c.items() if k!='pos_weight'}!={k:v for k,v in reference.items() if k!='pos_weight'}:
            raise ValueError('Only pos_weight may differ between training configs')
    winner=max(selected,key=lambda r:(r['metrics']['dice'],r['metrics']['average_precision'],-r['metrics']['brier_score']))
    if winner['pos_weight']!=10 or winner['threshold']!=FIXED_THRESHOLD:raise ValueError('Preannounced winner not reproduced')
    ten=next(r for r in selected if r['pos_weight']==10);thirty=next(r for r in selected if r['pos_weight']==30)
    return dict(schema='geofirex-unet-ablation-audit-v1',status='VERIFIED',archive_sha256=sha(archive),
        source_archive=str(Path(archive)),protocol_sha256=digest(protocol_bytes),
        variants=selected,winner=winner,per_event32=event_tables,initial_state_sha256=next(iter(initial)),
        protocol=protocol,delta_dice_10_vs_30=ten['metrics']['dice']-thirty['metrics']['dice'],
        delta_ap_10_vs_30=ten['metrics']['average_precision']-thirty['metrics']['average_precision'],
        independent_run_evidence='same recorded initial hash, different trained checkpoint hashes; no training replay',
        caveat='Very small Dice difference, one seed; weight30 has higher AP. No significance claim.',
        checkpoints_verified_without_retraining=True)


def compare(result,audited,historical=DEFAULT_ZIP):
    with zipfile.ZipFile(historical) as z:
        old=json.loads(z.read('comparison.json'))
        original=[r for r in old['rows'] if r['phase']=='holdout169' and r['mode']=='calibrated']
        protocol=json.loads(z.read('protocol.json'))
        if result['events']!=protocol['holdout_events']:raise ValueError('Confirmation event IDs differ')
        for model in ('B1','B2'):
            saved=json.loads(z.read(model+'_holdout169/evaluation.json'))
            lookup={r['event_id']:r for r in saved['per_event'] if r['threshold']==.95}
            for row in result['per_event']:
                if any(row[k]!=lookup[row['event_id']][k] for k in ('valid_pixels','positive_pixels','tiles','invalid_pixels')):
                    raise ValueError('Per-event confirmation pixel domain differs')
    current=dict(phase='holdout169',model='B2_w10',mode='confirmed_fixed',events=169,tiles=result['tiles'],**result['curves'][0])
    if result['tiles']!=65012 or len(result['curves'])!=1 or result['curves'][0]['threshold']!=FIXED_THRESHOLD:
        raise ValueError('Confirmation must use exactly 169 events, 65012 tiles and one fixed threshold')
    for row in original:
        if any(row[k]!=current[k] for k in ('valid_pixels','positive_pixels')):raise ValueError('Historical pixel domain differs')
    return dict(rows=[*original,current],macro_event=macro(result['per_event']),
                selection_source='32 events only; holdout scores never select winner or threshold',
                historical_archive_sha256=sha(historical),ablation_archive_sha256=audited['archive_sha256'])


def confirm(args):
    from geofirex.fast_compact import FastCompactTiles
    if args.device!='cpu':raise ValueError('This local confirmation runner permits CPU only; GPU requires separate explicit authorization')
    audited=audit(args.archive,args.historical)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'ablation_verified.json',audited)
    protocol=audited['protocol'];events=protocol['holdout_events']
    ds=FastCompactTiles(args.corpus,args.index,'val',cache_mb=128)
    ids=event_ids(ds)
    if len(events)!=169 or len(set(events))!=169 or any(not e.startswith('2020/') for e in events):raise ValueError('169 complete 2020 events required')
    if set(events)&set(protocol['calibration_events']) or int(np.isin(ids,events).sum())!=65012:raise ValueError('Confirmation domain differs')
    for name,key in [('index.json','index_sha256'),('normalization.json','normalization_sha256')]:
        if sha(Path(args.index)/name)!=protocol[key]:raise ValueError('Corpus contract differs')
    lock=dict(schema='geofirex-unet-confirmation-lock-v1',model='B2',pos_weight=10,threshold=FIXED_THRESHOLD,
              checkpoint_sha256=audited['winner']['checkpoint_sha256'],ablation_archive_sha256=audited['archive_sha256'],
              historical_archive_sha256=sha(args.historical),event_ids=events,tiles=65012,
              calibration_event_ids=protocol['calibration_events'],index_sha256=protocol['index_sha256'],
              normalization_sha256=protocol['normalization_sha256'],selection_scope='32 only; confirmation fixed before any holdout pixel read',
              execution_device='cpu',torch_version=torch.__version__,numpy_version=np.__version__,
              training_torch_version=audited['variants'][0].get('torch_version','2.11.0+cu130'),
              test_2021_locked=True,training_executed=False)
    save(out/'confirmation_locked.json',lock);locked_sha=sha(out/'confirmation_locked.json')
    torch.set_num_threads(2)
    with zipfile.ZipFile(args.archive) as z:
        state=torch.load(io.BytesIO(z.read(audited['winner']['checkpoint_member'])),map_location='cpu',weights_only=False)
    model=make_model('B2',8,3);model.load_state_dict(state['model'],strict=True);model.eval()
    cases=set(events[i] for i in np.linspace(0,len(events)-1,3,dtype=int))
    save(out/'status_started.json',dict(status='RUNNING',confirmation_lock_sha256=locked_sha,threshold=FIXED_THRESHOLD,device='cpu'))
    started=time.perf_counter()
    result=evaluate(model,ds,events,[FIXED_THRESHOLD],args,out/'B2_w10_holdout169','B2 w10 fixed0.78 CPU',cases)
    if sha(out/'confirmation_locked.json')!=locked_sha:raise ValueError('Confirmation lock changed')
    comparison=compare(result,audited,args.historical)
    assets=render_maps(result,out/'B2_w10_holdout169','B2_w10',FIXED_THRESHOLD,audited['winner']['checkpoint_sha256'])
    for case in assets:case['directory']='B2_w10_holdout169'
    save(out/'comparison.json',comparison);write_csv(out/'comparison.csv',comparison['rows'])
    new=result['new_fire_candidates'][0];old=result['observed_fire_candidates'][0];full=result['curves'][0]
    if any(new[k]+old[k]!=full[k] for k in ('tp','fp','fn','tn')):raise ValueError('Temporal domain partition differs')
    save(out/'new_fire.json',dict(new_fire_candidates=new,observed_fire_candidates=old,
                               definition='new fire: target=1, observed fire at J=0, joint valid mask'))
    manifest=dict(schema='geofirex-unet-confirmation-app-v1',threshold=FIXED_THRESHOLD,pos_weight=10,
        checkpoint_sha256=lock['checkpoint_sha256'],confirmation_lock_sha256=locked_sha,
        comparison=comparison['rows'],macro_event=comparison['macro_event'],maps=assets,
        ablation_archive_sha256=audited['archive_sha256'],historical_archive_sha256=sha(args.historical))
    save(out/'app_manifest.json',manifest)
    save(out/'status_complete.json',dict(status='COMPLETE',device='cpu',gpu_used=False,events=169,tiles=65012,
        threshold=FIXED_THRESHOLD,confirmation_lock_sha256=locked_sha,elapsed_seconds=time.perf_counter()-started,
        evaluation_sha256=sha(out/'B2_w10_holdout169/evaluation.json'),manifest_sha256=sha(out/'app_manifest.json'),
        training_executed=False,test_2021_pixels_opened=0))
    ds.close();print(json.dumps(dict(status='COMPLETE',metrics=full,out=str(out)),indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',default=str(ABLATION));p.add_argument('--historical',default=str(DEFAULT_ZIP))
    p.add_argument('--corpus',default=str(ROOT/'reports/mission_20261111/compact_full'))
    p.add_argument('--index',default=str(ROOT/'reports/gpu_acceleration_20261111_v1/index'))
    p.add_argument('--out',default=str(BASE/'confirmation_cpu'))
    p.add_argument('--device',choices=['cpu'],default='cpu');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--confirm',action='store_true')
    args=p.parse_args()
    if args.confirm:confirm(args)
    else:
        result=audit(args.archive,args.historical)
        print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','per_event32')},indent=2))


if __name__=='__main__':main()
