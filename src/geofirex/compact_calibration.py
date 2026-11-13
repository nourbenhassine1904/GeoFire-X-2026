"""Inference-only B1/B2 calibration. No optimizer, fitting or 2021 input.

Select thresholds using the checkpoint's historical 32 events, write an
immutable lock, then evaluate the disjoint 169 events at fixed thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from geofirex.calibration_io import save, sha
from geofirex.benchmark_metrics import ExternalAveragePrecision
from geofirex.benchmark_models import make_model


COARSE = np.round(np.arange(.10, .951, .05), 3)
# Precompute counts only; probabilities are inferred ONCE per tile/model.
DENSE = np.round(np.arange(.10, .9501, .005), 3)
METRICS = ('dice', 'iou', 'precision', 'recall')


def metrics(counts):
    tp, fp, fn, tn = map(int, counts)
    n, positive, predicted = tp + fp + fn + tn, tp + fn, tp + fp
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, valid_pixels=n,
                positive_pixels=positive, predicted_positive_pixels=predicted,
                prevalence=positive / n if n else None,
                dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (1. if n else None),
                iou=tp / (tp + fp + fn) if tp + fp + fn else (1. if n else None),
                precision=tp / predicted if predicted else None,
                recall=tp / positive if positive else None)


class ThresholdCounts:
    """Exact >= decisions, including ties; no [pixels, thresholds] allocation."""
    def __init__(self, thresholds):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        if not len(self.thresholds) or np.any(np.diff(self.thresholds) <= 0):
            raise ValueError('Thresholds must be increasing and unique')
        self.counts = np.zeros((len(self.thresholds), 4), dtype=np.int64)

    def update(self, p, y):
        p, y = np.asarray(p).ravel(), np.asarray(y).ravel().astype(bool)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid probabilities')
        # Match torch/numpy float32 comparison semantics at exact thresholds.
        bins = np.searchsorted(self.thresholds.astype(p.dtype), p, side='right')
        positives = np.bincount(bins[y], minlength=len(self.thresholds) + 1)
        negatives = np.bincount(bins[~y], minlength=len(self.thresholds) + 1)
        tp = np.cumsum(positives[::-1], dtype=np.int64)[::-1][1:]
        fp = np.cumsum(negatives[::-1], dtype=np.int64)[::-1][1:]
        self.counts += np.stack([tp, fp, int(y.sum()) - tp, int((~y).sum()) - fp], axis=1)

    def rows(self):
        return [dict(threshold=float(t), **metrics(c)) for t, c in zip(self.thresholds, self.counts)]


def choose_threshold(rows):
    """Predeclared: micro Dice; ties -> precision, then higher threshold."""
    def rank(row):
        return (row['dice'], row['precision'] if row['precision'] is not None else -1., row['threshold'])
    if not rows or rows[0]['positive_pixels'] == 0:
        raise ValueError('Calibration requires target-positive pixels')
    coarse = [r for r in rows if np.any(np.isclose(r['threshold'], COARSE, atol=1e-9))]
    best_coarse = max(coarse, key=rank)
    refined = [r for r in rows if abs(r['threshold'] - best_coarse['threshold']) <= .05000001]
    best = max(refined, key=rank)
    return dict(threshold=best['threshold'], coarse_best=best_coarse,
                selected=best, refined_candidates=refined,
                boundary_warning=best['threshold'] in (.1, .95),
                objective='micro Dice on historical 32 only; tie: precision, higher threshold',
                search='0.10..0.95 step 0.05; refine +/-0.05 step 0.005 within bounds')


class ProbabilityStats:
    """Brier and descriptive histograms use unmodified float32 probabilities."""
    def __init__(self):
        self.n = self.pos = 0
        self.squared_error = 0.
        self.hist = np.zeros((2, 100), dtype=np.int64)
        self.rel_n = np.zeros(20, dtype=np.int64)
        self.rel_p = np.zeros(20, dtype=np.float64)
        self.rel_y = np.zeros(20, dtype=np.float64)

    def update(self, p, y):
        p, y = np.asarray(p).ravel(), np.asarray(y).ravel()
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Targets must be binary')
        self.n += len(p)
        self.pos += int(y.sum())
        self.squared_error += float(np.square(p.astype(np.float64) - y).sum())
        h = np.minimum((p * 100).astype(int), 99)
        for label in (0, 1):
            self.hist[label] += np.bincount(h[y == label], minlength=100)
        bins = np.minimum((p * 20).astype(int), 19)
        self.rel_n += np.bincount(bins, minlength=20)
        self.rel_p += np.bincount(bins, weights=p, minlength=20)
        self.rel_y += np.bincount(bins, weights=y, minlength=20)

    def result(self):
        return dict(valid_pixels=self.n, positive_pixels=self.pos,
                    brier_score=self.squared_error / self.n if self.n else None,
                    histogram_edges=np.linspace(0, 1, 101).tolist(),
                    histogram_nonfire=self.hist[0].tolist(), histogram_fire=self.hist[1].tolist(),
                    reliability=[dict(lower=i / 20, upper=(i + 1) / 20, count=int(n),
                                      mean_probability=float(self.rel_p[i] / n) if n else None,
                                      fire_frequency=float(self.rel_y[i] / n) if n else None)
                                 for i, n in enumerate(self.rel_n)])


def macro(rows, keys=METRICS + ('average_precision', 'brier_score')):
    result = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key] = dict(mean=float(np.mean(values)) if values else None, defined_events=len(values),
                           q10=float(np.quantile(values, .1)) if values else None,
                           median=float(np.median(values)) if values else None,
                           q90=float(np.quantile(values, .9)) if values else None)
    return result


def write_csv(path, rows):
    if rows:
        with Path(path).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def event_ids(ds):
    return np.array([ds.event(i) for i in range(len(ds))])


def validate_contract(states, ds, train, expected=None):
    """Use explicit checkpoint IDs, never regenerate a convenient 32-event split."""
    all_val = sorted(set(event_ids(ds)))
    all_train = sorted(set(event_ids(train)))
    expected = expected or dict(train_events=250, train_tiles=93603, val_events=201,
                                val_tiles=78771, calibration_events=32, calibration_tiles=13759)
    if (len(all_train), len(train), len(all_val), len(ds)) != tuple(expected[k] for k in
            ('train_events', 'train_tiles', 'val_events', 'val_tiles')):
        raise ValueError('Unexpected compact corpus coverage')
    if any(e[:4] not in ('2018', '2019') for e in all_train) or any(not e.startswith('2020/') for e in all_val):
        raise ValueError('Year leakage; 2021 forbidden')
    ref = states['B1']['config']
    selected = ref['validation_events']
    if not isinstance(selected, list) or len(set(selected)) != expected['calibration_events'] or len(selected) != len(set(selected)):
        raise ValueError('Explicit historical validation event IDs required')
    if not set(selected) <= set(all_val):
        raise ValueError('Missing historical validation events')
    for name, state in states.items():
        c = state['config']
        if c['model'] != name or c['context'] != 3 or c['target'] != 'J+1' or c['seed'] != 42:
            raise ValueError('Model/temporal/seed contract differs')
        for key in ('validation_events', 'index_sha256', 'normalization_sha256', 'feature_policy'):
            if c[key] != ref[key]:
                raise ValueError(f'B1/B2 mismatch: {key}')
        if c['index_sha256'] != sha(ds.index / 'index.json') or c['normalization_sha256'] != sha(ds.index / 'normalization.json'):
            raise ValueError('Checkpoint/index/normalization mismatch')
        if state['normalization'] != ds.normalization:
            raise ValueError('Checkpoint normalization content differs')
        if state.get('updates') != 23404 or len(state.get('history', [])) != 4 or state.get('cursor') != 0 or state.get('epoch') != 5:
            raise ValueError('Expected completed epoch-4 checkpoint, 23404 updates, next epoch 5')
    fit_paths=ds.normalization.get('fit_paths',[])
    if not fit_paths or any(not any('/'+year+'/' in '/'+p.replace('\\','/')+'/' for year in ('2018','2019'))
                            or '/2020/' in p.replace('\\','/') or '/2021/' in p.replace('\\','/') for p in fit_paths):
        raise ValueError('Normalizer fit paths must contain train years only')
    ids = event_ids(ds)
    if int(np.isin(ids, selected).sum()) != expected['calibration_tiles']:
        raise ValueError('Historical validation tile coverage differs')
    holdout = sorted(set(all_val) - set(selected))
    return sorted(selected), holdout


def loader(ds, indices, args):
    return DataLoader(Subset(ds, list(map(int, indices))), batch_size=args.batch_size,
                      num_workers=args.workers, pin_memory=args.device == 'cuda',
                      prefetch_factor=2 if args.workers else None, shuffle=False,
                      generator=torch.Generator().manual_seed(42))


def audit_train(ds, args, out):
    """CPU-only, all train targets and joint masks. No predictions or fitting."""
    n = positive = observed_positive = new_positive = invalid = 0
    for step, (x, y, v) in enumerate(loader(ds, range(len(ds)), args)):
        mask = v.bool()
        if not torch.isin(y[mask], torch.tensor([0., 1.])).all():
            raise ValueError('Nonbinary training target')
        n += int(mask.sum())
        positive += int(y[mask].sum())
        obs = x[:, -1, 7:8]
        observed_positive += int(obs[mask].sum())
        new_positive += int(((y == 1) & (obs == 0) & mask).sum())
        invalid += int(mask.numel() - mask.sum())
        if step % 100 == 0:
            print(f'Train CPU audit: {min((step+1)*args.batch_size,len(ds))}/{len(ds)}', flush=True)
    result = dict(tiles=len(ds), events=len(set(event_ids(ds))), valid_pixels=n,
                  positive_pixels=positive, prevalence=positive/n, invalid_pixels=invalid,
                  observed_positive_pixels=observed_positive, new_fire_pixels=new_positive,
                  negative_to_positive=(n-positive)/positive if positive else None,
                  tile_categories={str(k): int(v) for k, v in Counter(ds.records['category'].tolist()).items()},
                  sampling='all tiles once; no positive oversampling; randomized locality blocks during training',
                  loss=dict(dice_weight=.5, bce_pos_weight=100,
                            evidence='scripts/train_compact_benchmark.py; SegmentationLoss(.5,100)',
                            checkpoint_stores_loss_hyperparameters=False),
                  feature_policy='already normalized; never normalize twice')
    save(out / 'train_audit.json', result)
    return result


def disk_guard(path, upcoming_pixels=0):
    # Two AP accumulators per batch; float64 score + two uint64 counts.
    required = 512 * 1024**2 + int(upcoming_pixels) * 48
    if shutil.disk_usage(path).free < required:
        raise RuntimeError('Insufficient AP scratch space. Retain outputs and rerun in a new output with more disk.')


def finish_ap(ap):
    # Hierarchical merge can temporarily need one additional copy of all runs.
    scratch_bytes=sum(p.stat().st_size for p in ap.owned)
    if shutil.disk_usage(ap.directory).free < scratch_bytes + 512*1024**2:
        raise RuntimeError('Insufficient scratch for exact AP merge; validation remains incomplete')
    return ap.compute()


def evaluate(model, ds, events, thresholds, args, out, label, map_events):
    """No threshold selection here. Holdout receives exactly baseline+locked."""
    out.mkdir(exist_ok=False)
    ids = event_ids(ds)
    pooled = ThresholdCounts(thresholds)
    pooled_new = ThresholdCounts(thresholds)
    pooled_old = ThresholdCounts(thresholds)
    stats = ProbabilityStats()
    global_ap = ExternalAveragePrecision(out / '_ap_global', retain_runs=False)
    per_event, new_rows, old_rows = [], [], []
    map_raw = []
    for number, event in enumerate(events, 1):
        started = time.perf_counter()
        indices = np.flatnonzero(ids == event)
        # Temporal/spatial locality for I/O only; exact same complete event domain.
        records = ds.records[indices]
        indices = indices[np.lexsort((records['day'], records['col'], records['row']))]
        event_counts, event_new, event_old = [ThresholdCounts(thresholds) for _ in range(3)]
        event_stats = ProbabilityStats()
        ap = ExternalAveragePrecision(out / f'_ap_event_{number:03d}', retain_runs=False)
        invalid = 0
        day = None
        tiles = []
        if event in map_events:
            days = sorted(set(int(ds.records[i]['day']) for i in indices))
            day = days[len(days)//2]  # Metadata-only case selection, not best metric.
        cursor = 0
        with torch.inference_mode():
            for x, y, v in loader(ds, indices, args):
                mask = v.numpy().astype(bool)
                yy = y.numpy()
                observed = x[:, -1, 7:8].numpy()
                pp = model(x.to(args.device, non_blocking=True)).sigmoid().cpu().numpy()
                if pp.shape != yy.shape or mask.shape != yy.shape or not np.isfinite(pp).all():
                    raise ValueError('Probability/target/mask mismatch')
                if not np.isin(yy[mask], [0, 1]).all() or not np.isin(observed[mask], [0, 1]).all():
                    raise ValueError('Nonbinary target/observed fire channel')
                p, target = pp[mask], yy[mask]
                event_counts.update(p, target)
                stats.update(p, target)
                event_stats.update(p, target)
                new = mask & (observed == 0)
                old = mask & (observed == 1)
                event_new.update(pp[new], yy[new])
                event_old.update(pp[old], yy[old])
                disk_guard(out, len(p))
                if len(p):
                    ap.update(p, target)
                    global_ap.update(p, target)
                invalid += int(mask.size - mask.sum())
                for j in range(len(x)):
                    record = ds.records[indices[cursor+j]]
                    if day is not None and int(record['day']) == day:
                        tiles.append((int(record['row']), int(record['col']), pp[j,0].copy(),
                                      yy[j,0].copy(), mask[j,0].copy(), observed[j,0].copy()))
                cursor += len(x)
        if cursor != len(indices):
            raise RuntimeError('Incomplete event, no official score')
        print(f'{label}: {event} inference complete; exact event AP...', flush=True)
        event_ap = finish_ap(ap)
        pooled.counts += event_counts.counts
        pooled_new.counts += event_new.counts
        pooled_old.counts += event_old.counts
        ep = event_stats.result()
        for row in event_counts.rows():
            per_event.append(dict(event_id=event, tiles=cursor, invalid_pixels=invalid,
                                  **row, average_precision=event_ap, brier_score=ep['brier_score']))
        for destination, counter in ((new_rows,event_new),(old_rows,event_old)):
            destination.extend(dict(event_id=event, **r) for r in counter.rows())
        save(out / f'event_{number:03d}_probabilities.json', dict(event_id=event, **ep))
        if tiles:
            map_raw.append(write_scene_raw(out, event, day, tiles))
        print(f'{label}: {number}/{len(events)} events; {cursor} tiles; {time.perf_counter()-started:.1f}s', flush=True)
    print(f'{label}: exact pooled AP (CPU/disk merge, no GPU training)...', flush=True)
    ap_value = finish_ap(global_ap)
    probability = stats.result()
    result = dict(status='COMPLETE', events=list(events), tiles=sum(r['tiles'] for r in per_event[::len(thresholds)]),
                  curves=[dict(**r, average_precision=ap_value, brier_score=probability['brier_score']) for r in pooled.rows()],
                  new_fire_candidates=pooled_new.rows(), observed_fire_candidates=pooled_old.rows(),
                  probabilities=probability, per_event=per_event,
                  per_event_new_fire=new_rows, per_event_observed_fire=old_rows, maps=map_raw)
    save(out / 'evaluation.json', result)
    write_csv(out / 'threshold_metrics.csv', result['curves'])
    write_csv(out / 'per_event.csv', per_event)
    write_csv(out / 'new_fire_per_event.csv', new_rows)
    return result


def write_scene_raw(out, event, day, tiles):
    h = max(r + p.shape[0] for r,c,p,y,v,o in tiles)
    w = max(c + p.shape[1] for r,c,p,y,v,o in tiles)
    if h*w > 40_000_000:
        raise ValueError('Illustration mosaic unexpectedly large')
    probability = np.full((h,w), np.nan, dtype=np.float32)
    target = np.zeros((h,w), dtype=np.uint8)
    observed = target.copy()
    valid = np.zeros((h,w), dtype=bool)
    occupied = valid.copy()
    for r,c,p,y,v,o in tiles:
        region = np.s_[r:r+p.shape[0], c:c+p.shape[1]]
        if occupied[region].any():
            raise ValueError('Overlapping compact tiles')
        probability[region], target[region], valid[region], observed[region] = p, y, v, o
        occupied[region] = True
    name = event.replace('/', '__') + '_' + date.fromordinal(day).isoformat()
    path = out / (name + '_raw.npz')
    np.savez_compressed(path, probability=probability, target=target, observed=observed, valid=valid)
    return dict(event_id=event, target_date=date.fromordinal(day).isoformat(), raw=path.name,
                shape=[h,w], tile_count=len(tiles), spatial_reference='pixel grid; compact does not store CRS/affine',
                selection='evenly spaced event IDs; median target date; all tiles, padding invalid')


def render_maps(result, out, model, threshold, checkpoint_sha):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    assets = []
    for case in result['maps']:
        with np.load(out / case['raw'], allow_pickle=False) as z:
            p, y, v, observed = [z[k] for k in ('probability','target','valid','observed')]
        segmentation = (p >= np.float32(threshold)) & v
        baseline = (p >= .5) & v
        # 0 TN, 1 TP, 2 FP, 3 FN, 255 invalid.
        error = np.where(segmentation, np.where(y,1,2), np.where(y,3,0)).astype(np.uint8)
        error[~v] = 255
        stem = case['raw'].removesuffix('_raw.npz')
        np.savez_compressed(out / (stem+'.npz'), probability=p, target=y, observed=observed,
                            valid=v, segmentation=segmentation, segmentation_05=baseline, error=error)
        fig, axes = plt.subplots(1,6,figsize=(21,4))
        panels = [(observed,'Feu observe J','inferno'), (y,'Verite J+1','inferno'),
                  (p,'Probabilite','inferno'), (baseline,'Seuil 0.50','inferno'),
                  (segmentation,f'Seuil {threshold:.3f}','inferno'),
                  (error,'Erreurs: TP vert, FP rouge, FN bleu',ListedColormap(['#16212b','#30bb80','#f25252','#3f8efc']))]
        for ax,(array,title,cmap) in zip(axes,panels):
            ax.imshow(np.ma.array(array,mask=~v),cmap=cmap,vmin=0,vmax=3 if array is error else 1,interpolation='nearest')
            ax.set_title(title,fontsize=9)
            ax.axis('off')
        fig.suptitle(f'{model} | {case["event_id"]} | {case["target_date"]} | cas metadata, pas representatif')
        fig.tight_layout()
        fig.savefig(out / (stem+'.png'),dpi=140)
        plt.close(fig)
        # Native resolution transparent layers for React canvas/image overlays.
        layers = {}
        for key,array,cmap,vmax in [('probability',p,plt.get_cmap('inferno'),1),
                                    ('target',y,plt.get_cmap('inferno'),1),
                                    ('segmentation',segmentation,plt.get_cmap('inferno'),1),
                                    ('errors',error,panels[-1][2],3)]:
            rgba = cmap(np.nan_to_num(array.astype(float),nan=0)/vmax)
            rgba[...,3] = v.astype(float)
            name = stem+'_'+key+'.png'
            plt.imsave(out/name,rgba)
            layers[key] = name
        assets.append(dict(**case, model=model, threshold=threshold, checkpoint_sha256=checkpoint_sha,
                           npz=stem+'.npz', preview=stem+'.png', layers=layers,
                           error_codes={'0':'TN','1':'TP','2':'FP','3':'FN','255':'invalid'}))
    save(out/'map_manifest.json', dict(schema='geofirex-calibration-maps-v1', cases=assets))
    return assets


def at(result, threshold):
    return next(r for r in result['curves'] if abs(r['threshold'] - threshold) < 1e-8)


def check_historical(state,result):
    historical=state['history'][-1]['validation']
    actual=at(result,.5)
    deltas={k:actual[k]-historical[k] for k in (*METRICS,'average_precision')
            if historical.get(k) is not None and actual.get(k) is not None}
    if historical.get('samples_seen') != result['tiles']:
        raise ValueError('Historical validation tile count not reproduced')
    if historical.get('pixels',actual['valid_pixels']) != actual['valid_pixels'] or historical.get('positive_pixels',actual['positive_pixels']) != actual['positive_pixels']:
        raise ValueError('Historical validation valid/positive pixel counts not reproduced')
    if any(abs(delta)>.002 for delta in deltas.values()):
        raise ValueError(f'Historical 0.5 scores not reproduced within 0.002: {deltas}')
    return dict(status='PASS',historical=historical,recomputed=actual,deltas=deltas,
                metric_absolute_tolerance=.002,reason='allow device/kernel floating point differences; identical pixel domain required')


def diagnosis(results,selected,train_stats,out):
    rows=[]
    for phase in results:
        for model,result in results[phase].items():
            threshold=selected[model]['threshold'];full=at(result,threshold)
            new=next(r for r in result['new_fire_candidates'] if abs(r['threshold']-threshold)<1e-8)
            old=next(r for r in result['observed_fire_candidates'] if abs(r['threshold']-threshold)<1e-8)
            for key in ('tp','fp','fn','tn','valid_pixels','positive_pixels'):
                if new[key]+old[key]!=full[key]:raise ValueError('Temporal subdomains do not partition valid pixels')
            prevalence=full['prevalence'];train_prior=train_stats['prevalence']
            rows.append(dict(phase=phase,model=model,threshold=threshold,
                prevalence=prevalence,predicted_positive_rate=full['predicted_positive_pixels']/full['valid_pixels'],
                false_discovery_rate=1-full['precision'] if full['precision'] is not None else None,
                new_fire_positive_pixels=new['positive_pixels'],new_fire_recall=new['recall'],
                new_fire_dice=new['dice'],new_fire_precision=new['precision'],
                new_fire_fp=new['fp'],observed_fire_fp=old['fp'],
                fp_share_on_previously_nonburning=new['fp']/full['fp'] if full['fp'] else None,
                brier_score=full['brier_score'],
                brier_constant_train_prior=prevalence*(1-train_prior)**2+(1-prevalence)*train_prior**2))
    save(out/'diagnosis.json',dict(loss=train_stats['loss'],train_prevalence=train_stats['prevalence'],rows=rows,
        conclusions_pending='Interpret observed histogram/reliability, threshold gain and temporal FP contribution together; no causal conclusion from weighting alone',
        future_experiments=[
            dict(id='E1',trigger='AP and validation loss still improve in checkpoint history',
                 treatment='continue existing checkpoint for 1 epoch, unchanged loss and data; bounded budget subject to authorization',
                 purpose='control for additional optimization; no architecture change'),
            dict(id='E2',trigger='FP remain high at selected threshold and reliability shows overprediction',
                 treatment='same initial checkpoint and 1-epoch budget as E1; change only pos_weight 100 to min(100,sqrt(train negative/positive pixel ratio)); Dice weight stays 0.5',
                 purpose='paired loss ablation; weight derived from train only; no additional temporal change')],
        validation_rule='future experiments selected on same 32 only; the now-observed 169 are a confirmation panel, no longer a fresh independent test',
        training_authorized_or_executed=False))


def common_comparison(results, selected):
    rows, distributions = [], []
    for phase in ('calibration32','holdout169'):
        for model in ('B1','B2'):
            result = results[phase][model]
            for mode,t in (('baseline_05',.5),('calibrated',selected[model]['threshold'])):
                per = [r for r in result['per_event'] if abs(r['threshold']-t)<1e-8]
                row = dict(phase=phase, model=model, mode=mode, events=len(result['events']),
                           tiles=result['tiles'], **at(result,t))
                rows.append(row)
                distributions.append(dict(phase=phase,model=model,mode=mode,threshold=t,macro_event=macro(per)))
        # Pixel counts and event coverage must match across both networks.
        one,two = results[phase]['B1'],results[phase]['B2']
        if one['events'] != two['events'] or one['tiles'] != two['tiles']:
            raise ValueError('B1/B2 event domain mismatch')
        for r,s in zip(one['per_event'][::len(one['curves'])],two['per_event'][::len(two['curves'])]):
            if (r['event_id'],r['valid_pixels'],r['positive_pixels']) != (s['event_id'],s['valid_pixels'],s['positive_pixels']):
                raise ValueError('B1/B2 valid-pixel domain mismatch')
    return rows,distributions


def diagnostic_figures(results, selected, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for model in ('B1','B2'):
        r=results['calibration32'][model]
        for ax,key in zip(axes,('dice','precision','recall')):
            ax.plot([c['threshold'] for c in r['curves']],[c[key] for c in r['curves']],label=model)
            ax.axvline(selected[model]['threshold'],alpha=.3)
            ax.set(xlabel='Seuil (32 calibration seulement)',ylabel=key)
            ax.legend()
    fig.tight_layout();fig.savefig(out/'calibration_curves.png',dpi=140);plt.close(fig)
    for phase in ('calibration32','holdout169'):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for model in ('B1','B2'):
            result=results[phase][model];ps=result['probabilities']
            for label,key in (('non-feu','histogram_nonfire'),('feu','histogram_fire')):
                values=np.array(ps[key]);den=values.sum()
                axes[0,0].plot(np.arange(100)/100+.005,values/den if den else values,label=model+' '+label)
            rel=[r for r in ps['reliability'] if r['count']]
            axes[0,1].plot([r['mean_probability'] for r in rel],[r['fire_frequency'] for r in rel],'.-',label=model)
            per=[r for r in result['per_event'] if abs(r['threshold']-selected[model]['threshold'])<1e-8]
            for ax,key in zip((axes[0,2],axes[1,0],axes[1,1]),('dice','precision','recall')):
                values=sorted(r[key] for r in per if r[key] is not None)
                ax.step(values,np.arange(1,len(values)+1)/max(1,len(values)),label=model)
                ax.set(xlabel=key,ylabel='CDF evenements')
            axes[1,2].scatter([r['prevalence'] for r in per],[r['fp']/r['valid_pixels'] if r['valid_pixels'] else 0 for r in per],label=model,alpha=.6)
        axes[0,0].set(xlabel='Probabilite',ylabel='Frequence conditionnelle par classe',yscale='log')
        axes[0,1].plot([0,1],[0,1],'k--');axes[0,1].set(xlabel='Probabilite moyenne',ylabel='Frequence feu')
        axes[1,2].set(xlabel='Prevalence par evenement',ylabel='FP / pixels valides')
        for ax in axes.flat:ax.legend()
        fig.suptitle(phase+' | seuils fixes pour holdout');fig.tight_layout()
        fig.savefig(out/(phase+'_diagnostics.png'),dpi=140);plt.close(fig)


def run(args):
    from geofirex.fast_compact import FastCompactTiles
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'status_started.json',dict(status='RUNNING',scores_official=False))
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; select --device cpu explicitly')
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    paths=dict(B1=Path(args.b1),B2=Path(args.b2))
    # Only the user's trusted training checkpoints; they contain optimizer/RNG
    # Python objects and therefore need weights_only=False. Nothing is resumed.
    states={m:torch.load(p,map_location='cpu',weights_only=False) for m,p in paths.items()}
    ds=FastCompactTiles(args.corpus,args.index,'val',cache_mb=args.cache_mb)
    train=FastCompactTiles(args.corpus,args.index,'train',cache_mb=args.cache_mb)
    calibration,holdout=validate_contract(states,ds,train)
    provenance={m:dict(path=str(paths[m]),sha256=sha(paths[m]),updates=s['updates'],epoch=s['epoch'],
                       history=s['history'],config=s['config']) for m,s in states.items()}
    save(out/'protocol.json',dict(schema='geofirex-threshold-calibration-v1',checkpoints=provenance,
        calibration_events=calibration,holdout_events=holdout,threshold_candidates=DENSE.tolist(),
        index_sha256=sha(ds.index/'index.json'),normalization_sha256=sha(ds.index/'normalization.json'),
        objective='micro Dice; tie precision then higher threshold',test_2021_pixels_opened=0,
        empty_policy='no valid pixels: null; empty-empty Dice/IoU=1; precision without predictions null; recall/AP without positives null; macro excludes null and reports defined count',
        ap_policy='exact noninterpolated AP, global score ties, unmodified float32 probabilities; AP and Brier independent of threshold',
        validity='intersection of target and all five historical day masks, including padding; unchanged FastCompactTiles',
        generalization='169 are untouched by threshold choice; 32 were already used during training; no independent 2021 test claim',
        versions=dict(torch=torch.__version__,numpy=np.__version__)))
    print('Verified checkpoints, 250 train / 32 calibration / 169 holdout events.',flush=True)
    train_stats=audit_train(train,args,out);train.close()
    results=dict(calibration32={},holdout169={});selected={};maps=[]
    case_events=set()
    for events in (calibration,holdout):
        case_events.update(events[i] for i in np.linspace(0,len(events)-1,3,dtype=int))
    for model in ('B1','B2'):
        network=make_model(model,8,3,**({'architecture':'spatial_cnn_v2'} if model=='B1' else {}))
        network.load_state_dict(states[model]['model'],strict=True);network.to(args.device).eval()
        result=evaluate(network,ds,calibration,DENSE,args,out/(model+'_calibration32'),model+' calibration32',case_events)
        results['calibration32'][model]=result
        save(out/(model+'_historical_reproduction.json'),check_historical(states[model],result))
        selected[model]=choose_threshold(result['curves'])
        write_csv(out/(model+'_calibration32')/'coarse_threshold_metrics.csv',
                  [r for r in result['curves'] if np.any(np.isclose(r['threshold'],COARSE,atol=1e-9))])
        write_csv(out/(model+'_calibration32')/'refined_threshold_metrics.csv',selected[model]['refined_candidates'])
        del network
    # Lock BOTH thresholds before opening holdout tiles with either network.
    lock=dict(schema='geofirex-threshold-lock-v1',thresholds={m:s['threshold'] for m,s in selected.items()},
              selection=selected,calibration_events=calibration,calibration_tiles=13759,
              checkpoint_sha256={m:p['sha256'] for m,p in provenance.items()},
              normalization_sha256=sha(ds.index/'normalization.json'),selection_scope='32 historical 2020 events only',
              decision='probability >= float32(threshold)',probability_calibration_applied=False)
    save(out/'thresholds_locked.json',lock);lock_hash=sha(out/'thresholds_locked.json')
    print('THRESHOLDS LOCKED:',lock['thresholds'],'SHA256:',lock_hash,flush=True)
    for model in ('B1','B2'):
        network=make_model(model,8,3,**({'architecture':'spatial_cnn_v2'} if model=='B1' else {}))
        network.load_state_dict(states[model]['model'],strict=True);network.to(args.device).eval()
        thresholds=sorted(set([.5,selected[model]['threshold']]))
        results['holdout169'][model]=evaluate(network,ds,holdout,thresholds,args,out/(model+'_holdout169'),model+' holdout169',case_events)
        if sha(out/'thresholds_locked.json')!=lock_hash:
            raise ValueError('Threshold lock changed during holdout')
        del network
    for phase in results:
        for model,result in results[phase].items():
            maps.extend(render_maps(result,out/(model+'_'+phase),model,selected[model]['threshold'],provenance[model]['sha256']))
    comparison,distribution=common_comparison(results,selected)
    save(out/'comparison.json',dict(rows=comparison,distributions=distribution))
    write_csv(out/'comparison.csv',comparison)
    diagnosis(results,selected,train_stats,out)
    diagnostic_figures(results,selected,out)
    assets=dict(schema='geofirex-calibration-app-v1',thresholds=lock['thresholds'],threshold_lock_sha256=lock_hash,
                probability_metrics_threshold_invariant=True,comparison=comparison,distributions=distribution,
                maps=[dict(**m,directory=next(model+'_'+phase for phase in results for model in results[phase]
                      if m['model']==model and m['event_id'] in results[phase][model]['events'])) for m in maps])
    save(out/'app_manifest.json',assets)
    save(out/'status_complete.json',dict(status='COMPLETE',threshold_lock_sha256=lock_hash,
        train_audit_complete=True,calibration_events=32,holdout_events=169,
        training_executed=False,thresholds=lock['thresholds'],test_2021_pixels_opened=0))
    ds.close()
    print(json.dumps(dict(status='COMPLETE',out=str(out),thresholds=lock['thresholds']),indent=2))
    return assets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True);p.add_argument('--index',required=True)
    p.add_argument('--b1',required=True);p.add_argument('--b2',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cache-mb',type=int,default=64)
    args=p.parse_args()
    if args.batch_size<1 or args.workers<0 or args.cache_mb<0:p.error('Invalid resource settings')
    run(args)


if __name__=='__main__':
    main()
