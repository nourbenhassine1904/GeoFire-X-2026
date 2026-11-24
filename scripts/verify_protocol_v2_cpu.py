"""New split distributions and CPU loader parity. Never builds a model.

Only the certified 2018-2020 compact corpus is decoded, once in date order.
This computes the NEW V2 distributions, not a repeat of the historical audit.
"""
from collections import Counter, defaultdict, deque
from datetime import date
import argparse
import io
import json
from pathlib import Path
import pickle
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from geofirex.fast_compact import FastCompactTiles
from geofirex.protocol_v2 import V2CompactTiles, make_v2_loaders, save, sha256, write_csv


def cpu_parity(corpus, index, old_index, out):
    loaders = make_v2_loaders(corpus, index, workers=0)
    old = {r: FastCompactTiles(corpus, old_index, r, cache_mb=8) for r in ('train', 'val')}
    lookup = {r: {tuple(map(int, rec.tolist()[:4])): i for i, rec in enumerate(ds.records)}
              for r, ds in old.items()}
    results = []
    for role, loader in zip(('train', 'val'), loaders):
        ds = loader.dataset
        picks = sorted({0, len(ds)-1, *[int(ids[0]) for k in range(3)
                                      if len(ids := np.flatnonzero(ds.records['category'] == k))]})
        batch = next(iter(DataLoader(Subset(ds, picks), batch_size=len(picks), num_workers=0)))
        for offset, i in enumerate(picks):
            rec = ds.records[i]; event = ds.event(i)
            historical_role = 'val' if event.startswith('2020/') else 'train'
            j = lookup[historical_role][tuple(map(int, rec.tolist()[:4]))]
            expected = old[historical_role][j]
            if not all(torch.equal(a[offset], b) for a, b in zip(batch, expected)):
                raise ValueError('V2 tensors/targets/domain differ from certified compact')
        if not all(torch.isfinite(t).all() for t in batch):
            raise ValueError('Nonfinite CPU batch')
        # Exercise the process serialization path without spawning model workers.
        ds.close(); restored = pickle.loads(pickle.dumps(ds))
        if not all(torch.equal(a, b) for a, b in zip(restored[picks[0]], ds[picks[0]])):
            raise ValueError('Worker serialization changes batch')
        restored.close()
        order = loader.sampler.order() if hasattr(loader.sampler, 'order') else np.arange(len(ds))
        if len(np.unique(order)) != len(ds) or set(order) != set(range(len(ds))):
            raise ValueError('Loader omits/duplicates tiles')
        results.append(dict(split=role, checked_indices=picks, batch_shape=list(batch[0].shape),
                            exact_x_y_valid_parity=True, worker_pickle_parity=True,
                            exhaustive_sampler_tiles=len(order)))
        ds.close()
    for ds in old.values():
        ds.close()
    save(out / 'cpu_loader_checks.json', dict(status='PASS', checks=results, device='cpu',
         model_forwards=0, optimizer_steps=0, test_pixels_opened=0,
         index_sha256=sha256(index / 'index.json')))


def distributions(corpus, index, out):
    started = time.perf_counter()
    meta = json.loads((index / 'index.json').read_text())
    # The certified historical indices include the embargoed events too.
    old_index = Path('reports/gpu_acceleration_20261111_v1/index')
    records = np.concatenate([np.load(old_index / (r + '.npy'), allow_pickle=False) for r in ('train', 'val')])
    by_archive = defaultdict(list)
    for rec in records:
        by_archive[int(rec['archive'])].append(rec)
    rows = []; opened = 0
    with (out / 'distribution_progress.jsonl').open('x', encoding='utf-8') as progress:
        for a, archive in enumerate(meta['archives']):
            event = meta['events'][archive]; role = meta['event_roles'][event]
            if event[:4] not in ('2018', '2019', '2020'):
                raise ValueError('2021 prohibited before any ZIP open')
            event_records = by_archive[a]
            wanted = {(int(r['day']), int(r['row']), int(r['col'])): int(r['category']) for r in event_records}
            counts = Counter(); categories = Counter(); window_days = set()
            day_members = defaultdict(list)
            with zipfile.ZipFile(corpus / archive) as z:
                for member in z.namelist():
                    day, row, col = member.removesuffix('.npz').split('_')
                    ordinal = date.fromisoformat(day).toordinal()
                    if ordinal >= date(2021, 1, 1).toordinal():
                        raise ValueError('Calendar 2021 prohibited')
                    day_members[ordinal].append((member, int(row), int(col)))
                history = deque(maxlen=6)
                for day, members in sorted(day_members.items()):
                    current = {}
                    for member, row, col in members:
                        with np.load(io.BytesIO(z.read(member)), allow_pickle=False) as npz:
                            fire = npz['x'][7:8].astype(bool); valid = npz['valid'].astype(bool)
                        opened += 1
                        current[row, col] = (fire, valid)
                    history.append((day, current))
                    if len(history) != 6 or [d for d, _ in history] != list(range(day-5, day+1)):
                        continue
                    for (row, col), (y, v) in current.items():
                        key = day, row, col
                        if key not in wanted:
                            raise ValueError('Unexpected complete compact window')
                        valid = v.copy()
                        for _, prior in list(history)[:-1]:
                            valid &= prior[row, col][1]
                        observed = history[-2][1][row, col][0]
                        new_domain = valid & ~observed
                        counts.update(tiles=1, valid_pixels=int(valid.sum()),
                                      positive_pixels=int((valid & y).sum()),
                                      new_candidate_pixels=int(new_domain.sum()),
                                      new_positive_pixels=int((new_domain & y).sum()),
                                      observed_positive_pixels=int((valid & observed).sum()))
                        categories[['negative', 'positive', 'hard_negative'][wanted[key]]] += 1
                        window_days.add(day)
                if counts['tiles'] != len(wanted):
                    raise ValueError('Distribution omitted compact tiles')
            result = dict(event_id=event, split=role, year=event[:4], windows=len(window_days),
                          **counts, **{k+'_tiles': categories[k] for k in ('negative', 'positive', 'hard_negative')})
            result['prevalence'] = counts['positive_pixels']/counts['valid_pixels'] if counts['valid_pixels'] else None
            result['new_prevalence'] = counts['new_positive_pixels']/counts['new_candidate_pixels'] if counts['new_candidate_pixels'] else None
            rows.append(result)
            progress.write(json.dumps(dict(event_id=event, events=len(rows), seconds=time.perf_counter()-started))+'\n'); progress.flush()
            if len(rows) % 30 == 0:
                print(f'distributions: {len(rows)}/451 events, {time.perf_counter()-started:.1f}s', flush=True)
    write_csv(out / 'event_distributions.csv', rows)
    pooled = {}
    integers = ['windows', 'tiles', 'valid_pixels', 'positive_pixels', 'new_candidate_pixels',
                'new_positive_pixels', 'observed_positive_pixels', 'negative_tiles', 'positive_tiles', 'hard_negative_tiles']
    for role in ('train', 'val', 'purged'):
        selected = [r for r in rows if r['split'] == role]
        total = {k: sum(r[k] for r in selected) for k in integers}
        total.update(events=len(selected), by_year=dict(Counter(r['year'] for r in selected)),
                     events_with_new_fire=sum(r['new_positive_pixels'] > 0 for r in selected),
                     prevalence=total['positive_pixels']/total['valid_pixels'],
                     new_prevalence=total['new_positive_pixels']/total['new_candidate_pixels'],
                     new_fire_count_quantiles=dict(zip(('min', 'q10', 'median', 'q90', 'max'),
                        np.quantile([r['new_positive_pixels'] for r in selected], [0, .1, .5, .9, 1]).tolist())))
        pooled[role] = total
    save(out / 'distributions.json', dict(status='COMPUTED_FROM_EXISTING_2018_2020_COMPACT',
         pooled=pooled, seconds=time.perf_counter()-started, compact_day_tiles_decoded=opened,
         test_pixels_opened=0, model_forwards=0, optimizer_steps=0,
         domain='Target and five previous validity footprints; inputs last three days; repeated window pixels, not unique geographic pixels',
         new_fire='valid & ~observed_J & target_J_plus_1; new detection, not proven physical ignition',
         cutoff_selection_used_labels=False, index_sha256=sha256(index / 'index.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--corpus', type=Path, default=Path('reports/mission_20261111/compact_full'))
    p.add_argument('--out', type=Path, default=Path('reports/protocol_v2_20261118'))
    p.add_argument('--mode', choices=['loaders', 'distributions'], required=True)
    a = p.parse_args(); index = a.out / 'index'
    torch.set_num_threads(2)
    if a.mode == 'loaders':
        cpu_parity(a.corpus, index, Path('reports/gpu_acceleration_20261111_v1/index'), a.out)
    else:
        distributions(a.corpus, index, a.out)
