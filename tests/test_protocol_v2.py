"""CPU-only checks of the new split and guarded compact reindexing."""
from datetime import date, timedelta
import io
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from geofirex.fast_compact import DTYPE
from geofirex.protocol_v2 import (V2CompactTiles, chronological_split, check_separation,
                                  event_intervals, make_v2_loaders, save, sha256, write_csv, build_v2)


def observation(event, day):
    return dict(event_id=event, date=day, relative_path=f'{event}/{day}.tif', status='ok',
                sha256=(event + day).encode().hex())


def test_chronological_cutoff_embargo_and_event_year_is_not_observation_year():
    intervals = {
        '2018/fire_1': ('2017-12-29', '2018-01-07'),
        '2019/fire_2': ('2019-02-01', '2019-02-09'),
        '2020/fire_3': ('2020-01-01', '2020-01-09'),
        '2020/fire_4': ('2020-02-01', '2020-02-09'),
        '2020/fire_5': ('2020-03-01', '2020-04-01'),
        '2020/fire_6': ('2020-04-01', '2020-04-09'),
        '2021/fire_7': ('2021-01-01', '2021-01-09'),
    }
    roles, best, candidates, clean = chronological_split(intervals)
    assert best['cutoff'] == '2020-04-01'
    assert roles['2020/fire_5'] == 'purged'  # includes the cutoff, not train
    assert roles['2021/fire_7'] == 'test'
    assert best['train_fraction'] == .8
    assert all(end < best['cutoff'] for e, (start, end) in intervals.items() if roles[e] == 'train')
    assert clean is not None and candidates


@pytest.mark.parametrize('mutation', ['duplicate_date', 'reused_path', 'crossing_train', 'test_train', 'same_sha'])
def test_metadata_leakage_is_rejected(mutation):
    rows = [observation('2018/fire_1', '2018-01-01'), observation('2020/fire_2', '2020-08-23'),
            observation('2021/fire_3', '2021-01-01')]
    roles = {'2018/fire_1': 'train', '2020/fire_2': 'val', '2021/fire_3': 'test'}
    if mutation == 'duplicate_date': rows.append(rows[0].copy())
    if mutation == 'reused_path': rows[1]['relative_path'] = rows[0]['relative_path']
    if mutation == 'crossing_train': roles['2020/fire_2'] = 'train'
    if mutation == 'test_train': roles['2021/fire_3'] = 'train'
    if mutation == 'same_sha': rows[1]['sha256'] = rows[0]['sha256']
    with pytest.raises(ValueError):
        check_separation(rows, roles, '2020-08-23')


@pytest.fixture
def compact(tmp_path):
    corpus = tmp_path / 'corpus'; index = tmp_path / 'v2' / 'index'
    corpus.mkdir(); index.mkdir(parents=True)
    archives = ['part0/train/2018/fire_1.zip', 'part0/val/2020/fire_2.zip']
    events = dict(zip(archives, ['2018/fire_1', '2020/fire_2']))
    routes = {}; roles = {}; expected = {}
    for a, (logical, event) in enumerate(events.items()):
        role = ('train', 'val')[a]; roles[event] = role
        day = date(2018, 1, 7) if a == 0 else date(2020, 9, 7)
        path = corpus / logical; path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'x') as z:
            for delta in range(-5, 1):
                x = np.full((8, 64, 64), delta, np.float32)
                x[7] = (delta in (-1, 0))
                valid = np.ones((1, 64, 64), bool)
                if delta == -5: valid[:, 0, 0] = False
                if delta == 0: valid[:, -1, -1] = False
                b = io.BytesIO(); np.savez_compressed(b, x=x, valid=valid)
                z.writestr(f'{day + timedelta(days=delta)}_0_0.npz', b.getvalue())
        routes[logical] = dict(alias=logical.replace('/', '__')+'.bin', bytes=path.stat().st_size,
                               sha256=sha256(path), partition='p01')
        records = np.array([(a, day.toordinal(), 0, 0, 1)], dtype=DTYPE)
        np.save(index / (role+'.npy'), records, allow_pickle=False)
        expected[role] = (day, logical)
    write_csv(index.parent / 'events.csv', [dict(event_id=e, split=r) for e, r in roles.items()])
    save(index / 'normalization.json', {'frozen': True})
    meta = dict(schema='geofirex-chronological-v2', archives=archives, events=events,
                event_roles=roles, cutoff='2020-08-23', routes=routes,
                events_csv_sha256=sha256(index.parent/'events.csv'),
                normalization_sha256=sha256(index/'normalization.json'),
                index_sha256={r:sha256(index/(r+'.npy')) for r in ('train', 'val')})
    save(index/'index.json', meta)
    return corpus, index, meta


def test_cpu_loader_shapes_six_day_validity_targets_and_no_rebalance(compact):
    corpus, index, _ = compact
    train, val = make_v2_loaders(corpus, index, batch_size=16)
    for loader in (train, val):
        x, y, valid = next(iter(loader))
        assert x.shape == (1, 3, 8, 64, 64)
        assert torch.equal(x[0, :, 0, 1, 1], torch.tensor([-3., -2., -1.]))
        assert y.all() and x[0, -1, 7].all()
        assert valid.sum().item() == 4094
        assert valid[0, 0, 0, 0] == 0 and valid[0, 0, -1, -1] == 0
        assert len(loader.dataset) == 1
        loader.dataset.close()


def test_worker_serialization_preserves_cpu_tensors(compact):
    corpus, index, _ = compact
    ds = V2CompactTiles(corpus, index, 'train', cache_mb=1)
    expected = ds[0]; copy = pickle.loads(pickle.dumps(ds))
    assert all(torch.equal(a, b) for a, b in zip(expected, copy[0]))
    ds.close(); copy.close()


@pytest.mark.parametrize('role', ['test', 'purged', '2021'])
def test_locked_roles_rejected_before_path_resolution(tmp_path, role):
    with pytest.raises(ValueError, match='locked'):
        V2CompactTiles(tmp_path / 'absent', tmp_path / 'absent', role)


@pytest.mark.parametrize('name', ['train.npy', 'normalization.json', '../events.csv'])
def test_changed_index_or_manifest_rejected(compact, name):
    corpus, index, _ = compact
    with (index / name).open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError, match='differs'):
        V2CompactTiles(corpus, index, 'train')


def test_wrong_event_role_rejected_even_with_valid_file_hashes(compact):
    corpus, index, meta = compact
    meta['event_roles']['2018/fire_1'] = 'val'
    (index/'index.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='role mismatch'):
        V2CompactTiles(corpus, index, 'train')


def test_opaque_publication_can_be_read_directly_without_copy(compact):
    corpus, index, meta = compact
    logical = meta['archives'][0]
    (corpus/logical).rename(corpus/meta['routes'][logical]['alias'])
    ds = V2CompactTiles(corpus, index, 'train')
    assert ds[0][0].shape == (3, 8, 64, 64)
    ds.close()


def test_ambiguous_mounts_rejected(compact):
    corpus, index, meta = compact
    logical = meta['archives'][0]
    (corpus/meta['routes'][logical]['alias']).write_bytes((corpus/logical).read_bytes())
    with pytest.raises(ValueError, match='ambiguous'):
        V2CompactTiles(corpus, index, 'train')


def test_real_v2_manifest_and_indices_metadata_only():
    root = Path('reports/protocol_v2_20261118')
    if not root.exists(): pytest.skip('Generated V2 delivery not present')
    summary = json.loads((root/'split_summary.json').read_text())
    checks = json.loads((root/'leakage_checks.json').read_text())
    assert summary['roles'] == {'train':318, 'val':80, 'purged':53, 'test':156}
    assert summary['selected']['cutoff'] == '2020-08-23'
    assert checks['status'] == 'PASS' and checks['test_pixels_opened'] == 0
    assert checks['train_last_date'] < checks['validation_first_date']
    assert checks['frozen_normalizer_subset_of_v2_train']
    assert len(np.load(root/'index/train.npy', allow_pickle=False)) == 115900
    assert len(np.load(root/'index/val.npy', allow_pickle=False)) == 30882


def test_real_split_rebuild_is_reproducible_and_never_opens_pixels(tmp_path, monkeypatch):
    root = Path('reports/protocol_v2_20261118')
    if not root.exists(): pytest.skip('Generated V2 delivery not present')
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata builder attempted a pixel/archive read')
    import rasterio
    monkeypatch.setattr(rasterio, 'open', forbidden)
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    target = tmp_path/'rebuilt'
    build_v2('reports/full_corpus_v1/prepared_v1', 'reports/gpu_acceleration_20261111_v1/index',
             'reports/mission_20261111/compact_full',
             'reports/gpu_acceleration_20261111_v1/publication', target)
    for name in ['events.csv', 'windows.csv', 'cutoff_candidates.csv', 'leakage_checks.json',
                 'split_summary.json', 'index/index.json', 'index/train.npy', 'index/val.npy',
                 'index/normalization.json']:
        assert sha256(root/name) == sha256(target/name)
