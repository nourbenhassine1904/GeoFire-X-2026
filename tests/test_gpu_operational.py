"""Deployment guards, normalization provenance, and complete-event batching."""
import copy
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_full_corpus import corpus, certify
from geofirex.corpus import json_new, sha256_file
from geofirex.full_corpus import prepare_corpus
from geofirex.gpu_operational import event_lots, load_training_normalizer, spread_windows, training_tokens
from geofirex.runner import load_config
from geofirex.runner_data import FoldData, fit_fold_normalizer


@pytest.fixture
def mounted(corpus, tmp_path):
    root, index, idx, audit, prepared = corpus
    certify(corpus)
    prepare_corpus(index, root, audit, prepared, engine='python')
    table = pq.read_table(prepared/'observations.parquet')
    selected = [r for r in table.to_pylist() if r['split'] != 'test']
    pq.write_table(pa.Table.from_pylist(selected, schema=table.schema),
                   tmp_path/'trainval.parquet')
    # A distinct physical root contains no test TIFFs. Keep the original fixture
    # untouched so this proves absence rather than merely filtering metadata.
    trainval_root = tmp_path/'trainval-rasters'
    for row in selected:
        destination = trainval_root/row['relative_path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/row['relative_path'], destination)
    from geofirex.runner import DEFAULTS
    c = dict(DEFAULTS, data_root=str(trainval_root), data_roots=[], manifest=str(tmp_path/'trainval.parquet'),
             index=str(idx), split=str(prepared/'fold_00.csv'), window_manifest=str(prepared/'windows.parquet'),
             qualification='pilot', corpus_scope='train_validation')
    return c


def test_training_scope_does_not_require_test_files(mounted):
    assert not (Path(mounted['data_root'])/'2021').exists()
    data = FoldData(mounted)
    assert not data.windows('test')
    assert {r['split'] for r in data.rows} == {'train', 'val'}
    # The deployment can still see the official split metadata for every event.
    assert any(v == 'test' for v in data.assignment.values())
    with pytest.raises(ValueError, match='607'):
        FoldData({**mounted, 'qualification': 'scientific'})


def test_scope_rejects_test_rows_and_aliases_before_raster_resolution(mounted, tmp_path):
    rows = pq.read_table(mounted['manifest']).to_pylist()
    rows[0]['relative_path'] = '2021/fire_1/2021-01-01.tif'
    path = tmp_path/'alias.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match='canonical relative'):
        FoldData({**mounted, 'manifest': str(path)})
    rows[0]['event_id'], rows[0]['date'] = '2021/fire_1', '2021-01-01'
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match='every train/val'):
        FoldData({**mounted, 'manifest': str(path)})


def test_cached_normalizer_is_train_only_pinned_and_relocatable(mounted, tmp_path):
    data = FoldData(mounted)
    norm = fit_fold_normalizer(data.groups['train'][5], size=8)
    tokens = sorted(training_tokens(data))
    artifact = dict(schema='train-normalizer-portable-v1', feature_policy=mounted['feature_policy'],
                    normalization={**norm.to_dict(), 'fit_paths': tokens},
                    bindings={k: data.hashes[k] for k in ('manifest', 'window_manifest', 'split', 'index')})
    path = tmp_path/'norm.json'
    json_new(path, artifact)
    c = dict(mounted, normalization_artifact=str(path), normalization_sha256=sha256_file(path))
    cached = load_training_normalizer(c, data)
    assert cached.mean == norm.mean and set(cached.fit_paths) == set(norm.fit_paths)
    assert not any('2020' in p or '2021' in p for p in cached.fit_paths)
    with pytest.raises(ValueError, match='SHA256'):
        load_training_normalizer({**c, 'normalization_sha256': '0'*64}, data)
    poisoned = copy.deepcopy(artifact)
    poisoned['normalization']['fit_paths'][0] = '2020/fire_1/2020-01-01.tif'
    other = tmp_path/'poison.json'
    json_new(other, poisoned)
    with pytest.raises(ValueError, match='fit set'):
        load_training_normalizer(dict(c, normalization_artifact=str(other), normalization_sha256=sha256_file(other)), data)
    poisoned = copy.deepcopy(artifact)
    poisoned['bindings']['manifest'] = '0'*64
    other = tmp_path/'wrong-tables.json'
    json_new(other, poisoned)
    with pytest.raises(ValueError, match='bindings'):
        load_training_normalizer(dict(c, normalization_artifact=str(other), normalization_sha256=sha256_file(other)), data)


def test_lots_are_complete_exclusive_bounded_and_never_test():
    rows = [dict(event_id=f'{role}/{i}', split=role, bytes=20, day=day)
            for role in ('train', 'val', 'test') for i in range(3) for day in range(2)]
    lots = event_lots(rows, 90)
    flattened = [r for _, lot in lots for r in lot]
    assert len(flattened) == 12 and all(r['split'] != 'test' for r in flattened)
    seen = set()
    for role, lot in lots:
        assert sum(r['bytes'] for r in lot) <= 90
        events = {r['event_id'] for r in lot}
        assert not seen & events
        seen |= events
        assert all(r['split'] == role for r in lot)
        assert all(sum(r['event_id'] == e for r in lot) == 2 for e in events)
    with pytest.raises(ValueError, match='exceeds lot'):
        event_lots(rows, 30)


def test_profiling_windows_cover_distinct_events(mounted):
    data = FoldData(mounted)
    chosen = spread_windows(data.windows('train'), 4)
    assert len(chosen) == 1 and data.windows('train')[chosen[0]].event_id.startswith('2018/')
