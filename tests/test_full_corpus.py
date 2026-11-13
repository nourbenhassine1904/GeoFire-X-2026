"""Synthetic integrity fixtures, never substitutes for the physical certificate."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from rasterio.transform import from_origin
import yaml

from geofirex.corpus import index_zip, sha256_file
from geofirex.full_corpus import (audit_corpus, audit_file, canonical_tables,
                                  patch_inventory, prepare_corpus)
from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.runner import load_config
from geofirex.runner_data import FoldData, TemporalStore, TrainTiles, fit_fold_normalizer


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / 'rasters'
    archive = tmp_path / 'official-fixture.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for year in (2018, 2020, 2021):
            for day in range(1, 9):
                name = f'{year}/fire_1/{year}-01-{day:02d}.tif'
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = np.full((23, 9, 11), year + day*.01, np.float32)
                raw[12] = day if year == 2018 else 1e6
                raw[13] = 90
                raw[14] = 100
                raw[0, 1, 1] = np.nan
                raw[22] = np.nan
                raw[22, 2:4, 3:day+2] = 2036
                with rasterio.open(path, 'w', driver='GTiff', count=23, height=9, width=11,
                                   dtype='float32', crs='EPSG:32610', transform=from_origin(100, 200, 375, 375)) as ds:
                    ds.write(raw)
                    ds.descriptions = RAW_BAND_NAMES
                z.write(path, name)
    with zipfile.ZipFile(archive) as z:
        index = index_zip(z)
    index_path = tmp_path / 'index.json'
    index_path.write_text(json.dumps(index))
    return root, index, index_path, tmp_path / 'audit', tmp_path / 'prepared'


def certify(corpus):
    root, index, _, audit, _ = corpus
    result = audit_corpus(index, root, audit)
    assert result['status'] == 'CERTIFIED'
    records = json.loads(next(audit.glob('certified-observations-*.json')).read_text())
    return records


def test_independent_crc_sha_all_band_streaming_and_resume(corpus):
    root, index, _, audit, _ = corpus
    hashes = {e['path']: sha256_file(root/e['path']) for e in index['entries']}
    first = audit_corpus(index, root, audit, limit=2)
    assert first['status'] == 'INCOMPLETE' and first['verified_observations'] == 2
    assert not list(audit.glob('certified-*.json'))
    log = next(audit.glob('audit-attempt-*.jsonl'))
    with log.open('ab') as f:
        f.write(b'{"kind":"audit_ok"')
    prior = log.read_bytes()
    result = audit_corpus(index, root, audit)
    assert result['reused_certificates'] == 2 and result['verified_observations'] == 24
    assert result['verified_events'] == 3 and result['raw_writes'] == result['network_bytes'] == 0
    assert log.read_bytes() == prior
    records = json.loads(next(audit.glob('certified-observations-*.json')).read_text())
    assert all(r['num_channels'] == 23 and len(r['finite_counts']) == 23 for r in records)
    assert all(r['nan_counts'][0] == 1 and r['nan_counts'][22] > 0 for r in records)
    assert next(r for r in records if r['year'] == 2021)['test_locked']
    assert hashes == {e['path']: sha256_file(root/e['path']) for e in index['entries']}
    assert not list(root.rglob('*.aux.xml'))


def test_mismatch_never_certified_and_raw_preserved(corpus):
    root, index, _, audit, _ = corpus
    index['entries'][0]['crc32'] = '00000000'
    before = sha256_file(root/index['entries'][0]['path'])
    result = audit_corpus(index, root, audit)
    assert result['status'] == 'INCOMPLETE' and len(result['errors']) == 1
    assert not list(audit.glob('certified-observations-*'))
    assert sha256_file(root/index['entries'][0]['path']) == before


def test_inventory_requires_exact_official_paths(corpus):
    root, index, _, audit, _ = corpus
    extra = root/'unexpected.tif'
    extra.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='inventory mismatch'):
        audit_corpus(index, root, audit)
    assert extra.read_bytes() == b'fixture'


def test_geometry_drift_prevents_complete_event(corpus):
    root, index, _, audit, _ = corpus
    records = certify(corpus)
    records[0]['transform'][2] += 1
    with pytest.raises(ValueError, match='spatially inconsistent'):
        canonical_tables(records, index)


def test_cross_split_duplicate_content_rejected(corpus):
    records = certify(corpus)
    records[-1]['sha256'] = records[0]['sha256']
    with pytest.raises(ValueError, match='cross annual'):
        canonical_tables(records, corpus[1])


def test_portable_parquet_is_consumed_and_test_never_normalizes(corpus, tmp_path):
    root, index, index_path, audit, out = corpus
    certify(corpus)
    report = prepare_corpus(index, root, audit, out, engine='python')
    assert report['window_counts']['train']['3'] == dict(natural=5, aligned=3, rejections={})
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(data_root=str(root), manifest=str(out/'observations.parquet'),
        window_manifest=str(out/'windows.parquet'), split=str(out/'fold_00.csv'), index=str(index_path),
        qualification='pilot', feature_policy='conservative_fire_terrain', fold=0, context=3)))
    c = load_config(config)
    data = FoldData(c)
    assert len(data.windows('train')) == len(data.windows('val')) == 3
    assert 'window_manifest' in data.hashes
    assert all(not Path(r['relative_path']).is_absolute() for r in data.rows)
    original_open = rasterio.open

    def guarded(path, *args, **kwargs):
        assert '2021' not in str(path)
        return original_open(path, *args, **kwargs)

    with patch('rasterio.open', guarded):
        norm = fit_fold_normalizer(data.groups['train'][5], size=4)
        assert max(norm.mean) < 1000 and not any('2020' in p or '2021' in p for p in norm.fit_paths)
        store = TemporalStore(data.windows('val'), norm, data.groups['val'][5])
        x, y, valid = TrainTiles(store, 8, 2, 42)[0]
    assert x.shape == (3, 8, 8, 8) and y.shape == valid.shape == (1, 8, 8)
    altered = pq.read_table(out/'windows.parquet').to_pylist()
    altered[0]['target_date'] = '2018-12-31'
    pq.write_table(pa.Table.from_pylist(altered), tmp_path/'tampered.parquet')
    with pytest.raises(ValueError, match='canonical'):
        FoldData({**c, 'window_manifest': str(tmp_path/'tampered.parquet')})


def test_temporal_gap_and_locked_test_no_window_rows(corpus):
    records = certify(corpus)
    observations, windows, counts = canonical_tables(records, corpus[1])
    assert {r['split'] for r in windows} == {'train', 'val'}
    assert all(max(r['context_dates']) < r['target_date'] for r in windows)
    # Keep certificate inventory consistent, but introduce an official-fixture gap.
    changed, index = copy.deepcopy(records), copy.deepcopy(corpus[1])
    changed[3]['date'] = index['entries'][3]['date'] = '2018-02-01'
    _, gap_windows, gap_counts = canonical_tables(changed, index)
    assert gap_counts['train']['3']['natural'] < counts['train']['3']['natural']


def test_patch_inventory_uses_exact_joint_mask_without_test(corpus):
    records = certify(corpus)
    _, windows, _ = canonical_tables(records, corpus[1])
    result = patch_inventory(records, windows, corpus[0], size=8)
    assert result['by_split']['train']['examples'] == 3
    assert result['by_split']['train']['usable_patches'] == 12
    # Force the streaming mask branch and make every input footprint empty.
    records[0]['valid_pixels'] = 0
    with patch('geofirex.runner_data.read_tile', return_value=(None, np.zeros((1, 8, 8)))) as reader:
        result = patch_inventory(records, windows, corpus[0], size=8)
    assert result['by_split']['train']['empty_patches'] == 4 and reader.called


def test_changed_certificate_file_blocks_preparation(corpus):
    certify(corpus)
    root, index, _, audit, out = corpus
    path = root/index['entries'][0]['path']
    with path.open('ab') as f:
        f.write(b'changed fixture')
    with pytest.raises(ValueError, match='Certified raster changed'):
        prepare_corpus(index, root, audit, out, engine='python')
    assert not out.exists()


def test_bounded_real_loader_and_all_model_forward_contracts_on_fixtures(corpus):
    from geofirex.full_corpus import validate_loaders, delivery_report, reuse_normalizer
    root, index, index_path, audit, out = corpus
    certify(corpus)
    prepare_corpus(index, root, audit, out, engine='python')
    original = load_config

    def fixture_config(path, overrides=None):
        return original(path, {**(overrides or {}), 'qualification': 'pilot'})

    # Synthetic 3-event fixture cannot claim the 607-event qualification.
    with patch('geofirex.runner.load_config', fixture_config):
        result = validate_loaders(out, root, index_path, audit)
    assert result['status'] == 'VALIDATED_CPU_ONLY'
    assert result['channels'] == 8 and result['raw_bands'] == 23
    assert result['test_paths_used'] == result['optimizer_updates'] == 0
    assert len(result['batches']) == 6 and all(r['patches'] == 48 for r in result['batches'])
    assert {r['model'] for r in result['forward_contracts']} == {'B0', 'B1', 'B2', 'B3'}
    assert all(not Path(p).name.startswith('2021') for p in json.loads((out/'normalization.json').read_text())['fit_paths'])
    first = delivery_report(out, audit)
    prior = Path(first['delivery']).read_bytes()
    second = delivery_report(out, audit)
    assert second['report'].endswith('REPORT-v2.md')
    assert Path(first['delivery']).read_bytes() == prior
    assert json.loads(Path(second['delivery']).read_text())['window_events_used'] == {'train': 1, 'val': 1}
    reused = out.parent/'prepared-reused'
    prepare_corpus(index, root, audit, reused, engine='python')
    with patch('geofirex.runner.load_config', fixture_config):
        with patch('geofirex.runner_data.fit_fold_normalizer', side_effect=AssertionError('Must reuse the certified train fit')):
            result = validate_loaders(reused, root, index_path, audit, reuse_from=out)
    assert result['normalization_mode'] == 'REUSED_VERIFIED_TRAIN_ONLY'
    assert result['normalization_reuse']['sha256'] == sha256_file(out/'normalization.json')
    assert sha256_file(reused/'normalization.json') == sha256_file(out/'normalization.json')
    with (out/'normalization.json').open('a') as stream:
        stream.write(' ')
    data = FoldData(fixture_config(reused/'gpu_configs/B1_T3_seed42.yaml', dict(device='cpu')))
    with pytest.raises(ValueError, match='SHA-256'):
        reuse_normalizer(out, reused, data, dict(continuous=[12, 14], landcover=False))


def test_spark_window_rows_match_independent_builder_and_are_materialized(corpus, tmp_path):
    pytest.importorskip('pyspark')
    from geofirex.full_corpus import spark_tables
    records = certify(corpus)
    observations, windows, _ = canonical_tables(records, corpus[1])
    stage = tmp_path/'spark-stage'
    stage.mkdir()
    result = spark_tables(observations, windows, stage)
    assert result['status'] == 'SPARK_EXECUTED' and result['window_rows'] == len(windows)
    assert pq.read_table(stage/'observations.parquet').num_rows == 24
    actual = pq.read_table(stage/'windows.parquet').to_pylist()
    assert {r['split'] for r in actual} == {'train', 'val'}
    assert len(actual) == 18 and all(not r['event_id'].startswith('2021') for r in actual)


def test_independent_sha_matches_or_rejects_prior_extraction_receipt(corpus):
    from geofirex.full_corpus import compare_extraction_certificate
    records = certify(corpus)
    root = corpus[0]
    session = root/'.extraction-session'
    session.mkdir()
    receipt = dict(files=[dict(path=r['indexed_path'], sha256=r['sha256'], crc32=r['crc32'], bytes=r['bytes']) for r in records],
                   archive=dict(md5='fixture', md5_status='VERIFIED'))
    path = session/'complete-fixture.json'
    path.write_text(json.dumps(receipt))
    assert compare_extraction_certificate(root, records)['members'] == 24
    records[-1]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='Independent SHA/CRC differs'):
        compare_extraction_certificate(root, records)


def test_absent_spark_does_not_create_outputs_or_start_network(corpus):
    import importlib.util
    original = importlib.util.find_spec
    root, index, _, audit, out = corpus
    with patch('importlib.util.find_spec', side_effect=lambda name: None if name == 'pyspark' else original(name)):
        with patch('urllib.request.urlopen') as network, pytest.raises(RuntimeError, match='PySpark absent'):
            prepare_corpus(index, root, audit, out)
    assert not out.exists() and not network.called
