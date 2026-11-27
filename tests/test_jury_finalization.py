"""Only completed results and authorized 2020 point diagnostics are read."""
import asyncio
import csv
import io
import json
from pathlib import Path
import numpy as np
import pytest
from geofirex import v2_execution as old
from geofirex.jury_app import create_jury_app,evidence,REPORT,ROOT
from geofirex.c8_comparison import rows_from_result
from test_unet_confirmation import request


def test_four_models_same_exact_80_events_targets_masks_and_original_criteria():
    data=json.loads(evidence('comparison.json'))
    assert [r['model'] for r in data['models']]==['C8','NEWPOS2','NEWPOS4','FIRE13']
    baseline=rows_from_result(old.read(REPORT/'C8_evaluation.json'))
    for row in data['models']:
        assert row['events']==80 and row['tiles']==30882 and row['threshold']==.78
        assert not row['promotion_authorized']
        result=old.read(REPORT/(row['model']+'_evaluation.json'))
        rows=rows_from_result(result)
        assert set(rows)==set(baseline)
        for e in rows:
            assert e.startswith('2020/')
            for domain in ('pooled','new_fire'):
                a,b=rows[e][domain],baseline[e][domain]
                assert a.sum()==b.sum() and a[0]+a[2]==b[0]+b[2]
    assert data['acceptance']==old.read(ROOT/'reports/fire13_20261122_v1/acceptance.json')
    fire=next(r for r in data['models'] if r['model']=='FIRE13')
    assert fire['FP_new']==95053 and fire['AP_new']==pytest.approx(.08418376163472618)
    assert sum(fire['scientific_gates'].values())==4
    assert data['live_model']['different_from_V2_reference'] and data['FIRE13_closed']


def test_real_diagnostic_same_pixels_all_models_and_error_categories():
    tile=json.loads(evidence('diagnostic_tile.json'))
    assert tile['selection'].startswith('lexicographically first') and not tile['new_Fire13_inference']
    cases=tile['cases'];first=cases[0]
    for case in cases:
        assert case['truth']==first['truth']
        assert case['result']['observed_fire']==first['result']['observed_fire']
        r=case['result'];assert r['crs'] is None and r['transform'] is None
        p=np.array(r['probability'],float);valid=np.isfinite(p)
        assert p.shape==(64,64) and np.isfinite(p[valid]).all()
        pred=np.array(r['predicted_fire'],float);observed=np.array(r['observed_fire'],float)
        new=np.array(r['new_predicted_fire'],float)
        assert np.array_equal(pred[valid],(p[valid]>=np.float32(.78)).astype(float))
        assert np.array_equal(new[valid],((pred[valid]==1)&(observed[valid]==0)).astype(float))


def test_jury_api_exports_health_and_no_model_promotion():
    app=create_jury_app()
    for path in ['/health','/science/v2/comparison','/science/v2/diagnostic-tile','/science/v2/provenance']:
        status,body=asyncio.run(request(app,path));assert status==200,path
        assert json.loads(body)
    status,body=asyncio.run(request(app,'/science/v2/export.csv'))
    rows=list(csv.DictReader(io.StringIO(body.decode('utf8'))))
    assert status==200 and len(rows)==4 and rows[-1]['FP_new']=='95053'
    assert asyncio.run(request(app,'/science/v2/no-such-result'))[0]==404
    assert asyncio.run(request(app,'/lab/predict','event_id=2021%2Flocked&as_of=2021-01-01'))[0]==403


def test_tampered_evidence_fails_closed(tmp_path):
    (tmp_path/'science_manifest.json').write_text(json.dumps({'files':{'comparison.json':'0'*64}}))
    (tmp_path/'comparison.json').write_text('{}')
    with pytest.raises(ValueError,match='integrity'):evidence('comparison.json',tmp_path)
    with pytest.raises(ValueError):evidence('../comparison.json',tmp_path)


def test_all_original_identities_and_archives_still_unchanged():
    old.identity(ROOT)
    before=old.read(REPORT/'preservation_before.json')['files']
    for name,digest in before.items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        from geofirex.protocol_v2 import sha256
        assert sha256(path)==digest
