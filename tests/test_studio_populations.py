"""Overview counts come from pinned metadata, without dataset/pixel access."""
import asyncio
import json
import pytest
from geofirex import jury_app
from geofirex.calibration_io import sha
from test_unet_confirmation import request


def test_population_endpoint_uses_verified_metadata_only(monkeypatch):
    from geofirex.studio_v2 import V2Predictor
    def forbidden(*args,**kwargs):
        raise AssertionError('Population metadata must not construct a predictor')
    monkeypatch.setattr(V2Predictor,'__init__',forbidden)
    status,body=asyncio.run(request(jury_app.create_jury_app(),'/science/v2/populations'))
    data=json.loads(body)
    assert status==200 and data['read_only'] and not data['new_inference']
    assert data['roles']=={'train':{'events':318,'tiles':115900},'val':{'events':80,'tiles':30882}}
    assert data['provenance']['index_sha256']==sha(jury_app.ROOT/data['provenance']['index'])


def test_missing_or_changed_index_fails_closed(tmp_path):
    config=tmp_path/'configs/protocol_v2/baseline_c8.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'index_sha256':'0'*64}))
    with pytest.raises(OSError):jury_app.protocol_populations(tmp_path)
    index=tmp_path/'reports/protocol_v2_20261118/index/index.json'
    index.parent.mkdir(parents=True);index.write_text('{}')
    with pytest.raises(ValueError,match='SHA-256'):jury_app.protocol_populations(tmp_path)
    config.write_text(json.dumps({'index_sha256':sha(index)}))
    with pytest.raises(KeyError):jury_app.protocol_populations(tmp_path)


def test_unavailable_metadata_is_explicit_503(monkeypatch):
    def unavailable():raise ValueError('index missing')
    monkeypatch.setattr(jury_app,'protocol_populations',unavailable)
    status,body=asyncio.run(request(jury_app.create_jury_app(),'/science/v2/populations'))
    assert status==503 and 'index missing' in json.loads(body)['detail']
