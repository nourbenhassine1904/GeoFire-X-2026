"""A verified FIRE13 probe is separate from permission to serve it interactively."""
import asyncio
from datetime import date
import json
import pytest
from geofirex.studio_v2 import V2Predictor,QUALIFICATION
from geofirex.jury_app import create_jury_app
from test_unet_confirmation import request


def test_fire13_probe_never_reads_target_or_refits(monkeypatch):
    from geofirex.fire13_features import Fire13Tiles
    p=V2Predictor('FIRE13',require_qualification=False)
    target=date(2020,9,10).toordinal();seen=[];original=p.dataset._day
    def traced(a,d,r,c):
        assert d<target
        seen.append(d)
        return original(a,d,r,c)
    def forbidden(*args,**kwargs):raise AssertionError('No target-reading __getitem__ in inference')
    monkeypatch.setattr(p.dataset,'_day',traced)
    monkeypatch.setattr(Fire13Tiles,'__getitem__',forbidden)
    try:
        result=p.predict('2020/fire_24332704','2020-09-09',512)
        assert set(seen)==set(range(target-5,target))
        assert len(result['channels'])==13 and result['model_id']=='FIRE13'
        assert result['checkpoint_sha256'].startswith('5afca67a87d8')
        assert result['new_inference'] and not result['target_pixels_read']
        assert result['derived_normalizer_sha256']=='e905dcd42c227b8d6910143624f2f14859d5bf7f14e16b9f785154790bec0516'
        assert p.parameters==125057 and p.fire13.stats['fit_role']=='train'
    finally:p.fire13.close()


def test_fire13_catalogue_obeys_parity_gate():
    status,raw=asyncio.run(request(create_jury_app(),'/studio/models'))
    assert status==200
    row=next(r for r in json.loads(raw)['models'] if r['id']=='FIRE13')
    q=json.loads(QUALIFICATION.read_text(encoding='utf8'))['models']['FIRE13']
    assert row['enabled']==(q['status']=='PASS')
    if q['status']!='PASS':
        assert 'archivé' in row['reason'] and 'tolérance' in row['reason']
        with pytest.raises(ValueError,match='archivé'):V2Predictor('FIRE13')


def test_fire13_2021_is_refused_before_loading(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No model or pixel access for 2021')
    monkeypatch.setattr(V2Predictor,'__init__',forbidden)
    app=create_jury_app()
    for endpoint in ('predict','observations','ground-truth'):
        assert asyncio.run(request(app,'/studio/v2/FIRE13/'+endpoint,'event_id=2021%2Ffire_1&as_of=2021-01-01'))[0]==403
