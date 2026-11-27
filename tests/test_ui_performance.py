"""Presentation APIs must expose archived metrics without model/test-pixel reads."""
import asyncio
import csv
import io
import json
import zipfile
import pytest
from geofirex.jury_app import create_jury_app,evidence
from geofirex.studio_v2 import V2Predictor
from test_unet_confirmation import request


def test_performance_exports_preserve_exact_archived_values_and_populations(monkeypatch):
    app=create_jury_app();opened=[];original=zipfile.ZipFile.open
    def traced(self,name,*args,**kwargs):
        opened.append(name.filename if isinstance(name,zipfile.ZipInfo) else name)
        return original(self,name,*args,**kwargs)
    def no_model(*args,**kwargs):raise AssertionError('Read-only metrics must not construct a predictor')
    monkeypatch.setattr(zipfile.ZipFile,'open',traced)
    monkeypatch.setattr(V2Predictor,'__init__',no_model)
    status,body=asyncio.run(request(app,'/science/performance'));data=json.loads(body)
    assert status==200 and data['read_only'] and not data['new_inference']
    assert len(data['validation']['per_event'])==80
    assert len(data['independent_test']['per_event'])==156
    assert all(e['event_id'].startswith('2020/') for e in data['validation']['per_event'])
    assert all(e['event_id'].startswith('2021/') for e in data['independent_test']['per_event'])
    validation=json.loads(evidence('C8_evaluation.json'))
    assert data['validation']['pooled']==validation['pooled']
    assert data['independent_test']['pooled']['average_precision']==.45500139551181146
    assert data['independent_test']['new_fire']['average_precision']==.17547636341982384
    for population in ('validation','independent_test'):
        r=data[population]
        for domain in ('pooled','new_fire'):
            for metric in ('tp','fp','fn','tn'):
                assert sum(e[domain][metric] for e in r['per_event'])==r[domain][metric]
    assert set(opened)=={'archive_manifest.json','evaluation.json','selection.json','CONSUMED.json','INFERENCE_COMPLETE.json'}
    assert not app.state.studio_v2
    status,body=asyncio.run(request(app,'/science/performance/export.json'))
    assert status==200 and json.loads(body)==data
    status,body=asyncio.run(request(app,'/science/performance/events.csv'))
    rows=list(csv.DictReader(io.StringIO(body.decode())))
    assert status==200 and len(rows)==472
    assert {r['population'] for r in rows}=={'validation','independent_test'}
    test_rows=[r for r in rows if r['year']=='2021' and r['domain']=='pooled']
    assert sum(int(r['fp']) for r in test_rows)==data['independent_test']['pooled']['fp']
    assert float(test_rows[0]['dice'])==data['independent_test']['per_event'][0]['pooled']['dice']
    assert len({r['source_archive_sha256'] for r in rows})==2
