"""HTTP/static and preservation checks only. Never run Spark or model inference."""
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

from geofirex.calibration_io import save,sha
from geofirex.propagation_experiment import preflight,atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/propagation_20261112_v1'


def main():
    checks=[];base='http://127.0.0.1:8014'
    def get(path,kind='json'):
        with urllib.request.urlopen(base+path,timeout=30) as response:
            content=response.read();assert response.status==200
            if kind=='png':assert content.startswith(b'\x89PNG')
            checks.append(dict(path=path,status=response.status,bytes=len(content),content_type=response.headers['Content-Type']))
            return json.loads(content) if kind=='json' else content
    health=get('/health');assert health['mode']=='scientific-results' and not health['live_inference'] and health['test_2021_locked']
    html=get('/','text');bundle=get('/app.js','text');get('/style.css','text')
    assert b'app.js' in html and 'Propagation J+1'.encode() in bundle
    summary=get('/science/summary');big=get('/science/big-data');unet=get('/science/unet-ablation');diagnosis=get('/science/propagation')
    get('/science/events?phase=calibration32');get('/science/events?phase=holdout169');get('/science/retained-model')
    assert unet['confirmation']['comparison'][2]['dice']==diagnosis['full']['dice']
    for phase in ('calibration32','holdout169'):
        for case in get('/science/maps?phase='+phase):
            for name in case['layers'].values():get('/science/assets/'+case['directory']+'/'+name,'png')
    for case in unet['confirmation']['maps']:
        for name in case['layers'].values():get('/science/unet-assets/'+case['directory']+'/'+name,'png')
    for case in diagnosis['cases']:
        for name in case['layers'].values():get('/science/propagation-assets/'+name,'png')
    previous=json.loads((ROOT/'reports/unet_confirmation_20261112_v1/delivery.json').read_text())
    assert summary['provenance']['archive_sha256']==previous['historical_zip_sha256']
    assert unet['provenance']['archive_sha256']==previous['source_ablation_sha256']
    assert big['parquet_files']==previous['spark_parquet_files']
    args=SimpleNamespace(corpus=ROOT/'reports/mission_20261111/compact_full',index=ROOT/'reports/gpu_acceleration_20261111_v1/index',
        protocol=OUT/'protocol.json',baseline32=OUT/'baseline32.json')
    plan,train,val=preflight(args);train.close();val.close();atomic_json(OUT/'preflight.json',plan)
    import xml.etree.ElementTree as ET
    xml=ET.parse(OUT/'tests.xml');suite=xml.getroot().find('testsuite')
    tests=dict(tests=int(suite.attrib['tests']),failures=int(suite.attrib['failures']),errors=int(suite.attrib['errors']))
    assert tests['failures']==tests['errors']==0
    react=json.loads((OUT/'react-component-check.json').read_text());assert react['status']=='PASS'
    files=['src/geofirex/propagation_experiment.py','src/geofirex/propagation_results.py','frontend/src/Propagation.mjs',
        'frontend/src/main.mjs','src/geofirex/science_results.py','frontend/dist/app.js',
        'src/geofirex/serving.py','src/geofirex/unet.py','src/geofirex/benchmark_models.py']
    output=dict(status='PREPARED_NOT_TRAINED',url=base,tests=tests,react_component=react,checks=checks,png_layers_checked=sum(c['content_type']=='image/png' for c in checks),
        historical_results_unchanged=True,spark_parquet_unchanged=True,checkpoints_archive_unchanged=True,
        new_training_executed=False,gpu_executed=False,historical_inference_rerun=False,spark_restarted=False,
        test_2021_pixels_opened=0,kaggle_published=False,git_commit_push_merge=False,
        browser_interactive_validation=dict(status='UNAVAILABLE',reason='The browser access host is unavailable',
            scope='HTTP, image bytes, build and automated contracts verified; interactive browser rendering not verified'),
        provenance=diagnosis['sources'],source_sha256={f:sha(ROOT/f) for f in files},
        notebook=json.loads((OUT/'notebook_build.json').read_text()),
        unresolved='Kaggle torch2.11.0 initial-main digest and actual GPU kernels/memory/throughput remain to verify before/within authorized run')
    atomic_json(OUT/'delivery.json',output)
    print(json.dumps(dict(status=output['status'],http_checks=len(checks),png_layers=output['png_layers_checked'],tests=tests),indent=2))


if __name__=='__main__':main()
