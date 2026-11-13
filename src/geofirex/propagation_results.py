"""Verified saved propagation diagnostics and whitelisted derived PNGs, no Torch."""
import json
from pathlib import Path
from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'reports/propagation_20261112_v1'


def verified(root=DEFAULT):
    root=Path(root);path=root/'diagnosis.json';manifest=root/'assets_manifest.json'
    if not path.is_file() or not manifest.is_file():return None
    seal=json.loads(manifest.read_text(encoding='utf-8'))
    if sha(path)!=seal['diagnosis_sha256']:raise ValueError('Propagation diagnosis integrity differs')
    data=json.loads(path.read_text(encoding='utf-8'))
    if any(Path(n).name!=n or not n.endswith('.png') for n in seal['assets']):
        raise ValueError('Unsafe propagation asset manifest')
    if data['status']!='SAVED_OUTPUTS_ONLY' or data['training_executed'] or data['inference_executed'] or data['test_2021_pixels_opened']:
        raise ValueError('Unexpected propagation diagnosis provenance')
    for name,digest in data['sources'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=digest:raise ValueError('Propagation source differs: '+name)
    for c in data['cases']:
        if not c['event_id'].startswith('2020/'):raise ValueError('Forbidden propagation map year')
        for name in c['layers'].values():
            if Path(name).name!=name or not name.endswith('.png') or name not in seal['assets']:
                raise ValueError('Unsafe propagation map')
    return data,seal


def attach_propagation(router):
    from fastapi import HTTPException,Response
    @router.get('/propagation-comparison')
    def comparison():
        path=ROOT/'configs/propagation_comparison_v1.json'
        if not path.is_file():raise HTTPException(404,'Expérience non importée')
        pins=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in pins['files'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Propagation comparison integrity differs')
        return json.loads((ROOT/pins['comparison']).read_text(encoding='utf-8'))
    @router.get('/propagation')
    def propagation():
        value=verified()
        if value is None:raise HTTPException(404,'Diagnostic de propagation indisponible')
        return value[0]
    @router.get('/propagation-assets/{name:path}')
    def asset(name:str):
        value=verified()
        if value is None or name not in value[1]['assets']:raise HTTPException(404,'Carte indisponible')
        path=DEFAULT/name
        if sha(path)!=value[1]['assets'][name]:raise ValueError('Propagation PNG integrity differs')
        return Response(path.read_bytes(),media_type='image/png',headers={'ETag':'"'+value[1]['assets'][name]+'"'})
