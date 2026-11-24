"""Jury entry point. Frozen historical application and checkpoints remain intact."""
import json
from pathlib import Path
from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/jury_20261202_v1'


def evidence(name,root=REPORT):
    root=Path(root);seal=json.loads((root/'science_manifest.json').read_text())
    if name not in seal['files'] or Path(name).name!=name:raise ValueError('Unknown evidence')
    if sha(root/name)!=seal['files'][name]:raise ValueError('Jury evidence integrity differs')
    return (root/name).read_bytes()


def create_jury_app():
    from fastapi import APIRouter,Response
    from starlette.routing import Mount
    from geofirex.science_results import create_results_app
    app=create_results_app()
    router=APIRouter(prefix='/science/v2',tags=['jury-validation2020'])
    @router.get('/comparison')
    def comparison():return json.loads(evidence('comparison.json'))
    @router.get('/diagnostic-tile')
    def tile():return json.loads(evidence('diagnostic_tile.json'))
    @router.get('/comparable-fp')
    def comparable_fp():return json.loads(evidence('FIRE13_comparable_fp_saved.json'))
    @router.get('/export.csv')
    def export_csv():return Response(evidence('comparison.csv'),media_type='text/csv; charset=utf-8',
        headers={'Content-Disposition':'attachment; filename="geofirex-validation2020-V2-80.csv"'})
    @router.get('/provenance')
    def provenance():
        comparison=json.loads(evidence('comparison.json'))
        return dict(sources=comparison['sources'],domain=comparison['evaluation_domain_sha256'],
            acceptance_sha256=comparison['acceptance_sha256'],test_2021_locked=True)
    idx=next(i for i,route in enumerate(app.router.routes) if isinstance(route,Mount) and route.path in ('','/'))
    app.router.routes[idx:idx]=router.routes
    from geofirex.studio_spatial import attach_spatial
    attach_spatial(app)
    return app
