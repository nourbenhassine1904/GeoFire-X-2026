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


def protocol_populations(root=ROOT):
    """Read pinned V2 metadata only; never construct a dataset or read pixels."""
    root=Path(root)
    baseline=json.loads((root/'configs/protocol_v2/baseline_c8.json').read_text())
    relative='reports/protocol_v2_20261118/index/index.json'
    path=root/relative
    identity=sha(path)
    if identity!=baseline['index_sha256']:
        raise ValueError('Identité SHA-256 de l’index V2 différente')
    index=json.loads(path.read_text())
    roles={role:{key:index['roles'][role][key] for key in ('events','tiles')} for role in ('train','val')}
    if any(type(n) is not int or n<=0 for values in roles.values() for n in values.values()):
        raise ValueError('Populations V2 absentes ou invalides')
    return dict(roles=roles,read_only=True,new_inference=False,
                provenance=dict(index=relative,index_sha256=identity))


def create_jury_app():
    from fastapi import APIRouter,Response,HTTPException
    from starlette.routing import Mount
    from geofirex.science_results import create_results_app
    app=create_results_app()
    router=APIRouter(prefix='/science/v2',tags=['jury-validation2020'])
    @router.get('/comparison')
    def comparison():return json.loads(evidence('comparison.json'))
    @router.get('/populations')
    def populations():
        try:return protocol_populations()
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise HTTPException(status_code=503,detail='Métadonnées V2 indisponibles : '+str(error)) from error
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
            acceptance_sha256=comparison['acceptance_sha256'],test_2021_locked=True,
            test_2021_status='COMPLETE_READ_ONLY',test_rerun_allowed=False)
    idx=next(i for i,route in enumerate(app.router.routes) if isinstance(route,Mount) and route.path in ('','/'))
    app.router.routes[idx:idx]=router.routes
    from geofirex.studio_spatial import attach_spatial
    attach_spatial(app)
    import os
    if os.environ.get('GEOFIREX_V2_ROOT'):
        for source in app.state.saved_scenes.sources.values():
            source['archive']=str(Path(os.environ['GEOFIREX_V2_ROOT'])/Path(source['archive']).name)
    from geofirex.studio_v2 import attach_studio_v2
    from geofirex.studio_final_results import attach_final_results
    attach_studio_v2(app)
    attach_final_results(app)
    # Current deployment status supplements the preserved historical health route.
    @router.get('/deployment-health')
    def deployment_health():
        from geofirex.studio_v2 import QUALIFICATION
        return dict(ready=True,mode='studio-v2-cpu',live_inference=True,
                    interactive_scope='validation2020_only',test_2021_status='COMPLETE_READ_ONLY',
                    test_rerun_allowed=False,qualification_available=QUALIFICATION.is_file())
    # The auxiliary health route is explicit; /health keeps compatibility.
    idx=next(i for i,route in enumerate(app.router.routes) if isinstance(route,Mount) and route.path in ('','/'))
    app.router.routes[idx:idx]=[router.routes[-1]]
    return app
