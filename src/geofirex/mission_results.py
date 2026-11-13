"""Optional read-only FastAPI router for executed mission artifacts.

Mount with app.include_router(mission_router('reports/mission_20261111')).
Historical prediction service and checkpoints remain unchanged.
"""
import json
from pathlib import Path
import numpy as np

def attach_mission(app,root):
    """Insert before the historical frontend's catch-all StaticFiles mount."""
    from starlette.routing import Mount
    index=next((i for i,r in enumerate(app.router.routes) if isinstance(r,Mount) and r.path in ('','/')),len(app.router.routes))
    app.router.routes[index:index]=mission_router(root).routes
    return app

def mission_router(root):
    from fastapi import APIRouter,HTTPException,Query
    root=Path(root).resolve();router=APIRouter(prefix='/mission',tags=['executed-science'])
    def read(name):
        path=root/name
        if not path.is_file():raise HTTPException(404,'Résultat pas encore disponible')
        return json.loads(path.read_text(encoding='utf-8'))

    @router.get('/events')
    def events():
        report=read('b0_v2/results.json')
        return dict(split='val',year=2020,events=report['per_event'],windows_evaluated=report['windows'])

    @router.get('/benchmark')
    def benchmark():
        b0=read('b0_v2/results.json')
        return dict(b0={k:v for k,v in b0.items() if k not in ('per_window','per_event')},b1_cpu_pilot=read('b1_cpu_pilot/summary.json'),matched_pilot=read('matched_b0_b1_pilot.json'),comparison_warning='B1 CPU couvre six événements val; utiliser matched_pilot pour une comparaison sur les mêmes cibles. Aucun résultat B2/B3 nouveau.',gpu_executed=False)

    @router.get('/big-data')
    def big_data():return read('spark/report.json')

    @router.get('/compact')
    def compact():
        full=root/'compact_full/report.json'
        return read('compact_full/report.json') if full.is_file() else read('compact_archive_pilot/report.json')

    @router.get('/map/{map_id}')
    def scene(map_id:int,stride:int=Query(1,ge=1,le=16)):
        if map_id not in range(4):raise HTTPException(404,'Carte non disponible')
        metadata=read(f'b0_v2/map_{map_id:02d}.json')
        with np.load(root/f'b0_v2/map_{map_id:02d}.npz',allow_pickle=False) as z:
            v=z['valid'][::stride,::stride]
            grids={k:np.where(v,z[k][::stride,::stride].astype(float),None).tolist() for k in ('observed','target','prediction','error')}
        return dict(**metadata,model='B0',preview_stride=stride,selection='first four error-containing tiles in deterministic evaluation order; illustrative cases, not a representative performance sample',**grids)
    return router
