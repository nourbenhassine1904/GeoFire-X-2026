"""Read-only completed 2021 results. Reads JSON members only; never predictions."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import zipfile
from geofirex.calibration_io import sha
from geofirex.prediction_lab import ROOT

RESULTS_SHA = 'dafc1972be8c77326d2ecde0840a3d8840468af62cc69e83f0b69fbc52742a3d'
CHECKPOINT_SHA = 'c2273fea252d709f4d316eba76515d669be97eac8bfa9302ff4831da2c9aedc1'
TEST_IDENTITY = '7d79162839fe18c1b032022f93839fb1d3dc8ef8d0ea6674d3d996e81108492a'
DEFAULT_ARCHIVE = Path('D:/GeoFire-X-Final-Evaluation-20261125-v1/results/final-2021-01-RESULTS.zip')


class CompletedResults:
    def __init__(self,path=None):
        backup=ROOT/'artifacts/final_test_v2_backups/GEOFIREX-C8-2021-20261125-01/final-2021-01-RESULTS.zip'
        default=DEFAULT_ARCHIVE if DEFAULT_ARCHIVE.is_file() else backup
        self.path=Path(path or os.environ.get('GEOFIREX_RESULTS_ARCHIVE',str(default)))
        self.stamp=None; self.value=None; self.lock=threading.Lock()

    def read(self):
        with self.lock:
            stat=self.path.stat(); stamp=(stat.st_size,stat.st_mtime_ns)
            if stamp==self.stamp: return self.value
            if sha(self.path)!=RESULTS_SHA: raise ValueError('SHA-256 officiel RESULTS incorrect')
            with zipfile.ZipFile(self.path) as z:
                manifest=json.loads(z.read('archive_manifest.json'))
                def member(name):
                    raw=z.read(name); row=manifest[name]
                    if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                        raise ValueError('Intégrité du résultat archivé incorrecte : '+name)
                    return json.loads(raw)
                evaluation=member('evaluation.json'); selection=member('selection.json')
                consumed=member('CONSUMED.json'); complete=member('INFERENCE_COMPLETE.json')
            if (evaluation['status']!='COMPLETE' or evaluation['tiles']!=75540
                    or len(evaluation['per_event'])!=156 or evaluation['threshold']!=.78
                    or evaluation['checkpoint_sha256']!=CHECKPOINT_SHA
                    or evaluation['test_identity']!=TEST_IDENTITY
                    or selection['checkpoint_sha256']!=CHECKPOINT_SHA
                    or consumed['test_identity']!=TEST_IDENTITY or complete['tiles']!=75540):
                raise ValueError('Identité ou protocole du résultat final incorrect')
            self.value=dict(status='COMPLETE',model='C8 V2',year=2021,events=156,tiles=75540,
                            threshold=.78,pooled=evaluation['pooled'],new_fire=evaluation['new_fire'],
                            macro_event_dice=evaluation['macro_event_dice'],read_only=True,
                            per_event=evaluation['per_event'],
                            new_inference=False,test_rerun_allowed=False,
                            provenance=dict(archive=str(self.path),archive_sha256=RESULTS_SHA,
                                            checkpoint_sha256=CHECKPOINT_SHA,test_identity=TEST_IDENTITY,
                                            evaluation_member_sha256=manifest['evaluation.json']['sha256'],
                                            freeze_sha256=evaluation['freeze_sha256']),
                            interpretation='Test indépendant terminé pour C8 seulement. Variantes : validation 2020 uniquement.')
            self.stamp=stamp
            return self.value


def attach_final_results(app):
    from fastapi import APIRouter,HTTPException,Response
    from starlette.routing import Mount
    router=APIRouter(prefix='/science/final-2021',tags=['completed-test-read-only'])
    store=CompletedResults()
    def read():
        try:return store.read()
        except (OSError,ValueError,KeyError,zipfile.BadZipFile) as e:
            raise HTTPException(503,'Résultats 2021 indisponibles : reconnecter ADATA ou configurer GEOFIREX_RESULTS_ARCHIVE vers une copie identique. '+str(e)) from e
    @router.get('')
    def result():return read()
    @router.get('/export.json')
    def export_json():
        return Response(json.dumps(read(),ensure_ascii=False,indent=2),media_type='application/json',
                        headers={'Content-Disposition':'attachment; filename="geofirex-C8-test2021-official.json"'})
    @router.get('/export.csv')
    def export_csv():
        r=read(); stream=io.StringIO(); writer=csv.writer(stream)
        writer.writerow(['model','year','domain','events','tiles','threshold','AP','Dice','recall','archive_sha256','checkpoint_sha256'])
        for domain in ('pooled','new_fire'):
            m=r[domain]; writer.writerow(['C8 V2',2021,domain,156,75540,.78,m['average_precision'],m['dice'],m['recall'],RESULTS_SHA,CHECKPOINT_SHA])
        return Response(stream.getvalue(),media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition':'attachment; filename="geofirex-C8-test2021-official.csv"'})
    performance_router=APIRouter(tags=['scientific-performance-read-only'])
    def performance_data():
        from geofirex.jury_app import evidence
        validation=json.loads(evidence('C8_evaluation.json'))
        comparison=json.loads(evidence('comparison.json'))
        source=next(r for r in comparison['sources'] if r['model']=='C8')
        row=next(r for r in comparison['models'] if r['model']=='C8')
        # AP_new was archived in the sealed comparison, not in the old accumulator.
        new_fire={**validation['new_fire'],'average_precision':row['AP_new']}
        events=[dict(event_id=r['event_id'],pooled={k:v for k,v in r.items() if k not in ('event_id','new_fire')},
                     new_fire=r['new_fire']) for r in validation['per_event']]
        return dict(model='C8 V2',threshold=.78,read_only=True,new_inference=False,
            validation=dict(year=2020,events=80,tiles=30882,status='ARCHIVED_VALIDATION',
                pooled=validation['pooled'],new_fire=new_fire,per_event=events,
                macro_event_dice=validation['macro_event_dice'],provenance=source),
            independent_test=read(),
            interpretation='Descriptive comparison of distinct event populations; not paired and not a new experiment.')
    @performance_router.get('/science/performance')
    def performance():return performance_data()
    @performance_router.get('/science/performance/export.json')
    def performance_json():
        return Response(json.dumps(performance_data(),ensure_ascii=False,indent=2),media_type='application/json',
                        headers={'Content-Disposition':'attachment; filename="geofirex-C8-performance-official.json"'})
    @performance_router.get('/science/performance/events.csv')
    def events_csv():
        data=performance_data();stream=io.StringIO();writer=csv.writer(stream)
        metrics=('average_precision','dice','iou','precision','recall','brier','tp','fp','fn','tn','valid_pixels')
        writer.writerow(['population','year','event_id','domain',*metrics,'source_archive_sha256'])
        for population in ('validation','independent_test'):
            result=data[population];provenance=result['provenance']
            for event in result['per_event']:
                for domain in ('pooled','new_fire'):
                    writer.writerow([population,result['year'],event['event_id'],domain,
                        *(event[domain].get(k) for k in metrics),provenance['archive_sha256']])
        return Response(stream.getvalue(),media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition':'attachment; filename="geofirex-C8-per-event-official.csv"'})
    idx=next(i for i,r in enumerate(app.router.routes) if isinstance(r,Mount) and r.path in ('','/'))
    app.router.routes[idx:idx]=router.routes+performance_router.routes
    app.state.completed_results=store
