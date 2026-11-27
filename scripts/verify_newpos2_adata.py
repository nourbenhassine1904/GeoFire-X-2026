"""Read-only NEWPOS2 archive audit; writes a new local receipt, never extracts pixels."""
from pathlib import Path
import json
import zipfile
from geofirex import v2_execution as old
from geofirex.protocol_v2 import save,sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/prop_hn2_20261122_v1'
SOURCE=Path('D:/GeoFire-X/GeoFire-X-V2')
RUN='B2_C8_V2_NEWPOS2_seed42'


def main():
    OUT.mkdir(exist_ok=True)
    rows=[];checkpoints=[]
    for stage in ('TRAINING-SAFE','COMPLETE'):
        paths=list(SOURCE.glob(f'{RUN}-{stage}-*.zip'))
        if len(paths)!=1:raise ValueError('Exactly one archive per expected stage required')
        path=paths[0];sidecar=path.with_suffix('.sha256')
        before=path.stat();side_before=sidecar.stat()
        expected=sidecar.read_text(encoding='ascii').split()[0]
        print('VERIFY READ-ONLY',path.name,flush=True)
        digest=old.verify_zip(path,expected)
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('artifact_manifest.json'))
            if manifest['stage']!=stage:raise ValueError('Archive stage differs')
            config=json.loads(z.read('inputs/configs/c8_newpos2_v1.json'))
            if config!=old.read(ROOT/'configs/c8_newpos2_v1.json'):raise ValueError('NEWPOS2 scientific identity differs')
            complete=json.loads(z.read('training_complete.json'))
            if complete['epochs']!=4 or complete['updates']!=28976:raise ValueError('Incomplete training')
            member=next(r for r in manifest['files'] if r['path']==complete['checkpoint'])
            if member['sha256']!=complete['checkpoint_sha256']:raise ValueError('Checkpoint receipt differs')
            checkpoints.append(member['sha256'])
            if stage=='COMPLETE':
                result=json.loads(z.read('evaluation/evaluation.json'))
                original=old.read(ROOT/'reports/c8_propagation_20261121_v1/baseline_reference.json')
                if (result['status']!='COMPLETE' or result['tiles']!=30882 or len(result['per_event'])!=80
                    or result['evaluation_domain_sha256']!=original['evaluation_domain_sha256']
                    or result['threshold']!=.78):raise ValueError('Validation domain differs')
                save(OUT/'newpos2_reference.json',result)
                report=json.loads(z.read('evaluation/paired_comparison/NEWPOS2_acceptance.json'))
                if report['all_scientific_gates_pass']:raise ValueError('Expected fixed NO-GO differs')
                save(OUT/'newpos2_fixed_decision.json',report)
                for name in ('resources_fit.json','runtime_budget.json'):
                    save(OUT/('newpos2_'+name),json.loads(z.read(name)))
        after=path.stat();side_after=sidecar.stat()
        if (before.st_size,before.st_mtime_ns,side_before.st_size,side_before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns,side_after.st_size,side_after.st_mtime_ns):
            raise ValueError('Source changed during verification')
        rows.append(dict(path=str(path),stage=stage,sha256=digest,bytes=before.st_size,mtime_ns=before.st_mtime_ns,
                         sidecar_sha256=sha256(sidecar),all_members_verified=True,unchanged_during_read=True))
    if len(set(checkpoints))!=1:raise ValueError('TRAINING-SAFE and COMPLETE checkpoints differ')
    save(OUT/'newpos2_archive_verification.json',dict(status='PASS',archives=rows,
         final_checkpoint_sha256=checkpoints[0],archive_modifications=0,test_pixels_opened=0,
         checkpoint_deserialized=False,scope='SHA256 + all member hashes/sizes + fixed identity + domain; no metric recomputation'))
    print('PASS: both immutable archives, same fixed checkpoint, validation domain and NO-GO',flush=True)


if __name__=='__main__':main()
