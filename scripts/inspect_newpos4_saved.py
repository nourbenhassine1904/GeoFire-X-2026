"""Verify saved NEWPOS4 artifacts and bounded diagnostics, never infer/train."""
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np
from geofirex import c8_experiments as c
from geofirex import newpos4_posthoc as n
from geofirex.c8_thresholds import counts,summarize,serial_rows,folds

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/newpos4_thresholds_20261121_v1'
ARCHIVES=Path('D:/GeoFire-X/GeoFire-X-V2')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    old=c.read(ROOT/c.REPORT/'preservation_before.json')['preserved_sha256']
    old.update(c.read(ROOT/c.REPORT/'delivery_manifest.json')['files'])
    old.update({p.relative_to(ROOT).as_posix():c.sha256(p) for p in (ROOT/c.REPORT).iterdir() if p.is_file()})
    for name,h in old.items():
        if c.sha256(ROOT/name)!=h:raise ValueError('Existing protected file changed: '+name)
    memory=c.old.host_memory()
    if memory[1]<512*1024**2:raise RuntimeError('Insufficient RAM for bounded verification')
    c.save(OUT/'preservation_before.json',dict(files=old,git_status=subprocess.check_output(
        ['git','-c','safe.directory='+ROOT.as_posix(),'status','--short'],cwd=ROOT,text=True),
        host_ram=memory,disks={d:dict(shutil.disk_usage(d)._asdict()) for d in ('C:/','D:/')},
        requested_directory_exists=(ARCHIVES/'C8-NEWPOS4-B').exists(),actual_directory=str(ARCHIVES)))
    c.save(OUT/'threshold_policy.json',n.cost_policy())
    c.save(OUT/'analysis_resource_gate.json',dict(available_ram_bytes=memory[1],required_available_ram_bytes=2*1024**3,
        full_threshold_analysis_allowed=memory[1]>=2*1024**3,
        bounded_verification_minimum_available_bytes=512*1024**2,
        bounded_tile_record_bytes=64*64*7,GPU_used=False,training_executed=False))
    baseline=next(ARCHIVES.glob('B2_C8_V2_seed42-COMPLETE-*.zip'))
    complete=next(ARCHIVES.glob('B2_C8_V2_NEWPOS4_seed42-COMPLETE-*.zip'))
    safe=next(ARCHIVES.glob('B2_C8_V2_NEWPOS4_seed42-TRAINING-SAFE-*.zip'))
    a=next(ARCHIVES.glob('C8_V2_THRESHOLD*_ANALYSIS-ANALYSIS-COMPLETE-*.zip'))
    receipts=[]
    for path in (complete,safe,baseline,a):
        print('Verifying',path.name,flush=True)
        receipts.append(n.verify_archive(path,path.with_suffix('.sha256')))
    c.save(OUT/'archive_verification.json',dict(archives=receipts,checkpoint_deserializations=0))
    with zipfile.ZipFile(baseline) as z:base=n.load_json(z,'evaluation/evaluation.json')
    with zipfile.ZipFile(a) as z:
        c8cross=n.load_json(z,'crossfit/crossfit.json');c8lock=n.load_json(z,'crossfit/thresholds_locked.json')
        if not n.load_json(z,'baseline_reproduction.json')['pass']:raise ValueError('C8 analysis did not reproduce reference')
    with zipfile.ZipFile(complete) as z:
        result=n.load_json(z,'evaluation/evaluation.json');receipt=n.load_json(z,'training_complete.json')
        config=n.load_json(z,'inputs/configs/c8_newpos4_v1.json')
        local_lock=c.read(ROOT/c.REPORT/'execution_manifest.json')
        if (config!=c.read(ROOT/'configs/c8_newpos4_v1.json') or receipt['checkpoint_sha256']!=n.CHECKPOINT
                or receipt['updates']!=28976 or receipt['epochs']!=4
                or n.load_json(z,'checkpoint_u0028976.json')['sha256']!=n.CHECKPOINT
                or n.load_json(z,'inputs/'+c.REPORT.as_posix()+'/execution_manifest.json')!=local_lock):
            raise ValueError('Scientific identity or final checkpoint differs')
        for name,digest in local_lock['files'].items():
            if hashlib.sha256(z.read('inputs/'+name)).hexdigest()!=digest:raise ValueError('Archived input identity differs')
        comparison=n.load_json(z,'evaluation/paired_comparison/comparison.json')
        boot=n.load_json(z,'evaluation/paired_comparison/bootstrap_events.json')
        runtime=n.load_json(z,'runtime_budget.json')
    with zipfile.ZipFile(safe) as z:
        if n.load_json(z,'training_complete.json')!=receipt:raise ValueError('TRAINING-SAFE/COMPLETE differ')
        if n.load_json(z,'checkpoint_u0028976.json')['sha256']!=n.CHECKPOINT:raise ValueError('Backup checkpoint differs')
    for name,value in [('baseline_reference.json',base),('newpos4_evaluation.json',result),('c8_crossfit.json',c8cross),
        ('c8_threshold_lock.json',c8lock),('archived_comparison.json',comparison),('archived_bootstrap.json',boot),
        ('archived_runtime.json',runtime)]:c.save(OUT/name,value)
    meta=c.read(ROOT/'reports/protocol_v2_20261118/index/index.json')
    records=np.load(ROOT/'reports/protocol_v2_20261118/index/val.npy',mmap_mode='r',allow_pickle=False)
    keys=[(meta['events'][meta['archives'][int(r['archive'])]],date.fromordinal(int(r['day'])).isoformat(),
        int(r['row']),int(r['col'])) for r in records]
    if any(meta['event_roles'][k[0]]!='val' for k in keys):raise ValueError('Non-validation key')
    store=n.ArchivedPredictions(complete,keys,expected_identity=dict(run_id=n.RUN,checkpoint_sha256=n.CHECKPOINT))
    if len(store.events)!=80 or len(keys)!=30882:raise ValueError('Incomplete validation')
    if c8lock['fold_mapping']!=folds(store.events):raise ValueError('C8 fold mapping differs')
    studio=OUT/'studio';studio.mkdir()
    rows={e:dict(pooled=np.zeros(4,np.int64),new_fire=np.zeros(4,np.int64),squared_error=0.) for e in store.events}
    domains={};best={};peak_rss=0;N=0
    # This verification reads one saved validation tile at a time, never raw data.
    with (studio/'tile_index.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['event_id','target_date','row','col','fold','archive_member','tile_index','new_tp','new_fp','new_fn','new_tn'])
        positions={e:{tuple(k):i for i,k in enumerate(row['keys'])} for e,row in store.files.items()};mapping=folds(store.events)
        for key,a in store.tiles():
            p,y,v,o=[a[k] for k in ('p','y','v','o')]
            if (not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not all(np.isin(x,[0,1]).all() for x in (y,v,o))):
                raise ValueError('Invalid prediction pixels')
            pooled=counts(p,y,v,.78);new=counts(p,y,(v==1)&(o==0),.78);r=rows[key[0]]
            r['pooled']+=pooled;r['new_fire']+=new;r['squared_error']+=float(np.square(p[v==1].astype(np.float64)-y[v==1]).sum())
            domains[key]=n.domain_tile(key,y,v,o);N+=1
            writer.writerow([*key[:2],key[2],key[3],mapping[key[0]],n.PREFIX+store.files[key[0]]['path'],positions[key[0]][key],*map(int,new)])
            for i,label in enumerate(('new_tp','new_fp','new_fn')):
                if label not in best or int(new[i])>best[label]['count']:
                    best[label]=dict(count=int(new[i]),key=key,data=a.copy())
            if N%1024==0:
                peak_rss=max(peak_rss,c.old.host_rss())
                if c.old.host_memory()[1]<384*1024**2:raise RuntimeError('RAM reserve exhausted during bounded verification')
    domain=hashlib.sha256(json.dumps(sorted(domains.items())).encode()).hexdigest()
    if domain!=n.DOMAIN:raise ValueError('Recomputed target/mask domain differs')
    recomputed=summarize(rows)
    for domain_name in ('pooled','new_fire'):
        if any(recomputed[domain_name][k]!=result[domain_name][k] for k in ('tp','fp','fn','tn')):raise ValueError('Archived counts differ')
    if abs(recomputed['pooled']['brier']-result['pooled']['brier'])>1e-10:raise ValueError('Archived Brier differs')
    original_rows={r['event_id']:r for r in result['per_event']}
    baseline_rows={r['event_id']:r for r in base['per_event']}
    for e,r in rows.items():
        for domain_name in ('pooled','new_fire'):
            expected=original_rows[e] if domain_name=='pooled' else original_rows[e]['new_fire']
            if list(map(int,r[domain_name]))!=[expected[k] for k in ('tp','fp','fn','tn')]:raise ValueError('Per-event counts differ')
            b=baseline_rows[e] if domain_name=='pooled' else baseline_rows[e]['new_fire']
            if sum(r[domain_name])!=sum(b[k] for k in ('tp','fp','fn','tn')) or r[domain_name][0]+r[domain_name][2]!=b['tp']+b['fn']:
                raise ValueError('C8/NEWPOS4 event target populations differ')
        if r['new_fire'][0]+r['new_fire'][2]!=store.files[e]['new_positive'] or r['new_fire'][1]+r['new_fire'][3]!=store.files[e]['new_negative']:
            raise ValueError('Prediction manifest population differs')
    c.save(OUT/'prediction_verification.json',dict(status='PASS',tiles=N,events=80,domain_sha256=domain,
        checkpoint_sha256=n.CHECKPOINT,source_archive_sha256=receipts[0]['sha256'],
        recomputed=recomputed,per_event=serial_rows(rows),peak_rss_sample_bytes=peak_rss,
        original_77_identities_unchanged=True,archived_execution_manifest_matches=True,
        AP='Reused exact archived AP; no AP sorter rerun',test_pixels_opened=0,
        inference_executed=False,threshold_crossfit_executed=False))
    from geofirex.newpos4_studio import export_cases
    export_cases(best,studio,receipts[0],n.CHECKPOINT)
    print('VERIFIED',N,'tiles; threshold analysis gate',memory[1]>=2*1024**3,flush=True)


if __name__=='__main__':main()
