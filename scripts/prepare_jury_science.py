"""Import completed validation2020 evidence, never launch training or recovery."""
from datetime import date,timedelta
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from geofirex import v2_execution as old
from geofirex.protocol_v2 import save,sha256,V2CompactTiles,sample_keys
from geofirex.c8_predictions import PIXEL
from geofirex.c8_comparison import rows_from_result
from geofirex.fire13_comparison import summary,gates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/jury_20261202_v1'
SOURCE=Path('D:/GeoFire-X/GeoFire-X-V2')
NAMES={'C8':'B2_C8_V2_seed42-COMPLETE-48e35c59.zip',
 'NEWPOS2':'B2_C8_V2_NEWPOS2_seed42-COMPLETE-193dafd8.zip',
 'NEWPOS4':'B2_C8_V2_NEWPOS4_seed42-COMPLETE-2193b5a8.zip',
 'FIRE13':'B2_C13_FIRE_V2_seed42_EVALUATION_RECOVERY_ONLY-EVALUATION-RECOVERY-COMPLETE-e1c5a248.zip'}
DOMAIN='efc995be596d7d520bac472ceb1b5122fe2111fe9d5a53c5edbec02851c1accc'


def load(z,name):return json.loads(z.read(name))


def saved_tile(archive,key):
    with zipfile.ZipFile(archive) as z:
        manifest=load(z,'evaluation/predictions/predictions_manifest.json')
        row=next(r for r in manifest['files'] if r['event']==key[0])
        i=[tuple(k) for k in row['keys']].index(key)
        with z.open('evaluation/predictions/'+row['path']) as stream:
            version=np.lib.format.read_magic(stream)
            header=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=header(stream)
            if fortran or dtype!=PIXEL or shape[1:]!=(64,64):raise ValueError('Prediction schema differs')
            stream.seek(i*64*64*PIXEL.itemsize,1)
            return np.frombuffer(stream.read(64*64*PIXEL.itemsize),PIXEL).reshape(64,64).copy()


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'science_manifest.json').exists():raise FileExistsError('Preserve sealed jury evidence')
    config,lock,_=old.identity(ROOT)
    policy_path=ROOT/'reports/fire13_20261122_v1/acceptance.json'
    policy=old.read(policy_path);models=[];results={};sources=[]
    for model,name in NAMES.items():
        p=SOURCE/name;expected=p.with_suffix('.sha256').read_text().split()[0]
        digest=old.verify_zip(p,expected)
        with zipfile.ZipFile(p) as z:
            artifact=load(z,'artifact_manifest.json')
            expected_stage='EVALUATION-RECOVERY-COMPLETE' if model=='FIRE13' else 'COMPLETE'
            if artifact['stage']!=expected_stage:raise ValueError('Wrong completed archive stage')
            result=load(z,'evaluation/evaluation.json');receipt=load(z,'training_complete.json')
            complete=load(z,'evaluation/complete.json')
            if (result['status']!='COMPLETE' or result['threshold']!=.78 or result['tiles']!=30882 or
                result['evaluation_domain_sha256']!=DOMAIN or len(result['per_event'])!=80 or
                any(not r['event_id'].startswith('2020/') for r in result['per_event']) or
                complete['checkpoint_sha256']!=receipt['checkpoint_sha256']):raise ValueError('Frozen V2 domain differs')
            checkpoint=next(r for r in artifact['files'] if r['path']==receipt['checkpoint'])
            if checkpoint['sha256']!=receipt['checkpoint_sha256']:raise ValueError('Checkpoint receipt differs')
            raw=z.read('evaluation/evaluation.json');dst=OUT/(model+'_evaluation.json')
            with dst.open('xb') as f:f.write(raw)
            sources.append(dict(model=model,archive=str(p),archive_sha256=digest,
                result_member='evaluation/evaluation.json',result_sha256=hashlib.sha256(raw).hexdigest(),
                checkpoint_sha256=receipt['checkpoint_sha256'],all_archive_members_verified=True))
            row=dict(model=model,scope='validation2020_V2_80',threshold=.78,events=80,tiles=30882,
                channels_per_day=13 if model=='FIRE13' else 8,updates=receipt['updates'],epochs=receipt['epochs'],
                status='reference_V2' if model=='C8' else 'experimental_not_promoted',
                promotion_authorized=False,**summary(result))
            if model!='C8':row['scientific_gates']=gates(result,policy)
            models.append(row);results[model]=result
            if model=='FIRE13':
                fp=load(z,'evaluation/comparable_fp/result.json')
                save(OUT/'FIRE13_comparable_fp_saved.json',fp)
                save(OUT/'FIRE13_acceptance_saved.json',load(z,'evaluation/paired_comparison/FIRE13_acceptance.json'))
    base=rows_from_result(results['C8'])
    for name,result in results.items():
        rows=rows_from_result(result)
        if set(rows)!=set(base):raise ValueError('Event coverage differs')
        for e in base:
            for domain in ('pooled','new_fire'):
                a,b=base[e][domain],rows[e][domain]
                if a.sum()!=b.sum() or a[0]+a[2]!=b[0]+b[2]:raise ValueError('Targets/masks differ')
    data=dict(schema='geofirex-jury-validation80-1',status='VERIFIED_SAVED_RESULTS',
        label='Validation 2020 — protocole V2, 80 événements déjà observés',
        evaluation_domain_sha256=DOMAIN,threshold=.78,models=models,sources=sources,
        acceptance=policy,acceptance_sha256=sha256(policy_path),
        conclusion='Aucune variante ne satisfait tous les critères. FIRE13 : absence de supériorité démontrée à FP comparable.',
        live_model=dict(protocol='legacy201',checkpoint_sha256='7295eaa6f957e59a31b7155403881a9b3bf19933a35b6872768896bda1f1b14e',
            label='U-Net poids10 historique, 201 événements (32 réglage + 169 confirmation observée)',
            different_from_V2_reference=True),test_2021_locked=True,automatic_promotion=False,
        FIRE13_closed=True,training_executed=False,new_aggregate_evaluation=False)
    save(OUT/'comparison.json',data)
    fields=['model','scope','status','threshold','events','tiles','recall_new','dice_new','precision_new','AP_new','FP_new','dice_global','AP_global','Brier','macro_event_dice']
    with (OUT/'comparison.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(models)
    # Outcome-independent first frozen key. Saved predictions only for experiments.
    with zipfile.ZipFile(SOURCE/NAMES['FIRE13']) as z:
        manifest=load(z,'evaluation/predictions/predictions_manifest.json')
        key=min(tuple(k) for row in manifest['files'] for k in row['keys'])
    arrays={name:saved_tile(SOURCE/NAMES[name],key) for name in ('NEWPOS2','NEWPOS4','FIRE13')}
    maps=[ROOT/f'reports/gpu_acceleration_20261111_v1/publication/p{i:02d}/mount-map.json' for i in range(11)]
    mounts,mount_receipt=old.verify_mounts(ROOT,maps)
    ds=V2CompactTiles('',ROOT/config['index'],'val',mounts=mounts)
    try:
        idx=sample_keys(ds).index(key);x,y,v=ds[idx]
    finally:ds.close()
    import torch
    from geofirex.benchmark_models import StackedUNet
    torch.set_num_threads(2)
    with zipfile.ZipFile(SOURCE/NAMES['C8']) as z:
        state=torch.load(io.BytesIO(z.read(load(z,'training_complete.json')['checkpoint'])),map_location='cpu',weights_only=False)
    model=StackedUNet(8,3,base_channels=config['base_channels'],depth=config['depth']).eval()
    model.load_state_dict(state['model'],strict=True)
    with torch.inference_mode():probability=model(x.unsqueeze(0)).sigmoid().numpy()[0,0]
    a=np.empty((64,64),PIXEL)
    for name,value in zip(('p','y','v','o'),(probability,y[0].numpy(),v[0].numpy(),x[-1,7].numpy())):a[name]=value
    arrays['C8']=a
    for value in arrays.values():
        for name in ('y','v','o'):
            if not np.array_equal(value[name],a[name]):raise ValueError('Diagnostic tile domain differs')
    cases=[]
    for name in NAMES:
        value=arrays[name];mask=value['v']==1
        nullable=lambda a:np.where(mask,a.astype(object),None).tolist()
        origin=list(key[2:]);as_of=(date.fromisoformat(key[1])-timedelta(days=1)).isoformat()
        result=dict(event_id=key[0],as_of=as_of,target_date=key[1],model=name+' V2',threshold=.78,
            scope='validation2020_V2_80_deterministic_tile',full_shape=[64,64],preview_stride=1,tile_origin=origin,
            probability=nullable(value['p']),observed_fire=nullable(value['o']),
            predicted_fire=nullable((value['p']>=np.float32(.78)).astype(np.uint8)),
            new_predicted_fire=nullable(((value['p']>=np.float32(.78))&(value['o']==0)).astype(np.uint8)),
            checkpoint_sha256=next(s['checkpoint_sha256'] for s in sources if s['model']==name),
            source_archive_sha256=next(s['archive_sha256'] for s in sources if s['model']==name),
            optimizer_updates=28976,crs=None,transform=None,
            provenance='CPU point inference from original C8 checkpoint' if name=='C8' else 'Saved COMPLETE predictions; no new inference',
            input_dates=[(date.fromisoformat(key[1])-timedelta(days=i)).isoformat() for i in (3,2,1)])
        cases.append(dict(model=name,result=result,truth=nullable(value['y'])))
    save(OUT/'diagnostic_tile.json',dict(key=list(key),selection='lexicographically first frozen validation key; outcome independent; not representative',
        cases=cases,spatial_reference='64x64 pixel tile, no CRS or affine',new_Fire13_inference=False,
        C8_point_inference=True,mount_receipt=mount_receipt))
    files={p.name:sha256(p) for p in OUT.glob('*.json') if p.name not in ('preservation_before.json','science_manifest.json')}
    files['comparison.csv']=sha256(OUT/'comparison.csv')
    save(OUT/'science_manifest.json',dict(files=files,source_archives=sources,original_scientific_identities=77,
        original_manifest_sha256=sha256(ROOT/old.REPORT/'execution_manifest.json')))
    print(json.dumps(dict(status='VERIFIED',models=models,diagnostic_key=key),indent=2))


if __name__=='__main__':main()
