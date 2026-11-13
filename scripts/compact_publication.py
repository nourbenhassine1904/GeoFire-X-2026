"""Independent private publication of the EXISTING compact corpus.

prepare/stage/verify are offline. publish requires --execute, a pinned plan SHA
and a durable body ceiling. Completed partitions are adopted by identity before
any upload; incomplete blob uploads use the CLI's resumable context. Never updates
or versions any dataset; never references the eight-lot orchestrator.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from geofirex.mission_streaming import save,sha

BASE=Path('reports/gpu_acceleration_20261111_v1')

def prepare(corpus,index,out):
    corpus,index,out=Path(corpus).resolve(),Path(index).resolve(),Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((corpus/'transfer-manifest.json').read_text());files=[{**r,'source':str(corpus/r['path'])} for r in manifest['files']]
    for name in ('report.json','transfer-manifest.json'):
        p=corpus/name;files.append(dict(path=name,source=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    original=sum(f['bytes'] for f in files)
    if original!=9845667110 or len(files)!=456:raise ValueError('Existing corpus transfer manifest changed')
    for name in ('train.npy','val.npy','index.json'):
        p=index/name;files.append(dict(path='index/'+name,source=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    groups=[];meta=[f for f in files if not f['path'].endswith('.zip')];groups.append(meta);current=[];n=0
    for f in files:
        if not f['path'].endswith('.zip'):continue
        if current and n+f['bytes']>1_000_000_000:groups.append(current);current=[];n=0
        current.append(f);n+=f['bytes']
    if current:groups.append(current)
    parts=[]
    for number,group in enumerate(groups):
        name=f'p{number:02d}';folder=out/name;folder.mkdir()
        for f in group:f['alias']=f['path'].replace('/','__')+'.bin'
        if len({f['alias'] for f in group})!=len(group):raise ValueError('Alias collision')
        canonical=json.dumps([{k:f[k] for k in ('path','alias','bytes','sha256')} for f in group],sort_keys=True,separators=(',',':')).encode();identity=hashlib.sha256(canonical).hexdigest()
        save(folder/'mount-map.json',dict(schema='opaque-compact-part-v1',identity=identity,files=[{k:f[k] for k in ('path','alias','bytes','sha256')} for f in group]))
        save(folder/'publication_identity.json',dict(identity=identity,files=len(group),payload_bytes=sum(f['bytes'] for f in group),schema='compact-t3-20261111-v1'))
        slug=f'geofirex-compact-t3-20261111-{name}';handle='nnnnbhss/'+slug
        save(folder/'dataset-metadata.json',dict(id=handle,title=slug,licenses=[dict(name='CC-BY-4.0')],description='Derived eight-channel WildfireSpreadTS, T=3/J+1. Source https://doi.org/10.5281/zenodo.8006177 (CC-BY-4.0). Private partition; .bin objects retain exact ZIP/NPZ/metadata bytes. See mount-map.json. No held-out 2021 pixels.'))
        controls=sum((folder/f).stat().st_size for f in ('mount-map.json','publication_identity.json'))
        parts.append(dict(number=number,folder=str(folder.resolve()),handle=handle,identity=identity,files=group,upload_file_bytes=sum(f['bytes'] for f in group)+controls))
    total=sum(p['upload_file_bytes'] for p in parts)
    plan=dict(schema='compact-private-publication-v1',corpus=str(corpus),index=str(index),original_files=456,original_file_bytes=original,parts=parts,payload_file_bytes=total,http_body_ceiling_bytes=12_000_000_000,headroom_bytes=12_000_000_000-total,uploads_executed=0,resume='identity-based partition skip + official resumable blob context; ambiguity stops; never create a version',remote_structure='flat opaque .bin objects + mount-map; reconstruct original paths using verified links, not unzip',license='CC-BY-4.0; existing docs/DATASET_VERSION.md')
    save(out/'plan.json',plan);print(json.dumps(dict(parts=len(parts),payload_file_bytes=total,http_body_ceiling_bytes=plan['http_body_ceiling_bytes'],plan_sha256=sha(out/'plan.json'))))

def stage(plan):
    for part in plan['parts']:
        folder=Path(part['folder'])
        for f in part['files']:
            source=Path(f['source']);target=folder/f['alias']
            if not source.is_file() or source.stat().st_size!=f['bytes']:raise ValueError('Existing source size differs')
            if target.exists():
                if not os.path.samefile(source,target):raise ValueError('Staging collision')
            else:os.link(source,target)
    print('Hardlink staging complete; no archive reconstructed or copied')

def restore(mounts,out,verify=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);seen={}
    for mount in map(Path,mounts):
        mapping=json.loads((mount/'mount-map.json').read_text())
        for f in mapping['files']:
            source=mount/f['alias'];logical=f['path'];target=out/logical
            if Path(logical).is_absolute() or '..' in Path(logical).parts:raise ValueError('Unsafe logical path')
            if logical in seen:raise ValueError('Duplicate logical object')
            if source.stat().st_size!=f['bytes'] or (verify and sha(source)!=f['sha256']):raise ValueError(f'Integrity mismatch: {logical}')
            target.parent.mkdir(parents=True,exist_ok=True)
            if os.name=='nt':os.link(source,target)
            else:target.symlink_to(source.resolve())
            seen[logical]=f['sha256']
    target=out/'index/normalization.json'
    if os.name=='nt':os.link(out/'normalization.json',target)
    else:target.symlink_to((out/'normalization.json').resolve())
    save(out/'mount-verification.json',dict(status='PASS',objects=len(seen),all_hashes_verified=verify,source='attached partition mounts',archives_unpacked=False))
    return out

def publish(plan_path,execute,pinned,ceiling):
    if not execute:print('OFFLINE: use prepare/stage; no upload permitted without --execute and pinned SHA');return
    if sha(plan_path)!=pinned:raise ValueError('Approved plan SHA mismatch')
    plan=json.loads(Path(plan_path).read_text())
    if ceiling!=plan['http_body_ceiling_bytes']:raise ValueError('Explicit budget differs from reviewed plan')
    sys.path[:0]=[str((BASE/'kaggle_sdk').resolve()),str(Path('scripts').resolve())]
    from acceleration_kaggle_readonly import client
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    from geofirex.upload_budget import BodyBudget
    journal=Path(plan_path).parent/'http-body.jsonl'
    with BodyBudget(journal,ceiling) as budget:
        undo=budget.install_requests_guard()
        try:
            # Authentication/control requests also share the durable ceiling.
            with client() as c:os.environ['KAGGLE_API_TOKEN']=c.api_token
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate()
            for part in plan['parts']:
                if not part['handle'].startswith('nnnnbhss/geofirex-compact-t3-20261111-'):raise ValueError('Destination outside compact namespace')
                owner,slug=part['handle'].split('/');request=ApiGetDatasetRequest();request.owner_slug=owner;request.dataset_slug=slug
                exists=False
                try:
                    with client() as c:r=c.datasets.dataset_api_client.get_dataset(request)
                    if r.is_private is not True:raise ValueError('Remote privacy differs')
                    exists=True
                except Exception as error:
                    status=getattr(getattr(error,'response',None),'status_code',None)
                    if status!=404:raise
                receipt=Path(part['folder'])/'publication-receipt.json'
                if exists:
                    probe=Path(part['folder']).parent/'remote-identities'/slug;probe.mkdir(parents=True,exist_ok=True)
                    api.dataset_download_file(part['handle'],'publication_identity.json',path=str(probe),force=True,quiet=True,unzip=True)
                    candidates=list(probe.glob('publication_identity.json'))
                    if len(candidates)!=1 or json.loads(candidates[0].read_text())['identity']!=part['identity']:raise ValueError('Existing dataset identity mismatch: no upload')
                    if not receipt.exists():save(receipt,dict(status='ADOPTED_EXISTING_PRIVATE_IDENTITY',handle=part['handle'],identity=part['identity']))
                    print('SKIP verified existing partition',part['number']);continue
                if receipt.exists():raise RuntimeError('Local receipt but remote absent: reconcile, never silently repeat')
                # Recheck only NEW publication objects, not historical raw TIFFs.
                for f in part['files']:
                    if sha(Path(part['folder'])/f['alias'])!=f['sha256']:raise ValueError('Staged compact changed')
                response=api.dataset_create_new(part['folder'],public=False,quiet=True,convert_to_csv=False,dir_mode='skip',ignore_patterns=['publication-receipt.json'])
                if getattr(response,'error',None):raise RuntimeError('Dataset create refused; reconcile remote state')
                save(receipt,dict(status='PRIVATE_CREATE_ACCEPTED_INTEGRITY_PENDING',handle=part['handle'],identity=part['identity'],reserved_body_bytes=budget.used))
                print('CREATE accepted; later mount verification required',part['number'])
        finally:undo();os.environ.pop('KAGGLE_API_TOKEN',None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','stage','publish','restore']);p.add_argument('--corpus',default='reports/mission_20261111/compact_full');p.add_argument('--index',default=str(BASE/'index'));p.add_argument('--out');p.add_argument('--plan',default=str(BASE/'publication/plan.json'));p.add_argument('--mounts',nargs='+');p.add_argument('--execute',action='store_true');p.add_argument('--approved-plan-sha256');p.add_argument('--max-http-body-bytes',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.corpus,a.index,a.out or BASE/'publication')
    elif a.mode=='stage':stage(json.loads(Path(a.plan).read_text()))
    elif a.mode=='restore':restore(a.mounts,a.out)
    else:publish(a.plan,a.execute,a.approved_plan_sha256,a.max_http_body_bytes)
