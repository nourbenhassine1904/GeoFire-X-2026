"""Build an additive offline recovery notebook; never launch GPU work."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from geofirex import fire13_experiment as c

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path('reports/fire13_recovery_20261123_v1')
ARCHIVES = Path('D:/GeoFire-X/GeoFire-X-V2')


def save_draft(path, data):
    path = Path(path).resolve()
    notebook = ROOT/'notebooks/GEOFIREX_FIRE13_EVALUATION_RECOVERY_ONLY.ipynb'
    if not path.is_relative_to(ROOT/REPORT) and path != notebook:
        raise ValueError('Recovery drafts only')
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')


def closure(root, seeds):
    pending, visited, external = list(seeds), {}, set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        path = root/('src/'+module.replace('.', '/')+'.py')
        if not path.exists():
            path = root/('src/'+module.replace('.', '/')+'/__init__.py')
        if not path.exists():
            raise FileNotFoundError(module)
        name = path.relative_to(root).as_posix()
        visited[module] = name
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            imports = []
            if isinstance(node, ast.Import):
                imports = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('Relative import needs explicit audit: '+name)
                if node.module == 'geofirex':
                    imports = ['geofirex.'+a.name for a in node.names]
                elif node.module:
                    imports = [node.module]
            for dependency in imports:
                if dependency.startswith('geofirex'):
                    pending.append(dependency)
                else:
                    external.add(dependency.split('.')[0])
    return visited, sorted(external)


def cell(kind, source, number):
    row = dict(cell_type=kind, id='recovery-'+str(number), metadata={}, source=source.splitlines(True))
    if kind == 'code':
        row.update(outputs=[], execution_count=None)
    return row


def main():
    out = ROOT/REPORT
    out.mkdir(exist_ok=True)
    _, _, _, lock, _ = c.identity(ROOT)
    source = ROOT/'reports/c8_propagation_20261121_v1/c8_experiments_code.zip'
    with zipfile.ZipFile(source) as z:
        historical = z.read('src/geofirex/c8_thresholds.py')
    if historical != (ROOT/'src/geofirex/c8_thresholds.py').read_bytes():
        raise ValueError('Historical threshold module differs')
    training = ARCHIVES/(c.B_ID+'-TRAINING-SAFE-c1145600.zip')
    partial = ARCHIVES/(c.B_ID+'-PARTIAL-373c40e9.zip')
    receipts = []
    before = c.read(out/'preservation_before.json')
    for archive in (training, partial):
        digest = c.old.verify_zip(archive, archive.with_suffix('.sha256').read_text().split()[0])
        with zipfile.ZipFile(archive) as z:
            complete = json.loads(z.read('training_complete.json'))
            checkpoint_bytes = z.read(complete['checkpoint'])
            if hashlib.sha256(checkpoint_bytes).hexdigest() != complete['checkpoint_sha256']:
                raise ValueError('Archived checkpoint identity differs')
            receipts.append(dict(path=str(archive), sha256=digest, members=len(z.namelist()),
                stage=json.loads(z.read('artifact_manifest.json'))['stage'], training_complete=complete,
                expanded_bytes=sum(i.file_size for i in z.infolist()),
                evaluation_members=[n for n in z.namelist() if n.startswith('evaluation/')]))
        for p in (archive, archive.with_suffix('.sha256')):
            before['files'][str(p)] = c.sha256(p)
        download = Path('C:/Users/nourb/Downloads')/archive.name
        if c.sha256(download) != digest:
            raise ValueError('Downloads / ADATA copies differ')
    if receipts[0]['training_complete'] != receipts[1]['training_complete']:
        raise ValueError('TRAINING-SAFE and PARTIAL final identities differ')
    (out/'preservation_before.json').write_text(json.dumps(before, indent=2)+'\n', encoding='utf8')
    save_draft(out/'archive_verification.json', dict(archives=receipts, downloads_identical=True,
        source_priority='ADATA HV300', training_archive_used=True, GPU_executed=False))
    modules, external = closure(ROOT, ['geofirex.fire13_recovery', 'geofirex.fire13_comparison',
        'geofirex.v2_propagation', 'geofirex'])
    files = dict(lock['files'])
    for path in modules.values():
        digest = c.sha256(ROOT/path)
        if path in files and files[path] != digest:
            raise ValueError('Attempted modification of frozen source: '+path)
        files[path] = digest
    additions = {n:h for n,h in files.items() if n not in lock['files']}
    save_draft(out/'dependency_audit.json', dict(modules=modules, external_import_roots=external,
        added_modules=additions, scope='Recursive AST, including function-local imports; isolated import smoke test also required'))
    save_draft(out/'module_provenance.json', dict(module='src/geofirex/c8_thresholds.py',
        sha256=hashlib.sha256(historical).hexdigest(), source=source.relative_to(ROOT).as_posix(),
        source_archive_sha256=c.sha256(source), member='src/geofirex/c8_thresholds.py',
        exact_bytes=True, historical_experiment='C8 Thresholds A', original_lock_modified=False))
    files[(REPORT/'dependency_audit.json').as_posix()] = c.sha256(out/'dependency_audit.json')
    files[(REPORT/'module_provenance.json').as_posix()] = c.sha256(out/'module_provenance.json')
    original = (c.REPORT/'execution_manifest.json').as_posix()
    files[original] = c.sha256(ROOT/original)
    files[(c.old.REPORT/'execution_manifest.json').as_posix()] = c.sha256(ROOT/c.old.REPORT/'execution_manifest.json')
    save_draft(out/'recovery_manifest.json', dict(schema='fire13-evaluation-recovery-1', files=files,
        original_manifest_sha256=c.sha256(ROOT/original), original_scientific_identities=77,
        training_archive_sha256=receipts[0]['sha256'], partial_archive_sha256=receipts[1]['sha256'],
        checkpoint_sha256=receipts[0]['training_complete']['checkpoint_sha256'],
        training_executed=False, evaluation_only=True, test_pixels_opened=0))
    for action in ('gpu-preflight', 'fire13-evaluate'):
        grant = c.read(ROOT/c.REPORT/('authorization_'+action+'_DISABLED.json'))
        grant.update(recovery_manifest_sha256=c.sha256(out/'recovery_manifest.json'),
            training_archive_sha256=receipts[0]['sha256'], checkpoint_sha256=receipts[0]['training_complete']['checkpoint_sha256'],
            authorization_id='', evaluation_only=True, output_name=c.B_ID+'_EVALUATION_RECOVERY_ONLY')
        save_draft(out/('authorization_'+action+'_NEW_DISABLED.json'), grant)
    buffer = io.BytesIO()
    names = set(files) | {(REPORT/'recovery_manifest.json').as_posix()}
    names |= {(REPORT/('authorization_'+a+'_NEW_DISABLED.json')).as_posix() for a in ('gpu-preflight','fire13-evaluate')}
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):
            info = zipfile.ZipInfo(name, (2026,11,23,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info, (ROOT/name).read_bytes())
    payload = buffer.getvalue(); payload_sha = hashlib.sha256(payload).hexdigest()
    (out/'fire13_recovery_code.zip').write_bytes(payload)
    (out/'fire13_recovery_code.sha256').write_text(payload_sha+'  fire13_recovery_code.zip\n', encoding='ascii')
    cells = [cell('markdown', '# FIRE13 EVALUATION RECOVERY ONLY\n\nInternet OFF. GPU T4, une seule carte visible. Attacher les onze partitions existantes p00-p10 et un dataset prive contenant TRAINING-SAFE + son .sha256. Aucun entrainement. Aucun pixel 2021. Seuil principal 0,78 et criteres historiques inchanges. Les deux autorisations ci-dessous sont nouvelles. Ne jamais supprimer un verrou pour recommencer.\n', 0),
        cell('code', "RUN_GPU_PREFLIGHT = False\nRUN_EVALUATION = False\nimport os,sys\nif 'torch' in sys.modules: raise RuntimeError('Restart kernel before CUDA configuration')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n", 1),
        cell('code', "from pathlib import Path\nimport base64,hashlib,io,zipfile,json\n"+
            f"DATA=base64.b64decode({base64.b64encode(payload).decode()!r})\nassert hashlib.sha256(DATA).hexdigest()=={payload_sha!r}\n"+
            "ROOT=Path('/kaggle/working/geofirex-fire13-recovery-code')\nROOT.mkdir(exist_ok=False)\nwith zipfile.ZipFile(io.BytesIO(DATA)) as z:\n    for n in z.namelist():\n        assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())\n    z.extractall(ROOT)\nsys.path.insert(0,str(ROOT/'src'))\nfrom geofirex import fire13_recovery as r\nc=r.c\nimport importlib\nfor module in c.read(ROOT/r.REPORT/'dependency_audit.json')['modules']:\n    importlib.import_module(module)\nprint(r.identity(ROOT)['checkpoint_sha256'])\n", 2),
        cell('code', "EXPECTED={v['map_sha256'] for v in c.read(ROOT/c.old.REPORT/'partitions.json').values()}\nMAPS=[p for p in Path('/kaggle/input').rglob('mount-map.json') if c.sha256(p) in EXPECTED]\nassert len(MAPS)==11\nprint(c.preflight(ROOT,MAPS))\n"+
            f"SOURCES=list(Path('/kaggle/input').rglob({training.name!r}))+list(Path('/kaggle/input').rglob({training.with_suffix('.bin').name!r}))\nassert len(SOURCES)==1, 'Attach exactly one original TRAINING-SAFE archive (zip or unchanged bin copy)'\nARCHIVE=SOURCES[0]\nSIDECAR=ARCHIVE.with_suffix('.sha256')\nassert SIDECAR.is_file()\nSOURCE=Path('/kaggle/working/fire13-restored-training-safe')\nOUT=Path('/kaggle/working')/r.OUT_NAME\nprint(r.resource_preflight(ROOT,ARCHIVE,OUT.parent))\n", 3),
        cell('markdown', 'Renseigner les observations ACTUELLES de quota et session dans la cellule suivante. Reserve conservative : 300 minutes et 10 heures-carte (deux T4 potentiellement facturees), 16 GiB RAM totale / 8 GiB disponibles, au moins 18 GiB disque plus restauration/copies. Delai inference + AP + comparaisons : 8100 secondes. Les reserves ne garantissent pas la duree. Activer explicitement CREATE_NEW_AUTHORIZATIONS une seule fois, puis RUN_GPU_PREFLIGHT et enfin RUN_EVALUATION. Ne pas reutiliser le grant du fit.\n', 4),
        cell('code', "CREATE_NEW_AUTHORIZATIONS = False\nOPERATOR = ''\nSESSION_REMAINING_MINUTES = None\nQUOTA_REMAINING_CARD_HOURS = None\nLOCAL_BACKUP = r'C:\\Users\\nourb\\Projects\\GeoFire-X-2026\\artifacts\\fire13'\nEXTERNAL_BACKUP = r'D:\\GeoFire-X\\GeoFire-X-V2'\nGPU_AUTH=OUT.parent/'fire13_recovery_gpu_authorization.json'\nEVAL_AUTH=OUT.parent/'fire13_recovery_evaluate_authorization.json'\nif CREATE_NEW_AUTHORIZATIONS:\n    from datetime import datetime,timezone\n    import uuid\n    assert OPERATOR.strip() and SESSION_REMAINING_MINUTES is not None and QUOTA_REMAINING_CARD_HOURS is not None\n    now=datetime.now(timezone.utc).isoformat()\n    for action,path in [('gpu-preflight',GPU_AUTH),('fire13-evaluate',EVAL_AUTH)]:\n        a=c.read(ROOT/r.REPORT/('authorization_'+action+'_NEW_DISABLED.json'))\n        a.update(authorized=True,authorized_by=OPERATOR,authorized_at=now,quota_checked_at=now,\n            authorization_id=str(uuid.uuid4()),session_remaining_minutes=SESSION_REMAINING_MINUTES,\n            quota_remaining_card_hours=QUOTA_REMAINING_CARD_HOURS,card_hours_reserved=10.,\n            local_backup_destination=LOCAL_BACKUP,external_backup_destination=EXTERNAL_BACKUP)\n        c.save(path,a)\n        r.authorize(ROOT,path,action,OUT.parent)\n", 5),
        cell('code', "if RUN_GPU_PREFLIGHT:\n    r.authorize(ROOT,GPU_AUTH,'gpu-preflight',OUT.parent)\n    print(c.preflight(ROOT,MAPS,OUT.parent,'cuda',gpu=True))\nelse: print('GPU preflight disabled')\n", 6),
        cell('code', "if RUN_EVALUATION:\n    assert RUN_GPU_PREFLIGHT\n    r.authorize(ROOT,EVAL_AUTH,'fire13-evaluate',OUT.parent)\n    r.resource_preflight(ROOT,ARCHIVE,OUT.parent)\n    r.restore(ROOT,ARCHIVE,SIDECAR,SOURCE)\n    print(r.verify_final(ROOT,SOURCE)[1])\n    r.recover(ROOT,MAPS,SOURCE,OUT,EVAL_AUTH,GPU_AUTH,ARCHIVE)\nelse: print('Evaluation disabled')\n", 7),
        cell('code', "from IPython.display import display,FileLink\nfor p in sorted(OUT.parent.glob(r.OUT_NAME+'-*.zip')):\n    print(c.old.verify_zip(p,p.with_suffix('.sha256').read_text().split()[0]))\n    display(FileLink(str(p)));display(FileLink(str(p.with_suffix('.sha256'))))\nfor p in OUT.parent.glob('fire13-evaluate-*.lock.json'): display(FileLink(str(p)))\nif (OUT/'evaluation/complete.json').exists():\n    print(json.dumps(c.read(OUT/'evaluation/paired_comparison/FIRE13_acceptance.json'),indent=2))\n", 8),
        cell('markdown', 'Sauvegarder ZIP final, SHA-256 et verrou de consommation dans les deux destinations, puis verifier les hashes. Les chemins Windows sont des destinations de telechargement, pas des montages accessibles a Kaggle. Conserver les archives originales et le verrou du fit sans modification. Conserver ce notebook execute et le quota reel avant/apres session. Aucun nouvel essai apres consommation sans revue explicite ; le verrou local ne peut pas empecher un nouvel essai dans une autre session Kaggle vierge. Ne pas relancer cette evaluation dans une autre session. Aucune integration Studio automatique.\n', 9)]
    notebook = ROOT/'notebooks/GEOFIREX_FIRE13_EVALUATION_RECOVERY_ONLY.ipynb'
    save_draft(notebook, dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'))))
    save_draft(out/'notebook_build.json', dict(payload_sha256=payload_sha,payload_bytes=len(payload),
        notebook_sha256=c.sha256(notebook),GPU_executed=False,published=False))
    save_draft(out/'kernel_metadata_NOT_PUBLISHED.json', dict(title='FIRE13 EVALUATION RECOVERY ONLY',
        code_file=notebook.name,language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_internet=False,dataset_sources=c.read(ROOT/c.old.REPORT/'dataset-sources.json'),
        additional_private_archive_dataset_required=True))
    print(json.dumps(dict(status='PREPARED_NO_GPU_LAUNCH',modules=len(modules),additions=additions,
        notebook=str(notebook),archives=receipts),indent=2))


if __name__ == '__main__':
    main()
