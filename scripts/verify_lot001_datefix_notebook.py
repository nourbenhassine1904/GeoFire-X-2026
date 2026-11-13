"""Offline check of the generated notebook; no raster opens or Kaggle calls."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile
import zlib


def main(folder):
    folder = Path(folder)
    notebook_bytes = (folder/'certify.ipynb').read_bytes()
    notebook = json.loads(notebook_bytes)
    code = ''.join(notebook['cells'][1]['source'])
    tree = ast.parse(code)
    assignments = {n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    payload_blob = assignments['payload'].args[0].args[0].args[0].value
    payload = json.loads(zlib.decompress(base64.b64decode(payload_blob)))
    archive_blob = assignments['archive'].args[0].args[0].args[0].value
    source_archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(archive_blob)))
    sources = {name:source_archive.read(name).decode('utf-8') for name in source_archive.namelist()}
    runtime_call = next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                        and isinstance(n.value.func,ast.Name) and n.value.func.id=='exec')
    runtime = runtime_call.value.args[0].args[0].value
    ast.parse(runtime)
    canonical = Path('reports/recovery_20261107_v3/finalized/deployment/bundle/metadata/lot-001.json')
    lot = json.loads(canonical.read_text())
    assert payload['reference'] == lot
    assert payload['allow_2017_context'] is True
    assert len(lot['files']) == 1269 and len(lot['events']) == 57
    assert all(payload['split'][event]=='train' for event in lot['events'])
    assert {r['relative_path'] for r in payload['observations']} == {r['path'] for r in lot['files']}
    source_manifest = ast.literal_eval(assignments['source_manifest'])
    assert all(hashlib.sha256(sources[n].encode()).hexdigest()==sha for n,sha in source_manifest.items())
    metadata = json.loads((folder/'kernel-metadata.json').read_text())
    assert metadata['dataset_sources'] == ['nnnnbhss/geofirex-wsts-fold0-tv-001-v1']
    assert metadata['is_private'] is True
    assert all(metadata[key] is False for key in ('enable_gpu','enable_tpu','enable_internet'))
    drive_references = []
    # Inspect decoded literals: serialized Python "except ... e:\\n" is not a drive.
    literals = '\n'.join(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str))
    for label, content in [('code_literals',literals),('runtime',runtime),('payload',json.dumps(payload)),*sources.items()]:
        for match in re.finditer(r'(?i)\b[a-z]:[\\/]',content):
            drive_references.append((label,content[max(0,match.start()-50):match.end()+100]))
    assert not drive_references, drive_references[:5]
    assert "dt.date.fromisoformat(p.stem).year != int(p.parts[0])" not in runtime
    for marker in ("len(expected) != 1269", "len(reference['events']) != 57", "hashlib.sha256()",
                   'zlib.crc32', "row != entry", 'validate_layout(ds)', "a.shape[0] != 23",
                   'Forbidden observation year', 'Held-out 2021 data attached', 'num_workers=0'):
        assert marker in runtime, marker
    from certify_kaggle_lot001_runtime import validate_lot_dates
    date_policy = validate_lot_dates(lot, payload['split'], allow_2017_context=True)
    result = dict(status='LOCAL_NOTEBOOK_VERIFIED_REMOTE_UNCERTIFIED', notebook=str((folder/'certify.ipynb').resolve()),
        notebook_bytes=len(notebook_bytes), notebook_sha256=hashlib.sha256(notebook_bytes).hexdigest(),
        canonical_manifest_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest(),
        files=1269,events=57,dataset_sources=metadata['dataset_sources'],gpu=False,tpu=False,internet=False,
        windows_drive_dependency=False,preserved_checks=['57 train events','1269 files','23-band layout',
            'sizes','SHA256','CRC32','no 2020/2021 observations','no attached test 2021','actual CPU DataLoader'],
        date_policy=date_policy)
    (folder.parent/'datefix-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
