"""Resolve eight certified read-only Kaggle lots without Windows mount assumptions.

Hashes are established by the saved CPU certificates. This mount step checks
immutable inventories, counts, roles, and sizes, without recertifying lot 001.
"""
import argparse
import hashlib
import json
from pathlib import Path


def mount(bundle,inputs='/kaggle/input'):
    import yaml
    bundle=Path(bundle)
    specs=json.loads((bundle/'certification-gates.json').read_text())
    inventories=list(Path(inputs).rglob('lot-inventory.json'))
    roots=[]
    events=set()
    files=set()
    for spec in specs['lots']:
        cert=spec['certificate']
        if cert is None: raise ValueError('Missing verified CPU certificate: '+spec['handle'])
        lot=json.loads((bundle/f"metadata/lot-{spec['number']:03d}.json").read_text())
        identity=hashlib.sha256(json.dumps({k:v for k,v in lot.items() if k!='identity'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if identity!=spec['identity'] or lot['identity']!=identity: raise ValueError('Pinned lot identity changed')
        matches=[p for p in inventories if json.loads(p.read_text()).get('identity')==identity]
        if len(matches)!=1 or json.loads(matches[0].read_text())!=lot: raise ValueError('Attached lot inventory mismatch')
        if events & set(lot['events']): raise ValueError('Event duplicated across lots')
        root=matches[0].parent/'rasters'
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        expected={r['path'] for r in lot['files']}
        if actual!=expected or files & expected: raise ValueError('Attached file set mismatch')
        for entry in lot['files']:
            path=root/entry['path']
            if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size!=entry['bytes']:
                raise ValueError('Attached size/path mismatch')
        events.update(lot['events']);files.update(expected);roots.append(str(root))
    if len(inventories)!=8 or len(events)!=451 or len(files)!=9490 or any(e.startswith('2021/') for e in events):
        raise ValueError('Require exactly 451 train/validation events and 9490 files, without test')
    all_tifs=[p for p in Path(inputs).rglob('*') if p.is_file() and p.suffix.lower() in ('.tif','.tiff')]
    if len(all_tifs)!=9490 or any('2021' in p.parts or p.stem.startswith('2021-') for p in all_tifs):
        raise ValueError('Unexpected/held-out 2021 rasters attached')
    for config in (bundle/'configs').glob('*.yaml'):
        c=yaml.safe_load(config.read_text())
        c.update(data_root=roots[0],data_roots=roots)
        config.write_text(yaml.safe_dump(c,sort_keys=False))
    (bundle/'mounted.json').write_text(json.dumps(dict(events=451,files=9490,test_files=0,roots=roots,
        hash_certification_reused=True,lot001_recertified=False),indent=2))
    return roots


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True)
    a=p.parse_args();print(json.dumps(mount(a.bundle),indent=2))
