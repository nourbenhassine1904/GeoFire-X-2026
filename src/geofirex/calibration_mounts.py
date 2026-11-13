"""Verify and link already-published compact objects. Never publish or unpack."""
import hashlib
import json
import os
from pathlib import Path
from geofirex.calibration_io import save, sha


def restore_verified(maps, expected, out):
    maps = list(map(Path, maps))
    mappings = [(p, json.loads(p.read_text(encoding='utf-8'))) for p in maps]
    identities = [m['identity'] for _,m in mappings]
    if len(identities) != 11 or len(set(identities)) != 11 or set(identities) != set(expected):
        raise ValueError('Attach exactly the 11 expected compact partitions; duplicate/missing/unknown identity')
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    seen = set()
    for mapping_path,mapping in mappings:
        files = mapping['files']
        canonical = json.dumps([{k:f[k] for k in ('path','alias','bytes','sha256')} for f in files],
                               sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(canonical).hexdigest() != mapping['identity']:
            raise ValueError('Partition manifest differs from pinned identity')
        for f in files:
            logical,alias = Path(f['path']),Path(f['alias'])
            if (logical.is_absolute() or '..' in logical.parts or alias.name != f['alias']
                    or '2021' in logical.parts or logical.as_posix() in seen):
                raise ValueError('Unsafe, forbidden-year or duplicate compact object')
            source = mapping_path.parent / alias
            if source.stat().st_size != f['bytes'] or sha(source) != f['sha256']:
                raise ValueError('Compact object SHA/size mismatch: '+f['path'])
            target = out / logical
            target.parent.mkdir(parents=True,exist_ok=True)
            if os.name == 'nt':os.link(source,target)
            else:target.symlink_to(source.resolve())
            seen.add(logical.as_posix())
        print('Verified partition',mapping['identity'][:12],len(files),'unchanged objects',flush=True)
    target = out/'index/normalization.json'
    if os.name == 'nt':os.link(out/'normalization.json',target)
    else:target.symlink_to((out/'normalization.json').resolve())
    save(out/'mount-verification.json',dict(status='PASS',objects=len(seen),all_hashes_verified=True,
         archives_unpacked=False,source='11 existing read-only compact dataset mounts'))
    return out
