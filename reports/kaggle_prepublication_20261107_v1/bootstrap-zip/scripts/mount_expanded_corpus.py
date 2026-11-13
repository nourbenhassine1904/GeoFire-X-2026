"""Verify pre-attached private lots on Kaggle CPU; create relocated runtime configs."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib
import yaml

from geofirex.corpus import json_new, sha256_file
from geofirex.runner_data import digest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', default='reports/gpu_operational_v1/deployment-v3/bundle')
    p.add_argument('--input-root', default='/kaggle/input')
    p.add_argument('--out', default='/kaggle/working/expanded-mount-v1')
    a = p.parse_args()
    bundle, inputs, out = Path(a.bundle), Path(a.input_root), Path(a.out)
    if not inputs.resolve().is_relative_to(Path('/kaggle/input').resolve()):
        p.error('Only pre-attached read-only Kaggle datasets are supported')
    plan = json.loads((bundle/'deployment.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    roots, verified, events = [], [], set()
    actual_tifs = list(inputs.rglob('*.tif'))
    if any('2021' in path.relative_to(inputs).parts for path in actual_tifs):
        raise ValueError('Held-out 2021 rasters are attached; remove their dataset before training')
    inventories = list(inputs.rglob('lot-inventory.json'))
    for spec in plan['lots']:
        canonical = json.loads((bundle/'metadata'/f"lot-{spec['number']:03d}.json").read_text())
        identity = canonical.pop('identity')
        if digest(canonical) != identity or spec['identity'] != identity:
            raise ValueError('Pinned lot metadata changed')
        canonical['identity'] = identity
        matches = []
        for path in inventories:
            if json.loads(path.read_text()).get('identity') == identity:
                matches.append(path.parent)
        if len(matches) != 1:
            raise ValueError(f"Expected one immutable attached copy of {spec['handle']}, found {len(matches)}")
        root = matches[0]/'rasters'
        expected = {f['path'] for f in canonical['files']}
        if {p.relative_to(root).as_posix() for p in root.rglob('*.tif')} != expected:
            raise ValueError('Attached lot inventory differs')
        for f in canonical['files']:
            path = root/f['path']
            if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size != f['bytes']:
                raise ValueError('Attached size/path mismatch')
            sha, crc = hashlib.sha256(), 0
            with path.open('rb') as fp:
                for block in iter(lambda: fp.read(1024*1024), b''):
                    sha.update(block)
                    crc = zlib.crc32(block, crc)
            if sha.hexdigest() != f['sha256'] or f'{crc & 0xffffffff:08x}' != f['crc32']:
                raise ValueError('Attached SHA256/CRC mismatch')
            with (out/'verified.jsonl').open('a', encoding='utf-8') as fp:
                fp.write(json.dumps(dict(path=f['path'], sha256=f['sha256'], lot_identity=identity))+'\n')
            verified.append(f['path'])
        if events & set(canonical['events']):
            raise ValueError('Event duplicated across lots')
        events.update(canonical['events'])
        roots.append(str(root))
    if len(verified) != len(set(verified)) or len(verified) != 9490 or len(events) != 451:
        raise ValueError('Mounted train/validation corpus incomplete')
    for source in (bundle/'configs').glob('*.yaml'):
        c = yaml.safe_load(source.read_text())
        c.update(data_root=roots[0], data_roots=roots)
        (out/source.name).write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    json_new(out/'mount-proof.json', dict(status='REMOTE_TRAIN_VALIDATION_VERIFIED', files=len(verified), events=len(events),
                                        bytes=plan['raw_bytes'], test_rasters=0, roots=roots,
                                        code_manifest_sha256=sha256_file('code-manifest.json')))
    print('Verified 451 complete train/validation events, 9490 TIFF; runtime configs:', out)
