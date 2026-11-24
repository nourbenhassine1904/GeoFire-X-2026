"""Verify preservation, then seal only the new V2 delivery and its sources."""
from pathlib import Path
import json

from geofirex.protocol_v2 import save, sha256


def seal():
    root = Path('reports/protocol_v2_20261118')
    evidence = Path('artifacts/protocol_v2_20261118')
    before = json.loads((evidence/'preservation_before.json').read_text())
    changes = []
    for name, expected in before['tracked'].items():
        path = Path(name)
        if not path.is_file():
            changes.append(name); continue
        s = path.stat()
        if (s.st_size != expected['bytes'] or s.st_mtime_ns != expected['mtime_ns']
                or ('sha256' in expected and sha256(path) != expected['sha256'])):
            changes.append(name)
    for name, expected in before['compact_archives'].items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != expected['bytes'] or path.stat().st_mtime_ns != expected['mtime_ns']:
            changes.append(name)
    if changes:
        raise ValueError('Historical files changed: '+str(changes))
    save(evidence/'preservation_after.json', dict(status='PASS',
         tracked_files_unchanged=len(before['tracked']),
         tracked_small_files_sha256_verified=sum('sha256' in r for r in before['tracked'].values()),
         other_tracked_files='size_and_mtime_identity_unchanged_no_massive_binary_rehash',
         compact_archives_size_and_mtime_unchanged=len(before['compact_archives']),
         published_11_partitions='not_modified_no_republication_no_massive_rehash',
         reference_sha256=sha256(evidence/'preservation_before.json'),
         git_commit_push_merge=False, test_pixels_opened=0))
    import xml.etree.ElementTree as ET
    suite = ET.parse(root/'tests-cpu-final.xml').getroot().find('testsuite')
    if suite is None or int(suite.attrib['failures']) or int(suite.attrib['errors']):
        raise ValueError('Final CPU test suite did not pass')
    decision = json.loads((root/'decision.json').read_text())
    for name, digest in decision['preregistration'].items():
        if sha256(name) != digest:
            raise ValueError('Preregistration changed without an updated preparation receipt')
    for name in ('events.csv', 'index/index.json', 'index/train.npy', 'index/val.npy',
                 'index/normalization.json', 'distributions.json'):
        path = root/name
        with Path(str(path)+'.sha256').open('x', encoding='ascii') as stream:
            stream.write(sha256(path)+'  '+path.name+'\n')
    additional = ['src/geofirex/protocol_v2.py', 'src/geofirex/v2_evaluation.py',
                  'scripts/prepare_protocol_v2.py', 'scripts/verify_protocol_v2_cpu.py',
                  'scripts/freeze_protocol_v2.py', 'scripts/seal_protocol_v2.py',
                  'tests/test_protocol_v2.py', 'tests/test_v2_evaluation.py']
    paths = [*root.rglob('*'), *Path('configs/protocol_v2').glob('*.json'),
             *(Path(p) for p in additional), evidence/'preservation_before.json', evidence/'preservation_after.json']
    paths = sorted({p for p in paths if p.is_file() and p.name not in ('delivery_manifest.json', 'SHA256SUMS')})
    files = [dict(path=p.as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in paths]
    manifest = root/'delivery_manifest.json'
    save(manifest, dict(schema='geofirex-v2-delivery-1', files=files,
         tests=dict(tests=int(suite.attrib['tests']), failures=0, errors=0),
         preservation='artifacts/protocol_v2_20261118/preservation_after.json',
         first_fit_scientific_decision='GO_BASELINE_C8', launch_decision='NO_GO_NOW',
         training_executed=False, model_forwards=0, optimizer_steps=0, test_pixels_opened=0))
    with (root/'SHA256SUMS').open('x', encoding='utf-8') as stream:
        for row in files:
            stream.write(row['sha256']+'  '+row['path']+'\n')
        stream.write(sha256(manifest)+'  '+manifest.as_posix()+'\n')
    with Path(str(manifest)+'.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha256(manifest)+'  '+manifest.name+'\n')
    # Check the sealed payload immediately, without touching historical binaries.
    if any(sha256(row['path']) != row['sha256'] for row in files):
        raise ValueError('V2 delivery identity mismatch')
    print('SEALED', len(files), 'files;', suite.attrib['tests'], 'CPU tests passed; historical files preserved')


if __name__ == '__main__':
    seal()
