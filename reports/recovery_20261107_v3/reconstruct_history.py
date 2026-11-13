"""Read historical snapshots and logs without importing test modules."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path.cwd()
out = root/'reports/recovery_20261107_v3'
old = root/'reports/recovery_20261107_v2'
active = old/'pytest-tmp/test_resume_across_changed_rea0'
snapshot = next((active/'runs').glob('*/source'))
nodes = []
for name in ('test_gpu_operational.py', 'test_gpu_resources.py', 'test_runner.py'):
    source = snapshot/'tests'/name
    for node in ast.parse(source.read_text(encoding='utf-8-sig')).body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith('test_'):
            continue
        cases = [[]]
        for decorator in reversed(node.decorator_list):
            if isinstance(decorator, ast.Call) and getattr(decorator.func, 'attr', None) == 'parametrize':
                values = ast.literal_eval(decorator.args[1])
                cases = [prior+list(value if isinstance(value, tuple) else (value,)) for prior in cases for value in values]
        nodes.extend(f'tests/{name}::{node.name}'+('['+'-'.join(map(str, case))+']' if case else '') for case in cases)
progress = (old/'tests-reference/output.log').read_text().strip()
assert progress == '.'*28
assert nodes[28] == 'tests/test_runner.py::test_resume_across_changed_readonly_mount_paths'
initial = ET.parse(root/'reports/gpu_operational_v1/tests-initial.xml')
proof = dict(last_command=json.loads((old/'tests-reference/supervisor.json').read_text())['command'],
             completed_nodes_reconstructed=nodes[:28], active_node_inferred=nodes[28],
             qualification='Default pytest order reconstructed from the actual v2 source snapshot, corroborated by its last incomplete fixture directory. No live stack was saved.',
             active_run_files=[str(p.relative_to(old)) for p in (active/'runs').rglob('*') if p.is_file() and 'source' not in p.relative_to(active/'runs').parts],
             historical_completed_suite=initial.find('testsuite').attrib,
             historical_failures=[dict(test=t.attrib, failure=f.attrib) for t in initial.iter('testcase') for f in t.findall('failure')])
(out/'history-reconstruction.json').open('x', encoding='utf-8').write(json.dumps(proof, indent=2))
files = []
for folder in ('reports/recovery_20261107_v1', 'reports/recovery_20261107_v2', 'reports/gpu_operational_v1'):
    files += [p for p in (root/folder).rglob('*') if p.is_file() and not any('tmp' in part for part in p.relative_to(root/folder).parts)]
files += list((root/'configs').glob('*.yaml'))
inventory = []
for path in sorted(set(files)):
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    inventory.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=value))
(out/'historical-inputs-before.json').open('x', encoding='utf-8').write(json.dumps(inventory, indent=2))
print(json.dumps(dict(completed=28, active=nodes[28], historical_files_pinned=len(inventory)), indent=2))
