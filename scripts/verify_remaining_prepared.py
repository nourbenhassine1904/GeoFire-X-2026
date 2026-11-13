"""Targeted prepared-input checks: byte sizes, embedded manifests, configs; no pixels."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import shutil
import zlib


def main(plan_path):
    import yaml
    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text())
    results=[]
    raw=Path(plan['raw_root']).resolve()
    for spec in plan['lots']:
        lot=json.loads(Path(spec['canonical']).read_text())
        sized=0
        for entry in lot['files']:
            p=raw/entry['path']
            if not p.resolve().is_relative_to(raw) or not p.is_file() or p.stat().st_size!=entry['bytes']:
                raise ValueError('Local source missing/size differs: '+entry['path'])
            sized+=p.stat().st_size
        folder=Path(spec['cpu_notebook']['folder'])
        nb=json.loads((folder/'certify.ipynb').read_text())
        source=''.join(nb['cells'][1]['source'])
        tree=ast.parse(source)
        payload_node=next(n.value for n in tree.body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='payload')
        blob=payload_node.args[0].args[0].args[0].value
        payload=json.loads(zlib.decompress(base64.b64decode(blob)))
        if payload['reference']!=lot or len(payload['observations'])!=spec['files']:
            raise ValueError('Notebook manifest or QC row mismatch')
        if any(payload['split'][e]!=spec['role'] for e in lot['events']): raise ValueError('Notebook role mismatch')
        # Compile embedded runtime/helper code as well as notebook top-level code.
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='compile':
                ast.parse(node.args[0].value)
        meta=json.loads((folder/'kernel-metadata.json').read_text())
        if meta['dataset_sources']!=[spec['handle']] or meta['enable_gpu'] or meta['enable_tpu'] or meta['enable_internet'] or not meta['is_private']:
            raise ValueError('CPU source/privacy/device mismatch')
        if hashlib.sha256((folder/'certify.ipynb').read_bytes()).hexdigest()!=spec['cpu_notebook']['sha256']:
            raise ValueError('CPU notebook pin mismatch')
        if 'D:/' in json.dumps(payload) or 'D:\\' in json.dumps(payload): raise ValueError('Windows path in CPU payload')
        results.append(dict(number=spec['number'],events=spec['events'],files=spec['files'],raw_bytes=sized,
            local_source_sizes_verified=True,CPU_notebook_compiles=True,role=spec['role']))
    training=plan_path.parent/'training'
    ready=json.loads((training/'readiness.json').read_text())
    for spec in ready['models']:
        folder=Path(spec['folder'])
        nb=json.loads((folder/'certify.ipynb').read_text())
        ast.parse(''.join(nb['cells'][1]['source']))
        c=yaml.safe_load((training/f"bundle/configs/{spec['model']}_T3_seed42.yaml").read_text())
        if (c['corpus_scope']!='train_validation' or c['context']!=3 or c['seed']!=42
            or c['normalization_sha256']!=plan['normalizer_sha256'] or c['epochs']!=1
            or any('D:/' in str(v) or 'D:\\' in str(v) for v in c.values())):
            raise ValueError('Training configuration differs or uses Windows path')
    result=dict(status='TARGETED_PREPARATION_CHECKS_PASSED',lots=results,
        remaining_raw_bytes=sum(x['raw_bytes'] for x in results),raw_pixels_read=0,network_requests=0,
        test_2021_events=0,cpu_notebooks=7,training_templates=4,
        remote_certification_remaining='Not executed; CPU automation gate is required before bulk upload',gpu_started=False)
    free=shutil.disk_usage(plan_path.parent).free
    required=plan['archive_ceiling_bytes']+20_000_000_000
    result.update(staging_filesystem_free_bytes=free,free_bytes_required_for_all_retained_archives=required,
                  sufficient_staging_capacity=free>=required)
    (plan_path.parent/'preparation-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main(sys.argv[1])
