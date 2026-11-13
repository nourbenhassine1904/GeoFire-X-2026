"""Prepare CPU-only private notebooks locally; no API calls or kernel pushes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(out, frozen):
    out, frozen=Path(out),Path(frozen)
    out.mkdir(parents=True,exist_ok=False)
    helper=Path('scripts/kaggle_certify_inputs.py').read_text(encoding='utf-8')
    review=json.loads(Path('reports/kaggle_prepublication_20261107_v1/frozen-review-final.json').read_text())
    payload_sha=review['package']['sha256']; manifest_sha=review['package']['manifest_sha256']
    code_handle='nnnnbhss/geofirex-expanded-code-recovery-v1'
    bundle='reports/recovery_20261107_v3/finalized/deployment/bundle'
    plan=json.loads((frozen/'deployment/bundle/deployment.json').read_text())
    receipts=[]
    for lot in [None]+plan['lots']+['all']:
        tag='code' if lot is None else ('all' if lot=='all' else f"lot-{lot['number']:03d}")
        folder=out/tag; folder.mkdir()
        sources=[code_handle]+([] if lot is None else ([x['handle'] for x in plan['lots']] if lot=='all' else [lot['handle']]))
        # Embed reviewed helper so the frozen code archive remains byte-identical.
        source=("import json, os\nfrom pathlib import Path\n"
                "namespace = {'__name__': 'geofirex_cpu_helpers'}\n"
                f"exec(compile({helper!r}, 'kaggle_certify_inputs.py', 'exec'), namespace)\n"
                f"work=namespace['bootstrap_code']('/kaggle/input/{code_handle.split('/')[1]}', '/kaggle/working/geofirex-reviewed', {payload_sha!r}, {manifest_sha!r})\n"
                f"runtime=namespace['check_runtime'](work, {bundle!r})\n"
                "Path('/kaggle/working/runtime-certificate.json').write_text(json.dumps(runtime,indent=2))\n"
                "print(json.dumps(runtime,indent=2))\n")
        if lot == 'all':
            source+=("import subprocess, sys\n"
                     "namespace['resource_check'](__import__('time').monotonic()+3600, min_available=4*1024**3)\n"
                     f"subprocess.run([sys.executable,'scripts/mount_expanded_corpus.py','--bundle',{bundle!r},'--out','/kaggle/working/expanded-mount-v1'],cwd=work,check=True,timeout=3600)\n")
        elif lot is not None:
            source+=(f"certificate=namespace['certify_lot'](work/{bundle!r}/'metadata/lot-{lot['number']:03d}.json', "
                     f"'/kaggle/input/{lot['slug']}', '/kaggle/working/{tag}-certificate', pinned_identity={lot['identity']!r}, seconds=3600)\n"
                     "print(json.dumps(certificate,indent=2))\n")
        ast.parse(source)
        notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=[
            f'# GeoFire-X CPU certification: {tag}\n',
            'Private CPU only. No Internet, installation, training or GPU. Missing dependency or checksum mismatch means STOP.\n']),
            dict(cell_type='code',metadata={},source=source.splitlines(keepends=True),execution_count=None,outputs=[])],
            metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
        file=folder/'certify.ipynb'; file.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
        metadata=dict(id=f'nnnnbhss/geofirex-certify-{tag}-20261107-v1',title=f'GeoFireX Certify {tag.replace("-"," ")} 20261107 V1',
            code_file='certify.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,
            enable_internet=False,dataset_sources=sources,competition_sources=[],kernel_sources=[])
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        receipts.append(dict(tag=tag,notebook_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),sources=sources,
                             kernel=metadata['id'],executed=False))
    (out/'prepared.json').write_text(json.dumps(dict(status='LOCAL_PREPARATION_ONLY',helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),
        frozen_payload_sha256=payload_sha,notebooks=receipts),indent=2),encoding='utf-8')
    return receipts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True); p.add_argument('--frozen',required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.out,a.frozen),indent=2))
