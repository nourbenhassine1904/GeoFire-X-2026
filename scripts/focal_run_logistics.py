"""Logistics around the byte-frozen Focal runner; default never trains.

No model/loss/loader changes. Training needs an external matching authorization.
The only hook saves the completed training BEFORE the unchanged evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid
import zipfile


GIB = 1024 ** 3
CONFIG_SHA = '48a13f21fc1bf7f1d581025f585f78fc254ef9bd298fe4492c469cdd1256defc'
CODE_SHA = '8f48deba428b11be05cc20ccaaf9c4ef3dec19445a4479ccfbf9dd19355a16bd'
PASS_SHA = 'd52dccc18e6caa71532c6c73de100d7294aff87d5d3a4fb785f401459a3ac3af'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def checked_pass(path, config):
    if sha(path) != PASS_SHA:
        raise ValueError('Downloaded PASS report changed')
    r = read(path)
    assert r['status'] == 'PASS' and r['causes'] == []
    assert not r['training_authorized'] and not r['RUN_TRAINING']
    assert r['optimizer_steps'] == r['forward_calls'] == r['test2021_pixels_opened'] == 0
    assert r['environment']['torch'] == config['torch_version']
    assert r['environment']['numpy'] == config['numpy_version']
    assert r['environment']['gpu_count'] == 2
    assert all(g['name'] == 'Tesla T4' for g in r['environment']['gpus'])
    init = r['initialization']
    assert init['status'] == 'PASS'
    assert init['expected_sha256'] == init['cpu_sha256'] == init['post_cuda_sha256'] == config['initial_state_sha256']
    corpus = r['corpus']
    assert corpus['status'] == 'PASS' and corpus['all_bytes_verified']
    assert set(corpus['partitions']) == {p['identity'] for p in config['dataset_partitions']}
    assert len(corpus['partitions']) == 11 and corpus['objects'] == 459
    assert corpus['sampler_order_sha256'] == config['sampler_order_sha256']
    assert r['frozen_config_sha256'] == CONFIG_SHA
    return r


def identity(workspace):
    w = Path(workspace)
    if sha(w / 'focal_config.json') != CONFIG_SHA or sha(w / 'focal-code.zip') != CODE_SHA:
        raise ValueError('Frozen scientific code/config differs')
    c = read(w / 'focal_config.json')
    checked_pass(w / 'precheck_report_kaggle_PASS.json', c)
    with zipfile.ZipFile(w / 'focal-code.zip') as z:
        for name in z.namelist():
            if (w / name).read_bytes() != z.read(name):
                raise ValueError('Extracted scientific source differs: ' + name)
    if sha(w / 'baseline_evaluation32.json') != c['baseline_evaluation_sha256']:
        raise ValueError('Saved baseline changed')
    return c


def disk_check(path, required=12 * GIB):
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError(f'Need {required} free bytes at {path}; have {free}')
    return free


def verify_zip(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('ZIP SHA256 mismatch')
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('artifact_manifest.json'))
        names = [r['path'] for r in manifest['files']]
        if len(names) != len(set(names)) or set(z.namelist()) != set(names) | {'artifact_manifest.json'}:
            raise ValueError('ZIP members differ from manifest')
        for r in manifest['files']:
            h = hashlib.sha256()
            n = 0
            with z.open(r['path']) as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b); n += len(b)
            if h.hexdigest() != r['sha256'] or n != r['bytes']:
                raise ValueError('ZIP member differs: ' + r['path'])
    return digest


def is_complete(out):
    return all((Path(out)/name).is_file() for name in
               ('complete.json', 'selection32.json', 'paired_comparison.json',
                'training_complete.json', 'development32/evaluation.json'))


def package(out, label, training_only=False):
    """New ZIP, checked member-by-member, no scratch or dataset copies."""
    out = Path(out).resolve()
    paths = [p for p in out.rglob('*') if p.is_file()
             and not any(part.startswith('_ap_') for part in p.relative_to(out).parts)
             and (not training_only or p.parent == out)]
    paths.sort()
    if any(p.is_symlink() or not p.resolve().is_relative_to(out) for p in paths):
        raise ValueError('Only own regular run outputs may be packaged')
    needed = sum(p.stat().st_size for p in paths) + 64 * 1024**2
    disk_check(out.parent, needed)
    final = out.parent / (out.name + '-' + label + '-' + uuid.uuid4().hex[:8] + '.zip')
    temporary = final.with_suffix('.zip.partial')
    manifest = dict(stage=label, complete=is_complete(out),
                    config_sha256=CONFIG_SHA, code_sha256=CODE_SHA,
                    files=[dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size,
                                sha256=sha(p)) for p in paths])
    with temporary.open('xb') as raw:
        with zipfile.ZipFile(raw, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in paths:
                z.write(p, p.relative_to(out).as_posix())
            z.writestr('artifact_manifest.json', json.dumps(manifest, indent=2))
        raw.flush(); os.fsync(raw.fileno())
    digest = verify_zip(temporary)
    if final.exists():
        raise FileExistsError(final)
    temporary.rename(final)
    with final.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(digest + '  ' + final.name + '\n')
    print('VERIFIED DOWNLOAD:', final, digest, flush=True)
    return final


def final_checkpoint(out, c):
    out = Path(out)
    r = read(out / 'training_complete.json')
    if r['status'] != 'COMPLETE' or r['updates'] != 23404 or r['epochs'] != 4 or len(r['history']) != 4:
        raise ValueError('Incomplete fixed epoch4 training')
    name = r['checkpoint']
    if Path(name).name != name:
        raise ValueError('Invalid checkpoint path')
    p = out / name
    if sha(p) != r['checkpoint_sha256'] or sha(p) != read(p.with_suffix('.json'))['sha256']:
        raise ValueError('Final checkpoint SHA differs')
    wanted = dict(config_sha256=CONFIG_SHA, code_sha256=CODE_SHA,
                  experiment=c['experiment'], initial_state_sha256=c['initial_state_sha256'])
    if r['identity'] != wanted:
        raise ValueError('Final identity differs')
    return p, wanted


def gate_authorization(path, c):
    a = read(path)
    required = dict(training_authorized=True, runs=1, experiment=c['experiment'],
                    config_sha256=CONFIG_SHA, code_sha256=CODE_SHA,
                    logistics_sha256=sha(__file__), max_training_seconds=2700,
                    session_reserve_minutes=120, gpu_count=1)
    if any(a.get(k) != v for k, v in required.items()):
        raise ValueError('Explicit authorization/config/code/logistics/budget does not match')
    if not all(isinstance(a.get(k), str) and a[k].strip() for k in
               ('checkpoint_backup_destination', 'local_backup_destination', 'authorized_by')):
        raise ValueError('Explicit local/external destinations and authorizer required')
    return a


def preserve_inputs(out, workspace, corpus):
    sources = [(Path(workspace)/name, name) for name in
               ('focal_config.json', 'focal-code.zip', 'baseline_evaluation32.json',
                'precheck_report_kaggle_PASS.json', 'focal_run_logistics.py')]
    sources += [(Path(corpus)/'index'/name, name) for name in ('normalization.json', 'index.json')]
    for source, name in sources:
        target = Path(out)/name
        if target.exists():
            if sha(target) != sha(source): raise ValueError('Preserved input changed: '+name)
        else:
            with target.open('xb') as dst, source.open('rb') as src:
                shutil.copyfileobj(src, dst)
    receipt = Path(out)/'mount_receipt.json'
    if not receipt.exists(): write_new(receipt, read(Path(corpus)/'mount-verification.json'))


def train(args):
    c = identity(args.workspace)
    gate_authorization(args.authorization, c)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Exactly physical GPU0 must be exposed before torch import')
    import torch
    from geofirex import focal_experiment as exp
    out = Path(args.out).resolve()
    if out.parent != Path('/kaggle/working').resolve() or out.exists():
        raise ValueError('NEW direct /kaggle/working output required')
    disk_check(out.parent); disk_check('/tmp')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise ValueError('Exactly one visible Tesla T4 required')
    run_args = SimpleNamespace(config=Path(args.workspace)/'focal_config.json',
        code_archive=Path(args.workspace)/'focal-code.zip', baseline=Path(args.workspace)/'baseline_evaluation32.json',
        corpus=args.corpus, index=str(Path(args.corpus)/'index'), authorization=args.authorization,
        out=out, execute_training=True, resume=None, resume_sha256=None)
    exp.verify_authorization(run_args, c)
    # A persistent session-level marker: never remove to launch a second candidate.
    lock = out.parent / 'GEOFIREX_FOCAL_SINGLE_RUN.lock.json'
    write_new(lock, dict(out=str(out), config_sha256=CONFIG_SHA, code_sha256=CODE_SHA))
    original = exp.evaluate
    def preserve_then_evaluate(*a, **kw):
        final_checkpoint(out, c)
        preserve_inputs(out, args.workspace, args.corpus)
        write_new(out / 'training_history.json', read(out/'training_complete.json')['history'])
        package(out, 'TRAINING-SAFE', training_only=True)
        disk_check(out.parent, 6 * GIB)
        return original(*a, **kw)  # exact existing evaluation, unchanged
    exp.evaluate = preserve_then_evaluate
    try:
        result = exp.run(run_args)
        if (out/'complete.json').exists():
            comparison = read(out/'paired_comparison.json')
            write_new(out/'selection32.json', dict(threshold=comparison['candidate']['threshold'],
                checkpoint_sha256=read(out/'training_complete.json')['checkpoint_sha256'],
                events=c['events'], scope='development32 only',
                objective='micro Dice; ties precision then higher threshold',
                boundary_inconclusive=comparison['boundary_inconclusive']))
        return result
    finally:
        exp.evaluate = original
        if out.exists():
            # Runs on ordinary exceptions, pause and KeyboardInterrupt as well.
            preserve_inputs(out, args.workspace, args.corpus)
            package(out, 'COMPLETE' if is_complete(out) else 'PARTIAL')


def evaluate_only(args):
    """Recovery of the SAME final checkpoint; no optimizer or training runner."""
    c = identity(args.workspace)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('One GPU0 required')
    import torch
    from geofirex import focal_experiment as exp
    from geofirex.fast_compact import FastCompactTiles
    if torch.__version__ != c['torch_version'] or exp.np.__version__ != c['numpy_version']:
        raise ValueError('Historical environment required')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise ValueError('Exactly one Tesla T4 required')
    source = Path(args.training_out)
    if (source/'complete.json').exists() or (source/'paired_comparison.json').exists():
        raise ValueError('Already evaluated; no redundant evaluation')
    checkpoint, wanted = final_checkpoint(source, c)
    if read(Path(args.corpus)/'mount-verification.json').get('all_hashes_verified') is not True:
        raise ValueError('Current verified mounts required')
    exp.preflight(Path(args.workspace)/'focal_config.json', args.corpus, Path(args.corpus)/'index')
    out = Path(args.out).resolve()
    if out.parent != Path('/kaggle/working').resolve() or out.exists():
        raise ValueError('NEW recovery output required')
    disk_check(out.parent)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (state['identity'] != wanted or state['updates'] != 23404 or state['epoch'] != 5
            or state['cursor'] != 0 or state['config'] != c):
        raise ValueError('Only exact final epoch4 state accepted')
    out.mkdir(exist_ok=False)
    ds = None
    try:
        for p in source.iterdir():
            if p.is_file() and (p.suffix in ('.json', '.jsonl', '.py', '.zip') or p == checkpoint):
                if p.name not in ('complete.json', 'paired_comparison.json', 'selection32.json'):
                    if p.name.startswith('checkpoint_') and p != checkpoint.with_suffix('.json') and p != checkpoint:
                        continue
                    with (out/p.name).open('xb') as dst, p.open('rb') as src:
                        shutil.copyfileobj(src, dst)
        network = exp.make_model('B2', 8, 3)
        network.load_state_dict(state['model']); network.to('cuda'); network.eval()
        ds = FastCompactTiles(args.corpus, Path(args.corpus)/'index', 'val')
        result = exp.evaluate(network, ds, c['events'], c['thresholds'],
            SimpleNamespace(batch_size=32, workers=4, device='cuda'), out/'development32', 'Focal recovery32', set())
        comparison = exp.compare_saved(result, read(Path(args.workspace)/'baseline_evaluation32.json'), c)
        write_new(out/'paired_comparison.json', comparison)
        write_new(out/'selection32.json', dict(threshold=comparison['candidate']['threshold'],
            checkpoint_sha256=sha(checkpoint), events=c['events'], scope='development32 only'))
        write_new(out/'complete.json', dict(status='COMPLETE', training_executed=False,
            checkpoint_sha256=sha(checkpoint), acceptance=comparison['acceptance']))
    finally:
        if ds is not None: ds.close()
        package(out, 'EVALUATION-RECOVERY')


def backup(archive, sidecar, local_root, external_root=None):
    """Read source; copy exclusively into NEW destination folders, hash both."""
    archive, sidecar = Path(archive).resolve(), Path(sidecar).resolve()
    expected = sidecar.read_text(encoding='ascii').split()[0]
    verify_zip(archive, expected)
    destinations = [Path(local_root).resolve()]
    if external_root is not None:
        destinations.append(Path(external_root).resolve())
    if len(set(destinations)) != len(destinations):
        raise ValueError('Backups must use distinct roots')
    receipts = []
    for root in destinations:
        if not root.is_dir():
            raise ValueError('Existing explicitly selected backup root required: ' + str(root))
        disk_check(root, archive.stat().st_size + 64*1024**2)
        dest = root / ('focal-preserved-' + uuid.uuid4().hex)
        dest.mkdir(exist_ok=False)
        copied = dest/archive.name
        with copied.open('xb') as dst, archive.open('rb') as src:
            shutil.copyfileobj(src, dst)
            dst.flush(); os.fsync(dst.fileno())
        verify_zip(copied, expected)
        with (dest/sidecar.name).open('xb') as dst, sidecar.open('rb') as src:
            shutil.copyfileobj(src, dst)
        write_new(dest/'backup_verified.json', dict(source=str(archive), sha256=expected,
                  bytes=copied.stat().st_size, archive_and_all_members_verified=True))
        receipts.append(str(copied))
    return receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('train', 'evaluate-only'):
        q = sub.add_parser(name)
        for key in ('workspace', 'corpus', 'out'): q.add_argument('--'+key, required=True)
        q.add_argument('--authorization' if name == 'train' else '--training-out', required=True)
    q = sub.add_parser('package'); q.add_argument('--out', required=True)
    q = sub.add_parser('backup')
    for key in ('archive', 'sidecar', 'local-root'): q.add_argument('--'+key, required=True)
    q.add_argument('--external-root')
    a = p.parse_args()
    if a.action == 'train': print(train(a))
    elif a.action == 'evaluate-only': evaluate_only(a)
    elif a.action == 'package': package(a.out, 'MANUAL-RESCUE')
    else: print(backup(a.archive, a.sidecar, a.local_root, a.external_root))


if __name__ == '__main__': main()
