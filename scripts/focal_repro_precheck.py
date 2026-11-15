"""Reproducibility diagnostics only: no fit, forward, optimizer or checkpoint load.

Used by a thin companion to the existing frozen Focal notebook. Corpus reads are
restricted to pinned compressed bytes and metadata indices, never NPZ pixels.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

from geofirex.benchmark_models import make_model


def model_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def byte_hash(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def rng_hash():
    return byte_hash(bytes(torch.get_rng_state().tolist()))


def environment():
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  byteorder=sys.byteorder, torch=torch.__version__, numpy=np.__version__,
                  cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  default_dtype=str(torch.get_default_dtype()),
                  default_device=str(torch.get_default_device()), threads=torch.get_num_threads(),
                  cuda_available=False, gpu_count=0, gpus=[], gpu_error=None,
                  torch_build=torch.__config__.show())
    try:
        result['cuda_available'] = torch.cuda.is_available()
        if result['cuda_available']:
            result['gpu_count'] = torch.cuda.device_count()
            for i in range(result['gpu_count']):
                prop = torch.cuda.get_device_properties(i)
                result['gpus'].append(dict(index=i, name=prop.name, memory_bytes=prop.total_memory,
                                           capability=[prop.major, prop.minor]))
    except Exception as exc:
        result['gpu_error'] = type(exc).__name__ + ': ' + str(exc)
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10, check=False)
        result['nvidia_smi'] = r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        result['nvidia_smi'] = 'unavailable; no installation attempted'
    return result


def initialization(config, env, *, cuda_transfer=True):
    """Replay exact seeds, model factory and historical hash, without a forward.

    Trace wrappers observe RNG states without changing any initializer. All
    CPU/Python/NumPy RNG states, thread and cuDNN settings are restored on exit.
    CUDA seeding/context can change in this dedicated diagnostic kernel.
    """
    if env['default_dtype'] != 'torch.float32' or env['default_device'] != 'cpu':
        return dict(status='BLOCKED', causes=['ambient default dtype/device differs; not silently overridden'])
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    conv_reset = torch.nn.Conv2d.reset_parameters
    norm_reset = torch.nn.GroupNorm.reset_parameters
    trace = []
    def traced(original, kind):
        def wrapped(module):
            before = rng_hash(); original(module); after = rng_hash()
            trace.append(dict(kind=kind, shape=list(module.weight.shape),
                              before=before, after=after, rng_consumed=before != after))
        return wrapped
    def seed(numpy_seed=42):
        random.seed(42); np.random.seed(numpy_seed); torch.manual_seed(42); torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    try:
        seed()
        before = rng_hash()
        np_before = np.random.get_state()
        torch.nn.Conv2d.reset_parameters = traced(conv_reset, 'Conv2d')
        torch.nn.GroupNorm.reset_parameters = traced(norm_reset, 'GroupNorm')
        try:
            model = make_model('B2', 8, 3)
        finally:
            torch.nn.Conv2d.reset_parameters = conv_reset
            torch.nn.GroupNorm.reset_parameters = norm_reset
        after = rng_hash()
        np_after = np.random.get_state()
        actual = model_hash(model)
        hash_does_not_consume_rng = rng_hash() == after
        raw = hashlib.sha256(); tensors = []
        for name, tensor in sorted(model.state_dict().items()):
            # Independent of NumPy's .tobytes implementation/conversion.
            data = bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())
            raw.update(name.encode()); raw.update(data)
            tensors.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                device=str(tensor.device), contiguous=tensor.is_contiguous(),
                                bytes=len(data), sha256=byte_hash(data)))
        seed(); repeat = model_hash(make_model('B2', 8, 3))
        seed(numpy_seed=123); numpy_control = model_hash(make_model('B2', 8, 3))
        seed(); torch.rand(1); intervening_draw = model_hash(make_model('B2', 8, 3))
        # Test the old Conv2d reset semantics with CURRENT backend, not torch2.11.
        def legacy_reset(module):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in:
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(module.bias, -bound, bound)
        seed(); torch.nn.Conv2d.reset_parameters = legacy_reset
        try: old_reset_control = model_hash(make_model('B2', 8, 3))
        finally: torch.nn.Conv2d.reset_parameters = conv_reset
        transfer_hash = None; transfer_error = None
        if cuda_transfer and env['cuda_available']:
            try:
                model.to('cuda:0')  # parameter copy only, no data/forward
                transfer_hash = model_hash(model)  # same historical hash boundary
                model.to('cpu')
            except Exception as exc:
                transfer_error = type(exc).__name__ + ': ' + str(exc)
        expected = config['initial_state_sha256']
        causes = []
        if actual != expected: causes.append('fresh CPU initialization hash differs from historical')
        if repeat != actual: causes.append('local replay is not deterministic')
        if raw.hexdigest() != actual: causes.append('NumPy and independent byte serialization disagree')
        if transfer_hash != expected: causes.append('historical post-CUDA-transfer boundary not reproduced')
        return dict(status='PASS' if not causes else 'BLOCKED', causes=causes,
                    expected_sha256=expected, cpu_sha256=actual, post_cuda_sha256=transfer_hash,
                    transfer_error=transfer_error, independent_bytes_sha256=raw.hexdigest(),
                    repeat_sha256=repeat, numpy_seed_control_sha256=numpy_control,
                    one_torch_draw_control_sha256=intervening_draw,
                    legacy_reset_current_backend_sha256=old_reset_control,
                    numpy_rng_unchanged_by_model=(np.array_equal(np_before[1], np_after[1])
                                                and np_before[2:] == np_after[2:]),
                    torch_rng_before=before, torch_rng_after=after,
                    hash_does_not_consume_rng=hash_does_not_consume_rng,
                    initializer_trace=trace, tensors=tensors,
                    parameters=sum(p.numel() for p in model.parameters()),
                    head_bias=float(model.network.head.bias.detach().cpu().item()),
                    forward_calls=0, optimizer_steps=0, checkpoint_loads=0)
    finally:
        torch.nn.Conv2d.reset_parameters = conv_reset
        torch.nn.GroupNorm.reset_parameters = norm_reset
        random.setstate(python_state); np.random.set_state(numpy_state); torch.set_rng_state(torch_state)
        torch.set_num_threads(threads)
        torch.backends.cudnn.benchmark = benchmark; torch.backends.cudnn.deterministic = deterministic


def verify_corpus(maps, config, sampler_class, *, all_bytes=True):
    """Validate existing mounts IN PLACE. No links, extraction or pixel read."""
    expected = {p['identity'] for p in config['dataset_partitions']}
    selected = []
    for path in maps:
        path = Path(path); mapping = json.loads(path.read_text(encoding='utf-8'))
        if mapping.get('identity') in expected: selected.append((path, mapping))
    if len(selected) != 11 or {m['identity'] for _, m in selected} != expected:
        raise ValueError('Need exactly the eleven EXISTING compact mounts, without duplicates')
    objects = {}; total = 0; parts = []
    # Validate ALL identities and year/path restrictions before reading payloads.
    for path, mapping in selected:
        canonical = json.dumps([{k:f[k] for k in ('path', 'alias', 'bytes', 'sha256')}
                                for f in mapping['files']], sort_keys=True, separators=(',', ':')).encode()
        if byte_hash(canonical) != mapping['identity']: raise ValueError('Partition identity differs')
        for f in mapping['files']:
            logical = Path(f['path']); alias = Path(f['alias'])
            if (logical.is_absolute() or '..' in logical.parts or '2021' in logical.parts
                    or alias.name != f['alias'] or '\\' in f['alias'] or f['path'] in objects):
                raise ValueError('Unsafe/duplicate/2021 compact object')
            if logical.suffix == '.zip' and not any('/'+y+'/' in '/'+logical.as_posix()
                                                   for y in ('2018', '2019', '2020')):
                raise ValueError('Archive outside permitted historical years')
            objects[f['path']] = (path.parent / alias, f)
        parts.append(mapping['identity'])
    if len(objects) != 459: raise ValueError('Expected 459 compact objects')
    for source, f in objects.values():
        if source.stat().st_size != f['bytes']: raise ValueError('Size differs: ' + f['path'])
        # Even diagnostic fast mode hashes all metadata used, never decodes ZIPs.
        if (all_bytes or not f['path'].endswith('.zip')) and sha(source) != f['sha256']:
            raise ValueError('SHA256 differs: ' + f['path'])
        total += f['bytes']
    index_path = objects['index/index.json'][0]
    if sha(index_path) != config['index_sha256']: raise ValueError('Index not the frozen index')
    if sha(objects['normalization.json'][0]) != config['normalization_sha256']:
        raise ValueError('Normalizer not the frozen normalizer')
    meta = json.loads(index_path.read_text(encoding='utf-8'))
    archives = meta['archives']
    if len(archives) != 451 or set(archives) != {p for p in objects if p.endswith('.zip')}:
        raise ValueError('Archive inventory differs')
    # These NPY files are small sample-coordinate indices, NOT raster arrays.
    train = np.load(objects['index/train.npy'][0], mmap_mode='r', allow_pickle=False)
    val = np.load(objects['index/val.npy'][0], mmap_mode='r', allow_pickle=False)
    counts = []
    for role, records, years in [('train', train, ('2018', '2019')), ('val', val, ('2020',))]:
        if len(records) != (93603 if role == 'train' else 78771): raise ValueError('Tile count differs')
        if np.any(records['archive'] >= len(archives)): raise ValueError('Bad archive coordinate')
        ids = np.array([meta['events'][archives[int(a)]] for a in records['archive']])
        if any(e[:4] not in years for e in ids): raise ValueError('Annual split differs')
        if len(set(ids)) != (250 if role == 'train' else 201): raise ValueError('Event count differs')
        if role == 'val':
            if (not set(config['events']) <= set(ids)
                    or int(np.isin(ids, config['events']).sum()) != 13759):
                raise ValueError('Development32 differs')
        counts.append(dict(role=role, tiles=len(records), events=len(set(ids))))
    class MetadataOnly:
        records = train
        def __len__(self): return len(self.records)
        def __getitem__(self, _): raise AssertionError('Pixel access forbidden')
    sampler = sampler_class(MetadataOnly(), seed=42)
    orders = []
    for epoch in range(4):
        sampler.epoch = epoch; order = sampler.order()
        if not np.array_equal(np.sort(order), np.arange(93603)): raise ValueError('Sampler omits/duplicates tiles')
        orders.append(byte_hash(order.tobytes()))
    matches = orders == config['sampler_order_sha256']
    return dict(status='PASS' if all_bytes and matches else 'BLOCKED',
                causes=([] if all_bytes else ['full compressed-byte SHA verification disabled'])
                       + ([] if matches else ['sampler order differs from frozen configuration']),
                partitions=sorted(parts), objects=len(objects), bytes=total, coverage=counts,
                all_bytes_verified=all_bytes, sampler_order_sha256=orders,
                sampler_order_dtype=str(order.dtype), real_raster_or_npz_arrays_decoded=0,
                new_mounts_created=0, datasets_downloaded=0, partitions_published=0)


def decision(config, env, initial, corpus):
    causes = list(initial.get('causes', [])) + list(corpus.get('causes', []))
    if env['torch'] != config['torch_version']: causes.append('PyTorch version differs')
    if env['numpy'] != config['numpy_version']: causes.append('NumPy version differs')
    if not env['cuda_available']: causes.append('CUDA GPU unavailable')
    if env.get('gpu_error'): causes.append('GPU metadata query failed: ' + env['gpu_error'])
    if initial.get('status') != 'PASS': causes.append('initialization prerequisite not PASS')
    if corpus.get('status') != 'PASS': causes.append('corpus/sampler prerequisite not PASS')
    return dict(status='PASS' if not causes else 'BLOCKED', causes=causes,
                scope='technical prerequisites ONLY; never permission or evidence of a trained improvement',
                training_authorized=False, RUN_TRAINING=False, authorization_generated=False,
                optimizer_steps=0, forward_calls=0, test2021_pixels_opened=0,
                environment=env, initialization=initial, corpus=corpus)
