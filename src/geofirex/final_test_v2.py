"""Independent, inference-only C8 V2 final evaluation.

No import-time I/O. Metadata commands never open a raster, including headers.
Historical development readers and their 2021 guards are deliberately untouched.
The real reader requires a consumed, durable, one-use authorization. This module
contains no optimizer, training loop, normalizer fitting or threshold search.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta, datetime, timezone
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile
import re
from contextlib import contextmanager

import numpy as np

INDEX = 'reports/protocol_v2_20261118/index'
EVENTS = 'reports/protocol_v2_20261118/events.csv'
OBSERVATIONS = 'reports/full_corpus_v1/prepared_v1/observations.parquet'
ARCHIVE_SHA = '37a8d69647ddebaa902d88e21fddf4df3f56eaa57e7094695b478d04116c1415'
CHECKPOINT_SHA = 'c2273fea252d709f4d316eba76515d669be97eac8bfa9302ff4831da2c9aedc1'
CHECKPOINT = 'checkpoint_u0028976.pt'
NORM_SHA = 'bf8b5adb6276126540786351e66076d35c2e82be09b21e14bea96a4c678b2f27'
EVENTS_SHA = '3245228c81ed52b807bbea27fe683a0a5c5aaae30393287f97d85030d693c38f'
OLD_MANIFEST_SHA = 'c150333d74baa454a8ac48763fc1665a8dc7bd853f768264a527f8b917c45989'
PIXEL = np.dtype([('p', '<f4'), ('y', 'u1'), ('v', 'u1'), ('o', 'u1')])
GIB = 1024**3
NEW_FILES = ('src/geofirex/final_test_v2.py', 'scripts/run_final_test_v2.py',
             'tests/test_final_test_v2.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    """Exclusive, flushed creation; never truncate scientific evidence."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def copy_exclusive(source,destination):
    with Path(source).open('rb') as src,Path(destination).open('xb') as dst:
        shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())


def safe_source(root, token):
    p = PurePosixPath(token)
    if p.is_absolute() or '..' in p.parts or '\\' in token or ':' in token:
        raise ValueError('Unsafe source path')
    root = Path(root).resolve(); result = (root / token).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Source escaped mount')
    return result


def metadata_rows(root):
    """Explicit whitelist: no label statistics or array-valued raster columns."""
    import pyarrow.parquet as pq
    columns = ['event_id', 'date', 'relative_path', 'height', 'width', 'bytes',
               'status', 'crs', 'transform', 'sha256', 'sidecars_json']
    rows = pq.read_table(Path(root)/OBSERVATIONS, columns=columns).to_pylist()
    for row in rows:
        row['date'] = str(row['date'])[:10]
    return rows


def inventory(rows, events, rasters=None):
    """Stat-only inventory; dates refer to the TARGET, inputs are days -3,-2,-1."""
    events = set(events); selected = sorted((r for r in rows if r['event_id'] in events),
                                           key=lambda r: (r['event_id'], r['date']))
    if {r['event_id'] for r in selected} != events:
        raise ValueError('Missing event metadata')
    seen = set(); paths = set(); geometry = {}; days = defaultdict(set)
    missing = size_errors = 0; byte_count = 0
    for r in selected:
        e, day, token = r['event_id'], r['date'], r['relative_path']
        key = e, day
        if (key in seen or token in paths or token != f'{e}/{day}.tif'
                or r['status'] != 'ok' or len(r['sha256']) != 64
                or min(r['height'], r['width']) <= 0 or not r['crs']):
            raise ValueError('Invalid/duplicate observation metadata')
        date.fromisoformat(day)
        if day[:4] != e[:4]:
            raise ValueError('Event/calendar year differs')
        g = [r['height'], r['width'], r['crs'], r['transform']]
        if geometry.setdefault(e, g) != g:
            raise ValueError('Metadata geometry changes inside event')
        # Sidecar identities are pinned as metadata too; no sidecar bytes read here.
        parse_sidecars(r)
        seen.add(key); paths.add(token); days[e].add(day); byte_count += r['bytes']
        if rasters is not None:
            path = safe_source(rasters, token)
            if not path.is_file(): missing += 1
            elif path.stat().st_size != r['bytes']: size_errors += 1
    keys = []
    for e in sorted(days):
        h, w = geometry[e][:2]
        for day in sorted(days[e]):
            d = date.fromisoformat(day)
            if all((d-timedelta(days=k)).isoformat() in days[e] for k in range(6)):
                keys.extend((e, day, r, c) for r in range(0, h, 64) for c in range(0, w, 64))
    per_event = Counter(k[0] for k in keys)
    pixels = len(keys)*64*64
    max_event_pixels = max(per_event.values(), default=0)*64*64
    # Two global and two current-event exact sorters, worst-case all unique.
    # Each sorter reserves input + merge output, 24 bytes/group.
    scratch = 4*24*(pixels+max_event_pixels)
    result = dict(events=len(events), observations=len(selected),
                  windows=len({k[:2] for k in keys}), geometric_tiles=len(keys),
                  upper_padded_pixels=pixels, max_event_padded_pixels=max_event_pixels,
                  raw_tiff_bytes=byte_count, exact_sort_scratch_upper_bytes=scratch,
                  prediction_upper_bytes=pixels*PIXEL.itemsize,
                  missing_paths=missing, size_errors=size_errors,
                  raster_headers_opened=0, raster_pixels_opened=0,
                  expected_tiff_sha256_recomputed=False,
                  format_check='historical certified metadata + .tif suffix; current headers deferred')
    return selected, keys, result


def parse_sidecars(row):
    sidecars = json.loads(row.get('sidecars_json') or '{}')
    if not isinstance(sidecars, dict):
        raise ValueError('Unsupported sidecar metadata schema')
    for name, value in sidecars.items():
        if PurePosixPath(name).name != name or '\\' in name or ':' in name:
            raise ValueError('Unsafe sidecar name')
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError('Invalid sidecar hash')
    return sidecars


def selected_checkpoint(archive, normalizer):
    """Only the project's hash-verified trusted checkpoint is deserialized."""
    import torch
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError('C8 COMPLETE archive SHA-256 differs')
    with zipfile.ZipFile(archive) as z:
        raw = z.read(CHECKPOINT)
        if hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA:
            raise ValueError('C8 checkpoint SHA-256 differs')
        complete = json.loads(z.read('training_complete.json'))
        if complete['epochs'] != 4 or complete['updates'] != 28976 or complete['status'] != 'COMPLETE':
            raise ValueError('C8 completion evidence differs')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if (state['epoch'], state['cursor'], state['updates']) != (4, 0, 28976):
        raise ValueError('Not the fixed epoch-4 checkpoint')
    if state['normalizer'] != normalizer:
        raise ValueError('Checkpoint normalizer differs from frozen train-only normalizer')
    return state


def verify_historical(root):
    root = Path(root)
    manifest = root/'reports/protocol_v2_execution/execution_manifest.json'
    if sha(manifest) != OLD_MANIFEST_SHA:
        raise ValueError('Historical execution manifest changed')
    for name, expected in load(manifest)['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Historical identity mismatch: '+name)
    science = load(root/'reports/jury_20261202_v1/science_manifest.json')
    for name, expected in science['files'].items():
        if sha(root/'reports/jury_20261202_v1'/name) != expected:
            raise ValueError('Saved selection evidence changed: '+name)
    if sha(root/EVENTS) != EVENTS_SHA or sha(root/INDEX/'normalization.json') != NORM_SHA:
        raise ValueError('Frozen event/normalizer identity differs')


def selection_evidence(root):
    """Recheck the seven numerical gates; one failed gate is sufficient NO-GO.

    Runtime is the eighth conjunct, never a reason to override a failed numerical
    gate. AP_new is reported but is not silently added to the original gates.
    """
    result=load(Path(root)/'reports/jury_20261202_v1/comparison.json')
    if result['threshold'] != .78 or len(result['models']) != 4:
        raise ValueError('Selection panel differs')
    rows={r['model']:r for r in result['models']}; b=rows['C8']
    policy=load(Path(root)/'configs/protocol_v2/evaluation.json')['acceptance_all_required']
    checks={}
    for name in ('NEWPOS2','NEWPOS4','FIRE13'):
        c=rows[name]
        if (c['threshold'],c['events'],c['tiles']) != (.78,80,30882): raise ValueError('Incomparable candidate panel')
        gates=dict(AP_pooled=c['AP_global']-b['AP_global']>=policy['delta_AP_min'],
                   Dice_pooled=c['dice_global']-b['dice_global']>=policy['delta_Dice_micro_min'],
                   recall_new=c['recall_new']-b['recall_new']>=policy['delta_new_fire_recall_min'],
                   Dice_new=c['dice_new']-b['dice_new']>=policy['delta_new_fire_Dice_min'],
                   FP_new=c['FP_new']<=b['FP_new']*policy['new_fire_FP_ratio_max'],
                   macro_Dice=c['macro_event_dice']-b['macro_event_dice']>=policy['delta_Dice_macro_event_min'],
                   Brier=c['Brier']-b['Brier']<=policy['delta_Brier_max'])
        if all(gates.values()): raise ValueError('Candidate numerical gates all pass; selection needs human review')
        checks[name]=dict(numerical_gates=gates,runtime_gate='not needed to establish NO_GO',all_required_pass=False)
    return dict(reference=b,candidates=checks,validation_domain_sha256=result['evaluation_domain_sha256'])


def freeze(root, archive, rasters, out):
    """Verify everything first, then create a distinct, exclusive freeze directory."""
    import csv
    from geofirex.unet_data import Normalizer
    root, out = Path(root).resolve(), Path(out)
    if out.exists(): raise FileExistsError(out)
    verify_historical(root)
    evidence=selection_evidence(root)
    cfg = load(root/'configs/protocol_v2/baseline_c8.json')
    norm = load(root/INDEX/'normalization.json')
    if (cfg['threshold'] != .78 or cfg['fixed_checkpoint_epoch'] != 4
            or cfg['channel_names'] != Normalizer(**norm).channel_names
            or cfg['context_days'] != 3 or cfg['tile_size'] != 64):
        raise ValueError('C8 configuration differs')
    with (root/EVENTS).open(encoding='utf-8', newline='') as stream:
        assignments = list(csv.DictReader(stream))
    roles = {r['event_id']:r['split'] for r in assignments}
    if len(roles) != 607 or Counter(roles.values()) != dict(train=318, val=80, test=156, purged=53):
        raise ValueError('V2 split differs')
    all_rows = metadata_rows(root)
    known = {r['relative_path'] for r in all_rows}
    for p in norm['fit_paths']:
        token = p.replace('\\', '/').split('/rasters/')[-1]
        e = '/'.join(token.split('/')[:2])
        if token not in known or roles.get(e) != 'train' or e[:4] not in ('2018', '2019'):
            raise ValueError('Normalizer is not train-only')
    state = selected_checkpoint(archive, norm)
    if state['config']['channel_names'] != cfg['channel_names']:
        raise ValueError('Checkpoint channel order differs')
    observations, keys, stats = inventory(all_rows, [e for e,r in roles.items() if r == 'test'], rasters)
    if (stats['events'], stats['observations'], stats['windows'], stats['geometric_tiles']) != (156, 4117, 3337, 75540):
        raise ValueError('Independent test identity/counts differ')
    if stats['missing_paths'] or stats['size_errors']:
        raise ValueError('Local test TIFF stat inventory incomplete')
    sources = {p:sha(root/p) for p in NEW_FILES}
    sources.update({p:sha(root/p) for p in load(root/'reports/protocol_v2_execution/execution_manifest.json')['files']})
    # Include current metric dependencies and selection evidence, not only the historical list.
    for p in ('src/geofirex/unet_data.py', 'src/geofirex/runner_data.py',
              'src/geofirex/mission_streaming.py', 'src/geofirex/v2_propagation.py',
              'reports/jury_20261202_v1/science_manifest.json',
              'reports/jury_20261202_v1/comparison.csv', OBSERVATIONS):
        sources[p] = sha(root/p)
    choice = dict(schema='geofirex-final-test-v2-freeze-1', selected='B2_C8_V2_seed42',
                  decision='B; no AUXNEW training; user selection confirmed before test',
                  selection_basis='C8 fallback; no candidate satisfied all frozen improvement gates',
                  selection_scope='already consulted validation; additional analyses exploratory',
                  selection_evidence=evidence,
                  checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
                  archive_sha256=ARCHIVE_SHA, epoch=4, updates=28976, threshold=.78,
                  channel_names=cfg['channel_names'], context_days=3, tile_size=64,
                  normalizer_sha256=NORM_SHA, normalization='unchanged 2018/19 subset of V2 train; never refit',
                  fit_paths=len(norm['fit_paths']), events_sha256=EVENTS_SHA,
                  observations_sha256=digest(observations), keys_sha256=digest(keys),
                  test_identity=digest(dict(events=sorted({k[0] for k in keys}), observations=observations)),
                  inventory=stats, files=sources, test_pixels_opened=0,
                  evaluation_authorized=False, training_authorized=False,
                  ledger_policy='one fixed durable shared ledger; no ephemeral Kaggle working/tmp ledger',
                  numerical_policy='float32; no AMP; validation parity required in actual runtime',
                  duration_budget_seconds=14400,
                  limitations=['one seed', 'weak new-fire recall', 'local prototype; not operational deployment'])
    out.mkdir(parents=True, exist_ok=False)
    save(out/'observations.json', observations)
    save(out/'selection.json', choice)
    save(out/'selection.sha256.json', {'selection.json':sha(out/'selection.json'),
                                      'observations.json':sha(out/'observations.json')})
    template = dict(schema='geofirex-final-test-v2-authorization-1', authorized=False,
                    purpose='one independent 2021 evaluation only; never training',
                    freeze_sha256=sha(out/'selection.json'), test_identity=choice['test_identity'],
                    authorization_id=None, authorized_by=None, authorized_at_utc=None,
                    preflight_sha256=None, qualification_sha256=None,
                    ledger_root=None, durable_shared_ledger_confirmed=False,
                    output=None, scratch_root=None, raster_root=None, archive=None,
                    device=None, session_seconds_remaining=None, quota_minutes_remaining=None,
                    quota_evidence=None, runtime_mounts_verified=False,
                    training_authorized=False,
                    max_seconds=14400, post_failure='stop; preserve partial results and ledger; no automatic rerun')
    save(out/'authorization_TEMPLATE_DISABLED.json', template)
    return choice


def verify_freeze(root, frozen):
    root, frozen = Path(root), Path(frozen)
    seals = load(frozen/'selection.sha256.json')
    for p, expected in seals.items():
        if sha(frozen/p) != expected: raise ValueError('Freeze bytes changed: '+p)
    f = load(frozen/'selection.json'); rows = load(frozen/'observations.json')
    if digest(rows) != f['observations_sha256']: raise ValueError('Frozen observations differ')
    for p, expected in f['files'].items():
        if sha(root/p) != expected: raise ValueError('Frozen source changed: '+p)
    _, keys, _ = inventory(rows, {r['event_id'] for r in rows})
    if digest(keys) != f['keys_sha256']: raise ValueError('Frozen exhaustive grid differs')
    return f, rows, keys


def memory_info():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (k, ctypes.c_ulonglong) for k in ('total', 'available', 'page', 'availpage', 'virtual', 'availvirtual', 'extended')]
        m = Memory(); m.length = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): raise ctypes.WinError()
        return dict(total_bytes=m.total, available_bytes=m.available)
    lines = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {k:int(lines[v].strip().split()[0])*1024 for k,v in [('total_bytes','MemTotal'),('available_bytes','MemAvailable')]}


def resource_plan(stats):
    # Keep 20% + 2 GiB beyond the sort upper bound; output + verified archive
    # coexist. The read-only 13.3 GiB TIFF mount is not included in writable disk.
    scratch = int(stats['exact_sort_scratch_upper_bytes']*1.2)+2*GIB
    output = 2*stats['prediction_upper_bytes']+2*GIB
    return dict(scratch_required_bytes=scratch, output_required_bytes=output,
                same_filesystem_required_bytes=scratch+output,
                ram_total_min_bytes=16_000_000_000, ram_available_min_bytes=8*GIB,
                readonly_tiff_bytes=stats['raw_tiff_bytes'],
                session_seconds=14400, quota_minutes=240,
                timing='4h conservative reservation, not measured 2021 throughput')


def preflight(root, frozen, archive, rasters, work, scratch, qualification=None):
    """CPU/stat-only even on Kaggle; no CUDA initialization or TIFF opening."""
    f, rows, _ = verify_freeze(root, frozen)
    _, _, stats = inventory(rows, {r['event_id'] for r in rows}, rasters)
    plan = resource_plan(stats); mem = memory_info()
    work, scratch = Path(work).resolve(), Path(scratch).resolve()
    if not work.is_dir() or not scratch.is_dir(): raise ValueError('Existing work/scratch parent directories required')
    same = os.stat(work).st_dev == os.stat(scratch).st_dev
    free_work, free_scratch = shutil.disk_usage(work).free, shutil.disk_usage(scratch).free
    disk_ok = (free_work >= plan['same_filesystem_required_bytes'] if same else
               free_work >= plan['output_required_bytes'] and free_scratch >= plan['scratch_required_bytes'])
    import importlib.metadata as md
    runtime = {p:md.version(p) for p in ('torch','numpy','rasterio','pyarrow')}
    q = load(qualification) if qualification else {}
    parity_ok = (q.get('status') == 'PASS' and q.get('freeze_sha256') == sha(Path(frozen)/'selection.json')
                 and q.get('runtime') == runtime and q.get('validation_parity', {}).get('passed') is True)
    checks = dict(frozen_identities='PASS', tiff_paths_sizes='PASS' if not stats['missing_paths'] and not stats['size_errors'] else 'FAIL',
                  c8_archive='PASS' if sha(archive) == ARCHIVE_SHA else 'FAIL',
                  ram='PASS' if mem['total_bytes'] >= plan['ram_total_min_bytes'] and mem['available_bytes'] >= plan['ram_available_min_bytes'] else 'FAIL',
                  disk='PASS' if disk_ok else 'FAIL', runtime_validation_parity='PASS' if parity_ok else 'BLOCKED',
                  independent_authorization='BLOCKED', durable_shared_lock='BLOCKED',
                  current_session_quota='BLOCKED', current_gpu='BLOCKED')
    return dict(schema='geofirex-final-test-v2-preflight-1', checks=checks,
                freeze_sha256=sha(Path(frozen)/'selection.json'), memory=mem, plan=plan, runtime=runtime,
                work=str(work), scratch=str(scratch), raster_root=str(Path(rasters).resolve()),
                archive=str(Path(archive).resolve()), same_filesystem=same,
                work_free_bytes=free_work, scratch_free_bytes=free_scratch, inventory=stats,
                qualification_sha256=sha(qualification) if qualification else None,
                created_utc=datetime.now(timezone.utc).isoformat(), test_pixels_opened=0)


@dataclass(frozen=True)
class PixelPermit:
    """Created by consume_authorization, never by metadata or qualification commands."""
    test_identity: str
    ledger_file: str
    nonce: str


def ephemeral_ledger(path):
    return any(p in str(path).lower().replace('\\','/') for p in ('/kaggle/working','/tmp/','/temp/'))


@contextmanager
def execution_deadline(seconds):
    """Also interrupt exact AP merges, reserving time for the safety archive."""
    if os.name != 'nt':
        from geofirex.v2_execution import evaluation_deadline
        with evaluation_deadline(seconds): yield
    else:
        import threading
        import _thread
        timer=threading.Timer(seconds,_thread.interrupt_main);timer.daemon=True;timer.start()
        try: yield
        finally: timer.cancel()


def consume_authorization(authorization, frozen, preflight_path, qualification, output, scratch, rasters, archive, device):
    a = load(authorization); f = load(Path(frozen)/'selection.json')
    if a.get('authorized') is not True:
        raise PermissionError('Independent 2021 authorization absent/disabled')
    expected = dict(freeze_sha256=sha(Path(frozen)/'selection.json'), test_identity=f['test_identity'],
                    preflight_sha256=sha(preflight_path), qualification_sha256=sha(qualification),
                    output=str(Path(output).resolve()), scratch_root=str(Path(scratch).resolve()),
                    raster_root=str(Path(rasters).resolve()), archive=str(Path(archive).resolve()), device=device)
    if any(a.get(k) != v for k,v in expected.items()): raise PermissionError('Authorization bindings differ')
    if (a.get('schema') != 'geofirex-final-test-v2-authorization-1'
            or not all(a.get(k) for k in ('authorization_id','authorized_by','authorized_at_utc'))
            or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}',a.get('authorization_id',''))
            or a.get('training_authorized') is not False
            or a.get('durable_shared_ledger_confirmed') is not True
            or a.get('runtime_mounts_verified') is not True
            or a.get('max_seconds') != 14400 or a.get('session_seconds_remaining',0) < 14400
            or (device.startswith('cuda') and (not a.get('quota_evidence') or a.get('quota_minutes_remaining',0) < 240))):
        raise PermissionError('Incomplete independent authorization/resource attestations')
    p = load(preflight_path); q = load(qualification)
    needed = ('frozen_identities','tiff_paths_sizes','c8_archive','ram','disk','runtime_validation_parity')
    freeze_sha=sha(Path(frozen)/'selection.json')
    if (any(p['checks'].get(k) != 'PASS' for k in needed) or q.get('device') != device
            or q.get('status')!='PASS' or q.get('freeze_sha256')!=freeze_sha
            or p.get('freeze_sha256')!=freeze_sha or p.get('runtime')!=q.get('runtime')):
        raise PermissionError('Technical preconditions not qualified on requested device')
    stamp=datetime.fromisoformat(p['created_utc'].replace('Z','+00:00'))
    age=(datetime.now(timezone.utc)-stamp).total_seconds()
    if not 0<=age<=1800: raise PermissionError('Preflight older than 30 minutes or future-dated')
    if Path(output).exists(): raise FileExistsError(output)
    ledger = Path(a['ledger_root']).resolve()
    if not ledger.is_dir() or ephemeral_ledger(ledger):
        raise PermissionError('Existing persistent ledger required; ephemeral path refused')
    lock = ledger/f"final-test-2021-{f['test_identity']}.lock.json"
    record = dict(status='CONSUMED_BEFORE_ANY_TEST_READ', **expected,
                  authorization_sha256=sha(authorization), authorization_id=a['authorization_id'],
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  retry='forbidden; preserve partials and request human incident review')
    save(lock, record)  # O_EXCL, fsync. Never remove even after failure.
    if os.name != 'nt':
        fd = os.open(ledger, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    return PixelPermit(f['test_identity'], str(lock), a['authorization_id'])


class FinalReader:
    """Bounded C8 raw reader. Target read requires a matching prediction receipt.

    With no permit ONLY 2020 validation data can be read. No TIFF header is
    opened during construction. Past context and spatial validity never depend
    on J+1. Inputs exactly match the existing compact encoder (including padding).
    """
    def __init__(self, rows, rasters, normalizer, *, permit=None, test_identity=None):
        from geofirex.runner_data import RasterReader
        from geofirex.unet_data import Normalizer
        years = {r['event_id'][:4] for r in rows}
        if years != {'2020'}:
            if years != {'2021'} or not isinstance(permit, PixelPermit) or permit.test_identity != test_identity:
                raise PermissionError('2021 reader requires consumed independent authorization')
            lock = load(permit.ledger_file)
            if lock.get('test_identity') != test_identity or lock.get('authorization_id') != permit.nonce:
                raise PermissionError('Durable ledger does not match pixel permit')
        self.rows = {(r['event_id'],r['date']):r for r in rows}
        if len(self.rows) != len(rows): raise ValueError('Duplicate reader observations')
        for r in rows:
            if r['relative_path']!=f"{r['event_id']}/{r['date']}.tif" or r['date'][:4]!=r['event_id'][:4]:
                raise PermissionError('Reader path/date/event mismatch')
        self.root = Path(rasters); self.normalizer = Normalizer(**normalizer)
        if self.normalizer.continuous != [12,14] or len(self.normalizer.channel_names) != 8:
            raise ValueError('Only frozen C8 profile supported')
        self.reader = RasterReader(64*1024**2)
        self.checked = set(); self.pending = {}; self.predicted = set(); self.counter = 0
        self.io_seconds = 0.; self.opened_2021 = set()
        self.target_cache=OrderedDict();self.target_cache_bytes=0

    def _verify_source(self,event,day):
        from geofirex.unet_data import validate_layout
        import rasterio
        r = self.rows[event,day]; path = safe_source(self.root, r['relative_path'])
        if event.startswith('2020/') and '2021' in path.parts:
            raise PermissionError('2020 qualification path resolves into 2021')
        started = time.perf_counter()
        if (event,day) not in self.checked:
            # Executed only for authorized 2021 or explicitly allowed 2020.
            if sha(path) != r['sha256']: raise ValueError('TIFF source content changed')
            if path.stat().st_size != r['bytes']: raise ValueError('TIFF source size changed')
            expected = parse_sidecars(r)
            with rasterio.open(path) as ds:
                validate_layout(ds)
                if (ds.height, ds.width, str(ds.crs), list(ds.transform)) != (r['height'],r['width'],r['crs'],r['transform']):
                    raise ValueError('TIFF geometry differs from frozen metadata')
                side_paths=[Path(p) for p in ds.files if Path(p).resolve()!=path.resolve()]
                if {p.name for p in side_paths}!=set(expected): raise ValueError('TIFF sidecar inventory changed')
                for p in side_paths:
                    if p.resolve().parent!=path.parent or (event.startswith('2020/') and '2021' in p.resolve().parts):
                        raise PermissionError('Sidecar escaped event directory')
                actual = {p.name:sha(p) for p in side_paths}
            if actual != expected: raise ValueError('TIFF sidecar identity changed')
            self.checked.add((event,day))
            if event.startswith('2021/'): self.opened_2021.add((event,day))
        self.io_seconds+=time.perf_counter()-started
        return path

    def _day(self, event, day, row, col):
        from geofirex.unet_data import padded_patch
        path=self._verify_source(event,day);started=time.perf_counter()
        raw, valid = self.reader.tile(str(path), row, col, 64)
        x = padded_patch(self.normalizer.transform(raw),0,0,64)
        valid = padded_patch(valid,0,0,64).astype(bool)
        self.io_seconds += time.perf_counter()-started
        return x, valid

    def _target(self,event,day,row,col):
        """Only fire band 23 + footprint, after prediction; no future encoding."""
        import rasterio
        from rasterio.enums import MaskFlags
        from rasterio.windows import Window
        from geofirex.unet_data import padded_patch
        path=self._verify_source(event,day);started=time.perf_counter();key=event,day
        full=self.rows[key]['height']*self.rows[key]['width']*5<=16*1024**2
        if key in self.target_cache:
            y,v=self.target_cache[key];self.target_cache.move_to_end(key)
        else:
            with rasterio.open(path) as ds:
                window=None if full else Window(col,row,min(64,ds.width-col),min(64,ds.height-row))
                fire=ds.read(23,window=window).astype(np.float32)
                if np.isinf(fire).any(): raise ValueError('Infinite target fire code')
                v=np.ones((1,*fire.shape),bool)
                if any(MaskFlags.per_dataset in f or MaskFlags.alpha in f for f in ds.mask_flag_enums):
                    v &= (ds.dataset_mask(window=window)>0)[None]
                nodata=ds.nodatavals[22]
                if nodata is not None and np.isfinite(nodata):
                    v &= (fire!=nodata)[None];fire[fire==nodata]=np.nan
                y=(~np.isnan(fire))[None].astype(np.float32)
            if full:
                n=y.nbytes+v.nbytes
                while self.target_cache and self.target_cache_bytes+n>16*1024**2:
                    _,(a,b)=self.target_cache.popitem(last=False);self.target_cache_bytes-=a.nbytes+b.nbytes
                self.target_cache[key]=(y,v);self.target_cache_bytes+=n
        if full: y=y[:,row:row+64,col:col+64];v=v[:,row:row+64,col:col+64]
        self.io_seconds+=time.perf_counter()-started
        return padded_patch(y,0,0,64),padded_patch(v,0,0,64)

    def context(self, key):
        event, day, r, c = key; target = date.fromisoformat(day)
        if key in self.pending or key in self.predicted: raise ValueError('Duplicate context')
        # Five past days for the common domain, only last three for the network.
        past = [self._day(event,(target-timedelta(days=k)).isoformat(),r,c) for k in (5,4,3,2,1)]
        x = np.stack([a for a,_ in past[-3:]])
        v = np.logical_and.reduce([v for _,v in past])
        observed = past[-1][0][7:8].copy()
        self.counter += 1; token = (key,self.counter)
        self.pending[key] = (token,v,observed)
        return x, token

    def record_prediction(self, token, probability):
        key = token[0]
        if key not in self.pending or self.pending[key][0] != token or key in self.predicted:
            raise ValueError('Unexpected/duplicate prediction receipt')
        p = np.asarray(probability)
        if p.shape != (1,64,64) or p.dtype != np.float32 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Prediction must be finite float32 [1,64,64]')
        self.predicted.add(key)

    def target_after_prediction(self, token):
        key = token[0]
        if key not in self.predicted or key not in self.pending or self.pending[key][0] != token:
            raise PermissionError('Target cannot be read before prediction')
        _, past_valid, observed = self.pending.pop(key)
        event,day,r,c = key
        y,v = self._target(event,day,r,c)
        return y, v & past_valid, observed


class BufferedAP:
    """Identical exact global-tie AP, bounded 250k-pixel sorting blocks."""
    def __init__(self, directory):
        from geofirex.benchmark_metrics import ExternalAveragePrecision
        self.acc = ExternalAveragePrecision(directory, retain_runs=False)
        self.p, self.y, self.n = [], [], 0

    def update(self, p, y):
        self.p.append(np.asarray(p).ravel().copy()); self.y.append(np.asarray(y).ravel().copy()); self.n += np.size(p)
        if self.n >= 250000: self.flush()

    def flush(self):
        if self.n: self.acc.update(np.concatenate(self.p), np.concatenate(self.y))
        self.p, self.y, self.n = [], [], 0

    def compute(self):
        self.flush(); return self.acc.compute()


class FinalMetrics:
    """Separate from guarded V2Metrics; exact AP both pooled and per event."""
    def __init__(self, scratch, expected_keys):
        self.expected = set(expected_keys)
        if not self.expected or len(self.expected) != len(expected_keys): raise ValueError('Unique nonempty exhaustive keys required')
        self.scratch = Path(scratch); self.scratch.mkdir(parents=True, exist_ok=False)
        self.ap = [BufferedAP(self.scratch/p) for p in ('pooled','new')]
        self.counts = np.zeros((2,4),np.int64); self.error = np.zeros(2,np.float64)
        self.event_counts = self.counts.copy(); self.event_error = self.error.copy()
        self.event_ap = None; self.event = None; self.per_event = []; self.completed_events = set()
        self.seen = set(); self.domains = {}; self.closed = False

    def _finish_event(self):
        from geofirex.compact_calibration import metrics
        if self.event is None: return
        rr = []
        for c,err,ap in zip(self.event_counts,self.event_error,self.event_ap):
            m = metrics(c); n = m['valid_pixels']; negatives = m['fp']+m['tn']
            m.update(average_precision=ap.compute(), brier=float(err/n) if n else None,
                     false_positive_rate=m['fp']/negatives if negatives else None)
            rr.append(m)
        self.per_event.append(dict(event_id=self.event, pooled=rr[0],new_fire=rr[1]))
        self.completed_events.add(self.event)

    def update(self, key, probability, target, valid, observed):
        if self.closed or key not in self.expected or key in self.seen: raise ValueError('Unexpected/duplicate/finalized tile')
        p,y,v,o = map(np.asarray,(probability,target,valid,observed))
        if not p.shape == y.shape == v.shape == o.shape: raise ValueError('Shapes differ')
        if p.dtype != np.float32 or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not all(np.isin(a,[0,1]).all() for a in (y,v,o)):
            raise ValueError('Finite float32 probability/binary domain required')
        if self.event != key[0]:
            self._finish_event()
            if key[0] in self.completed_events: raise ValueError('Events must be contiguous')
            self.event = key[0]; self.event_counts = np.zeros((2,4),np.int64); self.event_error = np.zeros(2,np.float64)
            self.event_ap = [BufferedAP(self.scratch/f'event_{len(self.per_event)}_{d}') for d in ('pooled','new')]
        truth = y.astype(bool); pred = p >= np.float32(.78)
        for i,mask in enumerate((v.astype(bool),v.astype(bool)&~o.astype(bool))):
            c = np.array([(pred&truth&mask).sum(),(pred&~truth&mask).sum(),(~pred&truth&mask).sum(),(~pred&~truth&mask).sum()],np.int64)
            err = np.square(p[mask].astype(np.float64)-y[mask]).sum()
            self.counts[i] += c; self.event_counts[i] += c; self.error[i] += err; self.event_error[i] += err
            if mask.any(): self.ap[i].update(p[mask],y[mask]); self.event_ap[i].update(p[mask],y[mask])
        h = hashlib.sha256(json.dumps([key,list(p.shape)]).encode())
        for a in (y,v,o): h.update(a.astype(bool).tobytes())
        self.domains[key] = h.hexdigest(); self.seen.add(key)

    def result(self):
        from geofirex.compact_calibration import metrics
        if self.closed or self.seen != self.expected: raise ValueError('Evaluation incomplete/already finalized')
        self._finish_event(); self.closed = True; rr=[]
        for c,err,ap in zip(self.counts,self.error,self.ap):
            m=metrics(c); n=m['valid_pixels']; neg=m['fp']+m['tn']
            m.update(average_precision=ap.compute(),brier=float(err/n) if n else None,
                     false_positive_rate=m['fp']/neg if neg else None); rr.append(m)
        dice=[r['pooled']['dice'] for r in self.per_event if r['pooled']['dice'] is not None]
        return dict(status='COMPLETE',threshold=.78,pooled=rr[0],new_fire=rr[1],per_event=self.per_event,
                    macro_event_dice=float(np.mean(dice)) if dice else None,defined_macro_events=len(dice),
                    tiles=len(self.seen),evaluation_domain_sha256=hashlib.sha256(json.dumps(sorted(self.domains.items())).encode()).hexdigest(),
                    ap='exact noninterpolated global ties on unmodified float32 probabilities')


class PredictionArchive:
    """Uncompressed event shards, append-only, durable event receipts, exact floats."""
    def __init__(self, out):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=False)
        self.event=None; self.stream=None; self.offset=0; self.rows=[]; self.receipts=[]

    def _finish(self):
        if self.stream is None: return
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        p=self.out/f'event_{len(self.receipts):03d}.bin'
        receipt=dict(event_id=self.event,path=p.name,sha256=sha(p),bytes=p.stat().st_size,tiles=self.rows)
        save(p.with_suffix('.json'),receipt); self.receipts.append(receipt); self.stream=None

    def add(self,key,p,y,v,o):
        p,y,v,o=map(np.asarray,(p,y,v,o))
        if (not p.shape==y.shape==v.shape==o.shape or p.dtype!=np.float32 or not np.isfinite(p).all()
                or np.any((p<0)|(p>1)) or not all(np.isin(a,[0,1]).all() for a in (y,v,o))):
            raise ValueError('Invalid lossless prediction payload')
        if self.event != key[0]:
            self._finish(); self.event=key[0]; self.offset=0; self.rows=[]
            self.stream=(self.out/f'event_{len(self.receipts):03d}.bin').open('xb')
        a=np.empty(p.shape,PIXEL)
        for name,value in zip(('p','y','v','o'),(p,y,v,o)): a[name]=value
        self.stream.write(a.tobytes()); self.rows.append(dict(key=key,shape=list(p.shape),offset=self.offset))
        self.offset += a.nbytes

    def finish(self):
        self._finish(); save(self.out/'manifest.json',dict(dtype=PIXEL.descr,events=self.receipts))


def result_package(out,kind='RESULTS'):
    """Exclusive verified archive; scratch is deliberately outside output."""
    if kind not in ('RESULTS','PARTIAL'): raise ValueError('Unknown archive kind')
    out=Path(out); target=out.with_name(out.name+'-'+kind+'.zip')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    manifest={p.relative_to(out).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:
        for p in files: z.write(p,p.relative_to(out).as_posix())
        z.writestr('archive_manifest.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(target) as z:
        for name,r in manifest.items():
            h=hashlib.sha256(); n=0
            with z.open(name) as stream:
                for b in iter(lambda:stream.read(1024**2),b''): h.update(b); n+=len(b)
            if h.hexdigest()!=r['sha256'] or n!=r['bytes']: raise ValueError('Result archive verification failed')
    save(target.with_suffix('.sha256.json'),dict(archive=target.name,sha256=sha(target),members_verified=len(manifest)))
    return str(target)


def qualify(root, frozen, archive, rasters, corpus, output, device='cpu'):
    """Deterministic, non-label-selected 2020 parity subset; no 2021 reads.

    Four grid corners from eight evenly spaced validation events. This is a
    software parity check, not another model-selection or performance estimate.
    A future CUDA runtime needs its own qualification, explicitly requested.
    """
    import importlib.metadata as md
    import torch
    from geofirex.benchmark_models import StackedUNet
    from geofirex.protocol_v2 import V2CompactTiles, sample_keys
    from geofirex.mission_streaming import peak_ram
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    f,_,_=verify_freeze(root,frozen)
    norm=load(Path(root)/INDEX/'normalization.json')
    state=selected_checkpoint(archive,norm)
    old=V2CompactTiles(corpus,Path(root)/INDEX,'val')
    all_keys=sample_keys(old)
    if any(k[0][:4]!='2020' or k[1][:4]!='2020' for k in all_keys):
        raise PermissionError('Parity restricted to 2020 validation')
    events=sorted({k[0] for k in all_keys})
    chosen=[events[i] for i in np.linspace(0,len(events)-1,min(8,len(events)),dtype=int)]
    selected=[]
    for event in chosen:
        ids=[i for i,k in enumerate(all_keys) if k[0]==event]
        day=all_keys[ids[len(ids)//2]][1]
        ids=[i for i in ids if all_keys[i][1]==day]
        rows=sorted({all_keys[i][2] for i in ids}); cols=sorted({all_keys[i][3] for i in ids})
        selected.extend(i for i in ids if all_keys[i][2] in (rows[0],rows[-1]) and all_keys[i][3] in (cols[0],cols[-1]))
    observations=[r for r in metadata_rows(root) if r['event_id'] in chosen]
    reader=FinalReader(observations,rasters,norm)
    torch.set_num_threads(2)
    model=StackedUNet(8,3,base_channels=8,depth=3); model.load_state_dict(state['model'],strict=True)
    model.eval(); model.requires_grad_(False); model.to(device)
    started=time.perf_counter(); checks=[]
    with __import__('rasterio').Env(GDAL_PAM_ENABLED='NO',GDAL_CACHEMAX=64*1024**2):
        for i in selected:
            key=all_keys[i]; x,token=reader.context(key); old_x,old_y,old_v=old[i]
            with torch.inference_mode():
                p=model(torch.from_numpy(x[None]).to(device)).sigmoid().cpu().numpy()[0]
                ref=model(old_x[None].to(device)).sigmoid().cpu().numpy()[0]
            reader.record_prediction(token,p); y,v,o=reader.target_after_prediction(token)
            check=dict(key=key,input_bit_equal=np.array_equal(x,old_x.numpy()),
                       target_bit_equal=np.array_equal(y,old_y.numpy()),valid_bit_equal=np.array_equal(v,old_v.numpy().astype(bool)),
                       observed_bit_equal=np.array_equal(o,old_x[-1,7:8].numpy()),
                       probability_max_abs_error=float(np.max(np.abs(p-ref))),
                       probability_equal=bool(np.allclose(p,ref,atol=1e-6,rtol=1e-6)),
                       threshold_classification_equal=bool(np.array_equal(p>=np.float32(.78),ref>=np.float32(.78))))
            checks.append(check)
    old.close()
    passed=all(all(r[k] for k in ('input_bit_equal','target_bit_equal','valid_bit_equal','observed_bit_equal','probability_equal','threshold_classification_equal')) for r in checks)
    result=dict(status='PASS' if passed else 'FAIL',freeze_sha256=sha(Path(frozen)/'selection.json'),
                runtime={p:md.version(p) for p in ('torch','numpy','rasterio','pyarrow')},device=device,
                device_name=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',
                validation_parity=dict(passed=passed,tiles=len(checks),events=len(chosen),checks=checks,
                  scope='deterministic 2020 subset; no new selection; not full validation evaluation'),
                elapsed_seconds=time.perf_counter()-started,peak_host_rss_bytes=peak_ram(),
                test_pixels_opened=0,training_executed=False)
    save(output,result)
    if not passed: raise ValueError('2020 parity failed; freeze remains inaccessible to real test')
    return result


def evaluate(root, frozen, archive, rasters, output, scratch, authorization, preflight_path, qualification, device='cuda:0'):
    """The sole real-test entry point. Never called by preparation/qualification."""
    f,rows,keys=verify_freeze(root,frozen)
    # Reject disabled grants BEFORE torch/CUDA, TIFF headers, output or scratch.
    if load(authorization).get('authorized') is not True: raise PermissionError('2021 evaluation is NOT authorized')
    live=preflight(root,frozen,archive,rasters,Path(output).parent,scratch,qualification)
    required=('frozen_identities','tiff_paths_sizes','c8_archive','ram','disk','runtime_validation_parity')
    if any(live['checks'][k]!='PASS' for k in required): raise PermissionError('Live technical preflight failed')
    if Path(output).with_name(Path(output).name+'-RESULTS.zip').exists(): raise FileExistsError('Result archive exists')
    norm=load(Path(root)/INDEX/'normalization.json'); state=selected_checkpoint(archive,norm)
    import torch
    from geofirex.benchmark_models import StackedUNet
    from geofirex.mission_streaming import peak_ram
    torch.set_num_threads(2)
    model=StackedUNet(8,3,base_channels=8,depth=3); model.load_state_dict(state['model'],strict=True)
    model.eval(); model.requires_grad_(False); model.to(device)
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    permit=consume_authorization(authorization,frozen,preflight_path,qualification,output,scratch,rasters,archive,device)
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    copy_exclusive(permit.ledger_file,out/'CONSUMED.json'); save(out/'live_preflight.json',live)
    copy_exclusive(Path(frozen)/'selection.json',out/'selection.json')
    copy_exclusive(authorization,out/'authorization.json');copy_exclusive(qualification,out/'qualification.json')
    run_scratch=Path(scratch)/('final-v2-'+permit.nonce)
    start=time.perf_counter(); infer_seconds=0.; scratch_peak=0
    try:
        with execution_deadline(14400-1800), __import__('rasterio').Env(GDAL_PAM_ENABLED='NO',GDAL_CACHEMAX=64*1024**2):
            reader=FinalReader(rows,rasters,norm,permit=permit,test_identity=f['test_identity'])
            writer=PredictionArchive(out/'predictions'); acc=FinalMetrics(run_scratch,keys)
            with (out/'progress.jsonl').open('x',encoding='utf-8') as log:
                for offset in range(0,len(keys),16):
                    if time.perf_counter()-start > 14400-1800: raise TimeoutError('Stop with 30min backup reserve; no automatic retry')
                    batch=keys[offset:offset+16]; contexts=[reader.context(k) for k in batch]
                    x=torch.from_numpy(np.stack([x for x,_ in contexts])).to(device)
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    t=time.perf_counter()
                    with torch.inference_mode(): p=model(x).sigmoid().cpu().numpy().astype(np.float32,copy=False)
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    infer_seconds+=time.perf_counter()-t
                    for key,(_,token),prob in zip(batch,contexts,p):
                        reader.record_prediction(token,prob)
                        y,v,o=reader.target_after_prediction(token)
                        writer.add(key,prob,y,v,o); acc.update(key,prob,y,v,o)
                    if offset//16%100==0:
                        scratch_peak=max(scratch_peak,sum(p.stat().st_size for p in run_scratch.rglob('*.bin')))
                        log.write(json.dumps(dict(tiles=min(offset+16,len(keys)),seconds=time.perf_counter()-start,peak_host_rss_bytes=peak_ram()))+'\n');log.flush()
                writer.finish()
            save(out/'INFERENCE_COMPLETE.json',dict(keys_sha256=f['keys_sha256'],tiles=len(keys),
                 source_observations_verified=len(reader.checked),test_observations_accessed=len(reader.opened_2021),
                 prediction_manifest_sha256=sha(out/'predictions/manifest.json')))
            result=acc.result()
            result.update(selected=f['selected'],checkpoint_sha256=CHECKPOINT_SHA,freeze_sha256=sha(Path(frozen)/'selection.json'),
                          test_identity=f['test_identity'],test_pixels_opened=True,
                          resources=dict(total_seconds=time.perf_counter()-start,inference_seconds=infer_seconds,
                          loader_io_seconds=reader.io_seconds,exact_metrics_and_other_seconds=time.perf_counter()-start-infer_seconds-reader.io_seconds,
                          peak_host_rss_bytes=peak_ram(),scratch_sampled_peak_bytes=scratch_peak,
                          scratch_reserved_upper_bytes=f['inventory']['exact_sort_scratch_upper_bytes'],
                          peak_vram_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                          gpu_card_hours_inference=infer_seconds/3600 if device.startswith('cuda') else 0,
                          gpu_session_hours_upper=(time.perf_counter()-start)/3600 if device.startswith('cuda') else 0))
            save(out/'evaluation.json',result)
            import csv
            with (out/'per_event.csv').open('x',newline='',encoding='utf-8') as stream:
                rows_csv=[dict(event_id=r['event_id'],**{domain+'_'+k:v for domain in ('pooled','new_fire') for k,v in r[domain].items()}) for r in result['per_event']]
                w=csv.DictWriter(stream,fieldnames=list(rows_csv[0]));w.writeheader();w.writerows(rows_csv)
        return result_package(out)
    except BaseException as exc:
        if 'writer' in locals() and writer.stream is not None:
            writer.stream.flush(); os.fsync(writer.stream.fileno()); writer.stream.close()
        save(out/'PARTIAL_NO_RETRY.json',dict(status='PARTIAL',error=repr(exc),ledger=permit.ledger_file,
             instruction='preserve outputs/scratch/ledger; no automatic 2021 reread; human incident review'))
        result_package(out,'PARTIAL')
        raise


def recover_metrics(root,frozen,partial,scratch,output):
    """CPU-only recovery from COMPLETE saved predictions, never raw TIFFs.

    Uses the already-consumed original authorization. It cannot repeat
    inference or repair incomplete predictions; the permanent ledger persists.
    """
    f,_,keys=verify_freeze(root,frozen);partial=Path(partial);out=Path(output)
    if out.exists() or (partial/'evaluation.json').exists(): raise FileExistsError('Existing/completed evaluation refused')
    consumed=load(partial/'CONSUMED.json');auth=load(partial/'authorization.json')
    complete=load(partial/'INFERENCE_COMPLETE.json')
    if (consumed['test_identity']!=f['test_identity'] or auth.get('authorized') is not True
            or consumed['authorization_sha256']!=sha(partial/'authorization.json')
            or consumed['freeze_sha256']!=sha(Path(frozen)/'selection.json')
            or complete['keys_sha256']!=f['keys_sha256'] or complete['tiles']!=len(keys)
            or complete['prediction_manifest_sha256']!=sha(partial/'predictions/manifest.json')):
        raise PermissionError('Original consumed authorization/inference evidence differs')
    ledger=Path(auth['ledger_root'])/f"final-test-2021-{f['test_identity']}.lock.json"
    if sha(ledger)!=sha(partial/'CONSUMED.json'): raise PermissionError('Durable ledger missing/changed')
    pred=partial/'predictions';manifest=load(pred/'manifest.json');seen=[]
    # Verify the entire immutable saved stream BEFORE accumulating any result.
    for event in manifest['events']:
        path=safe_source(pred,event['path'])
        if sha(path)!=event['sha256'] or path.stat().st_size!=event['bytes']: raise ValueError('Saved predictions changed')
        expected_offset=0
        for tile in event['tiles']:
            if tile['offset']!=expected_offset or tile['shape']!=[1,64,64]: raise ValueError('Saved shard layout differs')
            expected_offset+=4096*PIXEL.itemsize;seen.append(tuple(tile['key']))
        if expected_offset!=event['bytes']: raise ValueError('Saved shard length differs')
    if seen!=keys: raise ValueError('Saved predictions are not exhaustive frozen order')
    out.mkdir(parents=True,exist_ok=False);acc=FinalMetrics(Path(scratch)/('recovery-'+digest(str(out))[:16]),keys)
    with execution_deadline(14400-1800):
        for event in manifest['events']:
            with safe_source(pred,event['path']).open('rb') as stream:
                for tile in event['tiles']:
                    a=np.fromfile(stream,dtype=PIXEL,count=4096).reshape(1,64,64)
                    acc.update(tuple(tile['key']),a['p'],a['y'],a['v'],a['o'])
        r=acc.result();r.update(test_identity=f['test_identity'],freeze_sha256=sha(Path(frozen)/'selection.json'),
            recovery='CPU only from hash-verified complete predictions; no inference, no raw TIFF read',
            original_result_directory=str(partial.resolve()),raw_test_observations_opened=0)
        save(out/'evaluation.json',r);save(out/'CONSUMED.json',consumed)
    return result_package(out)
