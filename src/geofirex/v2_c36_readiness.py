"""Additive C36 CPU contracts. No model, GPU, raster pixels or training path.

The historical validator and all frozen C8 files deliberately remain unchanged.
Chronology checks validate evidence consistency, never invent its factual truth.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from geofirex.protocol_v2 import sha256

ADDED_BANDS = (0, 1, 2, 3, 4, 16)
ALL_BANDS = (0, 1, 2, 3, 4, 12, 13, 14, 16, 22)
PRODUCTS = {0: 'VNP09GA', 1: 'VNP09GA', 2: 'VNP09GA',
            3: 'VNP13A1', 4: 'VNP13A1', 16: 'MCD12Q1'}
EVENT = re.compile(r'(?:2018|2019|2020)/fire_[0-9]+')


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def frozen_inputs(root):
    root = Path(root)
    path = root/'reports/protocol_v2_execution/execution_manifest.json'
    lock = json.loads(path.read_text(encoding='utf8'))
    results = {name: sha256(root/name) == digest for name, digest in lock['files'].items()}
    if not all(results.values()):
        raise ValueError('Frozen C8 identity changed: '+str([p for p,v in results.items() if not v]))
    return dict(files=len(results), status='PASS', execution_manifest_sha256=sha256(path))


def safe_relative(value):
    p = PurePosixPath(value)
    if (not value or '\\' in value or ':' in value or p.is_absolute()
            or '..' in p.parts or '2021' in p.parts):
        raise ValueError('Unsafe path or reserved 2021 data')
    return p


def utc(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError('Explicit UTC timestamp required')
    return dt


class V2Contract:
    """Derive every admissible key from existing immutable indices, not evidence.

    Observations may contain test *metadata*, but no test paths can be admitted.
    Constructor accepts small certified-style fixtures for CPU tests.
    """
    def __init__(self, index, observations):
        self.index = Path(index)
        self.meta = json.loads((self.index/'index.json').read_text(encoding='utf8'))
        if self.meta['schema'] != 'geofirex-chronological-v2':
            raise ValueError('V2 schema required')
        self.index_sha256 = sha256(self.index/'index.json')
        self.events_sha256 = sha256(self.index.parent/'events.csv')
        if self.events_sha256 != self.meta['events_csv_sha256']:
            raise ValueError('Event manifest identity differs')
        self.observations = {}
        for r in observations:
            key = (r['event_id'], r['date'])
            if key in self.observations:
                raise ValueError('Duplicate observation metadata')
            self.observations[key] = r
        self.keys = {}; self.records = {}; self.inputs = {}; self.windows = set()
        self.input_owners = {}
        for role in ('train', 'val'):
            p = self.index/(role+'.npy')
            if sha256(p) != self.meta['index_sha256'][role]:
                raise ValueError('Frozen index differs')
            records = np.load(p, mmap_mode='r', allow_pickle=False)
            keys = []; inputs = {}
            for rec in records:
                event = self.meta['events'][self.meta['archives'][int(rec['archive'])]]
                target = date.fromordinal(int(rec['day']))
                if not EVENT.fullmatch(event) or self.meta['event_roles'].get(event) != role:
                    raise ValueError('Reserved year, embargo or wrong V2 role')
                if target >= date(2021, 1, 1):
                    raise ValueError('2021 target forbidden')
                cutoff = date.fromisoformat(self.meta['cutoff'])
                if ((role == 'train' and target >= cutoff)
                        or (role == 'val' and target-timedelta(days=5) < cutoff)):
                    raise ValueError('V2 chronological boundary differs')
                key = (event, target.isoformat(), int(rec['row']), int(rec['col']))
                if key[2] % 64 or key[3] % 64:
                    raise ValueError('Noncanonical tile offset')
                keys.append(key); self.windows.add((event, target.isoformat()))
                for delta in range(-5, 1):
                    day = (target+timedelta(days=delta)).isoformat()
                    obs = self.observations.get((event, day))
                    if obs is None or obs['status'] != 'ok' or day >= '2021-01-01':
                        raise ValueError('Unknown or uncertified observation date')
                    if not 0 <= key[2] < obs['height'] or not 0 <= key[3] < obs['width']:
                        raise ValueError('Tile outside certified grid')
                    if delta in (-3, -2, -1):
                        relative = safe_relative(obs['relative_path']).as_posix()
                        if relative != f'{event}/{day}.tif':
                            raise ValueError('Observation path/date/event mismatch')
                        if relative in self.input_owners and self.input_owners[relative] != role:
                            raise ValueError('Input path reused across roles')
                        self.input_owners[relative] = role
                        inputs[relative] = obs
            if not keys or len(keys) != len(set(keys)):
                raise ValueError('Empty or duplicated tile keys')
            self.keys[role] = keys; self.records[role] = records; self.inputs[role] = inputs

    @classmethod
    def from_repository(cls, root):
        import pyarrow.parquet as pq
        root = Path(root); frozen_inputs(root)
        config = json.loads((root/'configs/protocol_v2/candidate_c36.json').read_text())
        return cls(root/config['index'], pq.read_table(
            root/'reports/full_corpus_v1/prepared_v1/observations.parquet').to_pylist())

    def required_keys(self, bands=ADDED_BANDS):
        if not set(bands) <= set(ALL_BANDS) or not bands:
            raise ValueError('Unapproved C36 bands')
        return {(event, (date.fromisoformat(day)-timedelta(days=1)).isoformat(),
                 (date.fromisoformat(day)+timedelta(days=delta)).isoformat(), band)
                for event, day in self.windows for delta in (-3, -2, -1) for band in bands}

    def input_path(self, root, relative, role=None):
        """Fail before opening any nonallowlisted or reserved raster."""
        relative = safe_relative(relative).as_posix()
        owner = self.input_owners.get(relative)
        if owner is None or (role is not None and owner != role):
            raise ValueError('Raster outside authorized V2 input paths')
        root = Path(root).resolve(); path = (root/relative).resolve()
        if not path.is_relative_to(root) or '2021' in path.parts:
            raise ValueError('Raster resolves outside permitted data')
        return path

    def summary(self):
        return dict(index_sha256=self.index_sha256, events_sha256=self.events_sha256,
            roles={role:dict(tiles=len(self.keys[role]), unique_inputs=len(self.inputs[role]),
                events=len({k[0] for k in self.keys[role]}),
                ordered_keys_sha256=json_digest(self.keys[role])) for role in ('train', 'val')},
            added_band_evidence_keys=len(self.required_keys()),
            all_band_evidence_keys=len(self.required_keys(ALL_BANDS)), test_pixels_opened=0)


def validate_causal_evidence(contract, document, evidence_root):
    """Require exhaustive, versioned, canonical assets and actual proof documents.

    The evidence documents must have been reviewed against primary records by
    the named reviewer. A matching file hash does not by itself prove the claim.
    Synthetic fixtures exercise structure only and cannot qualify real assets.
    """
    if (document.get('schema') != 'geofirex-c36-causal-evidence-1'
            or document.get('index_sha256') != contract.index_sha256
            or document.get('events_sha256') != contract.events_sha256):
        raise ValueError('Evidence must bind the frozen V2 manifests')
    root = Path(evidence_root).resolve()
    expected = contract.required_keys(); seen = set(); canonical = {}; proof_cache = {}; primary_cache = {}
    for row in document.get('records', []):
        key = (row['event_id'], row['as_of_J'], row['input_date'], row['band'])
        if key not in expected or key in seen or type(row['band']) is not int:
            raise ValueError('Unexpected, duplicate or noninteger evidence key')
        seen.add(key)
        event,j,day,band = key
        if day > j or (event,day) not in contract.observations or not EVENT.fullmatch(event):
            raise ValueError('Future or unknown observation')
        decision = datetime.combine(date.fromisoformat(j)+timedelta(days=1), time.min,
                                    tzinfo=timezone.utc)
        if utc(row['decision_at']) != decision:
            raise ValueError('Decision is end of J, not end of target day')
        start,end,available = [utc(row[k]) for k in ('observation_start','observation_end','available_at')]
        if not start <= end <= available <= decision:
            raise ValueError('Asset incomplete or unavailable at J')
        kind = 'annual' if band == 16 else 'composite' if band in (3,4) else 'observed'
        if row.get('kind') != kind or row.get('source_product') != PRODUCTS[band]:
            raise ValueError('Wrong temporal kind or source product')
        if band == 16:
            year = row.get('product_year')
            if (type(year) is not int or start != datetime(year,1,1,tzinfo=timezone.utc)
                    or end != datetime(year+1,1,1,tzinfo=timezone.utc)):
                raise ValueError('Annual LC must state its complete observation year')
        for name in ('asset_sha256','evidence_sha256','raster_sha256','primary_record_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row.get(name,'')) or row[name] == '0'*64:
                raise ValueError('Nonempty SHA256 required: '+name)
        for name in ('asset_id','product_version','reviewer','justification','source_uri','primary_record_locator'):
            if not isinstance(row.get(name),str) or not row[name].strip():
                raise ValueError('Missing reviewed provenance: '+name)
        if not row['source_uri'].startswith('https://') or row.get('reviewed') is not True:
            raise ValueError('Reviewed primary-source reference required')
        utc(row['reviewed_at'])
        relative = f'{event}/{day}.tif'
        if row['raster_sha256'] != contract.observations[event,day]['sha256']:
            raise ValueError('Evidence not linked to certified raster bytes')
        identity = tuple(row[k] for k in ('asset_id','asset_sha256','source_product','product_version','raster_sha256'))
        canonical_key = (relative,band)
        if canonical_key in canonical and canonical[canonical_key] != identity:
            raise ValueError('An input path changes asset vintage between J values')
        canonical[canonical_key] = identity
        evidence_path = (root/safe_relative(row['evidence_path'])).resolve()
        if not evidence_path.is_relative_to(root):
            raise ValueError('Evidence escapes proof root')
        primary_path = (root/safe_relative(row['primary_record_path'])).resolve()
        if (not primary_path.is_relative_to(root) or primary_path == evidence_path
                or primary_path.suffix.lower() not in {'.pdf','.html','.txt','.json'}):
            raise ValueError('Separate archived primary document required, not a raster/archive or declaration alone')
        primary_key = (str(primary_path),row['primary_record_sha256'])
        if primary_key not in primary_cache:
            if sha256(primary_path) != row['primary_record_sha256']:
                raise ValueError('Archived primary record bytes differ')
            primary_cache[primary_key] = True
        cache_key = (str(evidence_path),row['evidence_sha256'])
        if cache_key not in proof_cache:
            if sha256(evidence_path) != row['evidence_sha256']:
                raise ValueError('Evidence bytes differ')
            proof_cache[cache_key] = json.loads(evidence_path.read_text(encoding='utf8'))
        proof = proof_cache[cache_key]
        fields = ('asset_id','asset_sha256','source_product','product_version','raster_sha256',
                  'observation_start','observation_end','available_at','source_uri',
                  'primary_record_sha256','primary_record_locator')
        if any(proof.get(k) != row[k] for k in fields):
            raise ValueError('Proof document does not substantiate the asset declaration')
    if seen != expected:
        raise ValueError(f'Causal coverage incomplete: {len(seen)}/{len(expected)}')
    return dict(status='REVIEWED_EVIDENCE_CONSISTENCY_PASS', records=len(seen),
        canonical_input_bands=len(canonical), proof_files=len(proof_cache),primary_files=len(primary_cache),
        evidence_sha256=json_digest(document), index_sha256=contract.index_sha256,
        factual_truth_requires_primary_source_review=True, training_authorized=False)
