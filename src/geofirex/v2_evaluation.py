"""V2 metric accumulator: explicit observed-J mask, no assumed fire channel.

Consumes probabilities only; contains no model or optimizer. Exact AP reuses
the certified global-tie external sorter. Never use with the locked 2021 set
until the separate final-evaluation decision.
"""
from collections import defaultdict
import hashlib
import json
import numpy as np

from geofirex.benchmark_metrics import ExternalAveragePrecision
from geofirex.compact_calibration import metrics


class V2Metrics:
    def __init__(self, ap_directory, expected_keys, threshold=.78):
        if not 0 < threshold < 1:
            raise ValueError('Invalid frozen operating threshold')
        self.expected = set(expected_keys)
        if not self.expected or len(self.expected) != len(expected_keys):
            raise ValueError('Nonempty unique exhaustive tile keys required')
        if any(key[0][:4] not in ('2018', '2019', '2020') or key[1] >= '2021-01-01' for key in self.expected):
            raise ValueError('V2 development metrics cannot open 2021')
        self.threshold = threshold; self.seen = set(); self.closed = False
        self.domains = {}
        self.ap = ExternalAveragePrecision(ap_directory, retain_runs=False)
        self.global_counts = np.zeros(4, np.int64)
        self.new_counts = np.zeros(4, np.int64)
        self.per_event = defaultdict(lambda: np.zeros(4, np.int64))
        self.squared_error = 0.

    def update(self, key, probability, target, valid, observed_J):
        if self.closed or key not in self.expected or key in self.seen:
            raise ValueError('Unexpected, duplicate or finalized evaluation tile')
        p, y, v, observed = map(np.asarray, (probability, target, valid, observed_J))
        if not p.shape == y.shape == v.shape == observed.shape:
            raise ValueError('Explicit probability/target/valid/observed shapes must match')
        if (p.dtype != np.float32 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
                or not all(np.isin(a, [0, 1]).all() for a in (y, v, observed))):
            raise ValueError('Finite float32 probabilities and binary masks/targets required')
        valid_mask = v.astype(bool); truth = y.astype(bool); pred = p >= np.float32(self.threshold)
        def count(mask):
            return np.array([int((pred & truth & mask).sum()), int((pred & ~truth & mask).sum()),
                             int((~pred & truth & mask).sum()), int((~pred & ~truth & mask).sum())], np.int64)
        counts = count(valid_mask)
        self.global_counts += counts; self.per_event[key[0]] += counts
        self.new_counts += count(valid_mask & ~observed.astype(bool))
        self.squared_error += float(np.square(p[valid_mask].astype(np.float64)-y[valid_mask]).sum())
        if valid_mask.any():
            self.ap.update(p[valid_mask], y[valid_mask])
        identity = hashlib.sha256(json.dumps([key, list(p.shape)]).encode())
        for a in (y, v, observed):
            identity.update(a.astype(bool).tobytes())
        self.domains[key] = identity.hexdigest()
        self.seen.add(key)

    def result(self):
        if self.closed or self.seen != self.expected:
            raise ValueError('Evaluation incomplete or already finalized')
        self.closed = True
        global_metrics = metrics(self.global_counts)
        global_metrics['average_precision'] = self.ap.compute()
        n = global_metrics['valid_pixels']
        global_metrics['brier'] = self.squared_error/n if n else None
        event_metrics = [dict(event_id=e, **metrics(c)) for e, c in sorted(self.per_event.items())]
        dice = [r['dice'] for r in event_metrics if r['dice'] is not None]
        domain_sha = hashlib.sha256(json.dumps(sorted(self.domains.items())).encode()).hexdigest()
        new_metrics = metrics(self.new_counts)
        negatives = new_metrics['fp'] + new_metrics['tn']
        new_metrics['false_positive_rate'] = new_metrics['fp']/negatives if negatives else None
        return dict(threshold=self.threshold, pooled=global_metrics, new_fire=new_metrics,
                    macro_event_dice=float(np.mean(dice)) if dice else None,
                    defined_macro_events=len(dice), per_event=event_metrics,
                    evaluation_domain_sha256=domain_sha,
                    tiles=len(self.seen), status='COMPLETE', test_pixels_opened=0)
