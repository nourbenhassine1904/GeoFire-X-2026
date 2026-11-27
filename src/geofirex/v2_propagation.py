"""Preregistered development PR diagnostics; never selects an operating threshold."""
import csv
import itertools
from collections import defaultdict
import numpy as np
from geofirex.benchmark_metrics import ExternalAveragePrecision, DTYPE, _merge
from geofirex.compact_calibration import metrics
from geofirex.v2_evaluation import V2Metrics


class PropagationMetrics(V2Metrics):
    def __init__(self, directory, expected_keys, policy):
        super().__init__(directory / 'pooled', expected_keys, threshold=.78)
        self.new_ap = ExternalAveragePrecision(directory / 'new', retain_runs=False)
        self.policy = policy
        self.new_event = defaultdict(lambda: np.zeros(4, np.int64))
        self.event_error = defaultdict(float)

    def update(self, key, probability, target, valid, observed_J):
        super().update(key, probability, target, valid, observed_J)
        p, y, v, observed = map(np.asarray, (probability, target, valid, observed_J))
        mask = v.astype(bool) & ~observed.astype(bool)
        if mask.any():
            self.new_ap.update(p[mask], y[mask])
        pred, truth = p >= np.float32(.78), y.astype(bool)
        self.new_event[key[0]] += [int((pred & truth & mask).sum()), int((pred & ~truth & mask).sum()),
                                   int((~pred & truth & mask).sum()), int((~pred & ~truth & mask).sum())]
        self.event_error[key[0]] += float(np.square(p[v.astype(bool)].astype(np.float64)-y[v.astype(bool)]).sum())

    def result(self):
        result = super().result()
        result['propagation_pr'] = exact_pr(self.new_ap, self.policy)
        for row in result['per_event']:
            e = row['event_id']; n = row['valid_pixels']
            row['brier'] = self.event_error[e]/n if n else None
            row['new_fire'] = metrics(self.new_event[e])
        return result


def exact_pr(acc, policy):
    """Global ties are indivisible. Max recall subject to an integer FP budget.

    Downsample the PR curve at predeclared recall crossings for bounded output.
    All groups still contribute to exact AP and budget operating envelopes.
    This same-panel envelope is descriptive/oracle, not out-of-sample performance.
    """
    paths = list(acc.paths); level = 0
    while len(paths) > acc.fan_in:
        merged = []
        for group_id, start in enumerate(range(0, len(paths), acc.fan_in)):
            group = paths[start:start+acc.fan_in]
            path = acc.directory / f'pr_merge_{level}_{group_id}.bin'
            with path.open('xb') as stream:
                source = _merge(group)
                while True:
                    block = list(itertools.islice(source, 4096))
                    if not block:
                        break
                    np.array(block, dtype=DTYPE).tofile(stream)
            acc.owned.add(path); acc._release(group); merged.append(path)
        paths = merged; level += 1
    positives = acc.positives; negatives = acc.pixels - positives
    budgets = [int(np.floor(rate * negatives)) for rate in policy['fpr_budgets']]
    def point(threshold, tp, fp):
        return dict(threshold=threshold, tp=tp, fp=fp, recall=tp/positives if positives else None,
                    precision=tp/(tp+fp) if tp+fp else None,
                    fpr=fp/negatives if negatives else None)
    best = [point(None, 0, 0) for _ in budgets]
    curve = [point(None, 0, 0)]; crossing = 1; resolution = policy['recall_curve_intervals']
    tp = seen = 0; ap = 0.
    for score, count, positive in _merge(paths):
        tp += positive; seen += count; fp = seen-tp
        if positives:
            ap += (positive/positives)*(tp/seen)
        current = point(score, tp, fp)
        for i, budget in enumerate(budgets):
            if fp <= budget and tp > best[i]['tp']:
                best[i] = current
        if positives and crossing <= resolution and tp/positives >= crossing/resolution:
            curve.append(current)
            while crossing <= resolution and tp/positives >= crossing/resolution:
                crossing += 1
    if seen and curve[-1] != point(score, tp, seen-tp):
        curve.append(point(score, tp, seen-tp))
    acc._release(list(acc.owned)); acc.finalized = True
    return dict(domain='valid & observed_J==0', positives=positives, negatives=negatives,
                average_precision=ap if positives else None,
                budgets=[dict(fpr_budget=rate, fp_budget=budget, **row) for rate,budget,row in
                         zip(policy['fpr_budgets'],budgets,best)], curve=curve,
                curve_policy='exact global-tie scan; downsampled at fixed recall crossings',
                interpretation='same-panel descriptive envelope; optimistic if used to choose and report a threshold',
                operational_threshold_selected=False)
