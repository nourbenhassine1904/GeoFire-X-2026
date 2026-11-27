"""Read all V2 metadata; compare ONLY the two authorized dates, then freeze."""
from pathlib import Path
import csv
import json
from collections import Counter
from geofirex.protocol_v2 import sha256, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/protocol_v2_execution'

def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'reports/protocol_v2_20261118'
    # Read complete files, including every row, without labels or raster access.
    for name in ('RAPPORT.md', 'PLAN_EXECUTION.md'):
        (source / name).read_text(encoding='utf8')
    candidates = list(csv.DictReader((source / 'cutoff_candidates.csv').open(encoding='utf8')))
    events = list(csv.DictReader((source / 'events.csv').open(encoding='utf8')))
    comparisons = []
    roles = {}
    for cutoff in ('2020-08-18', '2020-08-23'):
        assignment = {r['event_id']: ('test' if r['event_id'].startswith('2021/') else
                      'train' if r['last_date'] < cutoff else
                      'val' if r['first_date'] >= cutoff else 'purged') for r in events}
        counts = dict(Counter(assignment.values()))
        candidate = next(r for r in candidates if r['cutoff'] == cutoff)
        assert all(counts[k] == int(candidate[k]) for k in ('train', 'val', 'purged'))
        comparison = dict(cutoff=cutoff, events=counts,
            tiles={k: sum(int(r['tiles']) for r in events if assignment[r['event_id']] == k)
                   for k in ('train', 'val', 'purged')},
            train_last=max(r['last_date'] for r in events if assignment[r['event_id']] == 'train'),
            val_first=min(r['first_date'] for r in events if assignment[r['event_id']] == 'val'))
        assert comparison['train_last'] < comparison['val_first']
        comparisons.append(comparison); roles[cutoff] = assignment
    transitions = Counter((roles['2020-08-18'][e], roles['2020-08-23'][e]) for e in roles['2020-08-18'])
    protected = [p for p in source.rglob('*') if p.is_file()]
    protected += list((ROOT / 'configs/protocol_v2').glob('*'))
    protected += [ROOT / 'src/geofirex/protocol_v2.py', ROOT / 'src/geofirex/v2_evaluation.py']
    save(OUT / 'split_lock.json', dict(schema='v2-final-split-lock-1', date='2026-11-19',
        selected='2020-08-23', protocol='V2', comparisons=comparisons,
        transitions=[dict(before=a, after=b, events=n) for (a,b),n in sorted(transitions.items())],
        rationale='Preserve preregistered closest-80/20 rule, two additional train events and certified domains; accept nine fewer validation events and seven extra exclusions. 18 August has better retention; neither dominates all criteria.',
        selection_inputs='dates, event counts, tile counts, fixed geometry/domain rules only',
        labels_used=False, model_scores_used=False, further_split_exploration_allowed=False,
        protected_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(protected)}))
    print(json.dumps(comparisons, indent=2))

if __name__ == '__main__':
    main()
