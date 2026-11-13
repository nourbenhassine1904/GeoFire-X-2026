"""Inspect lot 001 date tokens and split metadata only; never open rasters."""
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path, PurePosixPath


def inspect():
    base = Path('reports/recovery_20261107_v3/finalized/deployment/bundle/metadata')
    lot = json.loads((base/'lot-001.json').read_text())
    with (base/'fold_00.csv').open() as stream:
        split = {r['event_id']:r['split'] for r in csv.DictReader(stream)}
    mismatches, outside_train = [], []
    years = Counter()
    for entry in lot['files']:
        p = PurePosixPath(entry['path'])
        year = dt.date.fromisoformat(p.stem).year
        years[str(year)] += 1
        row = dict(path=entry['path'], event=str(p.parent), folder_year=int(p.parts[0]),
                   observation_year=year, split=split.get(str(p.parent)))
        if year != int(p.parts[0]):
            mismatches.append(row)
        if year not in (2018, 2019):
            outside_train.append(row)
    result = dict(files=len(lot['files']), events=len(lot['events']),
        all_events_train=all(split.get(e)=='train' for e in lot['events']),
        observation_year_counts=dict(sorted(years.items())), mismatch_count=len(mismatches),
        mismatch_events=len({r['event'] for r in mismatches}), mismatches=mismatches,
        outside_train=outside_train, remote_certified=False)
    return result


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
