"""Quick manifest/config metadata inspection; no raster reads or network."""
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

base=Path('reports/recovery_20261107_v3/finalized/deployment/bundle')
for number in range(2,9):
    lot=json.loads((base/f'metadata/lot-{number:03d}.json').read_text())
    dates=Counter(str(dt.date.fromisoformat(Path(f['path']).stem).year) for f in lot['files'])
    print(json.dumps(dict(number=number,role=lot['role'],files=len(lot['files']),events=len(lot['events']),
        bytes=lot['bytes'],observation_years=dict(dates))))
print('RAW_ROOT_AVAILABLE',Path('D:/GeoFire-X/WildfireSpreadTS/rasters').is_dir())
for number in (2,8):
    lot=json.loads((base/f'metadata/lot-{number:03d}.json').read_text())
    p=Path('D:/GeoFire-X/WildfireSpreadTS/rasters')/lot['files'][0]['path']
    print('FIRST_RASTER',number,p.is_file(),p.stat().st_size if p.is_file() else None)
