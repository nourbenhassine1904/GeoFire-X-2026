"""Verify existing archives and seal UI provenance. No inference or evaluation run."""
import json
import hashlib
from pathlib import Path
import zipfile
from geofirex.jury_app import evidence,ROOT
from geofirex import v2_execution as old
from geofirex.calibration_io import sha

OUT=ROOT/'reports/studio_propagation_20261123_v1'
if __name__=='__main__':
    dest=OUT/'visualization_sources.json'
    if dest.exists():raise FileExistsError('Preserve source receipt')
    sources=json.loads(evidence('comparison.json'))['sources'];inventory=[]
    for s in sources:
        p=Path(s['archive']);side=p.with_suffix('.sha256').read_text().split()[0]
        assert side==s['archive_sha256']
        old.verify_zip(p,side)
        with zipfile.ZipFile(p) as z:
            name='evaluation/predictions/predictions_manifest.json'
            m=json.loads(z.read(name)) if name in z.namelist() else None
            inventory.append(dict(**s,saved_spatial=m is not None,events=len(m['files']) if m else 0,
                tiles=m['tiles'] if m else 0,manifest_sha256=hashlib.sha256(z.read(name)).hexdigest() if m else None))
    hero='reports/unet_confirmation_20261112_v1/confirmation_cpu/B2_w10_holdout169/2020__fire_24332880_2020-09-04_raw.npz'
    split='reports/protocol_v2_20261118/split_summary.json'
    data=dict(archives=inventory,hero=dict(path=hero,sha256=sha(ROOT/hero),event_id='2020/fire_24332880',as_of='2020-09-03',target_date='2020-09-04'),
        v2_split=dict(path=split,sha256=sha(ROOT/split),roles=json.loads((ROOT/split).read_text())['roles']),
        new_inference=False,test2021_pixels_opened=0,scientific_metrics_modified=False)
    dest.write_text(json.dumps(data,indent=2),encoding='utf8')
    (OUT/'visualization_manifest.json').write_text(json.dumps({'visualization_sources.json':sha(dest)},indent=2),encoding='utf8')
    print(json.dumps(inventory,indent=2))
