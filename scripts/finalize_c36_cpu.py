"""Seal additive C36 CPU progress, reusing prior preservation inventories."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from geofirex.protocol_v2 import sha256
from geofirex.v2_c36_readiness import frozen_inputs

ROOT=Path.cwd().resolve()
OUT=ROOT/'reports/c36_cpu_20261121_v1'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    if (OUT/'delivery_manifest.json').exists():raise FileExistsError('Preserve the sealed CPU report')
    previous=read(OUT/'preparation_before.json')
    changed=[name for name,digest in previous['sha256'].items() if sha256(ROOT/name)!=digest]
    if changed:raise ValueError('Pre-existing content changed: '+str(changed))
    frozen=frozen_inputs(ROOT)
    external=[]
    for a in read(ROOT/'reports/c36_feasibility_20261121_v1/c8_archive_verification.json')['archives']:
        p=Path(a['path']);s=p.stat()
        passed=(s.st_size,s.st_mtime_ns)==(a['bytes'],a['mtime_ns']) and sha256(p)==a['sha256']
        if not passed:raise ValueError('C8 archive altered')
        external.append(dict(path=str(p),sha256=a['sha256'],unchanged=True))
    tests_path=ROOT/'reports/c36_cpu_tests_20261121_v5/tests.xml'
    xml=ET.parse(tests_path).getroot();cases=xml.findall('.//testcase')
    if xml.findall('.//failure') or xml.findall('.//error') or len(cases)!=122:
        raise ValueError('Expected executed 122-test CPU PASS required')
    geometry=read(OUT/'geometry_headers.json')
    if geometry['status']!='PASS' or geometry['passed']!=6913 or geometry['pixel_reads']!=0:
        raise ValueError('Header coverage incomplete')
    meta=read(ROOT/'reports/protocol_v2_20261118/index/index.json')
    val=[e for e,r in meta['event_roles'].items() if r=='val']
    val.sort(key=lambda e:hashlib.sha256(('geofirex-threshold-v1:'+e).encode()).hexdigest())
    assert len(val)==80
    with (OUT/'proposal_validation_folds.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['event_id','global_role_unchanged','proposal_fold'])
        writer.writerows((e,'val',i%5) for i,e in enumerate(val))
    search_paths=[ROOT/'configs/multimodal_availability.example.json',
        ROOT/'reports/scientific_improvements_v1/feature_availability_matrix.csv',
        ROOT/'reports/scientific_improvements_v1/sources.json',
        ROOT/'reports/protocol_v2_20261118/index/index.json',
        ROOT/'reports/protocol_v2_execution/dataset-sources.json']
    logs=Path('D:/GeoFire-X/WildfireSpreadTS/logs/zip-v1')
    search_paths+=list(logs.glob('attempt-*.json'))+[logs/'body-transfer.jsonl']
    save('asset_evidence_search.json',dict(status='BLOCKED',
        inspected=[dict(path=str(p),sha256=sha256(p)) for p in search_paths],
        header_inputs_inspected=6913,header_global_tag_keys=geometry['tag_keys'],
        result='No reviewed, per-asset, historically dated release/composite dossier in these sources; transfer logs certify archive movement, not source product availability at J',
        missing_by_band={str(b):missing for b,missing in [(0,'asset/version/acquisition/publication'),
            (1,'asset/version/acquisition/publication'),(2,'asset/version/acquisition/publication'),
            (3,'asset/version/composite_end/publication'),(4,'asset/version/composite_end/publication'),
            (16,'asset/version/product_year/annual_interval/publication')]},
        required_added_band_keys=110106,verified_added_band_keys=0,
        no_download=True,claim_scope='Only the named existing sources, not an assertion about every file on the computer'))
    rows=[dict(blocker='A',status='BLOCKED',implemented_component='PASS',
               evidence='causal_status.json; asset_evidence_search.json',remaining='Actual reviewed causal assets for all six added bands'),
          dict(blocker='B',status='BLOCKED',implemented_component='PASS',
               evidence='geometry_headers.json; geometry_headers.csv; CPU tests',remaining='Real immutable causal overlay, after A'),
          dict(blocker='C',status='BLOCKED',implemented_component='PASS',
               evidence='CPU train-only/chunking/I-O guard tests',remaining='Real train-only statistics after A/B'),
          dict(blocker='D',status='BLOCKED',implemented_component='PASS',
               evidence='CPU target/mask/shared-channel/domain-digest tests; expected_contract.json',remaining='Exhaustive real 146782-tile C36 parity after A/B/C'),
          dict(blocker='E',status='BLOCKED',implemented_component='NOT_PREPARED',
               evidence='No C36 execution module or GPU notebook created',remaining='Runner and qualified budget after A-D')]
    save('qualification_table.json',rows)
    git=['git','-c','safe.directory='+ROOT.as_posix()]
    branch=subprocess.check_output(git+['branch','--show-current'],text=True).strip()
    assert branch=='codex/c36-cpu-readiness-20261121'
    new_files=['src/geofirex/v2_c36_readiness.py','src/geofirex/v2_c36_overlay.py',
               'scripts/prepare_c36_cpu.py','scripts/finalize_c36_cpu.py','tests/test_v2_c36_cpu.py',
               'docs/C36_CPU_PREPARATION.md','configs/c8_propagation_proposal_v1.json']
    save('verification_finale.json',dict(status='PASS_FOR_EXECUTED_CPU_CHECKS',
        C36_decision='NO_GO',branch=branch,preexisting_files_checked=len(previous['sha256']),
        preexisting_files_changed=changed,frozen_identity=frozen,external_C8_archives=external,
        cpu_tests=dict(passed=len(cases),new_c36_cases=sum(c.attrib.get('classname','').endswith('test_v2_c36_cpu') for c in cases),
            checkpoint_roundtrip_deselected=1,junit=str(tests_path.relative_to(ROOT)),sha256=sha256(tests_path)),
        actual_geometry_headers_passed=6913,actual_C36_pixel_parity_executed=False,
        real_C36_statistics_computed=False,runner_prepared=False,
        no_real_raster_pixels_opened=True,test_2021_pixels_opened=0,gpu_used=False,
        training_executed=False,committed=False,pushed=False,
        added_files=[dict(path=p,sha256=sha256(ROOT/p)) for p in new_files],
        git_status=subprocess.check_output(git+['status','--short'],text=True)))
    save('decision_final.json',dict(run_id='B2_C36_V2_seed42',status='NO_GO',
        reason='Causal evidence unresolved; dependent real overlay, normalization, full parity and runner remain blocked',
        blocker_status={r['blocker']:r['status'] for r in rows},
        implemented_component_checks_passed=['V2 membership and evidence gates','exact source geometry',
            'train-only moments and I/O gates on fixtures','channel/target/mask/domain parity on fixtures'],
        training_authorized=False,test_2021_reserved=True,
        next_authorized_action='CPU evidence validation when an actual primary-source dossier is available; otherwise CPU regression only',
        fallback_protocol='PROTOCOLE_SECOURS.md',fallback_experiments_executed=False))
    files=new_files+[str(p.relative_to(ROOT)) for p in sorted(OUT.iterdir()) if p.is_file()]
    files+=['reports/c36_cpu_tests_20261121_v5/tests.xml']
    save('delivery_manifest.json',dict(schema='geofirex-c36-cpu-delivery-1',
        files=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha256(ROOT/p)) for p in files],
        excludes='Synthetic pytest working directories and earlier diagnostic XML are not scientific data'))
    with (OUT/'delivery_manifest.sha256').open('x',encoding='ascii') as f:
        f.write(sha256(OUT/'delivery_manifest.json')+'  delivery_manifest.json\n')
    print(json.dumps(dict(C36='NO_GO',tests_passed=len(cases),header_geometry_passed=6913,
        preserved_files=len(previous['sha256']),frozen_files=frozen['files'],
        external_archives_preserved=len(external),files_delivered=len(files)),indent=2))


if __name__=='__main__':main()
