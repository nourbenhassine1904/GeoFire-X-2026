"""Stdlib-only preservation seal/package: never loads a model or pixel array."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/newpos4_thresholds_20261121_v1'


def read(p):return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def save(p,data):
    with p.open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def git(*args):
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),*args],cwd=ROOT,text=True).strip()


def verify_zip(path,expected):
    if sha(path)!=expected:raise ValueError('ZIP global SHA differs')
    with zipfile.ZipFile(path) as z:
        rows=read_zip_manifest(z);names=[r['path'] for r in rows]
        if len(names)!=len(set(names)) or set(z.namelist())!=set(names)|{'artifact_manifest.json'}:raise ValueError('ZIP membership differs')
        for r in rows:
            h=hashlib.sha256();size=0
            with z.open(r['path']) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block);size+=len(block)
            if h.hexdigest()!=r['sha256'] or size!=r['bytes']:raise ValueError('ZIP member differs')


def read_zip_manifest(z):return json.loads(z.read('artifact_manifest.json'))['files']


def main():
    if (OUT/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed mission')
    before=read(OUT/'preservation_before.json')['files']
    for name,digest in before.items():
        if sha(ROOT/name)!=digest:raise ValueError('Pre-existing file altered: '+name)
    pins=read(OUT/'analysis_inputs.json')
    for name,digest in pins['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Analysis source altered after qualification: '+name)
    archives=read(OUT/'archive_verification.json')['archives']
    for row in archives:
        p=Path(row['path']);s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']) or sha(p)!=row['sha256']:raise ValueError('Source archive altered')
    partial=read(OUT/'cpu_analysis/partial.json')
    if (partial['status']!='PARTIAL_NO_SELECTION' or (OUT/'cpu_analysis/complete.json').exists()
            or (OUT/'cpu_analysis/crossfit/thresholds_locked.json').exists()):raise ValueError('Unexpected local analysis state')
    partialzip=next(OUT.glob('cpu_analysis-PARTIAL-*.zip'))
    verify_zip(partialzip,partialzip.with_suffix('.sha256').read_text().split()[0])
    xml_path=ROOT/'reports/newpos4_threshold_tests_v1/tests.xml';xml=ET.parse(xml_path).getroot();cases=xml.findall('.//testcase')
    if len(cases)!=100 or any(xml.findall('.//'+k) for k in ('failure','error','skipped')):raise ValueError('100 executed PASS required')
    if git('branch','--show-current')!='codex/newpos4-thresholds-20261121':raise ValueError('Wrong branch')
    subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'diff','--check'],cwd=ROOT,check=True)
    frozen=read(ROOT/'reports/protocol_v2_execution/execution_manifest.json')['files']
    if len(frozen)!=77 or any(sha(ROOT/name)!=h for name,h in frozen.items()):raise ValueError('Frozen identities altered')
    receipt=dict(status='ARCHIVES_PREDICTIONS_PASS_THRESHOLD_ASSESSMENT_BLOCKED_RAM_NO_PROMOTION',
        preserved_files=len(before),original_scientific_files=77,changed_protected_files=[],archives_unchanged=archives,
        predictions_tiles=30882,prediction_events=80,tests=dict(passed=100,failed=0,skipped=0,receipt=xml_path.relative_to(ROOT).as_posix(),sha256=sha(xml_path)),
        local_threshold_analysis='PARTIAL_NO_SELECTION_BEFORE_HELDOUT_ASSESSMENT',
        partial_archive=str(partialzip.relative_to(ROOT)),partial_archive_sha256=sha(partialzip),
        notebook_CPU_prepared=True,notebook_launched=False,GPU_launched=False,training_executed=False,
        inference_executed=False,test_pixels_opened=0,promotion_authorized=False,
        branch=git('branch','--show-current'),head=git('rev-parse','HEAD'),git_diff_check='PASS',
        git_status=git('status','--short'),disks={d:dict(shutil.disk_usage(d)._asdict()) for d in ('C:/','D:/')})
    save(OUT/'verification_finale.json',receipt)
    save(OUT/'decision.json',dict(archive_integrity='PASS',prediction_coverage_and_domain='PASS',CPU_unit_tests='PASS_100',
        NEWPOS4_original_promotion='NO_GO',threshold_assessment='BLOCKED_RAM',
        reason='Resource guard stopped during calibration, before any held-out assessment; AP/Brier model gates already fail independently of threshold',
        next_action='Run prepared standalone notebook on Kaggle CPU with existing COMPLETE output; no raw datasets or inference',
        NEWPOS2='PROSPECTIVE_PROTOCOL_ONLY_NOT_AUTHORIZED',causal_channel_additions='BLOCKED_PENDING_PER_ASSET_EVIDENCE'))
    names=set(pins['files'])|{p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    names|={xml_path.relative_to(ROOT).as_posix(),'configs/c8_newpos2_prereg_20261121_v1.json',
        'notebooks/GEOFIREX_NEWPOS4_THRESHOLDS_CPU_DISABLED.ipynb','scripts/inspect_newpos4_saved.py',
        'scripts/prepare_newpos4_analysis.py','scripts/report_newpos4_saved.py','scripts/finalize_newpos4_saved.py'}
    save(OUT/'delivery_manifest.json',dict(schema='newpos4-saved-analysis-delivery-1',
        files={name:sha(ROOT/name) for name in sorted(names)},status=receipt['status'],source_archives_embedded=False))
    digest=sha(OUT/'delivery_manifest.json')
    with (OUT/'delivery_manifest.sha256').open('x',encoding='ascii') as f:f.write(digest+'  delivery_manifest.json\n')
    names|={(OUT/'delivery_manifest.json').relative_to(ROOT).as_posix(),(OUT/'delivery_manifest.sha256').relative_to(ROOT).as_posix()}
    artifact=OUT.parent/(OUT.name+'-PREPARATION-'+uuid.uuid4().hex[:8]+'.zip')
    members=[dict(path=name,bytes=(ROOT/name).stat().st_size,sha256=sha(ROOT/name)) for name in sorted(names)]
    with artifact.open('xb') as f:
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for row in members:z.write(ROOT/row['path'],row['path'])
            z.writestr('artifact_manifest.json',json.dumps(dict(stage='PREPARATION_NO_GPU',files=members)))
    artifact_sha=sha(artifact);verify_zip(artifact,artifact_sha)
    with artifact.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(artifact_sha+'  '+artifact.name+'\n')
    print(json.dumps(dict(status=receipt['status'],preserved_files=len(before),tests=100,
        artifact=str(artifact),artifact_sha256=artifact_sha,delivery_sha256=digest,
        disk_free={d:row['free'] for d,row in receipt['disks'].items()}),indent=2))


if __name__=='__main__':main()
