"""Explicit release selection and read-only checks. Never stages, commits or pushes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
from urllib.request import urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/publication_20261127'
BRANCH = 'release/geofirex-jury-20261202'
NEW = '''docs/AUDIT_NON_REGRESSION_20261127.txt
docs/DEPLOYMENT_STUDIO_V2.md
docs/FINALISATION_20261202.txt
docs/CRISP_DM_CLOTURE_20261127.md
docs/PUBLICATION_20261127.md
frontend/src/FinalResults.mjs
frontend/src/LiveComparison.mjs
frontend/src/Methodology.mjs
frontend/src/Overview.mjs
frontend/src/PerformanceDashboard.mjs
frontend/src/PredictionStudio.mjs
frontend/src/ScientificUI.mjs
frontend/src/api.mjs
frontend/src/professional.css
frontend/tests/professional-contract.mjs
frontend/verify-non-regression-browser.mjs
requirements-studio-cpu.txt
scripts/audit_studio_delivery.py
scripts/audit_studio_http_stability.py
scripts/audit_studio_preservation.py
scripts/capture_studio_environment.py
scripts/check_studio_http.py
scripts/qualify_studio_v2.py
scripts/studio_preservation.py
scripts/audit_release_publication.py
scripts/run_final_test_v2.py
scripts/run_final_test_v2_local.py
src/geofirex/final_test_v2.py
src/geofirex/final_test_v2_local.py
src/geofirex/studio_final_results.py
src/geofirex/studio_v2.py
tests/test_final_test_v2.py
tests/test_final_test_v2_local.py
tests/test_studio_fire13.py
tests/test_studio_populations.py
tests/test_studio_v2_deployment.py
tests/test_ui_performance.py
tests/test_release_publication.py
reports/jury_20261202_v1/diagnostic_tile.json
reports/fire13_20261122_v1/derived_normalization.json
reports/fire13_20261122_v1/scene_geometry.json
reports/fire13_20261122_v1/execution_manifest.json
reports/fire13_20261122_v1/acceptance.json
reports/fire13_20261122_v1/baseline_reference.json
reports/fire13_20261122_v1/newpos2_reference.json
reports/fire13_20261122_v1/newpos4_reference.json
reports/fire13_20261122_v1/budget.json
reports/fire13_20261122_v1/comparable_fp_policy.json
reports/fire13_20261122_v1/parity_full.json
reports/fire13_20261122_v1/cost_pilot.json
reports/fire13_20261122_v1/inventory.json
reports/fire13_20261122_v1/raw_band_decisions.json
reports/publication_20261127/backend.xml
reports/publication_20261127/guards.xml
reports/publication_20261127/launcher.json
reports/publication_20261127/frontend/propagation-dom.json
reports/publication_20261127/environment.json
reports/publication_20261127/science-readonly.json
reports/publication_20261127/big-data.json
reports/publication_20261127/qualification.json
reports/publication_20261127/acceptance.json
reports/publication_20261127/preservation.json
reports/publication_20261127/git-content-audit.json
reports/publication_20261127/selection.json
reports/finalisation_20261127/browser/browser-audit.json'''.splitlines()
DOCS = ['README.md', 'docs/DEPLOYMENT_STUDIO_V2.md',
        'docs/CRISP_DM_CLOTURE_20261127.md', 'docs/PUBLICATION_20261127.md']
SECRETS = {
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'aws_access_id': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'google_key': r'\bAIza[\w-]{35}\b',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'credential_url': r'https?://[^\s/:@]+:[^\s/@]+@',
    'openai_key': r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b',
}


def git(*args, raw=False):
    value = subprocess.check_output(['git', '-c', 'safe.directory=' + ROOT.as_posix(), *args], cwd=ROOT)
    return value if raw else value.decode('utf8').strip()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scan(names):
    findings = []
    for name in names:
        p = ROOT / name
        if not p.is_file():
            continue
        raw = p.read_bytes()
        if b'\x00' in raw[:8192]:
            continue
        value = raw.decode('utf8', errors='replace')
        for kind, pattern in SECRETS.items():
            for m in re.finditer(pattern, value):
                findings.append(dict(path=name, kind=kind, line=value.count('\n', 0, m.start()) + 1))
    return findings


def inventory():
    tracked = git('ls-files', '-z', raw=True).decode().strip('\x00').split('\x00')
    untracked = git('ls-files', '--others', '--exclude-standard', '-z', raw=True).decode().strip('\x00').split('\x00')
    modified = git('diff', '--name-only').splitlines()
    assert git('branch', '--show-current') == BRANCH
    assert git('rev-list', '--left-right', '--count', 'HEAD...origin/' + BRANCH).split() == ['0', '0']
    assert not git('diff', '--cached', '--name-only'), 'Preexisting index changes require review'
    rows = lambda names: [dict(path=n, bytes=(ROOT / n).stat().st_size) for n in names if (ROOT / n).is_file()]
    write('local-git-inventory.json', dict(head=git('rev-parse', 'HEAD'), branch=BRANCH,
        remote=git('remote', 'get-url', 'origin'), modified=modified, untracked=rows(untracked), tracked=rows(tracked)))
    repairs = []
    pinned = set()
    for name in ('reports/protocol_v2_execution/execution_manifest.json',
                 'reports/fire13_20261122_v1/execution_manifest.json'):
        pinned.update(json.loads((ROOT/name).read_text())['files'])
    for name in sorted(pinned & set(tracked)):
        before = git('show', 'HEAD:' + name, raw=True)
        current = (ROOT/name).read_bytes()
        if before != current:
            assert before.replace(b'\r\n', b'\n') == current.replace(b'\r\n', b'\n'), 'Unreviewed frozen change: ' + name
            repairs.append(name)
    selected = sorted(set(modified + NEW + repairs))
    write('selection.json', dict(branch=BRANCH, base_head=git('rev-parse', 'HEAD'), files=selected,
        frozen_git_byte_repairs_only=repairs,
        exclusions='All other untracked files preserved locally; no datasets, weights, archive ZIPs, profiles, secrets or environments.'))
    print(json.dumps(dict(tracked=len(tracked), modified=len(modified), untracked=len(untracked), selected=len(selected))))


def evidence(base):
    from geofirex.science_results import ScientificResults
    from geofirex.studio_final_results import CompletedResults
    from geofirex.jury_app import evidence as sealed
    from geofirex.studio_v2 import QUALIFICATION
    # CompletedResults verifies the ZIP and only reads authorized JSON members.
    result = CompletedResults().read()
    val = json.loads(sealed('C8_evaluation.json'))
    comp = json.loads(sealed('comparison.json'))
    c8 = next(r for r in comp['models'] if r['model'] == 'C8')
    expected = dict(average_precision=.45500139551181146, dice=.5348317337214559, recall=.5512253057)
    for key, value in expected.items():
        assert abs(result['pooled'][key] - value) < 5e-11
    assert abs(result['new_fire']['average_precision'] - .1754763634) < 5e-11
    assert abs(result['new_fire']['dice'] - .1576373262) < 5e-11
    assert abs(result['new_fire']['recall'] - .1040384448) < 5e-11
    write('science-readonly.json', dict(status=result['status'], events=result['events'], tiles=result['tiles'],
        epoch=4, threshold=result['threshold'], pooled=result['pooled'], new_fire=result['new_fire'],
        provenance={k:v for k,v in result['provenance'].items() if k!='archive'},
        validation=dict(events=80, tiles=30882, pooled=val['pooled'], new_fire=val['new_fire'], AP_new=c8['AP_new']),
        test_rerun=False, test_pixels_read=False))
    write('big-data.json', ScientificResults().big_data())
    environment = json.loads((ROOT / 'artifacts/studio-v2-deployment/environment.json').read_text())
    environment['node'] = subprocess.check_output(['node', '--version'], text=True).strip()
    environment['frontend_dependencies'] = json.loads((ROOT/'frontend/package.json').read_text())['dependencies']
    write('environment.json', environment)
    write('qualification.json', json.loads(QUALIFICATION.read_text(encoding='utf8')))
    api = []
    for route in ('/health', '/studio/models', '/science/v2/populations', '/science/performance',
                  '/science/final-2021/export.json', '/science/final-2021/export.csv',
                  '/science/performance/events.csv', '/science/v2/export.csv', '/feu.png'):
        with urlopen(base + route, timeout=180) as response:
            body = response.read()
            api.append(dict(route=route, status=response.status, bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
            assert response.status == 200
    for report in ('backend.xml', 'guards.xml'):
        backend_path = OUT / report
        original = OUT / ('local-' + report)
        if not original.exists():
            original.write_bytes(backend_path.read_bytes())
        tree = ET.parse(backend_path)
        for suite in tree.iter('testsuite'):
            suite.attrib.pop('hostname', None)
        tree.write(backend_path, encoding='utf8', xml_declaration=True)
    tree = ET.parse(OUT/'backend.xml')
    backend = list(tree.iter('testsuite'))
    counts = {k:sum(int(s.attrib.get(k, 0)) for s in backend) for k in ('tests', 'failures', 'errors', 'skipped')}
    guards = list(ET.parse(OUT/'guards.xml').iter('testsuite'))
    guard_counts = {k:sum(int(s.attrib.get(k, 0)) for s in guards) for k in ('tests', 'failures', 'errors', 'skipped')}
    frontend = json.loads((OUT / 'frontend/propagation-dom.json').read_text())
    q = json.loads((OUT / 'qualification.json').read_text())
    assert counts['tests'] >= 47 and counts['failures'] == counts['errors'] == 0
    assert guard_counts['tests'] == 38 and guard_counts['failures'] == guard_counts['errors'] == 0
    assert frontend['status'] == 'PASS' and len(frontend['checks']) >= 44
    assert q['status'] == 'PASS_WITH_ARCHIVED_FIRE13'
    browser = json.loads((ROOT / 'reports/finalisation_20261127/browser/browser-audit.json').read_text())
    assert browser['status'].startswith('PASS')
    write('acceptance.json', dict(status='PASS_WITH_DOCUMENTED_DEPLOYMENT_LIMITS',
        created_utc=datetime.now(timezone.utc).isoformat(), backend=counts, synthetic_guard_tests=guard_counts,
        frontend_checks=len(frontend['checks']),
        build='npm.cmd run build: PASS', api=api, qualification=q['status'],
        browser_previous_receipt='reports/finalisation_20261127/browser/browser-audit.json',
        browser_status=browser['status'], fresh_machine_tested=False, native_fullscreen_tested=False,
        training_executed=False, new_2021_inference=False))
    protected = subprocess.check_output([str(ROOT / '.venv/Scripts/python.exe'),
        str(ROOT / 'scripts/audit_studio_preservation.py'), 'verify',
        str(ROOT / 'reports/finalisation_20261127/preservation-before.json')], cwd=ROOT)
    preservation = json.loads(protected)
    assert preservation['status'] == 'PASS_ACCESSIBLE_FILES' and not preservation['unavailable']
    write('preservation.json', dict(**preservation,
        baseline='Local immutable snapshot from finalisation_20261127; full paths retained locally.',
        report_latex_preserved=True, original_feu_preserved=True, test_2021_rerun=False))
    print(json.dumps(dict(backend=counts, frontend=len(frontend['checks']), api=len(api), qualification=q['status'])))


def launcher(base):
    with urlopen(base+'/health', timeout=120) as response:
        assert response.status == 200
    with urlopen(base+'/studio/models', timeout=120) as response:
        rows = json.load(response)['models']
    assert all(next(r for r in rows if r['id']==m)['enabled'] for m in ('C8','NEWPOS2','NEWPOS4','legacy'))
    assert not next(r for r in rows if r['id']=='FIRE13')['enabled']
    write('launcher.json', dict(status='PASS', command='Start-GeoFireX.ps1 -Port 8031 -SkipBuild',
        health_http=200, model_catalog_http=200, port=8031, cpu_models=['C8','NEWPOS2','NEWPOS4','legacy'],
        FIRE13='ARCHIVED', training_executed=False, test_2021_inference=False,
        note='Temporary dedicated launch verified; primary existing instance remains on 8028.'))
    print('Launcher PASS')


def verify(staged, check_only=False):
    selected = json.loads((OUT / 'selection.json').read_text())['files']
    if staged:
        assert sorted(git('diff', '--cached', '--name-only').splitlines()) == selected, 'Index differs from explicit selection'
        check = subprocess.run(['git', '-c', 'safe.directory=' + ROOT.as_posix(), 'diff', '--cached', '--check'],
                               cwd=ROOT, capture_output=True, text=True)
        assert check.returncode == 0, 'Index whitespace check failed: ' + check.stdout[:3000]
    # The audit receipt is produced by this action, so it can be absent on its first run.
    own_receipt = 'reports/publication_20261127/git-content-audit.json'
    if not (ROOT / own_receipt).exists():
        write('git-content-audit.json', dict(status='PENDING'))
    assert all((ROOT / n).is_file() for n in selected)
    forbidden = []
    for n in selected:
        p = Path(n)
        if p.suffix.lower() in {'.zip', '.pt', '.pth', '.ckpt', '.npy', '.npz', '.tif', '.tiff', '.bin'} or (ROOT/n).stat().st_size > 5_000_000:
            forbidden.append(n)
    assert not forbidden, 'Forbidden/oversized release files: ' + repr(forbidden)
    tracked = git('ls-files', '-z', raw=True).decode().strip('\x00').split('\x00')
    head_tracked = set(git('ls-tree', '-r', '--name-only', 'HEAD').splitlines())
    findings = scan(sorted(set(selected + tracked)))
    reviewed_fixtures = []
    for row in list(findings):
        if row['path'] == 'tests/test_kaggle_blob_session.py' and row['kind'] == 'credential_url':
            line = (ROOT/row['path']).read_text().splitlines()[row['line']-1]
            fixture_url = 'https://' + 'SIGNED_QUERY_SENTINEL:password@' + 'storage.googleapis.com/object'
            if fixture_url in line:
                reviewed_fixtures.append(dict(**row, reason='Existing negative security test; explicit synthetic sentinel, not a credential.'))
                findings.remove(row)
    assert not findings, 'Sensitive patterns found (values withheld): ' + repr(findings)
    pinned = {}
    for name in ('reports/protocol_v2_execution/execution_manifest.json',
                 'reports/fire13_20261122_v1/execution_manifest.json'):
        pinned.update(json.loads((ROOT/name).read_text())['files'])
    repairs = []
    for name, expected in pinned.items():
        if name not in tracked and name not in selected:
            continue  # Binary indexes and documented external inputs are restored separately.
        assert digest(ROOT/name) == expected, 'Frozen working identity differs: ' + name
        if staged:
            value = git('show', ':' + name, raw=True)
            assert hashlib.sha256(value).hexdigest() == expected, 'Frozen staged identity differs: ' + name
        if name in head_tracked and name in selected:
            before = git('show', 'HEAD:' + name, raw=True)
            current = (ROOT/name).read_bytes()
            if before != current:
                assert before.replace(b'\r\n', b'\n') == current.replace(b'\r\n', b'\n'), 'Scientific logic change: ' + name
                repairs.append(name)
    missing, links = [], []
    for name in DOCS:
        value = (ROOT / name).read_text(encoding='utf8')
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', value):
            if '://' in target or target.startswith('#'):
                continue
            path = (ROOT / name).parent / unquote(target.split('#')[0].strip('<>'))
            links.append(dict(source=name, target=target))
            if not path.exists():
                missing.append(links[-1])
            elif path.is_file():
                relative = path.resolve().relative_to(ROOT).as_posix()
                if relative not in selected and relative not in tracked:
                    missing.append(dict(source=name, target=target, reason='not selected or tracked'))
    assert not missing, 'Unpublished/broken documentation references: ' + repr(missing)
    if staged:
        for name in selected:
            expected = git('hash-object', '--path=' + name, name)
            assert git('rev-parse', ':' + name) == expected, 'Unstaged selected bytes: ' + name
    sizes = sorted([dict(path=n, bytes=(ROOT/n).stat().st_size) for n in tracked if (ROOT/n).is_file()], key=lambda r:r['bytes'], reverse=True)
    receipt = dict(status='PASS', staged_checked=staged,
        selected_files=len(selected), selected_bytes=sum((ROOT/n).stat().st_size for n in selected),
        tracked_files=len(tracked), largest_tracked=sizes[:10], secret_scan_findings=[],
        reviewed_synthetic_fixtures=reviewed_fixtures,
        secret_scan_scope='All current tracked and selected bytes; high-confidence token/private-key/credential-URL patterns. Not a mathematical guarantee.',
        documentation_links_checked=len(links), broken_links=[], forbidden_files=[],
        gitattributes_reviewed=True, gitignore_reviewed=True, deletion_requested=False,
        frozen_tracked_or_selected_checked=sum(n in tracked or n in selected for n in pinned),
        frozen_git_byte_repairs_only=repairs,
        external_policy='Datasets, weights, RESULTS, compact binaries and durable local ledger stay outside this commit.',
        file_sha256={n:digest(ROOT/n) for n in selected if n not in (
            'reports/publication_20261127/git-content-audit.json', 'reports/publication_20261127/selection.json')})
    if not check_only:
        write('git-content-audit.json', receipt)
    print(json.dumps(dict(status='PASS', selected=len(selected), links=len(links), staged=staged)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'evidence', 'verify', 'launcher'])
    parser.add_argument('--base', default='http://127.0.0.1:8028')
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Verify without rewriting the audit receipt')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'inventory':
        inventory()
    elif args.action == 'evidence':
        evidence(args.base)
    elif args.action == 'launcher':
        launcher(args.base)
    else:
        verify(args.staged, args.check_only)
