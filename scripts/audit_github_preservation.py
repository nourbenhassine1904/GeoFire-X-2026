"""Read-only inventory and redacted secret checks; never stage, commit or push.

Run from the repository root. The first pass snapshots all untracked/ignored
files before ignore changes. Subsequent passes reuse that snapshot so exclusions
cannot hide files from review. Outputs under artifacts/ stay local.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'artifacts/github-preservation-audit'
REPORT = ROOT / 'reports/github_preservation_20261113'
GIT = ['git', '-c', 'safe.directory=' + ROOT.as_posix()]
TEXT = {'.py', '.ps1', '.mjs', '.js', '.jsx', '.ts', '.tsx', '.json', '.jsonl',
        '.csv', '.tsv', '.md', '.txt', '.xml', '.yaml', '.yml', '.toml', '.ini',
        '.cfg', '.ipynb', '.html', '.css', '.sql', '.sh', '.lock', '.gitignore'}
BINARY = {'.zip', '.gz', '.tar', '.7z', '.bin', '.pt', '.pth', '.ckpt', '.onnx',
          '.safetensors', '.npy', '.npz', '.tif', '.tiff', '.db', '.sqlite', '.sqlite3'}
ESSENTIAL_BINARY = {
    'reports/calibration_20261112_v1/GeoFireX_CALIBRATION_RESULTATS.zip',
    'reports/calibration_20261112_v1/calibration-code.zip',
    'reports/loss_ablation_20261112_v1/loss-ablation-code.zip',
    'reports/propagation_20261112_v1/propagation-code.zip',
    'reports/gpu_acceleration_20261111_v1/index/train.npy',
    'reports/gpu_acceleration_20261111_v1/index/val.npy',
    'artifacts/propagation-comparison/comparison.json',
    'artifacts/propagation-comparison/executed-training-code.zip',
}
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'provider_token': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{25,}|KGAT_[A-Za-z0-9_-]{15,})\b'),
    'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]{12,8192}\.[A-Za-z0-9_-]{12,8192}\.[A-Za-z0-9_-]{12,8192}\b'),
    'bearer_token': re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~-]{20,}'),
    'authenticated_url': re.compile(r'https?://[^\s/"\']+:[^\s/@"\']+@'),
    'signed_url': re.compile(r'(?i)[?&](?:X-Amz-Signature|X-Goog-Signature|sig|access_token|api_key)=[^\s&"\']{12,}'),
    'literal_credential': re.compile(r'''(?i)["']?(?:api[_-]?key|api[_-]?token|access[_-]?token|auth[_-]?token|secret[_-]?key|password|passwd|client[_-]?secret|kaggle_key)["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']'''),
    'email': re.compile(r'\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,24}\b'),
}


def git_files(*args):
    return [p for p in subprocess.check_output(GIT + list(args) + ['-z'], cwd=ROOT).decode().split('\0') if p]


def temporary(path):
    return any(p.startswith(('test-work', 'test_', 'pytest', 'tmp-', 'synthetic-test',
                             'essential-final', 'resume-check', 'continuous-check'))
               or p.endswith('-env') or p in {'__pycache__', '.pytest_cache', 'node_modules', '.venv',
                        '.ipynb_checkpoints', 'kaggle_sdk'} for p in Path(path).parts[:-1])


def reason(path, size, ignored=False):
    p = Path(path)
    if p.name.startswith('.env') and p.name != '.env.example': return 'local credentials/environment'
    if p.name in {'kaggle.json', 'credentials', 'id_rsa', 'id_ed25519'} or p.suffix in {'.pem', '.key'} or '.kaggle' in p.parts or '.aws' in p.parts: return 'local credentials'
    if temporary(path): return 'generated test workspace, cache or environment'
    if any(part.endswith('.egg-info') for part in p.parts): return 'generated Python packaging metadata'
    if p.name == 'memory.jsonl': return 'operational resource log; summary retained'
    if path in ESSENTIAL_BINARY or (path.startswith('reports/unet_confirmation_20261112_v1/confirmation_cpu/') and p.suffix == '.npz'): return ''
    if path.startswith('reports/gpu_acceleration_20261111_v1/publication-mount-test/'): return 'duplicate compact publication mount'
    if p.suffix == '.cjs': return 'generated bundled frontend verification output'
    if path.startswith('frontend/dist/'): return 'generated build'
    if path.startswith('mlruns/'): return 'MLflow experiment store; preserve off Git'
    if path.startswith(('data/raw/', 'data/interim/', 'data/processed/')) and p.name != '.gitkeep': return 'raw/derived local dataset'
    if p.suffix in BINARY or p.name.endswith(('.zip.bin', '.tar.gz.part')): return 'off-Git dataset, checkpoint, archive or numerical artifact'
    if p.suffix in {'.log', '.tmp', '.swp', '.part', '.pyc', '.pyo'}: return 'temporary output/log'
    if size > 10_000_000: return 'large metadata; preserve off Git with SHA-256'
    if ignored and path.startswith('artifacts/'): return 'local artifact; essential sources mirrored separately if needed'
    if ignored: return 'ignored local output'
    return ''


def sha(path, cache):
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(block)
        if path.stat().st_mtime_ns != stat.st_mtime_ns: raise RuntimeError('File changed during audit: ' + str(path))
        cache[key] = digest.hexdigest()
    return cache[key]


def scan_text(name, text, findings):
    lower = text.lower()
    needles = {
        'private_key': ['private key'], 'github_token': ['ghp_', 'gho_', 'ghu_', 'ghs_', 'ghr_', 'github_pat_'],
        'provider_token': ['sk-', 'akia', 'aiza', 'hf_', 'kgat_'], 'jwt': ['eyj'],
        'bearer_token': ['bearer'], 'authenticated_url': ['http'],
        'signed_url': ['signature=', 'sig=', 'access_token=', 'api_key='],
        'literal_credential': ['api', 'access', 'auth', 'secret', 'password', 'passwd', 'kaggle_key'], 'email': ['@'],
    }
    for kind, pattern in PATTERNS.items():
        if not any(n in lower for n in needles[kind]): continue
        for match in pattern.finditer(text):
            findings.append({'path': name, 'line': text.count('\n', 0, match.start()) + 1,
                             'rule': kind, 'fingerprint': hashlib.sha256(match.group().encode()).hexdigest()[:12]})


def write_csv(name, rows, fields):
    with (REPORT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--hash-excluded', action='store_true')
    args = parser.parse_args(); LOCAL.mkdir(parents=True, exist_ok=True); REPORT.mkdir(exist_ok=True)
    snapshot = LOCAL / 'initial_inventory.json'
    if snapshot.exists(): initial = json.loads(snapshot.read_text(encoding='utf-8'))
    else:
        initial = []
        for ignored, paths in [(False, git_files('ls-files', '--others', '--exclude-standard')),
                               (True, git_files('ls-files', '--others', '--ignored', '--exclude-standard'))]:
            for path in paths:
                p = ROOT / path
                if p.is_file(): initial.append(dict(path=path, bytes=p.stat().st_size, ignored=ignored))
        snapshot.write_text(json.dumps(initial, indent=2), encoding='utf-8')
    changed = git_files('diff', '--name-only', 'HEAD')
    current = git_files('ls-files', '--others', '--exclude-standard')
    known = {r['path']: r for r in initial}
    for path in changed + current:
        if (ROOT / path).is_file(): known[path] = dict(path=path, bytes=(ROOT/path).stat().st_size, ignored=False)
    findings, selected, excluded, skipped, errors = [], [], [], [], []
    cache_file = LOCAL/'hash_cache.json'
    cache = {tuple(k):v for k,v in json.loads(cache_file.read_text()).get('entries',[])} if cache_file.exists() else {}
    scanned = 0; archive_members = 0
    for i, row in enumerate(sorted(known.values(), key=lambda r:r['path'])):
        path, size = row['path'], row['bytes']; p = ROOT / path
        if i % 1000 == 0:
            print('Inventory progress',i,path,flush=True)
            cache_file.write_text(json.dumps({'entries':list(cache.items())}),encoding='utf-8')
        why = reason(path, size, row['ignored'])
        if path.startswith('artifacts/github-preservation-audit/'): why = 'local audit working files'
        # Generated reports are included by the final publication-plan builder.
        if path.startswith('reports/github_preservation_20261113/'): continue
        if not p.is_file(): errors.append(dict(path=path,error='missing since snapshot')); continue
        digest = ''
        if not why or (args.hash_excluded and not temporary(path) and not path.startswith(('frontend/dist/', 'mlruns/'))):
            try: digest = sha(p, cache)
            except OSError as e: errors.append(dict(path=path,error=type(e).__name__))
        out = dict(path=path,bytes=size,sha256=digest)
        if why: excluded.append(dict(**out,reason=why))
        else: selected.append(out)
        if temporary(path) or path.startswith(('frontend/dist/', 'mlruns/', 'artifacts/github-preservation-audit/')): continue
        if p.suffix in TEXT or p.name in {'.gitignore', '.env', 'credentials', 'kaggle.json'}:
            try:
                # Large textual manifests are scanned in bounded chunks, including overlap.
                with p.open('r',encoding='utf-8',errors='replace') as stream:
                    previous = ''
                    while chunk := stream.read(4*1024*1024):
                        scan_text(path, previous+chunk, findings); previous=chunk[-512:]
                scanned += 1
            except OSError as e: errors.append(dict(path=path,error=type(e).__name__))
        elif p.suffix == '.zip' and size < 100_000_000:
            # Dataset ZIPs contain rasters; inspect names first and only text members.
            try:
                with zipfile.ZipFile(p) as archive:
                    for member in archive.infolist():
                        if Path(member.filename).suffix in TEXT and member.file_size < 10_000_000:
                            scan_text(path+'::'+member.filename,archive.read(member).decode('utf-8',errors='replace'),findings)
                            archive_members += 1
            except (OSError,zipfile.BadZipFile) as e: errors.append(dict(path=path,error=type(e).__name__))
        elif not why: skipped.append(path)
        if i and i % 5000 == 0: print('Audited',i,'paths',flush=True)
    cache_file.write_text(json.dumps({'entries':list(cache.items())}),encoding='utf-8')
    write_csv('selected_files.csv',selected,['path','bytes','sha256'])
    write_csv('excluded_files.csv',excluded,['path','bytes','sha256','reason'])
    (LOCAL/'secret_findings.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
    summary=dict(initial_untracked_files=sum(not r['ignored'] for r in initial),
        initial_untracked_bytes=sum(r['bytes'] for r in initial if not r['ignored']),
        initial_ignored_files=sum(r['ignored'] for r in initial), selected_files=len(selected),
        selected_bytes=sum(r['bytes'] for r in selected),excluded_files=len(excluded),
        text_files_scanned=scanned,archive_text_members_scanned=archive_members,
        secret_findings=len(findings),secret_rules=dict(collections.Counter(f['rule'] for f in findings)),
        binary_selected_not_text_scanned=skipped,errors=errors,
        limits='Pattern scan is not a proof of absence. Cache/test workspaces and numerical binary payloads are not interpreted. Credential values are never written to reports.')
    (LOCAL/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='binary_selected_not_text_scanned'},indent=2),flush=True)


if __name__ == '__main__': main()
