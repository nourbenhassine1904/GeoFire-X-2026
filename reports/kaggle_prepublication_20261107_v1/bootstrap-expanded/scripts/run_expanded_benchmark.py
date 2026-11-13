"""One approved expanded-corpus run, with hard limits and no publication."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from geofirex.job_control import supervise

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--out', required=True, help='New supervisor receipt directory')
    p.add_argument('--seconds', type=int, default=7200)
    a = p.parse_args()
    if not 1 <= a.seconds <= 7200:
        p.error('Maximum 7200 seconds per explicitly approved run')
    import yaml
    c = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    profile = json.loads(Path(a.profile).read_text())
    if c['corpus_scope'] != 'train_validation' or c['device'] != 'cuda' or c['epochs'] != 1:
        p.error('Only a single-epoch train/validation deployment is supported')
    if (profile['model'] != c['model'] or profile['normalizer_sha256'] != c['normalization_sha256']
            or profile['input_hashes']['manifest'] != __import__('hashlib').sha256(Path(c['manifest']).read_bytes()).hexdigest()):
        p.error('Profile does not match model, normalization and input manifest')
    # A provisional forecast is a launch gate, never a runtime guarantee.
    estimate = 1200 + 1.5*(profile['projected_epoch_train_seconds']+profile['validation_linear_projection_seconds'])
    if estimate > a.seconds:
        p.error(f'Forecast {estimate:.0f}s exceeds approved wall budget; review profiling and budget first')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(c['evaluation_scratch_root'])
    scratch.mkdir(parents=True, exist_ok=True)

    def resources():
        if shutil.disk_usage(root).free < 5_000_000_000 or shutil.disk_usage(scratch).free < 5_000_000_000:
            raise RuntimeError('Less than 5 GB reserve; existing checkpoints and AP diagnostics preserved')
        if sum(f.stat().st_size for f in root.rglob('*') if f.is_file()) > 15_000_000_000:
            raise RuntimeError('Run output namespace exceeds 15 GB')

    if shutil.disk_usage(scratch).free < 40_000_000_000:
        p.error('40 GB free scratch required for exact AP merges; no input corpus copying')
    if shutil.disk_usage(root).free < 20_000_000_000:
        p.error('20 GB free output filesystem required before launch')
    resources()
    with (out/'training.log').open('x', encoding='utf-8') as log:
        result = supervise([sys.executable, 'scripts/run_benchmark.py', 'train', '--config', a.config],
                           seconds=a.seconds, log=log, check=resources)
    (out/'supervisor.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(0 if result['status'] == 'FINISHED' else 1)
