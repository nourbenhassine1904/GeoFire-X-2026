"""Offline QC, nested acquisition budgets and Parquet -> PyTorch contracts.

No network, training, MLflow writes, or overwrites. Run from repository root.
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import yaml

from geofirex.corpus import json_new, sha256_file, verify_member
from geofirex.benchmark import event_assignments
from geofirex.scientific_scaleup import (acquisition_stages, audit_observations,
                                       load_window_manifest, summarize_audit, write_metadata_tables, feature_contracts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--root', default='.')
    p.add_argument('--manifest', default='data/manifests/real/manifest_wildfirespreadts.csv')
    p.add_argument('--index', default='data/manifests/real/corpus_zip_index_v1.json')
    p.add_argument('--reuse-root', action='append', default=[],
                   help='Additional local archive-layout roots: CRC inventory only, no claim of pixel QC')
    args = p.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    root = Path(args.root).resolve()
    git = ['git', '-c', f'safe.directory={root.as_posix()}']
    def git_read(*cmd):
        return subprocess.check_output(git+list(cmd), cwd=root).decode('utf-8').strip()
    untracked = git_read('ls-files', '--others', '--exclude-standard').splitlines()
    json_new(output/'git_before.json', dict(commit=git_read('rev-parse', 'HEAD'),
        branch=git_read('branch', '--show-current'), status=git_read('status', '--short'),
        preserved_untracked={name: sha256_file(root/name) for name in untracked
                             if (root/name).is_file() and not (root/name).resolve().is_relative_to(output.resolve())},
        tracked_diff=git_read('diff', '--', 'README.md')))
    rows = list(csv.DictReader(Path(args.manifest).open(encoding='utf-8', newline='')))
    index = json.loads(Path(args.index).read_text(encoding='utf-8'))
    from geofirex.cloud_corpus import validate_index
    validate_index(index,official=True)
    observations = audit_observations(rows, index, root)
    audit = summarize_audit(observations, index)
    audit['input_hashes'] = {k: sha256_file(v) for k,v in [('manifest',args.manifest),('index',args.index)]}
    json_new(output/'rarity.json', audit)
    json_new(output/'feature_contracts.json', feature_contracts())
    write_metadata_tables(observations,audit['aligned_windows'],output)
    loaded = pq.read_table(output/'windows.parquet').to_pylist()
    clean = [r for r in observations if r['status'] == 'ok']
    assignment = event_assignments(sorted({r['event_id'] for r in clean}), 0)
    for role in ('train', 'val'):
        for t in (1,3,5):
            load_window_manifest(loaded, clean, assignment, role, t)
    expected = {(e['event_id'], e['date']): e for e in index['entries']}
    verified = [expected[(r['event_id'],r['date'])]['path'] for r in clean]
    reuse_inventory = []
    for base_path in args.reuse_root:
        for e in index['entries']:
            path = Path(base_path)/e['path']
            if path.is_file():
                valid = verify_member(path,e)
                reuse_inventory.append(dict(path=e['path'],root=str(Path(base_path).resolve()),
                                            crc_verified=valid,bytes=path.stat().st_size,
                                            sha256=sha256_file(path)))
                if valid: verified.append(e['path'])
    json_new(output/'additional_local_inventory.json',reuse_inventory)
    stages = acquisition_stages(index, verified)
    for stage in stages:
        json_new(output/f"acquisition_{stage['target_events']}.json", stage)
    with (output/'events.csv').open('x', newline='', encoding='utf-8') as f:
        fields = ['event_id','split','observations','expected_observations','complete','positive_days',
                  'zero_detection_days','entirely_invalid_days','positive_pixels','valid_pixels','positive_percent']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(audit['events'])
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    events = audit['events']
    names = [e['event_id'] for e in events]
    axes[0].barh(names,[e['positive_percent'] or 0 for e in events])
    axes[0].set_xlabel('Detected fire / valid pixels (%)')
    axes[1].barh(names,[e['positive_days'] for e in events],label='With detection')
    axes[1].barh(names,[e['zero_detection_days'] for e in events],
                 left=[e['positive_days'] for e in events],label='No detection')
    axes[1].set_xlabel('Audited observations'); axes[1].legend()
    fig.suptitle('Acquired pilot only; no inference about unacquired rasters')
    fig.savefig(output/'rarity.png',dpi=160); plt.close(fig)
    # Existing runner configurations; partial corpus remains explicitly pilot.
    base = yaml.safe_load(Path('configs/runner_sample_gpu_pilot.yaml').read_text(encoding='utf-8'))
    base.update(protocol_version='annual-v2-runner', experiment='geofirex-scaleup-v1',
                qualification='pilot', device='cuda', amp=False, workers=0,
                output_root='artifacts/scientific-scaleup-runs-v1', storage_policy='persistent',
                tracking_uri='sqlite:///artifacts/scientific-scaleup-runs-v1/tracking.db',
                artifact_root='file:./artifacts/scientific-scaleup-runs-v1/mlflow_artifacts',
                data_root='.', manifest='REPLACE_WITH_CERTIFIED_STAGE_MANIFEST.csv',
                split='splits/benchmark/annual_v1/fold_00.csv', index=args.index)
    configs = output/'gpu_configs'; configs.mkdir()
    runs = []
    for level, seeds in [(1,[42]),(2,[42,43,44])]:
        for model in ('B0','B1','B2','B3'):
            for seed in (seeds if model != 'B0' else [42]):
                config = {**base,'model':model,'seed':seed}
                config.pop('model_kwargs',None)
                if model == 'B1': config['model_kwargs'] = dict(architecture='spatial_cnn_v2',width=16)
                if model in ('B2','B3'): config['model_kwargs'] = dict(upsampling='nearest_v1')
                name = f'L{level}_{model}_seed{seed}.yaml'
                (configs/name).write_text(yaml.safe_dump(config, sort_keys=False),encoding='utf-8')
                from geofirex.runner import load_config, runner_model
                from geofirex.unet_data import Normalizer
                validated = load_config(configs/name)
                model_instance = runner_model(validated, Normalizer([12,14],[0,0],[1,1],[1,1],[]))
                runs.append(dict(config=name, level=level, model=model,seed=seed,
                                 parameters=sum(p.numel() for p in model_instance.parameters()),
                                 updates=0 if model == 'B0' else 1280, approved=False))
    json_new(output/'gpu_plan.json',dict(status='NOT_APPROVED_NOT_SUBMITTED',runs=runs,
        level3='Deferred: modality declarations and harmonized T=1/3/5 required before generating runs',
        duration='Unknown until bounded throughput and full-scene validation profiling on acquired stage',
        quotas='Historical quota receipts are not current quota authorization',
        checkpoint_policy='Existing runner: best validation AP, optimizer/RNG resume; new output directories',
        limitation='Runner scientific qualification currently requires all 607 events, including test inventory; partial stages must remain pilot'))
    json_new(output/'execution.json',dict(elapsed_seconds=time.perf_counter()-started,
        observations=len(observations), parquet_window_rows=len(loaded), roundtrip='passed',
        scope='local CPU only; no transfer, no training, no MLflow writes',
        source_hashes={str(f):sha256_file(f) for f in (Path(__file__),Path('src/geofirex/scientific_scaleup.py'))},
        hashes={f.relative_to(output).as_posix():sha256_file(f) for f in output.rglob('*') if f.is_file()}))
    print(json.dumps(dict(statuses=audit['statuses'],by_split=audit['by_split'],
                         stages=[{k:s[k] for k in ('target_events','observations','additional_payload_bytes',
                                                   'additional_extracted_bytes','proposed_transfer_cap_bytes')}
                                 for s in stages]),indent=2))


if __name__ == '__main__':
    main()
