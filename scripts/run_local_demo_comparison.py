"""Short real-data CPU diagnostics; never a substantial pilot or final test.

Reuse one existing two-epoch spatial CNN, then run the remaining baselines with
the same seed, target set and budget. No architecture selection lock is made.
"""
import argparse
import json
from pathlib import Path

from geofirex.corpus import json_new
from geofirex.runner import load_config, train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b1-run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reuse-records', action='store_true', help='Reuse recorded completed B0/B2/B3 runs after an interrupted report')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.reuse_records)
    runs = {'B1': Path(args.b1_run)}
    for model in ('B0', 'B2', 'B3'):
        if args.reuse_records:
            runs[model] = Path(json.loads((out/(model+'-run.json')).read_text())['run'])
            continue
        config = load_config('configs/runner_demo_local.yaml', {'model': model, 'model_kwargs': {}})
        runs[model] = train(config)
        json_new(out/(model+'-run.json'), {'run': str(runs[model])})
        if model == 'B0':
            reference = json.loads((runs['B0']/'summary.json').read_text())
            original = json.loads((runs['B1']/'summary.json').read_text())
            if reference['comparison_hash'] != original['comparison_hash']:
                runs['B1'] = train(load_config('configs/runner_demo_local.yaml'))
    summaries = {model: json.loads((run/'summary.json').read_text()) for model, run in runs.items()}
    assert len({s['comparison_hash'] for s in summaries.values()}) == 1
    assert len({s['evaluation_target_hash'] for s in summaries.values()}) == 1
    rows = []
    for model in ('B0', 'B1', 'B2', 'B3'):
        summary = summaries[model]
        validation = json.loads((runs[model]/f"validation_{summary['best_epoch']:04d}.json").read_text())
        rows.append({'model': model, 'run': str(runs[model]), 'run_id': summary['run_id'],
            'optimizer_updates': summary['optimizer_updates'], 'best_epoch': summary['best_epoch'],
            'seconds': summary['seconds'], 'threshold': summary['threshold'],
            'average_precision': summary['validation_average_precision'], 'validation': validation['micro']})
    json_new(out/'comparison.json', {'qualification': 'diagnostic', 'real_data': True,
        'training': 'CPU, seed 42, 2 epochs x 32 patches, B0 untrained',
        'corpus_counts': summaries['B1']['corpus_counts'], 'test_2021': 'locked; no final-test command used',
        'scientific_architecture_selection': False, 'comparison_hash': summaries['B1']['comparison_hash'],
        'rows': rows})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in ('B1', 'B2', 'B3'):
        history = summaries[model]['history']
        axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], marker='o', label=model)
        axes[1].plot([r['epoch'] for r in history], [r['validation_average_precision'] for r in history], marker='o', label=model)
    axes[0].set_ylabel('Training loss')
    axes[1].set_ylabel('Validation micro AP')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_xticks([1, 2])
        ax.legend()
    fig.suptitle('Real-data CPU diagnostics · one seed · 32 patches/epoch · not a scientific ranking')
    fig.tight_layout()
    fig.savefig(out/'learning-curves.png', dpi=160)
    plt.close(fig)
    print(out/'comparison.json')
