"""Presentation figures from saved GPU results only; never trains or downloads."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prepare(source, destination):
    report = json.loads(Path(source).read_text(encoding='utf-8'))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows = report['models']
    labels = ['Persistence', 'CNN B1', 'U-Net B2', 'ConvLSTM B3']
    colors = ['#64748b', '#ea580c', '#2563eb', '#16a34a']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, title in zip(axes, ['average_precision', 'dice'], ['Validation micro Average Precision', 'Validation Dice']):
        values = [r['metrics'][metric] for r in rows]
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, .6)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=15)
        for i, value in enumerate(values): ax.text(i, value+.01, f'{value:.4f}', ha='center')
    fig.suptitle('GeoFire-X GPU pilot | 3 train events / 2 validation events | seed 42')
    fig.text(.5, .01, 'Validation-only selection; no test 2021. B1 has best AP; Persistence has slightly higher Dice.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    for suffix in ('png', 'svg'): fig.savefig(destination/f'comparison.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for row, color in zip(rows[1:], colors[1:]):
        history = row['summary']['history']
        epochs = [h['epoch'] for h in history]
        for ax, key in zip(axes, ['train_loss', 'validation_average_precision']):
            ax.plot(epochs, [h[key] for h in history], marker='o', label=row['model'], color=color)
            ax.set_xlabel('Epoch'); ax.set_xticks(range(1, 11)); ax.legend()
    axes[0].set_title('Training loss (weighted BCE + Dice)')
    axes[1].set_title('Validation full-scene micro AP')
    axes[1].axhline(rows[0]['metrics']['average_precision'], color=colors[0], linestyle='--', label='Persistence')
    axes[1].legend()
    fig.suptitle('Actual GPU runs | 10 epochs / 1,280 updates per architecture')
    fig.tight_layout()
    for suffix in ('png', 'svg'): fig.savefig(destination/f'learning-curves.{suffix}', dpi=180)
    plt.close(fig)
    return [str(p) for p in destination.iterdir()]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='artifacts/sample-launch-20261101/verified-output/campaign.json')
    p.add_argument('--out', default='reports/figures/suivi3_gpu')
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.out), indent=2))
