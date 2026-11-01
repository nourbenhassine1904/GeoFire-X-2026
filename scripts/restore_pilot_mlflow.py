"""Restore Kaggle MLflow locally without changing the certified backup.

Copies SQLite, rewrites artifact locations in the copy, and verifies four real
runs through MlflowClient. No training, network access or artifact download.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

from geofirex.corpus import sha256_file


def restore(source_root, destination):
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    source = source_root/'runs/tracking.db'
    original_hash = sha256_file(source)
    destination.mkdir(parents=True, exist_ok=False)
    db = destination/'tracking.db'
    with sqlite3.connect(f'file:{source.as_posix()}?mode=ro', uri=True) as src, sqlite3.connect(db) as dst:
        src.backup(dst)
        prefix = 'file:///kaggle/working/sample-pilot/runs/mlflow_artifacts'
        local = (source_root/'runs/mlflow_artifacts').as_uri()
        dst.execute('UPDATE experiments SET artifact_location = replace(artifact_location, ?, ?)', (prefix, local))
        dst.execute('UPDATE runs SET artifact_uri = replace(artifact_uri, ?, ?)', (prefix, local))
        # The unused default experiment must also remain local.
        dst.execute('UPDATE experiments SET artifact_location = ? WHERE experiment_id = 0',
                    ((destination/'default-artifacts').as_uri(),))
        dst.commit()
        assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Metric
    uri = 'sqlite:///'+db.as_posix()
    client = MlflowClient(tracking_uri=uri)
    campaign = json.loads((source_root/'campaign.json').read_text())
    checked = []
    for row in campaign['models']:
        summary = row['summary']
        run = client.get_run(summary['run_id'])
        assert run.info.status == 'FINISHED'
        assert run.data.params['model'] == row['model']
        assert any(m.value == summary['optimizer_updates'] for m in client.get_metric_history(summary['run_id'], 'optimizer_updates'))
        assert any(m.value == summary['validation_average_precision'] for m in client.get_metric_history(summary['run_id'], 'validation_average_precision'))
        # Original latest metrics can describe the last epoch, while the summary
        # describes the best epoch. Preserve history; add unambiguous display keys.
        selected = {'selected_validation_'+k: v for k, v in row['metrics'].items()
                    if k in ('dice', 'iou', 'precision', 'recall', 'average_precision')}
        selected.update(total_optimizer_updates=summary['optimizer_updates'], completed_epochs=row['epochs'])
        client.log_batch(summary['run_id'], metrics=[Metric(k, float(v), int(time.time()*1000), 0) for k, v in selected.items()])
        client.set_tag(summary['run_id'], 'restoration', 'Local paths + selected_* summary metrics; original histories preserved')
        history = client.get_metric_history(summary['run_id'], 'train_loss')
        assert len(history) == row['epochs']
        items = {f.path for f in client.list_artifacts(summary['run_id'])}
        assert summary['checkpoint'] in items
        if row['model'] != 'B0': assert 'curves.png' in items
        from urllib.parse import urlparse, unquote
        from urllib.request import url2pathname
        artifact = Path(url2pathname(unquote(urlparse(run.info.artifact_uri).path)))
        assert sha256_file(artifact/summary['checkpoint']) == summary['checkpoint_sha256']
        assert (artifact/'example.png').is_file()
        checked.append({'model': row['model'], 'run_id': summary['run_id'], 'epochs': row['epochs'],
                        'metric_history': len(history), 'artifacts': len(items), 'artifact_uri': run.info.artifact_uri,
                        'selected_metrics': selected})
    assert sha256_file(source) == original_hash
    result = {'tracking_uri': uri, 'source_sha256_unchanged': original_hash, 'runs': checked,
              'restored_database': str(db), 'note': 'One MLflow experiment with four B0/B1/B2/B3 runs; original backup untouched'}
    (destination/'restoration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', default='artifacts/sample-launch-20261101/verified-output')
    p.add_argument('--out', default='artifacts/suivi3-mlflow-restored')
    args = p.parse_args()
    print(json.dumps(restore(args.source_root, args.out), indent=2))
