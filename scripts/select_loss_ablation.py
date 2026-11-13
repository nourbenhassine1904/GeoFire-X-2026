"""Choose once among six completed independent runs, using selection32 only."""
import argparse
import json
from pathlib import Path
from geofirex.calibration_io import save,sha


def select(paths,out):
    paths=[Path(p) for p in paths]
    if len(paths)!=6:raise ValueError('Expected six runs: two models x weights 100,30,10')
    records=[];initial={};contracts=set();seen=set()
    for directory in paths:
        row=json.loads((directory/'selection32.json').read_text())
        config=json.loads((directory/'config.json').read_text())
        if row['status']!='COMPLETE' or row['updates']!=23404 or row['holdout_pixels_opened'] or row['test_2021_pixels_opened']:
            raise ValueError('Incomplete or contaminated run')
        key=(row['model'],row['pos_weight'])
        if (config['model'],config['pos_weight'])!=key or (config['seed'],config['epochs'],config['batch_size'])!=(42,4,16):
            raise ValueError('Model/weight/seed/budget differs from predeclared experiment')
        if key in seen:raise ValueError('Duplicate candidate')
        seen.add(key)
        if config['initialization']!='fresh make_model seed42 per independent run; never load historical trained weights':
            raise ValueError('Independent initialization required')
        if row['initial_state_sha256']!=config['initial_state_sha256']:raise ValueError('Initialization record differs')
        if row['checkpoint_sha256']!=sha(directory/'checkpoint_latest.pt'):raise ValueError('Checkpoint digest differs')
        previous=initial.setdefault(row['model'],row['initial_state_sha256'])
        if previous!=row['initial_state_sha256']:raise ValueError('Paired initial states differ within architecture')
        contracts.add(tuple(config[k] for k in ('seed','epochs','batch_size','index_sha256','normalization_sha256','protocol_sha256','torch_version','device')))
        records.append(dict(**row,run=str(directory)))
    if seen!={(m,w) for m in ('B1','B2') for w in (100,30,10)} or len(contracts)!=1:
        raise ValueError('Incomplete weight grid or incomparable protocol')
    def rank(r):
        metrics=r['metrics']
        return (-metrics['dice'],-metrics['average_precision'],metrics['brier_score'],r['pos_weight'],r['model'])
    records.sort(key=rank)
    result=dict(schema='geofirex-loss-ablation-selection-v1',selection_scope='32 calibration events only',
        criterion='micro Dice at each selected threshold; tie AP, Brier, lower weight, model ID',
        winner=records[0],candidates=records,holdout_used_for_selection=False,test_2021_locked=True,
        next_step='Freeze this record before one confirmatory evaluation of the winner; no retuning on 169')
    save(out,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs=6,required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(select(a.runs,a.out),indent=2))
