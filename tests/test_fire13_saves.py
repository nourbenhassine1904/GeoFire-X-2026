"""FIRE13 checkpoint/backup fixtures. Untrained CPU models, never results."""
from pathlib import Path
import torch
import pytest
from geofirex import fire13_experiment as exp


def synthetic_final(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    _,config,_,lock,_=exp.identity('.')
    out=tmp_path/'UNTRAINED_FIXTURE';out.mkdir()
    model=exp.make_model(config,'cpu');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    exp.preserve('.',out,lock)
    state=dict(epoch=4,cursor=0,updates=28976,history=['SYNTHETIC UNTRAINED FIXTURE'])
    p=exp.old.checkpoint(out,model,optimizer,state,config,lock,{'fixture':True})
    exp.save(out/'training_complete.json',dict(status='COMPLETE',epochs=4,updates=28976,
        checkpoint=p.name,checkpoint_sha256=exp.sha256(p)))
    return out


@pytest.mark.parametrize('outcome',['complete','timeout','missing_coverage'])
def test_training_safe_precedes_evaluation_and_failure_stays_partial(tmp_path,monkeypatch,outcome):
    out=synthetic_final(tmp_path,monkeypatch)
    def callback():
        archives=list(tmp_path.glob('*-TRAINING-SAFE-*.zip'));assert len(archives)==1
        exp.old.verify_zip(archives[0])
        if outcome=='timeout':raise TimeoutError('Synthetic interruption')
        if outcome=='complete':
            (out/'evaluation').mkdir();exp.save(out/'evaluation/complete.json',dict(status='COMPLETE'))
        return 'fixture'
    if outcome=='complete':assert exp.safeguarded_candidate_evaluation('.',out,callback)=='fixture'
    else:
        with pytest.raises((TimeoutError,ValueError)):exp.safeguarded_candidate_evaluation('.',out,callback)
    stage='COMPLETE' if outcome=='complete' else 'PARTIAL'
    assert len(list(tmp_path.glob(f'*-{stage}-*.zip')))==1
    if outcome!='complete':assert not list(tmp_path.glob('*-COMPLETE-*.zip'))


def test_synthetic_checkpoint_restore_and_two_verified_backup_roots(tmp_path,monkeypatch):
    out=synthetic_final(tmp_path,monkeypatch)
    archive=exp.old.package(out,'TRAINING-SAFE');sidecar=archive.with_suffix('.sha256')
    restored=exp.restore_candidate('.',archive,sidecar,tmp_path/'restored')
    state,p=exp.final_candidate('.',restored)
    assert state['history']==['SYNTHETIC UNTRAINED FIXTURE']
    assert exp.sha256(p)==exp.read(out/'training_complete.json')['checkpoint_sha256']
    with pytest.raises(FileExistsError):exp.restore_candidate('.',archive,sidecar,restored)
    roots=[tmp_path/'synthetic_local',tmp_path/'synthetic_external']
    for r in roots:r.mkdir()
    copies=exp.old.backup(archive,sidecar,*roots)
    assert len(copies)==2 and all(exp.sha256(p)==exp.sha256(archive) for p in copies)
    with pytest.raises(ValueError):exp.old.backup(archive,sidecar,roots[0],roots[0])
    wrong=tmp_path/'wrong.sha256';wrong.write_text('0'*64)
    with pytest.raises(ValueError):exp.restore_candidate('.',archive,wrong,tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists()


def test_atomic_lock_survives_removed_run_directory(tmp_path):
    p=exp.new_output(tmp_path/exp.B_ID,exp.B_ID);p.rmdir()
    with pytest.raises(FileExistsError):exp.new_output(p,exp.B_ID)


def test_new_outputs_refuse_baseline_names_and_reuse(tmp_path):
    with pytest.raises(ValueError):exp.new_output(tmp_path/'B2_C8_V2_seed42',exp.B_ID)
    p=exp.new_output(tmp_path/exp.B_ID,exp.B_ID)
    with pytest.raises(ValueError):exp.new_output(p,exp.B_ID)


def test_locked_acceptance_exact_boundaries():
    from copy import deepcopy
    from geofirex.fire13_comparison import gates
    policy=exp.read(exp.REPORT/'acceptance.json');r=deepcopy(exp.baseline_json('.'))
    r['propagation_pr']['average_precision']=policy['AP_new_min']
    r['pooled'].update(average_precision=policy['AP_pooled_min'],dice=policy['Dice_pooled_min'],brier=policy['Brier_max'])
    r['new_fire'].update(recall=policy['recall_new_min'],dice=policy['Dice_new_min'],fp=policy['FP_new_max'])
    r['macro_event_dice']=policy['macro_Dice_min']
    assert all(gates(r,policy).values())
    r['new_fire']['fp']+=1
    assert not gates(r,policy)['FP_new']
