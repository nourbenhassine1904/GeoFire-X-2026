import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from geofirex.benchmark import aligned_windows,annual_folds,event_assignments
from geofirex.benchmark_metrics import (ExternalAveragePrecision,new_fire_metrics,
                                       rank_validation_candidates)
from geofirex.benchmark_models import make_model


@pytest.fixture(autouse=True,scope="module")
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_all_twelve_folds_match_authors_order_and_disjoint_years():
    folds=annual_folds()
    assert len(folds)==12
    assert folds[0]=={"fold_id":0,"train_years":[2018,2019],"val_years":[2020],"test_years":[2021]}
    assert folds[-1]["train_years"]==[2020,2021] and folds[-1]["test_years"]==[2018]
    assert len({(tuple(f['train_years']),tuple(f['val_years']),tuple(f['test_years'])) for f in folds})==12
    for fold in folds:
        years=fold["train_years"]+fold["val_years"]+fold["test_years"]
        assert sorted(years)==[2018,2019,2020,2021]
        assignment=event_assignments([f"{year}/fire_1" for year in years],fold["fold_id"])
        assert sorted(assignment.values())==["test","train","train","val"]


def manifest_rows():
    return [{"event_id":"2018/fire_1","date":f"2018-01-{day:02d}",
             "relative_path":f"fixture/2018/fire_1/2018-01-{day:02d}.tif","height":"20","width":"24"}
            for day in range(1,9)]


def test_contexts_share_exact_target_dates_and_consecutive_inputs():
    rows=manifest_rows()
    groups,_=aligned_windows(rows,{"2018/fire_1":"train"},"train")
    assert [w.target_date for w in groups[1]]==[w.target_date for w in groups[3]]==[w.target_date for w in groups[5]]
    assert len(groups[1])==3 and groups[1][0].context_dates==("2018-01-05",)
    assert len(groups[5][0].context_dates)==5 and groups[5][0].target_date=="2018-01-06"
    assert all(w.target_date not in w.context_dates for group in groups.values() for w in group)


def test_alignment_handles_gaps_without_short_context_advantage():
    rows=[r for r in manifest_rows() if r["date"]!="2018-01-04"]
    groups,rejections=aligned_windows(rows,{"2018/fire_1":"train"},"train")
    assert all(len(group)==0 for group in groups.values())
    assert rejections[5]["date_gap"]>0


def test_alignment_rejects_reused_paths_and_unassigned_events():
    rows=manifest_rows()
    rows[-1]["relative_path"]=rows[0]["relative_path"]
    with pytest.raises(ValueError,match="reused"):
        aligned_windows(rows,{"2018/fire_1":"train"},"train")
    with pytest.raises(ValueError,match="assigned"):
        aligned_windows(manifest_rows(),{},"train")


@pytest.mark.parametrize("name",["B1","B2","B3"])
@pytest.mark.parametrize("context",[1,3,5])
def test_model_common_contract_finite_backward_and_odd_shapes(name,context):
    torch.manual_seed(42)
    model=make_model(name,channels=8,context=context)
    x=torch.randn(2,context,8,17,19,requires_grad=True)
    logits=model(x)
    assert logits.shape==(2,1,17,19)
    torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_persistence_is_exact_last_observation_and_temporal_model_uses_history():
    x=torch.rand(1,3,8,17,19,requires_grad=True)
    persistence=make_model("B0",channels=8,context=3)
    assert torch.equal(persistence.predict_proba(x),x[:,-1,7:8])
    recurrent=make_model("B3",channels=8,context=3)
    recurrent(x).sum().backward()
    assert x.grad[:,0].abs().sum()>0
    assert not torch.allclose(recurrent(x.detach()),recurrent(x.detach().flip(1)))
    with pytest.raises(NotImplementedError):
        make_model("B4")


@pytest.mark.parametrize("ties",[False,True])
def test_external_ap_matches_sklearn_across_chunks_ties_and_merge_levels(tmp_path,ties):
    rng=np.random.default_rng(42)
    p=rng.random(127)
    if ties:
        p=np.round(p,1)
    y=(rng.random(127)>.8).astype(np.uint8)
    accumulator=ExternalAveragePrecision(tmp_path/"ap",chunk_pixels=7,fan_in=2)
    for i in range(0,len(p),19):
        accumulator.update(p[i:i+19],y[i:i+19])
    assert accumulator.compute()==pytest.approx(average_precision_score(y,p),abs=1e-14)
    with pytest.raises(RuntimeError):
        accumulator.update(p,y)


def test_external_ap_empty_and_all_positive_conventions(tmp_path):
    empty=ExternalAveragePrecision(tmp_path/"empty")
    empty.update(np.ones(4),np.zeros(4))
    assert empty.compute() is None
    positive=ExternalAveragePrecision(tmp_path/"positive")
    positive.update(np.ones(4),np.ones(4))
    assert positive.compute()==1


def test_new_fire_candidate_domain_excludes_current_fire():
    r=new_fire_metrics(np.array([.9,.7,.1,.3]),np.array([1,1,1,0]),np.array([1,0,0,0]),.5)
    assert r["candidate_pixels"]==3 and r["new_positive_pixels"]==2
    assert r["new_tp"]==r["new_fn"]==1


def test_model_selection_ignores_test_metrics_and_rejects_other_protocols():
    shared={"protocol_version":"v1","dataset_hash":"same","fold_id":0,
            "evaluation_target_hash":"same","feature_policy":"same","status":"FINISHED"}
    candidates=[{**shared,"run_id":"a","validation_average_precision":.3,"test_average_precision":.9},
                {**shared,"run_id":"b","validation_average_precision":.5,"test_average_precision":.1}]
    assert rank_validation_candidates(candidates)[0]["run_id"]=="b"
    candidates[1]["fold_id"]=1
    with pytest.raises(ValueError,match="protocol"):
        rank_validation_candidates(candidates)


@pytest.mark.parametrize("probability,target",[(np.array([.5]),np.array([np.nan])),
                                               (np.array([1.1]),np.array([1])),
                                               (np.array([.5]),np.array([2]))])
def test_scoring_rejects_invalid_probabilities_or_labels(probability,target):
    from geofirex.unet_evaluation import score
    with pytest.raises(ValueError):
        score(probability,target,.5)


def test_architecture_ranking_requires_equal_seeds():
    from geofirex.benchmark_metrics import rank_architectures
    shared={"protocol_version":"v1","dataset_hash":"same","fold_id":0,
            "evaluation_target_hash":"same","feature_policy":"same","status":"FINISHED","context_days":1}
    candidates=[{**shared,"model":model,"seed":seed,"run_id":f"{model}_{seed}","validation_average_precision":ap}
                for model,ap in (("B1",.2),("B2",.4)) for seed in (42,43,44)]
    assert rank_architectures(candidates)[0]["model"]=="B2"
    with pytest.raises(ValueError,match="seeds"):
        rank_architectures(candidates[:-1])
