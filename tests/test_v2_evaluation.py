import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from geofirex.v2_evaluation import V2Metrics


def test_pooled_ap_global_ties_explicit_observed_mask_and_event_macro(tmp_path):
    keys = [('2020/fire_1', '2020-09-01', 0, 0), ('2020/fire_2', '2020-09-01', 0, 0)]
    acc = V2Metrics(tmp_path/'ap', keys)
    acc.update(keys[0], np.array([.9, .8, .8, .1], np.float32), np.array([1, 0, 1, 0]),
               np.array([1, 1, 1, 0]), np.array([1, 0, 0, 0]))
    with pytest.raises(ValueError, match='incomplete'):
        acc.result()
    acc.update(keys[1], np.array([.8, .2], np.float32), np.array([0, 1]),
               np.array([1, 1]), np.array([0, 0]))
    result = acc.result()
    expected_ap = average_precision_score([1,0,1,0,1], [.9,.8,.8,.8,.2])
    assert result['pooled']['average_precision'] == pytest.approx(expected_ap)
    assert result['pooled']['dice'] == pytest.approx(4/7)
    assert result['new_fire']['tp'] == 1 and result['new_fire']['fp'] == 2
    assert result['new_fire']['fn'] == 1 and result['new_fire']['recall'] == .5
    assert result['macro_event_dice'] == pytest.approx(.4)
    assert result['pooled']['brier'] == pytest.approx((.01+.64+.04+.64+.64)/5)


def test_duplicate_and_changed_domain_are_rejected(tmp_path):
    key = ('2020/fire_1', '2020-09-01', 0, 0)
    acc = V2Metrics(tmp_path/'ap', [key])
    acc.update(key, np.array([.5], np.float32), np.array([0]), np.array([1]), np.array([0]))
    with pytest.raises(ValueError, match='duplicate'):
        acc.update(key, np.array([.5], np.float32), np.array([0]), np.array([1]), np.array([0]))
    result = acc.result()
    assert result['pooled']['average_precision'] is None
    assert result['pooled']['dice'] == 1
    assert result['new_fire']['recall'] is None


def test_test_year_is_refused_before_creating_scratch(tmp_path):
    with pytest.raises(ValueError, match='2021'):
        V2Metrics(tmp_path/'ap', [('2021/fire_1', '2021-09-01', 0, 0)])
    assert not (tmp_path/'ap').exists()


def test_same_keys_with_changed_pixels_have_different_domain_identity(tmp_path):
    key = ('2020/fire_1', '2020-09-01', 0, 0)
    identities = []
    for i, valid in enumerate(([1, 1], [1, 0])):
        acc = V2Metrics(tmp_path/str(i), [key])
        acc.update(key, np.array([.9, .2], np.float32), np.array([1, 0]),
                   np.array(valid), np.array([0, 0]))
        identities.append(acc.result()['evaluation_domain_sha256'])
    assert identities[0] != identities[1]
