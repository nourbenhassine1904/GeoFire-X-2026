import pytest

from geofirex.real_splits import (
    is_sample_too_small_for_science,
    make_smoke_test_split,
    split_events_by_year,
)
from geofirex.splits import check_no_leakage


def test_split_events_by_year_each_fold_holds_out_exactly_one_year():
    event_years = {
        "2018/a": "2018", "2018/b": "2018",
        "2019/c": "2019",
        "2020/d": "2020", "2020/e": "2020",
    }
    folds = split_events_by_year(event_years)

    assert set(folds.keys()) == {"2018", "2019", "2020"}
    fold_2019 = folds["2019"]
    assert fold_2019["2019/c"] == "test"
    assert fold_2019["2018/a"] == "train"
    assert fold_2019["2020/d"] == "train"
    check_no_leakage(fold_2019)  # must not raise


def test_make_smoke_test_split_is_deterministic():
    event_ids = [f"2020/fire_{i}" for i in range(5)]
    a = make_smoke_test_split(event_ids, seed=7)
    b = make_smoke_test_split(event_ids, seed=7)
    assert a == b


def test_make_smoke_test_split_assigns_every_event_exactly_once():
    event_ids = [f"2020/fire_{i}" for i in range(5)]
    assignment = make_smoke_test_split(event_ids, seed=1)
    assert set(assignment.keys()) == set(event_ids)
    check_no_leakage(assignment)


def test_make_smoke_test_split_handles_tiny_counts_without_crashing():
    assert make_smoke_test_split([], seed=1) == {}
    one = make_smoke_test_split(["e1"], seed=1)
    assert one == {"e1": "train"}
    two = make_smoke_test_split(["e1", "e2"], seed=1)
    assert set(two.values()) == {"train", "test"}


def test_is_sample_too_small_for_science_flags_five_real_events():
    five_events = [f"2020/fire_{i}" for i in range(5)]
    assert is_sample_too_small_for_science(five_events) is True


def test_is_sample_too_small_for_science_false_for_large_enough_sample():
    many_events = [f"2020/fire_{i}" for i in range(50)]
    assert is_sample_too_small_for_science(many_events) is False
