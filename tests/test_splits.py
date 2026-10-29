from geofirex.splits import check_no_leakage, split_events


def test_every_event_assigned_exactly_once():
    event_ids = [f"2020/fire_{i}" for i in range(20)]
    assignment = split_events(event_ids, seed=42)
    assert set(assignment.keys()) == set(event_ids)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_no_event_in_two_splits():
    event_ids = [f"2020/fire_{i}" for i in range(20)]
    assignment = split_events(event_ids, seed=42)
    check_no_leakage(assignment)  # must not raise


def test_split_is_deterministic_given_seed():
    event_ids = [f"2020/fire_{i}" for i in range(20)]
    a = split_events(event_ids, seed=7)
    b = split_events(event_ids, seed=7)
    assert a == b


def test_ratios_must_sum_to_one():
    import pytest

    with pytest.raises(ValueError):
        split_events(["a", "b"], seed=1, ratios=(0.5, 0.5, 0.5))
