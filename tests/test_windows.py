from geofirex.windows import (
    REASON_DATE_GAP,
    REASON_EVENT_TOO_SHORT,
    REASON_SHAPE_MISMATCH,
    build_windows,
)


def _row(event_id, date, height=10, width=10, path=None):
    return {
        "event_id": event_id,
        "date": date,
        "relative_path": path or f"{event_id}/{date}.tif",
        "height": height,
        "width": width,
    }


def test_consecutive_days_produce_one_window_per_valid_start():
    rows = [_row("e1", d) for d in ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04")]
    windows, rejections = build_windows(rows, context_days=2)

    assert len(windows) == 2  # starts at 01-01 and 01-02
    assert rejections == {}
    w = windows[0]
    assert w.context_dates == ("2020-01-01", "2020-01-02")
    assert w.target_date == "2020-01-03"
    assert w.event_id == "e1"


def test_event_too_short_is_rejected_with_reason():
    rows = [_row("e1", "2020-01-01"), _row("e1", "2020-01-02")]
    windows, rejections = build_windows(rows, context_days=3)

    assert windows == []
    assert rejections == {REASON_EVENT_TOO_SHORT: 1}


def test_date_gap_is_rejected_with_reason():
    rows = [_row("e1", d) for d in ("2020-01-01", "2020-01-02", "2020-01-05")]  # gap before 01-05
    windows, rejections = build_windows(rows, context_days=2)

    assert windows == []
    assert rejections.get(REASON_DATE_GAP) == 1


def test_shape_mismatch_is_rejected_with_reason():
    rows = [
        _row("e1", "2020-01-01", height=10, width=10),
        _row("e1", "2020-01-02", height=20, width=20),  # inconsistent shape
    ]
    windows, rejections = build_windows(rows, context_days=1)

    assert windows == []
    assert rejections.get(REASON_SHAPE_MISMATCH) == 1


def test_windows_never_cross_events():
    rows = [
        _row("e1", "2020-01-01"),
        _row("e1", "2020-01-02"),
        _row("e2", "2020-01-02"),  # same date, DIFFERENT event - must not merge with e1
        _row("e2", "2020-01-03"),
    ]
    windows, _ = build_windows(rows, context_days=1)

    assert all(w.event_id in ("e1", "e2") for w in windows)
    # e1 has exactly 1 valid window (01-01 -> 01-02); e2 has exactly 1 (01-02 -> 01-03)
    assert len(windows) == 2


def test_context_days_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        build_windows([], context_days=0)


def test_mixed_reasons_are_each_counted_independently():
    rows = (
        [_row("short", "2020-01-01")]  # too short for context_days=2
        + [_row("gap", d) for d in ("2020-02-01", "2020-02-02", "2020-02-05")]  # gap
        + [_row("ok", d) for d in ("2020-03-01", "2020-03-02", "2020-03-03")]  # valid
    )
    windows, rejections = build_windows(rows, context_days=2)

    assert len(windows) == 1
    assert windows[0].event_id == "ok"
    assert rejections[REASON_EVENT_TOO_SHORT] == 1
    assert rejections[REASON_DATE_GAP] == 1
