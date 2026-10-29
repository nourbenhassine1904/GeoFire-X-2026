from pathlib import Path

from geofirex.constants import NUM_CHANNELS
from geofirex.io import list_events, read_event
from geofirex.synthetic import generate_synthetic_dataset


def test_read_event_shape(tmp_path: Path):
    generate_synthetic_dataset(tmp_path, num_events=1, min_days=5, max_days=5, height=16, width=16, seed=1)
    events = list_events(tmp_path)
    assert len(events) == 1

    arr, dates = read_event(events[0])
    assert arr.shape == (5, NUM_CHANNELS, 16, 16)
    assert len(dates) == 5
    assert dates == sorted(dates)  # chronological order
