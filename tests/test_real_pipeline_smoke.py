"""Smoke test against the ACTUAL real WildfireSpreadTS sample on disk (108
files, gitignored - never present on a fresh clone). Skipped automatically
when the sample is absent, so this never breaks CI or a teammate's machine
that has not run scripts/download_real_sample.py.
"""

from pathlib import Path

import pytest
import torch

from geofirex.real_dataset import RealGeoFireXDataset
from geofirex.real_splits import make_smoke_test_split, save_split_csv

MANIFEST = Path("data/manifests/real/manifest_wildfirespreadts.csv")
REAL_SAMPLE_DIR = Path("data/raw/WildfireSpreadTS/real_sample")

pytestmark = pytest.mark.skipif(
    not (MANIFEST.exists() and REAL_SAMPLE_DIR.exists()),
    reason="real WildfireSpreadTS sample not present locally (gitignored, run "
    "scripts/download_real_sample.py to fetch it)",
)


@pytest.fixture(scope="module")
def smoke_split_path(tmp_path_factory):
    import csv

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    event_ids = sorted({r["event_id"] for r in rows})
    assignment = make_smoke_test_split(event_ids, seed=42)
    path = tmp_path_factory.mktemp("smoke") / "split.csv"
    save_split_csv(assignment, path)
    return path


def test_real_sample_loads_without_nan_or_shape_errors(smoke_split_path):
    ds = RealGeoFireXDataset(MANIFEST, smoke_split_path, split_name="train", context_days=1)
    assert len(ds) > 0, "smoke-test split produced zero windows on the real sample"

    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape[1] == 23
        assert y.shape[0] == 1
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()
        assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_real_sample_window_report_has_no_unexplained_rejections(smoke_split_path):
    ds = RealGeoFireXDataset(MANIFEST, smoke_split_path, split_name="train", context_days=1)
    # The real sample was already verified to have zero date gaps (Phase 2
    # audit) - any rejection here would indicate a regression worth investigating.
    assert ds.rejections == {}
