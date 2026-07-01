from pathlib import Path

import numpy as np
import pytest

from osc_tools.ml.ct_saturation_dataset import (
    CTSaturationFile,
    current_feature_count,
    deterministic_split,
    load_ct_saturation_mat,
)


class FakeTimeseries:
    """Минимальная имитация MATLAB timeseries для модульного теста адаптера."""
    def __init__(self, data, time):
        self.Data = data
        self.Time = time


def test_load_mat_record_extracts_secondary_and_three_labels():
    n = 8
    fake = {
        "I2_CT1": FakeTimeseries(np.ones((n, 3)), np.arange(n) * 1e-5),
        "flag_sat_phsA": np.arange(n) >= 4,
        "flag_sat_phsB": np.zeros(n, dtype=bool),
        "flag_sat_phsC": np.zeros(n, dtype=bool),
    }
    record = load_ct_saturation_mat("unused.mat", loadmat_fn=lambda *a, **k: fake)
    assert record["secondary_a"].shape == (n, 3)
    assert record["labels"].shape == (n, 3)
    assert record["fs_hz"] == pytest.approx(100_000.0)


def test_split_is_stable_and_feature_count_is_84():
    files = [CTSaturationFile(Path(f"{i}.mat"), i, 0.015) for i in range(100)]
    a = deterministic_split(files, seed=7)
    b = deterministic_split(reversed(files), seed=7)
    assert {x.record_id for x in a[0]} == {x.record_id for x in b[0]}
    assert {x.record_id for x in a[1]} == {x.record_id for x in b[1]}
    assert current_feature_count() == 84
