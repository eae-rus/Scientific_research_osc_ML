from pathlib import Path

import numpy as np
import pytest
import torch

from osc_tools.ml.ct_saturation_dataset import (
    CTSaturationFile,
    CTSaturationLazyDataset,
    ct_spectral_feature_count,
    current_feature_count,
    deterministic_split,
    load_ct_saturation_mat,
)
from scripts.phase4_experiments.ct_saturation.analyze_ct_saturation import binary_metrics
from scripts.phase4_experiments.ct_saturation.analyze_ct_saturation import real_normalization_profile
from scripts.phase4_experiments.ct_saturation.train_ct_saturation import CONFIG, create_ct_model


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


def test_binary_article_metrics_use_expected_confusion_counts():
    result = binary_metrics(
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([1, 0, 1, 0], dtype=bool),
    )
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def _synthetic_ct_record():
    samples_per_period = 32
    time = np.arange(160) / 1600.0
    phases = np.column_stack([
        np.sin(2 * np.pi * 50 * time - shift)
        for shift in (0, 2 * np.pi / 3, 4 * np.pi / 3)
    ]).astype(np.float32)
    labels = np.zeros_like(phases, dtype=np.float32)
    labels[80:88, 0] = 1.0
    return {
        "secondary_a": phases * 5.0,
        "voltage_v": phases * 100.0,
        "current_error_a": None,
        "labels": labels,
        "time_s": time,
        "fs_hz": float(samples_per_period * 50),
    }


def test_v2_spectral_and_raw_contracts_have_40_ms_window():
    file_info = CTSaturationFile(Path("synthetic.mat"), 1, 0.015)
    loader = lambda _: _synthetic_ct_record()
    spectral = CTSaturationLazyDataset(
        [file_info], num_periods=2, stride_fraction=32,
        num_harmonics=9, sub_periods=(2,), input_mode="spectral",
        include_voltage=True, phase_permutation=False, loader=loader,
    )
    x_spectral, y_spectral = spectral[0]
    assert x_spectral.shape == (ct_spectral_feature_count(9, (2,), True), 64)
    assert y_spectral.shape == (64, 3)

    raw = CTSaturationLazyDataset(
        [file_info], num_periods=2, input_mode="raw", include_voltage=True,
        raw_target_spp=32, phase_permutation=False, loader=loader,
    )
    x_raw, y_raw = raw[0]
    assert x_raw.shape == (6, 64)
    assert y_raw.shape == (64, 3)


def test_real_normalization_matches_normosc_formulas():
    lookup = {
        "abc": {"norm": "YES", "1Ip_base": "5", "1Ub_base": "100", "1Uc_base": "110"}
    }
    assert real_normalization_profile("abc_Bus 1", lookup, voltage_source="BB") == (100.0, 300.0)
    assert real_normalization_profile("abc_Bus 1", lookup, voltage_source="CL") == (100.0, 330.0)


def test_v2_models_preserve_main_ozz_transformer_depth_and_output_grid():
    spectral_cfg = dict(CONFIG)
    spectral_model = create_ct_model(spectral_cfg)
    assert len(spectral_model.encoder_blocks) == 6
    x_spectral = torch.zeros(2, ct_spectral_feature_count(9, (2, 4, 6, 10), True), 64)
    assert spectral_model(x_spectral, mode="classify")["classify"].shape == (2, 64, 3)

    raw_cfg = dict(CONFIG)
    raw_cfg["input_mode"] = "raw"
    raw_model = create_ct_model(raw_cfg)
    assert len(raw_model.encoder_blocks) == 6
    x_raw = torch.zeros(2, 6, 64)
    assert raw_model(x_raw, mode="classify")["classify"].shape == (2, 64, 3)
