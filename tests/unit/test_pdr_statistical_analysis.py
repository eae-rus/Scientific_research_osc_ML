"""Лёгкие проверки статистики PDR без чтения полного датасета."""

import numpy as np
import pytest

from scripts.phase5_experiments.analyze_pdr_dataset_study import (
    _agreement_metrics,
    _temporal_metrics,
)


pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from scripts.phase5_experiments.review_pdr_analysis_results import (
    ALGORITHMS,
    SIGNAL_METRICS,
    build_review_tables,
)


def test_agreement_metrics_report_class_imbalance_explicitly() -> None:
    confusion = np.asarray([[80, 10], [5, 5]], dtype=np.int64)
    metrics = _agreement_metrics(confusion)

    assert metrics["agreement"] == pytest.approx(0.85)
    assert metrics["jaccard_reverse"] == pytest.approx(80 / 95)
    assert metrics["jaccard_forward"] == pytest.approx(5 / 20)
    assert metrics["forward_prevalence_left"] == pytest.approx(0.10)
    assert metrics["forward_prevalence_right"] == pytest.approx(0.15)
    assert -1.0 <= metrics["mcc"] <= 1.0


def test_temporal_metrics_preserve_unlabeled_gap_and_find_chatter() -> None:
    labels = np.asarray([-999, 0, 0, 1, 0, 0], dtype=np.int16)
    samples = np.arange(len(labels), dtype=np.int64)
    metrics = _temporal_metrics(labels, samples, f_adc=10.0)

    assert metrics["valid_windows"] == 5
    assert metrics["transitions"] == 2
    assert metrics["chatter_returns_le_100ms"] == 1
    assert metrics["max_switches_in_0_5s"] == 2
    assert metrics["median_run_duration_sec"] == pytest.approx(0.2)
    assert metrics["p05_run_duration_sec"] == pytest.approx(0.11)
    assert metrics["state_entropy_bits"] == pytest.approx(0.7219280948873623)


def test_temporal_metrics_constant_state_has_zero_entropy() -> None:
    labels = np.zeros(20, dtype=np.int16)
    metrics = _temporal_metrics(labels, np.arange(20), f_adc=1000.0)

    assert metrics["transitions"] == 0
    assert metrics["state_entropy_bits"] == 0.0
    assert np.isnan(metrics["lag1_autocorrelation"])
    assert metrics["median_run_duration_sec"] == pytest.approx(0.020)


def test_review_builds_sampling_and_weighting_profiles(tmp_path) -> None:
    records = []
    signals = []
    for record_id, source, spp in ((0, "open_ee", 24), (1, "french_rte", 32)):
        record = {
            "source": source,
            "record_id": record_id,
            "file_name": f"record_{record_id}",
            "f_adc": spp * 50,
            "voltage_basis": "phase",
            "input_sha256": f"hash_{record_id}",
            "split": "train",
            "duration_sec": 1.0 + record_id,
            "n_windows": 100 + record_id,
            "disagreement_fraction": 0.1 * record_id,
            "total_transitions": record_id,
            "interest_score": 0.2,
            "categories": "stable",
            "low_coverage": False,
            "spp": spp,
        }
        for algorithm in ALGORITHMS:
            record[f"{algorithm}__coverage_fraction"] = 1.0
            record[f"{algorithm}__forward_fraction"] = 0.25 + 0.1 * record_id
            record[f"{algorithm}__valid_windows"] = 100
            record[f"{algorithm}__transitions_per_second"] = float(record_id)
            record[f"{algorithm}__transitions"] = record_id
        records.append(record)
        signal = {
            "source": source,
            "record_id": record_id,
            "file_name": f"record_{record_id}",
            "f_adc": spp * 50,
            "voltage_basis": "phase",
            "missing_current_group": False,
            "missing_voltage_group": False,
        }
        signal.update({metric: 0.1 + record_id for metric in SIGNAL_METRICS})
        signals.append(signal)

    pd.DataFrame(records).to_csv(tmp_path / "record_statistics.csv", index=False)
    pd.DataFrame(signals).to_csv(tmp_path / "signal_record_statistics.csv", index=False)
    pd.DataFrame([
        {
            "source": source,
            "left": ALGORITHMS[1],
            "right": ALGORITHMS[2],
            "disagreement_fraction": 0.1,
            "disagreement_transitions": 1,
        }
        for source in ("open_ee", "french_rte")
    ]).to_csv(tmp_path / "pairwise_record_agreement.csv", index=False)
    pd.DataFrame(columns=["input_sha256", "records"]).to_csv(
        tmp_path / "duplicate_groups.csv", index=False
    )

    outputs = build_review_tables(tmp_path)

    assert outputs["sampling_profiles"].exists()
    assert outputs["weighting_sensitivity"].exists()
    assert outputs["metric_guide"].exists()
    profiles = pd.read_csv(outputs["sampling_profiles"])
    assert set(profiles["spp"]) == {24, 32}
