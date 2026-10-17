"""Метрики РНМ должны оставаться информативными при дисбалансе классов."""

import numpy as np

from osc_tools.pdr.pdr_trainer import _binary_metrics


def test_binary_metrics_include_both_classes_and_mcc() -> None:
    metrics = _binary_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int8),
        np.asarray([0, 1, 1, 1], dtype=np.int8),
    )

    assert metrics["tp"] == 2
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["f1"] == 0.8
    assert metrics["balanced_accuracy"] == 0.75
    assert np.isclose(metrics["macro_f1"], 11.0 / 15.0)
    assert np.isclose(metrics["mcc"], 1.0 / np.sqrt(3.0))


def test_all_reverse_prediction_exposes_forward_failure() -> None:
    metrics = _binary_metrics(
        np.asarray([0] * 127 + [1] * 32, dtype=np.int8),
        np.zeros(159, dtype=np.int8),
    )

    assert np.isclose(metrics["accuracy"], 127.0 / 159.0)
    assert metrics["f1"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["mcc"] == 0.0
