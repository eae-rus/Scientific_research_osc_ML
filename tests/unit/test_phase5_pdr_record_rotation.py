"""Ротация weak-пулов должна покрывать весь split без скрытого исключения."""

import numpy as np

from scripts.phase5_experiments.run_phase5_pdr_training import (
    _epoch_record_pool,
    _mandatory_record_sample_indices,
)
from scripts.phase5_experiments.evaluate_pdr_full_weak_validation import (
    _combine_source_metrics,
)
from osc_tools.pdr.pdr_dataset import _select_stratified_window_indices


def test_epoch_record_pool_covers_every_record_in_one_cycle() -> None:
    values = list(range(23))
    pools = [
        _epoch_record_pool(values, 5, 42, "test", epoch)[0]
        for epoch in range(5)
    ]

    assert sorted(value for pool in pools for value in pool) == values
    assert [len(pool) for pool in pools] == [5, 5, 5, 4, 4]


def test_epoch_record_pool_is_deterministic_and_cycles() -> None:
    values = list(range(11))
    first = _epoch_record_pool(values, 4, 7, "test", 0)
    repeated = _epoch_record_pool(values, 4, 7, "test", 3)

    assert first[0] == repeated[0]
    assert first[1:] == (0, 3)
    assert repeated[1:] == (0, 3)


def test_epoch_record_pool_uses_all_records_when_limit_is_large() -> None:
    pool, pool_index, pool_count = _epoch_record_pool(
        [4, 2, 8], 100, 3, "test", 9
    )

    assert sorted(pool) == [2, 4, 8]
    assert pool_index == 0
    assert pool_count == 1


def test_epoch_pool_keeps_shard_blocks_local() -> None:
    values = list(range(640))  # десять блоков по 64 записи
    pool, _, pool_count = _epoch_record_pool(values, 128, 42, "test", 0)

    assert pool_count == 5
    assert len(pool) == 128
    assert len({record_id // 64 for record_id in pool}) <= 2


def test_full_validation_combines_source_confusions_without_second_pass() -> None:
    template = {
        "mae_margin": 0.0,
        "n_margin_samples": 0,
        "n_samples": 2,
        "n_applicability_samples": 2,
        "n_direction_records": 1,
        "n_applicability_records": 1,
        "record_macro_accuracy": 1.0,
        "record_macro_f1_score": 0.5,
        "record_present_class_f1_score": 1.0,
        "applicability_record_macro_accuracy": 1.0,
        "applicability_record_macro_f1_score": 0.5,
        "applicability_record_present_class_f1_score": 1.0,
    }
    metrics = _combine_source_metrics({
        "left": template | {
            "tp": 1, "tn": 1, "fp": 0, "fn": 0,
            "applicability_tp": 2, "applicability_tn": 0,
            "applicability_fp": 0, "applicability_fn": 0,
        },
        "right": template | {
            "tp": 0, "tn": 1, "fp": 0, "fn": 1,
            "applicability_tp": 1, "applicability_tn": 1,
            "applicability_fp": 0, "applicability_fn": 0,
        },
    })

    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 1.0
    assert metrics["applicability_accuracy"] == 1.0
    assert metrics["n_samples"] == 4


def test_mandatory_sampler_selects_one_point_per_record_with_global_offsets() -> None:
    class FakeDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

    groups = [
        ("left", FakeDataset([(10, 0), (10, 1), (11, 0)])),
        ("right", FakeDataset([(20, 0), (20, 1), (21, 0)])),
    ]
    selected = _mandatory_record_sample_indices(
        groups, seed=42, samples_per_record=1
    )

    assert len(selected) == 4
    assert sum(index in (0, 1) for index in selected) == 1
    assert 2 in selected
    assert sum(index in (3, 4) for index in selected) == 1
    assert 5 in selected


def test_transition_stratification_keeps_transition_and_stable_zones() -> None:
    directions = np.asarray([0] * 100 + [1] * 101, dtype=np.int16)
    samples = np.arange(201, dtype=np.int64)
    valid = np.arange(201, dtype=np.int64)

    selected, selected_near, available_near, switches = (
        _select_stratified_window_indices(
            valid,
            directions,
            samples,
            limit=20,
            seed=42,
            transition_radius_samples=10,
            transition_fraction=0.5,
        )
    )

    assert len(selected) == 20
    assert selected_near == 10
    assert available_near == 21
    assert switches == 1
    assert sum(abs(int(index) - 100) <= 10 for index in selected) == 10


def test_multiple_mandatory_points_are_taken_from_each_record() -> None:
    class FakeDataset:
        samples = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

        def __len__(self):
            return len(self.samples)

    selected = _mandatory_record_sample_indices(
        [("source", FakeDataset())], seed=7, samples_per_record=2
    )

    assert len(selected) == 4
    assert sum(index < 3 for index in selected) == 2
    assert sum(index >= 3 for index in selected) == 2
