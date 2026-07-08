"""Тесты базовых контрактов Phase 5."""

import pytest

from osc_tools.ml.phase5_contracts import (
    CHANNEL_ORDER,
    ChannelProvenance,
    TimebaseContract,
    available_harmonics,
    snapshot_indices,
)


def test_channel_order_and_provenance_are_explicit() -> None:
    assert CHANNEL_ORDER == ("IA", "IB", "IC", "IN", "UA", "UB", "UC", "UN")
    assert ChannelProvenance.MISSING != ChannelProvenance.MEASURED
    assert ChannelProvenance.DERIVED != ChannelProvenance.MEASURED


@pytest.mark.parametrize(
    ("sampling_rate", "network_frequency", "expected_spp", "expected_stride"),
    [
        (600.0, 50.0, 12, 2),
        (916.0, 50.0, 18, 2),
        (1600.0, 50.0, 32, 4),
        (1920.0, 60.0, 32, 4),
        (6400.0, 50.0, 128, 16),
    ],
)
def test_timebase_contract_for_dataset_groups(
    sampling_rate: float,
    network_frequency: float,
    expected_spp: int,
    expected_stride: int,
) -> None:
    contract = TimebaseContract.create(sampling_rate, network_frequency)

    assert contract.spp == expected_spp
    assert contract.window_samples == expected_spp * 10
    assert contract.stride_samples == expected_stride
    assert contract.to_metadata()["actual_stride_periods"] == pytest.approx(
        expected_stride / expected_spp
    )


def test_available_harmonics_respects_nyquist() -> None:
    assert available_harmonics(spp=12, requested=9) == (1, 2, 3, 4, 5, 6)
    assert available_harmonics(spp=32, requested=9) == tuple(range(1, 10))


def test_snapshot_modes_include_boundaries_without_duplicates() -> None:
    assert snapshot_indices(10, 89, 2) == (10, 89)
    indices = snapshot_indices(10, 89, 5)
    assert indices == (10, 30, 50, 69, 89)
    assert len(set(indices)) == 5


def test_snapshot_mode_rejects_too_short_sequence() -> None:
    with pytest.raises(ValueError, match="Недостаточно"):
        snapshot_indices(3, 5, 5)


@pytest.mark.parametrize(
    "sampling_rate,network_frequency",
    [(0.0, 50.0), (1600.0, 0.0), (float("nan"), 50.0)],
)
def test_timebase_rejects_invalid_frequencies(
    sampling_rate: float,
    network_frequency: float,
) -> None:
    with pytest.raises(ValueError):
        TimebaseContract.create(sampling_rate, network_frequency)
