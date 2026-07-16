"""Связка lazy real-source contract и feature contract v2."""

import numpy as np
import pytest

from osc_tools.ml.lazy_multi_dataset import (
    LazyMultiSourceDataset,
    SpectralMultiSourceDataset,
)
from osc_tools.ml.phase5_sources import DatasetSource
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig


class _SyntheticSource(DatasetSource):
    name = "synthetic"

    def __init__(self, spp: int = 20, periods: int = 40) -> None:
        self.spp = spp
        t = np.arange(spp * periods, dtype=np.float32) / spp
        phases = np.stack([
            np.sin(2 * np.pi * t),
            np.sin(2 * np.pi * t - 2 * np.pi / 3),
            np.sin(2 * np.pi * t + 2 * np.pi / 3),
        ])
        self.raw = np.vstack((phases, np.zeros((1, len(t))), phases, np.zeros((1, len(t))))).astype(np.float32)

    def __len__(self) -> int:
        return 2

    def get_metadata(self, idx: int) -> dict[str, object]:
        return {"spp": self.spp, "voltage_basis": "phase", "record_id": idx}

    def load_signal(self, idx: int) -> np.ndarray:
        return self.raw


def _raw_dataset(history_periods: float = 10.0) -> LazyMultiSourceDataset:
    source = _SyntheticSource()
    return LazyMultiSourceDataset(
        {"synthetic": source}, {"synthetic": 1.0}, samples_per_epoch=4,
        window_periods=10.0, history_periods=history_periods, seed=7,
    )


def test_raw_sampling_is_deterministic_and_includes_causal_history() -> None:
    dataset = _raw_dataset()
    first = dataset[2]
    repeated = dataset[2]
    assert np.array_equal(first["raw"], repeated["raw"])
    assert first["metadata"] == repeated["metadata"]
    assert first["raw"].shape == (8, 400)
    assert first["metadata"]["history_samples"] == 200


@pytest.mark.parametrize("mode, expected_length", [("snapshot_2", 2), ("snapshot_5", 5), ("sequence_1_8", 67)])
def test_spectral_wrapper_returns_v2_features(mode: str, expected_length: int) -> None:
    dataset = SpectralMultiSourceDataset(
        _raw_dataset(),
        SpectralFeatureBuilder(SpectralFeatureConfig("B")),
        temporal_mode=mode,
    )
    sample = dataset[0]
    assert sample["features"].shape == (expected_length, 156)
    assert sample["target"].shape == sample["features"].shape
    assert sample["missing_mask"].shape == sample["features"].shape
    assert sample["provenance"].shape == sample["features"].shape
    assert set(np.unique(sample["provenance"])) <= {0, 1, 2}
    assert sample["metadata"]["feature_contract"] == "feature_contract_v2_b"


def test_spectral_wrapper_rejects_noncausal_short_history() -> None:
    with pytest.raises(ValueError, match="history_periods"):
        SpectralMultiSourceDataset(
            _raw_dataset(history_periods=0.0),
            SpectralFeatureBuilder(SpectralFeatureConfig("A")),
        )
