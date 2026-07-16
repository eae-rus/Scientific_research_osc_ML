"""Физические contracts feature builder v2."""

import numpy as np

from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig


def _balanced_raw(spp: int = 32) -> np.ndarray:
    t = np.arange(spp) / spp
    ia = np.sin(2 * np.pi * t)
    return np.stack((ia, np.sin(2 * np.pi * t - 2*np.pi/3), np.sin(2 * np.pi * t + 2*np.pi/3),
                     np.zeros(spp), ia, np.sin(2 * np.pi * t - 2*np.pi/3), np.sin(2 * np.pi * t + 2*np.pi/3), np.zeros(spp)), axis=1)


def test_version_a_preserves_220_external_features() -> None:
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("A"))
    features, mask, metadata = builder.build(_balanced_raw(), spp=32)
    assert features.shape == (1, 220)
    assert mask.shape == features.shape
    assert metadata["feature_contract"] == "feature_contract_v2_a"


def test_version_b_has_only_symmetric_features_and_masks_unavailable_harmonics() -> None:
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("B"))
    features, mask, metadata = builder.build(_balanced_raw(12), spp=12)
    assert features.shape == (1, 156)
    assert metadata["available_harmonics"] == [1, 2, 3, 4, 5, 6]
    assert mask.any()


def test_balanced_positive_sequence_dominates_h1_symmetric_components() -> None:
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("B", low_periods=()))
    features, _, metadata = builder.build(_balanced_raw(64), spp=64)
    names = metadata["feature_names"]
    magnitudes = {name: features[0, index] for index, name in enumerate(names) if name.endswith("h1_magnitude")}
    assert magnitudes["I1_h1_magnitude"] > 0.99
    assert magnitudes["I2_h1_magnitude"] < 1e-5
    assert magnitudes["I0_h1_magnitude"] < 1e-5


def test_same_physical_phase_at_50_and_60_hz() -> None:
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("B", low_periods=()))
    at_50, _, metadata_50 = builder.build(_balanced_raw(32), spp=32)
    at_60, _, metadata_60 = builder.build(_balanced_raw(32), spp=32)
    assert metadata_50["spp"] == metadata_60["spp"] == 32
    assert np.allclose(at_50, at_60, atol=1e-6, equal_nan=True)


def test_line_voltage_basis_masks_only_zero_sequence_voltage() -> None:
    raw = _balanced_raw(32)
    raw[:, 4] = raw[:, 4] - raw[:, 5]
    raw[:, 5] = raw[:, 5] - raw[:, 6]
    raw[:, 6] = -(raw[:, 4] + raw[:, 5])
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("B", standard_harmonics=1, low_periods=()))
    features, mask, metadata = builder.build(raw, spp=32, voltage_basis="line")
    u0_indices = [i for i, name in enumerate(metadata["feature_names"]) if name.startswith("U0_")]
    other_indices = [i for i in range(features.shape[1]) if i not in u0_indices]
    assert mask[:, u0_indices].all()
    assert not mask[:, other_indices].any()
    assert all(metadata["feature_provenance"][i] == 0 for i in u0_indices)
    derived_indices = [i for i, name in enumerate(metadata["feature_names"]) if name.startswith(("I1_", "U1_"))]
    assert all(metadata["feature_provenance"][i] == 2 for i in derived_indices)
