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
