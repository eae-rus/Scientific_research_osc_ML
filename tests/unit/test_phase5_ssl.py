"""Masked spectral modeling и checkpoint passport Phase 5."""

import numpy as np
import pytest

from osc_tools.ml.checkpoint_contracts import FeatureContractPassport
from osc_tools.ml.phase5_ssl import MaskedSpectralDataset, SpectralMaskingConfig, apply_group_mask
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig


def test_group_mask_is_deterministic_and_never_targets_missing() -> None:
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("B", standard_harmonics=1, low_periods=()))
    features = np.ones((5, len(builder.schema.names)), dtype=np.float32)
    missing = np.zeros_like(features, dtype=bool)
    missing[:, :2] = True
    first, first_mask = apply_group_mask(features, missing, builder.schema, SpectralMaskingConfig(0.25), 17)
    second, second_mask = apply_group_mask(features, missing, builder.schema, SpectralMaskingConfig(0.25), 17)
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert not np.any(first_mask & missing)
    assert np.all(first[first_mask] == 0.0)
    assert np.all(features[~first_mask] == first[~first_mask])


def test_checkpoint_passport_detects_feature_mismatch() -> None:
    a = SpectralFeatureBuilder(SpectralFeatureConfig("A"))
    b = SpectralFeatureBuilder(SpectralFeatureConfig("B"))
    passport_a = FeatureContractPassport.create(a.schema, "sequence_1_8", True, True)
    passport_a.assert_compatible(passport_a.to_dict())
    with pytest.raises(ValueError, match="Несовместимый"):
        passport_a.assert_compatible(
            FeatureContractPassport.create(b.schema, "sequence_1_8", True, True).to_dict()
        )


def test_masked_dataset_retries_fully_missing_feature_sample() -> None:
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("B", standard_harmonics=1, low_periods=()))
    channels = len(builder.schema.names)

    class _Dataset:
        feature_builder = builder

        def __len__(self):
            return 2

        def __getitem__(self, index):
            missing = np.ones((2, channels), dtype=bool) if index == 0 else np.zeros((2, channels), dtype=bool)
            features = np.full((2, channels), np.nan, dtype=np.float32) if index == 0 else np.ones((2, channels), dtype=np.float32)
            return {
                "features": features,
                "target": features.copy(),
                "missing_mask": missing,
                "provenance": np.where(missing, 0, 2).astype(np.uint8),
                "metadata": {"source": "synthetic", "record_index": index},
            }

    sample = MaskedSpectralDataset(_Dataset(), SpectralMaskingConfig(0.25), max_attempts=2)[0]
    assert sample["metadata"]["requested_dataset_index"] == 0
    assert sample["metadata"]["sampled_dataset_index"] == 1
    assert sample["metadata"]["ssl_resample_attempt"] == 1
    assert sample["reconstruction_mask"].any()
