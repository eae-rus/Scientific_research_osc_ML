import numpy as np
import pytest
from osc_tools.ml.phase5_ozz_input_augmentation import augment_current_channels
from osc_tools.ml.phase5_hierarchical_ozz import supervised_window
from scripts.phase5_experiments.run_phase5_hierarchical_ozz import learning_rate_factor, validate_resume_contract


@pytest.mark.parametrize('zero', [0., 1.])
def test_all_current_corruption_retains_voltages_labels_and_usable_tokens(zero):
    raw = np.random.default_rng(1).normal(size=(320, 8)).astype(np.float32)
    provenance = np.ones(8, np.uint8)
    raw[:, 3] = np.nan; provenance[3] = 0
    original = raw.copy()
    x, state, _ = augment_current_channels(raw, provenance, np.random.default_rng(2), 1., 0., zero)
    np.testing.assert_array_equal(x[:, 4:], original[:, 4:])
    np.testing.assert_array_equal(raw, original)
    assert np.isnan(x[:, 3]).all()
    assert np.all(x[:, :3] == 0) if zero else np.isnan(x[:, :3]).all()
    labels = np.zeros((320, 4), np.float32); labels[50, 1] = 1
    before = supervised_window(raw, labels, 32, 1, provenance=provenance)
    after = supervised_window(x, labels, 32, 1, provenance=state)
    np.testing.assert_array_equal(after['target'], before['target'])
    np.testing.assert_array_equal(after['supervised'], before['supervised'])
    assert (after['state'][7:, :, 3:] != 0).any()
    if not zero:
        assert not after['state'][:, :, :3].any()


def test_single_phase_only_and_reproducible():
    raw = np.ones((32, 8), np.float32)
    for seed in range(30):
        first = augment_current_channels(raw, None, np.random.default_rng(seed), 0., 1., .5)
        second = augment_current_channels(raw, None, np.random.default_rng(seed), 0., 1., .5)
        np.testing.assert_array_equal(first[0], second[0])
        assert np.count_nonzero(~np.all(first[0] == raw, axis=0)) == 1
        np.testing.assert_array_equal(first[0][:, 3:], raw[:, 3:])


def test_disabled_recipe_consumes_no_randomness():
    rng = np.random.default_rng(7); original = np.random.default_rng(7)
    augment_current_channels(np.ones((2, 8)), None, rng)
    assert rng.random() == original.random()


def test_warmup_cosine_and_strict_resume():
    config = dict(lr_warmup_epochs=10., lr_total_epochs=300., lr_initial_ratio=.05, lr_min_ratio=.05)
    for epoch, expected in ((0, .05), (5, .525), (10, 1.), (155, .525), (300, .05)):
        assert learning_rate_factor(config, epoch) == pytest.approx(expected)
    assert validate_resume_contract(config, dict(config)) is False
    with pytest.raises(ValueError):
        validate_resume_contract(config, dict(config, lr_total_epochs=400.))
