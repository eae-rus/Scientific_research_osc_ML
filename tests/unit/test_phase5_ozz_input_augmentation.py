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


def test_adaptation_schedule_and_strict_resume_migration():
    config = dict(lr_schedule='adaptation_cosine', lr_restart_epoch=300., lr_restart_warmup=2.,
                  lr_restart_initial=.1, lr_restart_peak=1., lr_decay_start=302., lr_decay_end=400., lr_min_ratio=.1)
    for epoch, expected in ((300, .1), (301, .55), (302, 1.), (351, .55), (400, .1), (450, .1)):
        assert learning_rate_factor(config, epoch) == pytest.approx(expected)
    old = dict(implementation_sha256='709b69182cb4bb5bf47dbe36cc7e09bdc4f07ca10fff1b876f2b85ebbeaa02c4',
               balanced_events=True, backbone_lr=1e-5, lr_schedule='flat_cosine', lr_decay_start=150., lr_decay_end=300., lr_min_ratio=.1)
    new = dict(old, **{k: v for k, v in config.items()}, current_dropout_all=.2, current_dropout_phase=.25,
               current_zero_probability=.5, validation_current_samples=40)
    new['implementation_sha256'] = 'current'
    assert validate_resume_contract(old, new, True)
    with pytest.raises(ValueError):
        validate_resume_contract(old, dict(new, backbone_lr=1e-3), True)
    with pytest.raises(ValueError):
        validate_resume_contract(new, dict(new, lr_restart_epoch=400.), True)
