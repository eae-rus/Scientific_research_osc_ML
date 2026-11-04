import numpy as np
import pytest
import torch
from osc_tools.ml.phase5_ozz_balance import event_start, average_precision_metrics, balanced_window_loss, estimate_balance
from osc_tools.ml.phase5_hierarchical_ozz import supervised_window


@pytest.mark.parametrize('spp', [24., 32., 18.32])
def test_anchor_survives_each_blocks_fourier_warmup(spp):
    rng = np.random.default_rng(5)
    labels = np.zeros((round(100*spp), 4), np.float32)
    labels[round(31*spp), 2] = 1
    for blocks in [1, 2, 5, 10]:
        length = round(blocks*10*spp)
        for _ in range(10):
            start = event_start(labels, length, spp, 2, rng)
            batch = supervised_window(np.ones((length, 8), np.float32), labels[start:start+length], spp, blocks)
            assert batch['target'][batch['supervised'], 2].any()


def test_ap_ties_absent_and_perfect_ranking():
    y = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
    s = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
    r = average_precision_metrics(s, y)
    assert r['average_precision_per_class'] == [1., None, .25]
    assert r['macro_ap'] == .625


def test_smoothing_does_not_turn_background_into_high_weight_positive():
    z = torch.zeros(1, 4, requires_grad=True)
    balance = dict(positive_weight=[100.]*4, normalizer=[2.]*4)
    balanced_window_loss(z, torch.zeros_like(z), .05, balance).backward()
    assert (z.grad > 0).all()  # Gradient descent lowers negative-example logits.
    short = balanced_window_loss(z.detach(), torch.zeros_like(z), .05, balance)
    long = balanced_window_loss(z.detach().repeat(100, 1), torch.zeros(100, 4), .05, balance)
    torch.testing.assert_close(short, long)


def test_frequency_estimate_weights_windows_equally_and_caps_rare_classes():
    def batch(y):
        return dict(target=y, valid=torch.ones(len(y), dtype=torch.bool), supervised=torch.ones(len(y), dtype=torch.bool))
    data = [batch(torch.ones(1, 4)), batch(torch.zeros(100, 4))]
    result = estimate_balance(data)
    assert result['positive_fraction'] == [.5]*4
    assert result['positive_weight'] == [1.]*4
    with pytest.raises(ValueError, match='absent'):
        estimate_balance(data[1:])


def test_focal_downweights_easy_examples_more_than_errors():
    balance = dict(positive_weight=[1.]*4, normalizer=[1.]*4)
    y = torch.ones(1, 4)
    ratios = []
    for value in [-5., 5.]:
        z = torch.full((1, 4), value, requires_grad=True)
        plain = balanced_window_loss(z, y, .05, balance)
        focal = balanced_window_loss(z, y, .05, balance, 1.)
        ratios.append(float((focal/plain).detach()))
        focal.backward()
        assert torch.isfinite(z.grad).all()
    assert ratios[0] > .99 and ratios[1] < .01


def test_upgrade_only_accepts_known_code_and_explicit_recipe_change():
    from scripts.phase5_experiments.run_phase5_hierarchical_ozz import validate_resume_contract
    old = dict(implementation_sha256='ab4098ec60d58712f4b33100df327c6acd7baed4510f06fa5a9641921d9b6d3a', backbone_lr=1e-5)
    new = dict(old, implementation_sha256='new', balanced_events=True, balance_samples=400,
               positive_weight_cap=100., focal_gamma=1.)
    assert validate_resume_contract(old, new, True)
    assert not validate_resume_contract(new, new)
    with pytest.raises(ValueError):
        validate_resume_contract(old, new)
    with pytest.raises(ValueError):
        validate_resume_contract(old, dict(new, backbone_lr=1e-3), True)
    with pytest.raises(ValueError):
        validate_resume_contract(dict(old, implementation_sha256='unknown'), new, True)
