import pytest
from scripts.phase5_experiments.run_phase5_hierarchical_ozz import learning_rate_factor, validate_resume_contract


def test_flat_cosine_is_continuous_monotone_and_keeps_floor():
    c = dict(lr_schedule='flat_cosine', lr_decay_start=150., lr_decay_end=300., lr_min_ratio=.1)
    assert learning_rate_factor(c, 0) == 1
    assert learning_rate_factor(c, 149.9) == 1
    assert learning_rate_factor(c, 150) == 1
    assert learning_rate_factor(c, 225) == pytest.approx(.55)
    assert learning_rate_factor(c, 300) == pytest.approx(.1)
    assert learning_rate_factor(c, 350) == pytest.approx(.1)
    values = [learning_rate_factor(c, t/10) for t in range(4000)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_known_checkpoint_can_add_schedule_but_cannot_change_existing_schedule():
    old = dict(implementation_sha256='60de3a4bbe9e4320e5cc60b79a5fea7f4c3c83460ee0c81fd9931c35a4974b62', balanced_events=True, backbone_lr=1e-5)
    new = dict(old, implementation_sha256='current', lr_schedule='flat_cosine',
               lr_decay_start=150., lr_decay_end=300., lr_min_ratio=.1)
    assert validate_resume_contract(old, new, True)
    with pytest.raises(ValueError):
        validate_resume_contract(old, dict(new, backbone_lr=1e-3), True)
    with pytest.raises(ValueError):
        validate_resume_contract(new, dict(new, lr_decay_end=400.), True)
