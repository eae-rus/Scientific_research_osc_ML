import pytest
from scripts.phase5_experiments.run_phase5_hierarchical_ozz import learning_rate_factor


def test_cosine_has_a_single_peak_and_preserves_floor():
    config = dict(lr_warmup_epochs=10., lr_total_epochs=300., lr_initial_ratio=.05, lr_min_ratio=.05)
    values = [learning_rate_factor(config, i/10) for i in range(4001)]
    assert all(a <= b for a, b in zip(values[:100], values[1:101]))
    assert all(a >= b for a, b in zip(values[100:], values[101:]))
    assert values[100] == 1.
    assert values[-1] == pytest.approx(.05)
