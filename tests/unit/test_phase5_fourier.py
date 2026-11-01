import numpy as np
import pytest
from osc_tools.ml.phase5_hierarchical_data import build_local_features, build_local_features_reference


@pytest.mark.parametrize('spp', [8., 12., 18.32, 24., 32., 106.6666666667, 128.])
@pytest.mark.parametrize('basis', ['phase', 'line'])
@pytest.mark.parametrize('blocks', [1, 5, 17])
def test_batched_features_exact_reference(spp, basis, blocks):
    raw = np.random.default_rng(83).normal(size=(int(np.floor(blocks*10*spp+.5)), 8)).astype(np.float32)
    raw[:, 3] = np.nan
    raw[7:25, 1] = np.nan
    raw[27, 5] = np.inf
    provenance = np.array([1, 1, 1, 0, 1, 1, 1, 1], np.uint8)
    args = raw, spp, blocks, basis, provenance
    reference = build_local_features_reference(*args)
    actual = build_local_features(*args)
    np.testing.assert_array_equal(actual[0], reference[0])
    np.testing.assert_array_equal(actual[1], reference[1])
    np.testing.assert_array_equal(actual[3], reference[3])
    assert actual[2] == reference[2]


def test_batched_features_cannot_read_previous_block():
    raw = np.random.default_rng(1).normal(size=(640, 8)).astype(np.float32)
    before = build_local_features(raw, 32., 2, 'phase', np.ones(8, np.uint8))
    raw[:320] *= 100
    after = build_local_features(raw, 32., 2, 'phase', np.ones(8, np.uint8))
    np.testing.assert_array_equal(before[0][80:], after[0][80:])


def test_fourier_checkpoint_migration_is_narrow():
    from scripts.phase5_experiments.run_phase5_hierarchical import assert_resume_contract, model_data_hash
    old = dict(model_data_sha256='8858f68cfeacd482ea92fc1dc8ddfa79930141bbca29c461426b997e57789776', lr=.0002)
    new = dict(old, model_data_sha256=model_data_hash())
    assert assert_resume_contract(old, new)
    with pytest.raises(ValueError): assert_resume_contract(old, dict(new, lr=.001))
    with pytest.raises(ValueError): assert_resume_contract(old, dict(new, model_data_sha256='unverified'))
