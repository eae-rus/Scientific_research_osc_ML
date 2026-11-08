import numpy as np
import torch
import pytest
from osc_tools.ml.phase5_ozz_features import continuous_features, expand_labels, cached_expanded_labels
from osc_tools.ml.phase5_hierarchical_data import ExactFrequencyBuilder, token_positions
from osc_tools.ml.phase5_hierarchical_ozz import supervised_window
from osc_tools.ml.phase5_ozz_inference import prepare_chunks


@pytest.mark.parametrize('spp,blocks', [(24., 3), (13.7, 4), (128., 60)])
def test_continuous_fourier_matches_direct_whole_history(spp, blocks):
    n = int(np.floor(10*spp*blocks+.5))
    raw = np.random.default_rng(12).normal(size=(n, 8)).astype(np.float32)
    provenance = np.ones(8, np.uint8)
    f, missing, _, positions = continuous_features(raw, spp, blocks, 'phase', provenance)
    expected, mask, _ = ExactFrequencyBuilder(spp).build(raw, round(spp), positions,
        voltage_basis='phase', channel_provenance=provenance, phasor_backend='prefix')
    np.testing.assert_array_equal(missing, mask)
    np.testing.assert_allclose(f, expected, atol=2e-5, rtol=2e-5, equal_nan=True)


def test_no_repeated_warmup_and_window_uses_same_prior_history():
    raw = np.random.default_rng(3).normal(size=(960, 8)).astype(np.float32)
    labels = np.zeros((960, 4), np.float32)
    full = supervised_window(raw, labels, 32., 3, feature_mode='continuous_record')
    assert not full['supervised'][:7].any()
    assert full['supervised'][7:].all()
    window = supervised_window(raw[320:], labels[320:], 32., 2,
                               feature_mode='continuous_record', history=raw[:320])
    torch.testing.assert_close(window['z'], full['z'][80:])
    assert window['supervised'].all()


def test_context_boundary_retains_fourier_history():
    raw = np.ones((960, 8), np.float32)
    chunks = list(prepare_chunks(raw, 32., 2, feature_mode='continuous_record'))
    assert len(chunks) == 2
    assert chunks[1][3].all()


@pytest.mark.parametrize('spp', [24., 32., 13.7])
def test_label_tail_is_forward_union_for_all_classes(spp):
    y = np.zeros((150, 4), np.float32)
    y[10:14, 0] = 1; y[[20, 30], 1] = 1; y[70, 2] = 1; y[149, 3] = 1
    result = expand_labels(y, spp)
    expected = y.astype(bool)
    for c in range(4):
        for point in np.flatnonzero(y[:, c]):
            expected[point:min(150, point+int(np.floor(spp))+1), c] = True
    np.testing.assert_array_equal(result, expected)
    assert result[14:20, 0].all()  # A cropped window retains the preceding event's tail.
    assert result[149, 3]
    assert not result[:10].any()


def test_label_cache_survives_reload_without_repeating_expansion(tmp_path, monkeypatch):
    import osc_tools.ml.phase5_ozz_features as module
    y = np.zeros((32000, 4)); y[100:500, 0] = 1
    original = y.copy()
    first = cached_expanded_labels(y, 32., tmp_path)
    monkeypatch.setattr(module, 'expand_labels', lambda *a: (_ for _ in ()).throw(AssertionError('cache miss')))
    np.testing.assert_array_equal(cached_expanded_labels(y, 32., tmp_path), first)
    np.testing.assert_array_equal(y, original)
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) < 10000
