import numpy as np
import torch
from osc_tools.ml.phase5_ozz_inference import prepare_chunks, predict_record, mark_record
from osc_tools.ml.phase5_hierarchical_ozz import supervised_window
from osc_tools.ml.phase5_ozz_evaluation import event_counts, threshold_report


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return torch.ones((*batch['valid'].shape, 4))


def test_features_match_training_and_tail_is_not_valid():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(351, 8)).astype(np.float32)
    chunks = list(prepare_chunks(raw, 32., 150))
    assert len(chunks) == 1
    batch, starts, ends, usable = chunks[0]
    reference = supervised_window(raw[:320], np.zeros((320, 4)), 32., 1)
    for key in ('z', 'state', 'times'):
        torch.testing.assert_close(batch[key][:80], reference[key], rtol=0, atol=0)
    assert not usable[ends > len(raw)].any()
    assert starts[0] == 0 and ends[79] == 320
    assert usable[:80].sum() == reference['supervised'].sum()


def test_noninteger_spp_chunk_partition():
    raw = np.ones((971, 8), np.float32)
    chunks = list(prepare_chunks(raw, 12.3, 2))
    left = np.concatenate([r[1] for r in chunks])
    right = np.concatenate([r[2] for r in chunks])
    assert np.array_equal(left[1:], right[:-1])
    assert np.max(right-left)-np.min(right-left) == 1


def test_shared_cache_ignores_threshold_but_not_input(tmp_path, monkeypatch):
    import osc_tools.ml.phase5_ozz_inference as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    model = ConstantModel()
    raw = np.ones((640, 8), np.float32)
    config = dict(inference_sha256='test', max_seconds=30., amp_bf16=False)
    p, coverage = mark_record(model, raw, config, torch.device('cpu'), 1600.)
    assert model.calls == 1
    assert coverage.max() == 1
    assert np.isnan(p[coverage == 0]).all()
    predict_record(model, raw, dict(config, threshold=.2), torch.device('cpu'), 1600.)
    assert model.calls == 1
    raw[50, 0] += 1
    predict_record(model, raw, config, torch.device('cpu'), 1600.)
    assert model.calls == 2


def test_one_token_events_and_tolerance_do_not_relabel():
    truth = np.array([0, 0, 1, 0, 0], bool)
    pred = np.array([0, 1, 0, 0, 0], bool)
    valid = np.ones(5, bool)
    assert event_counts(pred, truth, valid).tolist() == [0, 1, 1, 0, 1]
    assert event_counts(pred, truth, valid, 1).tolist() == [1, 0, 0, 1, 1]


def test_missing_warmup_does_not_split_true_long_event():
    valid = np.array([1, 1, 0, 0, 1, 1], bool)
    truth = np.ones(6, bool)
    assert event_counts(truth, truth, valid).tolist() == [1, 0, 0, 0, 0]


def test_one_long_prediction_cannot_claim_many_events():
    truth = np.array([0, 1, 0, 1, 0], bool)
    assert event_counts(np.ones(5, bool), truth, np.ones(5, bool)).tolist() == [1, 0, 1, 1, 2]


def test_lower_threshold_reports_extra_false_alarms():
    p = np.full((5, 4), .4)
    record = dict(p=p, y=np.zeros_like(p), v=np.ones(5, bool), group=4)
    report = threshold_report([record], (.3, .5))
    assert report['thresholds']['0.3']['background_fp_files'] == 1
    assert report['thresholds']['0.5']['background_fp_files'] == 0


def test_real_statistics_keeps_absolute_time_and_unknown_gaps():
    from scripts.phase4_experiments.real_ozz.collect_real_ozz_statistics import compute_file_statistics
    probs = np.zeros((100, 4))
    probs[:20] = np.nan
    probs[30:40, 0] = 1
    probs[40:50] = np.nan
    probs[50:60, 0] = 1
    result = compute_file_statistics(probs, 1000., expansion_zones=0)
    key = 'Stable_SPGF'
    assert result['temporal'][key]['onset_ms'] == 30.
    assert result['events'][key]['n_events'] == 2
    assert result['events'][key]['total_samples'] == 20


def test_vectorized_dilation_matches_reference():
    from scripts.phase4_experiments.real_ozz.collect_real_ozz_statistics import expand_zones
    rng = np.random.default_rng(3)
    for size in (1, 8, 100):
        mask = rng.random(size) > .7
        for radius in (0, 1, 16, 200):
            expected = mask.copy()
            for i in np.flatnonzero(mask):
                expected[max(0, i-radius):min(size, i+radius+1)] = True
            np.testing.assert_array_equal(expand_zones(mask, radius), expected)


def test_unobserved_events_are_reported_not_counted_as_misses():
    y = np.zeros((5, 4)); y[1, 0] = 1
    record = dict(p=np.zeros_like(y), y=y, v=np.array([1, 0, 1, 1, 1], bool), group=0)
    report = threshold_report([record], (.5,))
    assert report['unobserved_events_per_class'] == [1, 0, 0, 0]
    assert report['thresholds']['0.5']['single_token_events'] == [0, 0, 0, 0]


def test_stratified_weights_and_nested_simulated_selection(tmp_path, monkeypatch):
    import json
    from osc_tools.ml.phase5_ozz_evaluation import select_records
    class Source:
        def get_metadata(self, i):
            return dict(n_samples=1200*(1 if i < 10 else 20))
    import osc_tools.ml.dataset_registry as registry
    monkeypatch.setattr(registry, 'create_source', lambda *a: Source())
    records = [dict(group=g, path=f'{g}_{i}') for g in range(4) for i in range(20)]
    records += [dict(group=4, index=i, registry='test', name=f'bg_{i}', spp=24., f_network=50.)
                for i in range(12)]
    (tmp_path/'split.json').write_text(json.dumps(dict(validation=records)))
    small = select_records(tmp_path/'checkpoint.pt', 2, 6)
    large = select_records(tmp_path/'checkpoint.pt', 5, 6)
    assert {r['path'] for r in small if r['group'] < 4} <= {r['path'] for r in large if r['group'] < 4}
    background = [r for r in small if r['group'] == 4]
    assert len(background) == 6
    assert sum(r['population_weight'] for r in background) == 12
    assert len({r['evaluation_stratum'] for r in background}) == 2
