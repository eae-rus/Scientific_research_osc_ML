"""OZZ transfer, label alignment and missing-input contracts."""
from dataclasses import replace
import numpy as np
import polars as pl
import torch
import pytest
from osc_tools.ml.models.phasor_hierarchical import HierarchicalConfig, PhasorHierarchicalKAN
from osc_tools.ml.phase5_hierarchical_ozz import supervised_window, normalized_real
from scripts.phase5_experiments.run_phase5_hierarchical_ozz import OZZClassifier


def test_supervised_input_matches_ssl_builder_without_hidden_targets():
    from osc_tools.ml.phase5_hierarchical_data import build_local_features
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(640, 8)).astype(np.float32)
    raw[:, 3] = np.nan
    labels = np.zeros((640, 4), np.float32)
    labels[39, 2] = 1
    batch = supervised_window(raw, labels, 32, 2)
    features, missing, _, _ = build_local_features(raw, 32, 2, 'phase', np.isfinite(raw).any(0).astype(np.uint8))
    polar = np.nan_to_num(features).reshape(160, 13, 6, 2)
    expected = np.stack((polar[..., 0]*np.cos(polar[..., 1]), polar[..., 0]*np.sin(polar[..., 1])), -1)
    expected[missing.reshape(160, 13, 6, 2).any(-1)] = 0
    np.testing.assert_allclose(batch['z'], expected)
    assert not (batch['state'] == 3).any()
    assert batch['target'][9, 2] == 1 and batch['target'].sum() == 1
    assert not batch['supervised'][:7].any() and not batch['supervised'][80:87].any()


def test_real_scaling_uses_correct_bus_and_preserves_missing():
    frame = pl.DataFrame({'file_name': ['example_Bus 2 _event N1']*2, 'IA': [100., 200.],
                          'UA BB': [300., 600.], 'UB BB': [None, None], 'UB CL': [600., 1200.]})
    nominal = {'example': {'norm': 'YES', '2Ip_base': 5., '2Ub_base': 100., '2Uc_base': 200., '1Ip_base': 99.}}
    raw = normalized_real(frame, nominal)
    np.testing.assert_allclose(raw[:, 0], [1, 2])
    np.testing.assert_allclose(raw[:, 4], [1, 2])
    np.testing.assert_allclose(raw[:, 5], [1, 2])
    assert np.isnan(raw[:, 3]).all()
    assert normalized_real(frame, {}) is None


def test_ozz_head_uses_global_context_and_backpropagates_without_changing_ssl():
    cfg = replace(HierarchicalConfig(), channel_width=8, width=16, channel_layers=1,
                  local_layers=1, global_layers=1, fusion_layers=1, dropout=0, block_tokens=4)
    backbone = PhasorHierarchicalKAN(cfg)
    model = OZZClassifier(backbone).train()
    batch = dict(z=torch.randn(1, 7, 13, 6, 2)*.1, state=torch.ones(1, 7, 13, 6, dtype=torch.long),
                 times=torch.arange(7)[None].float()/8, valid=torch.ones(1, 7, dtype=torch.bool))
    logits = model(batch)
    assert logits.shape == (1, 7, 4)
    logits[:, :4].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.global_blocks.parameters())
    assert model.classifier.weight.grad.abs().sum() > 0
    assert not backbone.head._forward_pre_hooks
    before = logits.detach()
    batch['z'][:, 4:] *= 10
    after = model(batch).detach()
    assert not torch.allclose(before[:, :4], after[:, :4], atol=1e-8, rtol=1e-8)


def test_manual_ozz_entry_selects_phase5_without_training(monkeypatch):
    from scripts.phase4_experiments.sim_ozz import run_phase4_finetune_sim_ozz as entry
    from scripts.phase5_experiments import run_phase5_hierarchical_ozz as engine
    seen = []
    monkeypatch.setattr(engine, 'run', lambda cfg: seen.append(cfg))
    entry.run_manual()
    assert len(seen) == 1 and seen[0]['resume']
    assert seen[0]['background_repeats'] == 2
    assert 'phase5' in seen[0]['output'] and seen[0]['ssl_checkpoint'].endswith('latest_checkpoint.pt')
    assert seen[0]['balanced_events'] and seen[0]['allow_balance_upgrade']

@pytest.mark.parametrize('balanced,upgrade,loader_failure', [(False, False, False), (True, False, False), (True, True, False), (True, False, True), (True, False, 'optimizer')])
def test_ozz_resume_matches_uninterrupted_training(tmp_path, monkeypatch, balanced, upgrade, loader_failure):
    from dataclasses import asdict
    from scripts.phase5_experiments import run_phase5_hierarchical_ozz as engine
    cfg = replace(HierarchicalConfig(), channel_width=8, width=16, channel_layers=1,
                  local_layers=1, global_layers=1, fusion_layers=1, dropout=.05)
    checkpoint = tmp_path/'ssl.pt'
    torch.save(dict(contract={'architecture': asdict(cfg), 'data_contract': 'phase5_hierarchical_b_reim_v1'},
                    model=PhasorHierarchicalKAN(cfg).state_dict(), epoch=1, step=7), checkpoint)
    raw = np.random.default_rng(8).normal(size=(320, 8)).astype(np.float32)*.1
    records = [dict(kind='real', name='fixture', raw=raw, group=4)]
    monkeypatch.setattr(engine, 'prepare_records', lambda *a: ({'train': records, 'validation': records}, {'fixture': True}))
    monkeypatch.setattr(engine, 'export_curves', lambda *a: None)
    monkeypatch.setattr(engine, 'estimate_balance', lambda *a: dict(positive_weight=[4.]*4, normalizer=[1.6]*4))
    config = dict(ssl_checkpoint=str(checkpoint), output=str(tmp_path/'continuous'), resume=False,
                  epochs=1, samples_per_epoch=5, validation_samples=2, target_tokens=160, max_seconds=.2,
                  backbone_lr=1e-5, head_lr=1e-4, use_real_no_ozz=False, workers=0, save_every=1,
                  console_interval=2., seed=42, label_error_probability=.05, drop_path_rate=.3,
                  balanced_events=balanced, focal_gamma=1. if balanced else 0.)
    engine.run(config)
    config['output'] = str(tmp_path/'resumed')
    save_original = engine.atomic_save
    def interrupt_after_first_step(state, path):
        save_original(state, path)
        if state['step'] == 1:
            (path.parent/'STOP').touch()
    original_loader = engine.loader
    original_step = torch.optim.AdamW.step
    if loader_failure == 'optimizer':
        def failing_step(self, *args, **kwargs):
            original_step(self, *args, **kwargs)
            raise RuntimeError('synthetic failure after optimizer mutated tensors')
        monkeypatch.setattr(torch.optim.AdamW, 'step', failing_step)
    elif loader_failure:
        def failing_loader(dataset, workers, cursor=0):
            for i, batch in enumerate(original_loader(dataset, workers, cursor)):
                if dataset.train and i == 1:
                    raise OSError('synthetic disk failure during accumulation')
                yield batch
        monkeypatch.setattr(engine, 'loader', failing_loader)
    else:
        monkeypatch.setattr(engine, 'atomic_save', interrupt_after_first_step)
    engine.run(config)
    if loader_failure:
        failed = torch.load(tmp_path/'resumed/latest_checkpoint.pt', weights_only=False)
        assert failed['cursor'] == 0 and failed['step'] == 0
        monkeypatch.setattr(engine, 'loader', original_loader)
        monkeypatch.setattr(torch.optim.AdamW, 'step', original_step)
    else:
        (tmp_path/'resumed/STOP').unlink()
    monkeypatch.setattr(engine, 'atomic_save', save_original)
    config['resume'] = True
    if upgrade:
        old_path = tmp_path/'resumed/latest_checkpoint.pt'
        old = torch.load(old_path, weights_only=False)
        for key in ('balanced_events', 'balance_samples', 'positive_weight_cap', 'focal_gamma'):
            old['contract'].pop(key)
        old['contract']['implementation_sha256'] = 'ab4098ec60d58712f4b33100df327c6acd7baed4510f06fa5a9641921d9b6d3a'
        old.pop('balance')
        torch.save(old, old_path)
        config['allow_balance_upgrade'] = True
    # Continued OZZ does not depend on an available SSL checkpoint.
    checkpoint.unlink()
    engine.run(config)
    full = torch.load(tmp_path/'continuous/latest_checkpoint.pt', weights_only=False)
    resumed = torch.load(tmp_path/'resumed/latest_checkpoint.pt', weights_only=False)
    if upgrade:
        assert (tmp_path/'resumed/before_balance_step_1.pt').exists()
        assert len(resumed['balance_history']) == 1
    assert (full['epoch'], full['cursor'], full['step'], full['best']) == (resumed['epoch'], resumed['cursor'], resumed['step'], resumed['best'])
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)


def test_expert_containment_excludes_all_buses_events_and_both_verifications(tmp_path):
    from osc_tools.ml.phase5_hierarchical_ozz import select_real_background
    report = tmp_path/'report.csv'
    report.write_text('filename;Проверка\nabc123.cfg;+\ndef456;-\n', encoding='utf-8-sig')
    names = ['prefix_ABC123_Bus 1', 'abc123_Bus 2 _event N3', 'def456_Bus 1', 'other_Bus 2']
    selected, audit = select_real_background(names, report)
    assert selected == ['other_Bus 2']
    assert len(audit['excluded_files']) == 3 and audit['report_identifiers'] == 2


def test_binary_smoothing_keeps_independent_outputs():
    from osc_tools.ml.phase5_ozz_regularization import smooth_binary_targets
    target = torch.tensor([[0., 0., 0., 0.], [1., 1., 0., 0.]])
    result = smooth_binary_targets(target, .05)
    torch.testing.assert_close(result, torch.tensor([[.05]*4, [.95, .95, .05, .05]]))
    torch.testing.assert_close(smooth_binary_targets(target, 0), target)


def test_unlabeled_population_uses_all_records_and_preserves_ssl_groups(tmp_path, monkeypatch):
    from osc_tools.ml import phase5_hierarchical_ozz as data
    registry = tmp_path/'registry.json'
    registry.write_text('{}')
    report = tmp_path/'report.csv'
    report.write_text('filename;Проверка\nexpert;+\nfalse_alarm;-\n')
    names = ['train_Bus 1', 'train_Bus 2', 'val_Bus 1', 'held_Bus 1',
             'prefix_expert_Bus 1', 'false_alarm_Bus 1', 'short_Bus 1']
    class Source:
        manifest_path = registry
        def __len__(self): return len(names)
        def get_metadata(self, i):
            return dict(file_name=names[i], f_network=60., f_adc=1200.,
                        n_samples=100 if i == 6 else 1200, normalized=True)
    monkeypatch.setattr(data, 'create_source', lambda *a: Source())
    splits = {'sources': {'open_ee': {'splits': {'train': [0, 4, 5, 6], 'validation': [1, 2], 'holdout': [3]}}}}
    records, audit = data.prepare_unlabeled_background(tmp_path, dict(registry=registry, expert_report=report), splits)
    assert [r['index'] for r in records['train']] == [0, 1]
    assert [r['index'] for r in records['validation']] == [2]
    assert audit['holdout_indices'] == [3] and audit['unusable_indices'] == [6]
    assert all(r['spp'] == 20 and r['f_network'] == 60 for r in records['train'])


def test_background_traversal_visits_every_record_across_epochs(monkeypatch):
    from osc_tools.ml import phase5_hierarchical_ozz as data
    records = [dict(kind='real', name=str(i), raw=np.full((320, 8), i, np.float32), group=4) for i in range(7)]
    seen = []
    def build(raw, *a, **kw):
        seen.append(int(raw[0, 0]))
        return {}
    monkeypatch.setattr(data, 'supervised_window', build)
    dataset = data.OZZDataset(records, 3, 42, .2, True)
    # Disable amplitude augmentation while retaining the epoch-dependent schedule.
    class RNG:
        def choice(self, choices): return choices[0]
        def integers(self, *a): return 0
        def random(self): return 1.
    monkeypatch.setattr(np.random, 'default_rng', lambda *a: RNG())
    for serial in range(7):
        dataset.epoch = serial//3
        dataset[serial%3]
    assert sorted(seen) == list(range(7))


def test_stochastic_depth_preserves_ssl_weights_eval_and_skips_whole_block(monkeypatch):
    from osc_tools.ml.phase5_ozz_regularization import OZZRegularizedBackbone
    cfg = replace(HierarchicalConfig(), channel_width=8, width=16, channel_layers=1,
                  local_layers=1, global_layers=1, fusion_layers=1, dropout=0, block_tokens=4)
    original = PhasorHierarchicalKAN(cfg).eval()
    model = OZZRegularizedBackbone(cfg, .5).eval()
    model.load_state_dict(original.state_dict(), strict=True)
    assert model.state_dict().keys() == original.state_dict().keys()
    args = (torch.randn(1, 7, 13, 6, 2)*.1, torch.ones(1, 7, 13, 6, dtype=torch.long),
            torch.arange(7)[None].float()/8, torch.ones(1, 7, dtype=torch.bool))
    torch.testing.assert_close(model(*args)['prediction'], original(*args)['prediction'], rtol=0, atol=0)
    x = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.tensor([[True]*4, [True, True, False, False]])
    model.train()
    monkeypatch.setattr(torch, 'rand', lambda *a, **kw: torch.tensor(0.))
    monkeypatch.setattr(model.channel_blocks[0], 'forward', lambda *a: (_ for _ in ()).throw(AssertionError('executed skipped block')))
    y = model.run_block(model.channel_blocks[0], x, valid)
    torch.testing.assert_close(y, x*valid[..., None])
    y.sum().backward()
    torch.testing.assert_close(x.grad, valid[..., None].expand_as(x).float())
