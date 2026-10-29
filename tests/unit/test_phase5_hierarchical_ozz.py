"""OZZ transfer, label alignment and missing-input contracts."""
from dataclasses import replace
import numpy as np
import polars as pl
import torch
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
    assert len(seen) == 1 and not seen[0]['resume']
    assert 'phase5' in seen[0]['output'] and seen[0]['ssl_checkpoint'].endswith('best_model.pt')

def test_ozz_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
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
    config = dict(ssl_checkpoint=str(checkpoint), output=str(tmp_path/'continuous'), resume=False,
                  epochs=1, samples_per_epoch=5, validation_samples=2, target_tokens=160, max_seconds=.2,
                  backbone_lr=1e-5, head_lr=1e-4, use_real_no_ozz=False, workers=0, save_every=1,
                  console_interval=2., seed=42)
    engine.run(config)
    config['output'] = str(tmp_path/'resumed')
    save_original = engine.atomic_save
    def interrupt_after_first_step(state, path):
        save_original(state, path)
        if state['step'] == 1:
            (path.parent/'STOP').touch()
    monkeypatch.setattr(engine, 'atomic_save', interrupt_after_first_step)
    engine.run(config)
    (tmp_path/'resumed/STOP').unlink()
    monkeypatch.setattr(engine, 'atomic_save', save_original)
    config['resume'] = True
    # Continued OZZ does not depend on an available SSL checkpoint.
    checkpoint.unlink()
    engine.run(config)
    full = torch.load(tmp_path/'continuous/latest_checkpoint.pt', weights_only=False)
    resumed = torch.load(tmp_path/'resumed/latest_checkpoint.pt', weights_only=False)
    assert (full['epoch'], full['cursor'], full['step'], full['best']) == (resumed['epoch'], resumed['cursor'], resumed['step'], resumed['best'])
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
