from dataclasses import replace
import torch
from osc_tools.ml.models import phasor_hierarchical as module


def test_hierarchical_positions_use_local_and_global_period_time(monkeypatch):
    cfg = replace(module.HierarchicalConfig(), channel_width=8, width=16, heads=2,
                  channel_layers=1, local_layers=1, global_layers=1, fusion_layers=1, dropout=0.)
    model = module.PhasorHierarchicalKAN(cfg).eval()
    original = module.positional
    calls = []
    def capture(t, width):
        calls.append((t.clone(), width))
        return original(t, width)
    monkeypatch.setattr(module, 'positional', capture)
    z = torch.randn(1, 161, 13, 6, 2)*.1
    state = torch.ones(1, 161, 13, 6, dtype=torch.long)
    times = torch.arange(1, 162)[None].float()/8
    valid = torch.ones(1, 161, dtype=torch.bool)
    with torch.no_grad(): result = model(z, state, times, valid)
    local, global_time = calls[0][0], calls[1][0]
    torch.testing.assert_close(local[0], torch.arange(80).float()/8)
    torch.testing.assert_close(local[1], local[0])
    torch.testing.assert_close(global_time, torch.tensor([[.125, 10.125, 20.125]]))
    assert result['prediction'].shape[1] == 161
    assert torch.isfinite(result['prediction']).all()


def test_30_second_positional_codes_do_not_repeat_between_blocks():
    # 150 ten-period blocks at 50 Hz; no learned table length restriction.
    starts = .125 + torch.arange(150).float()*10
    pe = module.positional(starts, 128)
    assert pe.dtype == torch.float32 and torch.isfinite(pe).all()
    assert torch.unique(pe, dim=0).shape[0] == 150
