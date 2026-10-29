"""Small reproducible diagnostics; does not train or alter legacy models."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from osc_tools.ml.layers.transformer_blocks import TransformerEncoderBlock, PhysicalKANFeedForward
from osc_tools.ml.models.transformer import PhysicalKANTransformer


class ZeroAttention(nn.Module):
    def forward(self, query, key, value, **kwargs):
        return torch.zeros_like(query), None


class ZeroFFN(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def main():
    torch.manual_seed(42)
    block = TransformerEncoderBlock(64, 4, ZeroFFN(), ZeroAttention(), dropout=0).eval()
    x = torch.randn(2, 5, 64, requires_grad=True)
    y = block(x)
    y.sum().backward()
    result = {
        "torch": torch.__version__,
        "identity_output_max_error": (y - x).abs().max().item(),
        "identity_gradient_max_error": (x.grad - 1).abs().max().item(),
        "scope": "synthetic CPU diagnostics; not a checkpoint quality assessment",
    }
    model = PhysicalKANTransformer(156, d_model=64, num_heads=4, num_layers=4,
                                   d_ff=256, dropout=0, max_seq_len=256,
                                   cyclic_angle_encoding=True, use_provenance_embedding=True).eval()
    observed = []
    def capture(module, inputs):
        amp = inputs[0][..., :module.d_complex]
        observed.append({"negative_fraction": (amp < 0).float().mean().item(), "min": amp.min().item()})
    handles = [m.register_forward_pre_hook(capture) for m in model.modules() if isinstance(m, PhysicalKANFeedForward)]
    inp = torch.rand(2, 156, 5)
    inp[:, 1::2] = (inp[:, 1::2] - .5) * (2 * torch.pi)
    out = model(inp, mode="ssl", provenance=torch.ones_like(inp, dtype=torch.long))["ssl"]
    out.square().mean().backward()
    result.update(parameters=sum(p.numel() for p in model.parameters()),
                  physical_ffn_inputs=observed, finite_output=bool(torch.isfinite(out).all()),
                  finite_parameter_gradients=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
    for handle in handles:
        handle.remove()
    assert result["identity_output_max_error"] == 0
    assert result["identity_gradient_max_error"] == 0
    assert any(item["negative_fraction"] > 0 for item in observed)
    destination = ROOT / "reports/phase5/architecture_contract_audit_20261028.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
