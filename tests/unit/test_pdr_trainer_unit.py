"""Контракт PDR trainer с PhysicalKANTransformer-подобным backbone."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from osc_tools.pdr.pdr_trainer import (
    PDRTaskHead,
    extract_backbone_features,
    pdr_combined_loss,
)
from osc_tools.ml.models.transformer import PhysicalKANTransformer


class _ContractBackbone(nn.Module):
    def forward(self, x, mode="features", provenance=None):
        assert x.ndim == 3  # (B,C,T)
        assert provenance is not None and provenance.shape == x.shape
        assert mode == "features"
        return {"features": torch.nan_to_num(x.transpose(1, 2), nan=0.0)}


def test_extract_backbone_features_transposes_and_passes_provenance() -> None:
    batch = {
        "features": torch.randn(2, 5, 12),
        "provenance": torch.ones(2, 5, 12, dtype=torch.long),
    }
    latent = extract_backbone_features(_ContractBackbone(), batch, "cpu")
    assert latent.shape == (2, 5, 12)
    assert PDRTaskHead(d_model=12)(latent)["logits"].shape == (2, 2)


def test_combined_loss_uses_confidence_weights() -> None:
    outputs = {
        "logits": torch.tensor([[5.0, -5.0], [-5.0, 5.0]], requires_grad=True),
        "margin": torch.tensor([0.0, 10.0], requires_grad=True),
    }
    targets = {
        "target_class": torch.tensor([0, 0]),
        "pdr_margin": torch.tensor([0.0, 0.0]),
        "pdr_confidence": torch.tensor([1.0, 0.0]),
    }
    loss = pdr_combined_loss(outputs, targets)
    assert loss.item() < 0.01


def test_actual_phase5_backbone_pdr_forward_backward() -> None:
    model = PhysicalKANTransformer(
        num_input_channels=12,
        ssl_output_channels=12,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        max_seq_len=16,
        cyclic_angle_encoding=True,
        use_provenance_embedding=True,
    )
    batch = {
        "features": torch.randn(3, 5, 12),
        "provenance": torch.ones(3, 5, 12, dtype=torch.long),
        "target_class": torch.tensor([0, 1, 0]),
        "pdr_margin": torch.tensor([-1.0, 1.0, -0.5]),
        "pdr_confidence": torch.ones(3),
    }
    head = PDRTaskHead(d_model=16)

    latent = extract_backbone_features(model, batch, "cpu")
    loss = pdr_combined_loss(head(latent), batch)
    loss.backward()

    assert latent.shape == (3, 5, 16)
    assert torch.isfinite(loss)
