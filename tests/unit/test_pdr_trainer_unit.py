"""Контракт PDR trainer с PhysicalKANTransformer-подобным backbone."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from osc_tools.pdr.pdr_trainer import (
    PDRTaskHead,
    decode_pdr_predictions,
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


def test_evaluation_prediction_callback_preserves_metrics() -> None:
    from osc_tools.pdr.pdr_trainer import evaluate_pdr_metrics
    batch = {
        "features": torch.ones(3, 2, 12), "provenance": torch.ones(3, 2, 12, dtype=torch.long),
        "target_class": torch.tensor([0, 1, 0]), "target_applicable": torch.tensor([True, True, False]),
        "pdr_margin": torch.full((3,), float("nan")), "record_id": torch.tensor([1, 1, 2]),
    }
    model, head = _ContractBackbone(), PDRTaskHead(d_model=12)
    expected = evaluate_pdr_metrics(model, head, [batch])
    captured = []
    actual = evaluate_pdr_metrics(model, head, [batch], prediction_callback=lambda b, o: captured.append(o["logits"].shape))
    assert actual == expected
    assert captured == [torch.Size([3, 2])]
    assert actual["n_samples"] == 2
    assert actual["n_applicability_samples"] == 3


def test_extract_backbone_features_transposes_and_passes_provenance() -> None:
    batch = {
        "features": torch.randn(2, 5, 12),
        "provenance": torch.ones(2, 5, 12, dtype=torch.long),
    }
    latent = extract_backbone_features(_ContractBackbone(), batch, "cpu")
    assert latent.shape == (2, 5, 12)
    outputs = PDRTaskHead(d_model=12)(latent)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["applicability_logit"].shape == (2,)


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


def test_unlabeled_trains_applicability_but_not_direction_or_margin() -> None:
    outputs = {
        "logits": torch.tensor([[0.0, 0.0], [-100.0, 100.0]], requires_grad=True),
        "applicability_logit": torch.tensor([5.0, 5.0], requires_grad=True),
        "margin": torch.tensor([0.0, 1000.0], requires_grad=True),
    }
    targets = {
        "target_class": torch.tensor([0, 0]),
        "target_applicable": torch.tensor([True, False]),
        "pdr_margin": torch.tensor([0.0, float("nan")]),
        "pdr_confidence": torch.tensor([1.0, 0.0]),
    }
    loss = pdr_combined_loss(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    # На UNLABELED нет градиента ни направления, ни margin.
    assert torch.equal(outputs["logits"].grad[1], torch.zeros(2))
    assert outputs["margin"].grad[1].item() == 0.0
    # Но ошибка применимости на этой точке обучает отдельную голову.
    assert outputs["applicability_logit"].grad[1].abs().item() > 0.0


def test_decode_abstains_before_returning_direction() -> None:
    outputs = {
        "logits": torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        "applicability_logit": torch.tensor([2.0, 2.0, -2.0]),
    }
    assert decode_pdr_predictions(outputs).tolist() == [1, 0, -999]


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
        "target_applicable": torch.tensor([True, True, False]),
        "pdr_margin": torch.tensor([-1.0, 1.0, -0.5]),
        "pdr_confidence": torch.ones(3),
    }
    head = PDRTaskHead(d_model=16)

    latent = extract_backbone_features(model, batch, "cpu")
    loss = pdr_combined_loss(head(latent), batch)
    loss.backward()

    assert latent.shape == (3, 5, 16)
    assert torch.isfinite(loss)
def test_evaluation_detects_cross_split_input_duplicates():
    from scripts.phase5_experiments.evaluate_pdr_expert_holdout import _split_overlap_audit
    rows = [dict(source="open_ee", record_id=i, split=s, input_sha256=h)
            for i, s, h in [(1, "train", "a"), (2, "validation", "a"), (3, "holdout", "b")]]
    result = _split_overlap_audit(rows)
    assert not result["passed"]
    assert result["training_overlap"][0]["record_id"] == 2
    assert result["training_overlap"][0]["training_record_ids"] == [1]
