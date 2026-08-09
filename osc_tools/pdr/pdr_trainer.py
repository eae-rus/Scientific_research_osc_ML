"""Модуль обучения и оценки моделей на задачу РНМ (PDRTrainer).

Предоставляет функции тонкой настройки (fine-tuning) Physical KAN-Transformer
на псевдометках РНМ с комбинацией классификационной CrossEntropy и регрессии запаса (margin).
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class PDRTaskHead(nn.Module if HAS_TORCH else object):
    """Головка решения для задачи РНМ над выходами KAN-Transformer backbone."""

    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 2,
        use_margin_head: bool = True,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch не установлен.")
        super().__init__()
        self.use_margin_head = use_margin_head
        # Классификатор направления БАВР (0: REVERSE / Разрешение, 1: FORWARD / Блокировка)
        self.class_head = nn.Linear(d_model, num_classes)
        # Регрессор углового/мощностного запаса срабатывания (margin)
        self.margin_head = nn.Linear(d_model, 1) if use_margin_head else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch_size, num_tokens, d_model) или (batch_size, d_model)
        if x.ndim not in (2, 3):
            raise ValueError(f"PDRTaskHead ожидает (B,D) или (B,T,D), получено {tuple(x.shape)}")
        last_token = x[:, -1, :] if x.ndim == 3 else x
        logits = self.class_head(last_token)
        out = {"logits": logits}
        if self.margin_head is not None:
            out["margin"] = self.margin_head(last_token).squeeze(-1)
        return out


def pdr_combined_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    margin_loss_weight: float = 0.5,
) -> torch.Tensor:
    """Комбинированный лосс для обучения РНМ (CrossEntropy + Huber Margin Loss)."""
    logits = outputs["logits"]
    target_class = targets["target_class"]

    sample_weight = targets.get("pdr_confidence")
    loss_ce_per_sample = F.cross_entropy(logits, target_class, reduction="none")
    if sample_weight is not None:
        weight = sample_weight.to(loss_ce_per_sample.dtype).clamp(min=0.0)
        loss_ce = (loss_ce_per_sample * weight).sum() / weight.sum().clamp(min=1.0)
    else:
        loss_ce = loss_ce_per_sample.mean()
    total_loss = loss_ce

    if "margin" in outputs and "pdr_margin" in targets:
        pred_margin = outputs["margin"]
        target_margin = targets["pdr_margin"]
        loss_huber_per_sample = F.huber_loss(
            pred_margin,
            target_margin,
            delta=0.1,
            reduction="none",
        )
        if sample_weight is not None:
            weight = sample_weight.to(loss_huber_per_sample.dtype).clamp(min=0.0)
            loss_huber = (loss_huber_per_sample * weight).sum() / weight.sum().clamp(min=1.0)
        else:
            loss_huber = loss_huber_per_sample.mean()
        total_loss = total_loss + margin_loss_weight * loss_huber

    return total_loss


def extract_backbone_features(
    model: Optional[nn.Module],
    batch: Dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Привести PDR batch к контракту backbone и вернуть latent (B,T,D)."""
    features_time_first = batch["features"].to(device)
    if model is None:
        return torch.nan_to_num(features_time_first, nan=0.0)

    features = features_time_first.transpose(1, 2)  # (B,C,T)
    provenance = batch.get("provenance")
    provenance_channels_first = (
        provenance.to(device).transpose(1, 2) if provenance is not None else None
    )
    try:
        result = model(
            features,
            mode="features",
            provenance=provenance_channels_first,
        )
    except TypeError:
        # Совместимость с простыми пользовательскими backbone без mode/provenance.
        result = model(features)
    if isinstance(result, dict):
        if "features" not in result:
            raise KeyError("Backbone не вернул обязательный ключ 'features'")
        return result["features"]
    if not isinstance(result, torch.Tensor):
        raise TypeError("Backbone должен вернуть Tensor или dict с ключом 'features'")
    return result


def train_pdr_epoch(
    model: Optional[nn.Module],
    head: PDRTaskHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    margin_loss_weight: float = 0.5,
) -> float:
    """Одна эпоха fine-tuning с единым контрактом Phase 5 backbone."""
    if model is not None:
        model.train()
    head.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        latent = extract_backbone_features(model, batch, device)
        outputs = head(latent)
        targets = {
            "target_class": batch["target_class"].to(device),
            "pdr_margin": batch["pdr_margin"].to(device),
        }
        if "pdr_confidence" in batch:
            targets["pdr_confidence"] = batch["pdr_confidence"].to(device)
        loss = pdr_combined_loss(outputs, targets, margin_loss_weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1
    return total_loss / max(1, n_batches)


def evaluate_pdr_metrics(
    model: Optional[nn.Module],
    head: PDRTaskHead,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Оценка качества модели на тестовом/валидационном датасете РНМ."""
    if not HAS_TORCH:
        return {}

    if model is not None:
        model.eval()
    head.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []
    margin_errors: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            target_cls = batch["target_class"].to(device)
            target_margin = batch["pdr_margin"].to(device)

            latent = extract_backbone_features(model, batch, device)
            out = head(latent)

            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(target_cls.cpu().tolist())

            if "margin" in out:
                err = (out["margin"] - target_margin).abs().cpu().tolist()
                margin_errors.extend(err)

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    acc = float(np.mean(all_preds_arr == all_targets_arr)) if len(all_targets_arr) > 0 else 0.0
    mae_margin = float(np.mean(margin_errors)) if margin_errors else 0.0

    # Вычисление Precision, Recall и F1-score для класса FORWARD (1)
    tp = float(np.sum((all_preds_arr == 1) & (all_targets_arr == 1)))
    fp = float(np.sum((all_preds_arr == 1) & (all_targets_arr == 0)))
    fn = float(np.sum((all_preds_arr == 0) & (all_targets_arr == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mae_margin": mae_margin,
        "n_samples": len(all_targets),
    }
