"""Модуль обучения и оценки моделей на задачу РНМ (PDRTrainer).

Предоставляет функции тонкой настройки (fine-tuning) Physical KAN-Transformer
на псевдометках РНМ. Применимость органа и направление моделируются раздельно.
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


def decode_pdr_predictions(
    outputs: Dict[str, torch.Tensor],
    applicability_threshold: float = 0.5,
) -> torch.Tensor:
    """Декодировать иерархический выход в -999/REVERSE/FORWARD."""
    if not 0.0 <= applicability_threshold <= 1.0:
        raise ValueError("applicability_threshold должен быть в диапазоне [0, 1]")
    direction = torch.argmax(outputs["logits"], dim=-1)
    applicable = (
        torch.sigmoid(outputs["applicability_logit"])
        >= applicability_threshold
    )
    unlabeled = torch.full_like(direction, -999)
    return torch.where(applicable, direction, unlabeled)


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
        # Физическая определимость направления: 0 = UNLABELED, 1 = направление доступно.
        self.applicability_head = nn.Linear(d_model, 1)
        # Регрессор углового/мощностного запаса срабатывания (margin)
        self.margin_head = nn.Linear(d_model, 1) if use_margin_head else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch_size, num_tokens, d_model) или (batch_size, d_model)
        if x.ndim not in (2, 3):
            raise ValueError(f"PDRTaskHead ожидает (B,D) или (B,T,D), получено {tuple(x.shape)}")
        last_token = x[:, -1, :] if x.ndim == 3 else x
        logits = self.class_head(last_token)
        out = {
            "logits": logits,
            "applicability_logit": self.applicability_head(last_token).squeeze(-1),
        }
        if self.margin_head is not None:
            out["margin"] = self.margin_head(last_token).squeeze(-1)
        return out


def pdr_combined_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    margin_loss_weight: float = 0.5,
    applicability_loss_weight: float = 1.0,
) -> torch.Tensor:
    """BCE применимости + маскированные CE направления и Huber margin."""
    logits = outputs["logits"]
    target_class = targets["target_class"]
    target_applicable = targets.get("target_applicable")
    if target_applicable is None:
        target_applicable = torch.ones_like(target_class, dtype=torch.bool)
    valid = target_applicable.to(device=logits.device, dtype=torch.bool)

    applicability_logit = outputs.get("applicability_logit")
    if applicability_logit is None:
        loss_applicability = logits.sum() * 0.0
    else:
        loss_applicability = F.binary_cross_entropy_with_logits(
            applicability_logit,
            valid.to(dtype=applicability_logit.dtype),
        )

    sample_weight = targets.get("pdr_confidence")
    loss_ce_per_sample = F.cross_entropy(logits, target_class, reduction="none")
    if sample_weight is not None:
        weight = sample_weight.to(loss_ce_per_sample.dtype).clamp(min=0.0) * valid
        loss_ce = (loss_ce_per_sample * weight).sum() / weight.sum().clamp(min=1.0)
    else:
        loss_ce = (loss_ce_per_sample * valid).sum() / valid.sum().clamp(min=1)
    total_loss = applicability_loss_weight * loss_applicability + loss_ce

    if "margin" in outputs and "pdr_margin" in targets:
        pred_margin = outputs["margin"]
        target_margin = targets["pdr_margin"]
        margin_valid = valid & torch.isfinite(target_margin)
        if bool(margin_valid.any()):
            loss_huber_per_sample = F.huber_loss(
                pred_margin[margin_valid],
                target_margin[margin_valid],
                delta=0.1,
                reduction="none",
            )
            if sample_weight is not None:
                weight = sample_weight[margin_valid].to(loss_huber_per_sample.dtype).clamp(min=0.0)
                loss_huber = (
                    (loss_huber_per_sample * weight).sum()
                    / weight.sum().clamp(min=1.0)
                )
            else:
                loss_huber = loss_huber_per_sample.mean()
        else:
            loss_huber = pred_margin.sum() * 0.0
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
            "target_applicable": batch["target_applicable"].to(device),
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
    all_applicable_preds: list[int] = []
    all_applicable_targets: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            target_cls = batch["target_class"].to(device)
            target_applicable = batch.get("target_applicable")
            if target_applicable is None:
                target_applicable = torch.ones_like(target_cls, dtype=torch.bool)
            else:
                target_applicable = target_applicable.to(device=device, dtype=torch.bool)
            target_margin = batch["pdr_margin"].to(device)

            latent = extract_backbone_features(model, batch, device)
            out = head(latent)

            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds[target_applicable].cpu().tolist())
            all_targets.extend(target_cls[target_applicable].cpu().tolist())

            applicable_preds = (
                torch.sigmoid(out["applicability_logit"]) >= 0.5
            )
            all_applicable_preds.extend(applicable_preds.cpu().to(torch.int8).tolist())
            all_applicable_targets.extend(target_applicable.cpu().to(torch.int8).tolist())

            if "margin" in out:
                margin_valid = target_applicable & torch.isfinite(target_margin)
                err = (
                    out["margin"][margin_valid]
                    - target_margin[margin_valid]
                ).abs().cpu().tolist()
                margin_errors.extend(err)

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    applicable_preds_arr = np.asarray(all_applicable_preds, dtype=np.int8)
    applicable_targets_arr = np.asarray(all_applicable_targets, dtype=np.int8)

    acc = float(np.mean(all_preds_arr == all_targets_arr)) if len(all_targets_arr) > 0 else 0.0
    mae_margin = float(np.mean(margin_errors)) if margin_errors else 0.0

    # Вычисление Precision, Recall и F1-score для класса FORWARD (1)
    tp = float(np.sum((all_preds_arr == 1) & (all_targets_arr == 1)))
    fp = float(np.sum((all_preds_arr == 1) & (all_targets_arr == 0)))
    fn = float(np.sum((all_preds_arr == 0) & (all_targets_arr == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    applicability_accuracy = (
        float(np.mean(applicable_preds_arr == applicable_targets_arr))
        if len(applicable_targets_arr) > 0 else 0.0
    )
    app_tp = float(np.sum((applicable_preds_arr == 1) & (applicable_targets_arr == 1)))
    app_fp = float(np.sum((applicable_preds_arr == 1) & (applicable_targets_arr == 0)))
    app_fn = float(np.sum((applicable_preds_arr == 0) & (applicable_targets_arr == 1)))
    applicability_precision = app_tp / (app_tp + app_fp) if (app_tp + app_fp) > 0 else 0.0
    applicability_recall = app_tp / (app_tp + app_fn) if (app_tp + app_fn) > 0 else 0.0
    applicability_f1 = (
        2.0 * applicability_precision * applicability_recall
        / (applicability_precision + applicability_recall)
        if (applicability_precision + applicability_recall) > 0 else 0.0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mae_margin": mae_margin,
        "n_samples": len(all_targets),
        "n_applicability_samples": len(applicable_targets_arr),
        "applicability_accuracy": applicability_accuracy,
        "applicability_precision": applicability_precision,
        "applicability_recall": applicability_recall,
        "applicability_f1_score": applicability_f1,
    }
