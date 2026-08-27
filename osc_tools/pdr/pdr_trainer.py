"""Модуль обучения и оценки моделей на задачу РНМ (PDRTrainer).

Предоставляет функции тонкой настройки (fine-tuning) Physical KAN-Transformer
на псевдометках РНМ. Применимость органа и направление моделируются раздельно.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Any, Optional
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


def _binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Полный набор бинарных метрик без зависимости от sklearn."""

    targets = np.asarray(targets, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = float(np.sum((predictions == 1) & (targets == 1)))
    tn = float(np.sum((predictions == 0) & (targets == 0)))
    fp = float(np.sum((predictions == 1) & (targets == 0)))
    fn = float(np.sum((predictions == 0) & (targets == 1)))

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    precision_1 = ratio(tp, tp + fp)
    recall_1 = ratio(tp, tp + fn)
    f1_1 = ratio(2.0 * precision_1 * recall_1, precision_1 + recall_1)
    precision_0 = ratio(tn, tn + fn)
    recall_0 = ratio(tn, tn + fp)
    f1_0 = ratio(2.0 * precision_0 * recall_0, precision_0 + recall_0)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": float(np.mean(predictions == targets)) if targets.size else 0.0,
        "precision": precision_1,
        "recall": recall_1,
        "f1": f1_1,
        "negative_precision": precision_0,
        "specificity": recall_0,
        "negative_f1": f1_0,
        "macro_f1": 0.5 * (f1_0 + f1_1),
        "balanced_accuracy": 0.5 * (recall_0 + recall_1),
        "mcc": ratio(tp * tn - fp * fn, float(mcc_denominator)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_support": tp + fn,
        "negative_support": tn + fp,
    }


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
    supervised = targets.get("supervised_mask")
    if supervised is None:
        supervised = torch.ones_like(target_class, dtype=torch.bool)
    supervised = supervised.to(device=logits.device, dtype=torch.bool)
    valid = target_applicable.to(device=logits.device, dtype=torch.bool) & supervised

    applicability_logit = outputs.get("applicability_logit")
    if applicability_logit is None:
        loss_applicability = logits.sum() * 0.0
    else:
        applicability_per_sample = F.binary_cross_entropy_with_logits(
            applicability_logit,
            target_applicable.to(device=logits.device, dtype=applicability_logit.dtype),
            reduction="none",
        )
        loss_applicability = (
            (applicability_per_sample * supervised).sum()
            / supervised.sum().clamp(min=1)
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
        if "expert_train_mask" in batch:
            targets["supervised_mask"] = batch["expert_train_mask"].to(device)
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
    progress_callback: Optional[Callable[[int], None]] = None,
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
    all_direction_record_ids: list[int] = []
    all_applicability_record_ids: list[int] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
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
            record_ids = batch.get("record_id")
            if record_ids is not None:
                record_ids = torch.as_tensor(record_ids, device=device)
                source_ids = batch.get("source_id")
                if source_ids is not None:
                    source_ids = torch.as_tensor(source_ids, device=device)
                    record_ids = record_ids + source_ids * 1_000_000_000
                all_direction_record_ids.extend(record_ids[target_applicable].cpu().tolist())

            applicable_preds = (
                torch.sigmoid(out["applicability_logit"]) >= 0.5
            )
            all_applicable_preds.extend(applicable_preds.cpu().to(torch.int8).tolist())
            all_applicable_targets.extend(target_applicable.cpu().to(torch.int8).tolist())
            if record_ids is not None:
                all_applicability_record_ids.extend(record_ids.cpu().tolist())

            if "margin" in out:
                margin_valid = target_applicable & torch.isfinite(target_margin)
                err = (
                    out["margin"][margin_valid]
                    - target_margin[margin_valid]
                ).abs().cpu().tolist()
                margin_errors.extend(err)
            if progress_callback is not None:
                progress_callback(batch_index)

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    applicable_preds_arr = np.asarray(all_applicable_preds, dtype=np.int8)
    applicable_targets_arr = np.asarray(all_applicable_targets, dtype=np.int8)

    direction_metrics = _binary_metrics(all_targets_arr, all_preds_arr)
    applicability_metrics = _binary_metrics(applicable_targets_arr, applicable_preds_arr)
    mae_margin = float(np.mean(margin_errors)) if margin_errors else 0.0

    def record_macro(
        record_ids: list[int], targets: np.ndarray, predictions: np.ndarray,
    ) -> tuple[float, float, float, int]:
        if not record_ids:
            return 0.0, 0.0, 0.0, 0
        ids = np.asarray(record_ids, dtype=np.int64)
        per_record = [
            _binary_metrics(targets[ids == record_id], predictions[ids == record_id])
            for record_id in np.unique(ids)
        ]
        # Обычный binary macro-F1 назначает нулевой F1 отсутствующему в записи
        # классу. Поэтому безошибочная осциллограмма только с REVERSE получила бы
        # 0.5. Отдельно сохраняем средний F1 лишь по фактически присутствующим в
        # каждой записи классам — он пригоднее для пофайловой интерпретации.
        present_class_f1 = []
        for item in per_record:
            scores = []
            if item["positive_support"] > 0:
                scores.append(item["f1"])
            if item["negative_support"] > 0:
                scores.append(item["negative_f1"])
            present_class_f1.append(float(np.mean(scores)) if scores else 0.0)
        return (
            float(np.mean([item["accuracy"] for item in per_record])),
            float(np.mean([item["macro_f1"] for item in per_record])),
            float(np.mean(present_class_f1)),
            len(per_record),
        )

    (
        record_accuracy,
        record_macro_f1,
        record_present_class_f1,
        direction_record_count,
    ) = record_macro(all_direction_record_ids, all_targets_arr, all_preds_arr)
    (
        app_record_accuracy,
        app_record_macro_f1,
        app_record_present_class_f1,
        applicability_record_count,
    ) = record_macro(
        all_applicability_record_ids, applicable_targets_arr, applicable_preds_arr
    )

    return {
        "accuracy": direction_metrics["accuracy"],
        "precision": direction_metrics["precision"],
        "recall": direction_metrics["recall"],
        "f1_score": direction_metrics["f1"],
        "specificity": direction_metrics["specificity"],
        "negative_f1_score": direction_metrics["negative_f1"],
        "macro_f1_score": direction_metrics["macro_f1"],
        "balanced_accuracy": direction_metrics["balanced_accuracy"],
        "mcc": direction_metrics["mcc"],
        "tp": direction_metrics["tp"],
        "tn": direction_metrics["tn"],
        "fp": direction_metrics["fp"],
        "fn": direction_metrics["fn"],
        "forward_support": direction_metrics["positive_support"],
        "reverse_support": direction_metrics["negative_support"],
        "record_macro_accuracy": record_accuracy,
        "record_macro_f1_score": record_macro_f1,
        "record_present_class_f1_score": record_present_class_f1,
        "n_direction_records": direction_record_count,
        "mae_margin": mae_margin,
        "n_margin_samples": len(margin_errors),
        "n_samples": len(all_targets),
        "n_applicability_samples": len(applicable_targets_arr),
        "applicability_accuracy": applicability_metrics["accuracy"],
        "applicability_precision": applicability_metrics["precision"],
        "applicability_recall": applicability_metrics["recall"],
        "applicability_f1_score": applicability_metrics["f1"],
        "applicability_specificity": applicability_metrics["specificity"],
        "applicability_negative_f1_score": applicability_metrics["negative_f1"],
        "applicability_macro_f1_score": applicability_metrics["macro_f1"],
        "applicability_balanced_accuracy": applicability_metrics["balanced_accuracy"],
        "applicability_mcc": applicability_metrics["mcc"],
        "applicability_tp": applicability_metrics["tp"],
        "applicability_tn": applicability_metrics["tn"],
        "applicability_fp": applicability_metrics["fp"],
        "applicability_fn": applicability_metrics["fn"],
        "applicability_record_macro_accuracy": app_record_accuracy,
        "applicability_record_macro_f1_score": app_record_macro_f1,
        "applicability_record_present_class_f1_score": app_record_present_class_f1,
        "n_applicability_records": applicability_record_count,
    }
