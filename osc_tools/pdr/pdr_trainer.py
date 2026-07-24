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
        num_classes: int = 3,
        use_margin_head: bool = True,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch не установлен.")
        super().__init__()
        self.use_margin_head = use_margin_head
        # Линейные/KAN классификатор направления (0: REVERSE, 1: BLOCK, 2: FORWARD)
        self.class_head = nn.Linear(d_model, num_classes)
        # Регрессор запаса срабатывания (margin)
        self.margin_head = nn.Linear(d_model, 1) if use_margin_head else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch_size, num_tokens, d_model) -> берем последний токен
        last_token = x[:, -1, :]
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

    loss_ce = F.cross_entropy(logits, target_class)
    total_loss = loss_ce

    if "margin" in outputs and "pdr_margin" in targets:
        pred_margin = outputs["margin"]
        target_margin = targets["pdr_margin"]
        loss_huber = F.huber_loss(pred_margin, target_margin, delta=0.1)
        total_loss = total_loss + margin_loss_weight * loss_huber

    return total_loss


def evaluate_pdr_metrics(
    model: nn.Module,
    head: PDRTaskHead,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Оценка качества модели на тестовом/валидационном датасете РНМ."""
    if not HAS_TORCH:
        return {}

    model.eval()
    head.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []
    margin_errors: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            feats = batch["features"].to(device)
            target_cls = batch["target_class"].to(device)
            target_margin = batch["pdr_margin"].to(device)

            # Передача через модель backbone (если доступна) или линейную проекцию
            feats_out = model(feats) if hasattr(model, "forward") else feats
            out = head(feats_out)

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

    return {
        "accuracy": acc,
        "mae_margin": mae_margin,
        "n_samples": len(all_targets),
    }
