from __future__ import annotations

import torch.nn as nn


class BaseModel(nn.Module):
    """Базовый класс для всех ML-моделей проекта."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):  # pragma: no cover - контрактный метод
        raise NotImplementedError

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Возвращает количество параметров модели."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
