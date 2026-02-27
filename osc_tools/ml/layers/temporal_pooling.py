"""
Модуль временного пулинга для свёрточных моделей.

Решает проблему "размазывания" — когда AdaptiveAvgPool1d(1) усредняет
ВСЮ временную ось и модель теряет информацию о том, ГДЕ произошло событие.

Стратегии:
- global_avg: Стандартный AdaptiveAvgPool1d(1) — обратная совместимость.
- attention:  Обучаемое внимание — модель сама выбирает, какие моменты важны.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):
    """
    Обучаемый Attention Pooling по временной оси.

    Вычисляет веса важности для каждого временного шага,
    затем возвращает взвешенную сумму.

    Args:
        channels: Количество каналов (размерность признаков).
    """
    def __init__(self, channels: int):
        super().__init__()
        # Одномерная свёртка 1x1 для вычисления скоров внимания
        # Вход: (B, C, T) → вычисляем скор для каждого T
        self.attention_conv = nn.Conv1d(channels, 1, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.attention_conv.weight)
        nn.init.zeros_(self.attention_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Channels, Time)
        Returns:
            (Batch, Channels) — взвешенная сумма по времени
        """
        # Вычисляем скоры внимания: (B, 1, T)
        scores = self.attention_conv(x)
        # Softmax по временной оси — получаем распределение вероятностей
        weights = F.softmax(scores, dim=-1)  # (B, 1, T)
        # Взвешенная сумма: (B, C, T) * (B, 1, T) → sum по T → (B, C)
        pooled = (x * weights).sum(dim=-1)  # (B, C)
        return pooled


class TemporalPooling(nn.Module):
    """
    Универсальный модуль временного пулинга.

    Заменяет AdaptiveAvgPool1d(1) в свёрточных моделях,
    предоставляя выбор стратегии через параметр.

    Args:
        channels: Количество входных каналов (нужно для attention).
        strategy: Стратегия пулинга:
            - "global_avg" — стандартный, обратная совместимость
            - "attention" — обучаемое внимание по времени

    Свойства:
        output_scale: Множитель выходной размерности (всегда 1 для текущих стратегий).
    """
    def __init__(
        self,
        channels: int,
        strategy: str = "global_avg",
    ):
        super().__init__()

        self.strategy = strategy
        self.channels = channels

        # Множитель размерности выхода (для корректного создания классификатора)
        self._output_scale = 1

        if strategy == "global_avg":
            self._pool = nn.AdaptiveAvgPool1d(1)

        elif strategy == "attention":
            self._attn_pool = TemporalAttentionPool(channels)

        else:
            raise ValueError(
                f"Неизвестная стратегия пулинга: '{strategy}'. "
                f"Допустимые: global_avg, attention"
            )

    @property
    def output_scale(self) -> int:
        """Множитель выходной размерности относительно входных каналов."""
        return self._output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Channels, Time)
        Returns:
            (Batch, Channels * output_scale) — вектор признаков без временной оси
        """
        if self.strategy == "global_avg":
            return self._pool(x).flatten(1)

        elif self.strategy == "attention":
            return self._attn_pool(x)  # (B, C)

        # Не должно произойти из-за проверки в __init__
        raise RuntimeError(f"Необработанная стратегия: {self.strategy}")

    def extra_repr(self) -> str:
        """Строковое представление для print(model)."""
        parts = [f"strategy='{self.strategy}'", f"channels={self.channels}"]
        return ", ".join(parts)
