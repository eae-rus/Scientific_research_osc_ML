"""
Функции потерь для Фазы 4 (Physical KAN-Transformer).

Содержит:
- ComplexMSELoss: потеря через расстояние на комплексной плоскости
  (автоматически решает проблему скачка 2π и взвешивает ошибку угла по амплитуде)
- SpectralReconstructionLoss: полная SSL-потеря с нормализацией, порогами шума
  и весовыми коэффициентами для текущего/будущего окна
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ComplexMSELoss(nn.Module):
    """Потеря через расстояние между векторами на комплексной плоскости.

    Вместо раздельного сравнения амплитуд и углов, переводим предсказание
    и цель в комплексную форму:
        Z = A · cos(φ) + i · A · sin(φ)
    и считаем MSE между Z_pred и Z_true.

    Преимущества:
    - Автоматически решает проблему скачка 2π (sin/cos периодичны)
    - Автоматически взвешивает ошибку угла по амплитуде:
      если A≈0, то вектор короткий и ошибка угла = 0

    Args:
        reduction: 'mean', 'sum' или 'none'
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_amp: torch.Tensor,
        pred_phase: torch.Tensor,
        true_amp: torch.Tensor,
        true_phase: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_amp: (B, C, T) — предсказанные амплитуды
            pred_phase: (B, C, T) — предсказанные углы
            true_amp: (B, C, T) — целевые амплитуды
            true_phase: (B, C, T) — целевые углы
            mask: (B, C, T) — True для позиций, где Loss НЕ считается
                              (отсутствующие каналы, padding и т.д.)

        Returns:
            scalar loss (если reduction='mean' или 'sum')
            или (B, C, T) тензор (если reduction='none')
        """
        # Перевод в комплексную плоскость
        z_pred_re = pred_amp * torch.cos(pred_phase)
        z_pred_im = pred_amp * torch.sin(pred_phase)
        z_true_re = true_amp * torch.cos(true_phase)
        z_true_im = true_amp * torch.sin(true_phase)

        # Квадрат расстояния между векторами
        diff_sq = (z_pred_re - z_true_re) ** 2 + (z_pred_im - z_true_im) ** 2

        if mask is not None:
            # Обнуляем Loss для замаскированных позиций
            diff_sq = diff_sq * (~mask).float()

        if self.reduction == 'mean':
            if mask is not None:
                valid_count = (~mask).float().sum().clamp(min=1.0)
                return diff_sq.sum() / valid_count
            return diff_sq.mean()
        elif self.reduction == 'sum':
            return diff_sq.sum()
        else:
            return diff_sq


class SpectralReconstructionLoss(nn.Module):
    """Полная SSL-потеря для self-supervised обучения трансформера.

    Реализует 4 шага из PHASE_4_PLAN:
    1. Векторная разность (Complex MSE) — расстояние между векторами
    2. Относительная нормализация — деление на максимальную амплитуду фазы
    3. Пороги шума — минимальный нормировочный делитель
    4. Весовые коэффициенты — линейная зависимость от амплитуды (1.0 → 0.1)

    Args:
        noise_threshold_current: порог шума для токов (1/2000)
        noise_threshold_voltage: порог шума для напряжений (1/300)
        num_current_channels: число каналов амплитуд токов
        weight_current_window: вес для текущих 10 периодов
        weight_future_window: вес для предсказываемых 2 периодов
        min_signal_weight: минимальный вес для слабых сигналов
    """

    def __init__(
        self,
        noise_threshold_current: float = 1.0 / 2000,
        noise_threshold_voltage: float = 1.0 / 300,
        num_current_channels: int | None = None,
        weight_current_window: float = 1.0,
        weight_future_window: float = 0.1,
        min_signal_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.noise_threshold_I = noise_threshold_current
        self.noise_threshold_U = noise_threshold_voltage
        self.num_current_channels = num_current_channels
        self.weight_current = weight_current_window
        self.weight_future = weight_future_window
        self.min_signal_weight = min_signal_weight

    def forward(
        self,
        pred_amp: torch.Tensor,
        pred_phase: torch.Tensor,
        true_amp: torch.Tensor,
        true_phase: torch.Tensor,
        mask: torch.Tensor | None = None,
        current_len: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_amp: (B, C, T) — предсказанные амплитуды
            pred_phase: (B, C, T) — предсказанные углы
            true_amp: (B, C, T) — целевые амплитуды
            true_phase: (B, C, T) — целевые углы
            mask: (B, C, T) — True где Loss НЕ считается
            current_len: длина текущего окна (остаток — будущее); если None, всё = текущее

        Returns:
            scalar loss
        """
        B, C, T = pred_amp.shape

        # Шаг 1: Векторная разность (модуль разности комплексных векторов)
        z_pred_re = pred_amp * torch.cos(pred_phase)
        z_pred_im = pred_amp * torch.sin(pred_phase)
        z_true_re = true_amp * torch.cos(true_phase)
        z_true_im = true_amp * torch.sin(true_phase)

        # Модуль разности (не квадрат — для более честной метрики)
        vector_diff = torch.sqrt(
            (z_pred_re - z_true_re) ** 2 + (z_pred_im - z_true_im) ** 2 + 1e-12
        )

        # Шаг 2: Относительная нормализация
        # Максимальная амплитуда по времени для каждого канала
        # TODO: стоит проверить / обудмать, как будут определяеться эти "каналы",
        # ведь речь во многом о том, что это "ТОК фазы А", и не важно какая гармоника - а их много.
        max_amp_per_channel = true_amp.amax(dim=-1, keepdim=True)  # (B, C, 1)

        # Шаг 3: Пороги шума (минимальный нормировочный делитель)
        noise_thresholds = self._get_noise_thresholds(C, pred_amp.device)
        max_amp_clamped = torch.clamp(max_amp_per_channel, min=noise_thresholds)

        # Относительная ошибка
        relative_error = vector_diff / max_amp_clamped

        # Шаг 4: Весовые коэффициенты (линейно от амплитуды)
        # От 1.0 (крупные) до min_signal_weight (микро-сигналы на уровне порога)
        signal_strength = (max_amp_per_channel / max_amp_clamped).clamp(0, 1)
        amplitude_weight = self.min_signal_weight + (1.0 - self.min_signal_weight) * signal_strength
        weighted_error = relative_error * amplitude_weight

        # Маскирование пропущенных каналов
        if mask is not None:
            weighted_error = weighted_error * (~mask).float()

        # Временные веса (текущее окно vs будущее)
        if current_len is not None and current_len < T:
            time_weights = torch.ones(1, 1, T, device=pred_amp.device)
            time_weights[:, :, :current_len] = self.weight_current
            time_weights[:, :, current_len:] = self.weight_future
            weighted_error = weighted_error * time_weights

        # Финальная агрегация
        if mask is not None:
            valid = (~mask).float().sum().clamp(min=1.0)
            return weighted_error.sum() / valid
        return weighted_error.mean()

    def _get_noise_thresholds(
        self, num_channels: int, device: torch.device
    ) -> torch.Tensor:
        """Создать тензор порогов шума для каждого канала.

        Returns:
            (1, C, 1) тензор порогов
        """
        thresholds = torch.full(
            (1, num_channels, 1), self.noise_threshold_U, device=device
        )
        if self.num_current_channels is not None:
            thresholds[:, :self.num_current_channels, :] = self.noise_threshold_I
        return thresholds
