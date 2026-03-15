"""
SSL Dataset для self-supervised обучения Physical KAN-Transformer (Фаза 4).

Адаптер над PrecomputedDataset, который:
1. Подготавливает данные в формате (B, C, T) для Transformer
2. Реализует Masked Spectral-Temporal Modeling (маскирование временных шагов/каналов)
3. Поддерживает предсказание будущего (2 периода вперёд)
4. Разделяет амплитуды и углы для корректного расчёта Loss

Использование:
    from osc_tools.ml.ssl_dataset import SSLSpectralDataset

    dataset = SSLSpectralDataset(
        dataframe=df,
        indices=train_indices,
        window_size=320,
        future_steps=2,          # +2 периода для предсказания
        mask_ratio=0.25,         # 25% шагов маскируются
    )
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from osc_tools.ml.precomputed_dataset import PrecomputedDataset

# Заметки по расширению спектральных признаков (Étape 1):
# 1. Низшие гармоники (периоды 2, 4, 6, 10) — пока НЕ предрассчитаны в pipeline.
#    Потребуется расширение FFT-калькулятора для вычисления интергармоник.
#    ВАЖНО: при добавлении низших гармоник они ДОЛЖНЫ включаться в ту же группу
#    физического сигнала (например, IA_h0.5 → группа IA вместе с IA_h1..IA_h9),
#    чтобы нормализация Loss и build_channel_groups_phase_polar() были корректны.
# 2. Симметричные составляющие — УЖЕ поддерживаются через feature_mode='symmetric_polar'.
#    PrecomputedDataset принимает list[str], например ['phase_polar', 'symmetric_polar'].
# 3. Группировка каналов по физическим сигналам для нормализации Loss —
#    реализована в losses.py через build_channel_groups_phase_polar().
#    SSLSpectralDataset предоставляет метод get_channel_groups() для передачи в Loss.
class SSLSpectralDataset(Dataset):
    """Dataset для self-supervised обучения Transformer на спектральных данных.

    Выдаёт:
    - input: (C, T) — входные данные (с маскированными шагами заменёнными на mask_value)
    - target: (C, T_full) — полный целевой сигнал (текущее окно + будущие шаги)
    - mask: (C, T_full) — True для позиций где Loss не считается (маскированные + padding)
    - current_len: int — длина текущего окна (для разделения current/future в Loss)

    Стратегия маскирования (Masked Spectral-Temporal Modeling):
    - Скрываем mask_ratio случайных временных шагов
    - На скрытых шагах все каналы заменяются на mask_value (learnable в модели)

    Args:
        dataframe: предрассчитанный DataFrame с признаками
        indices: индексы начал окон (или кортежи для random sampling)
        window_size: размер основного окна (10 периодов = 320 точек при 1600 Гц)
        feature_mode: режим признаков (по умолчанию 'phase_polar')
        num_harmonics: число гармоник
        downsampling_stride: шаг прореживания (16 = полпериода)
        future_periods: число будущих периодов для предсказания (0 = без предсказания)
        samples_per_period: число отсчётов в одном периоде (32 при 1600 Гц и 50 Гц)
        mask_ratio: доля маскируемых временных шагов (0.0–1.0)
        mask_value: значение для замены маскированных данных
    """

    def __init__(
        self,
        dataframe: Union[pl.DataFrame, pl.LazyFrame],
        indices: Union[List[int], List[Tuple[int, int]]],
        window_size: int = 320,
        feature_mode: str = 'phase_polar',
        num_harmonics: int = 9,
        downsampling_stride: int = 16,
        future_periods: int = 2,
        samples_per_period: int = 32,
        mask_ratio: float = 0.25,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.downsampling_stride = downsampling_stride
        self.future_periods = future_periods
        self.samples_per_period = samples_per_period
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

        # Будущие шаги в сырых отсчётах
        self.future_raw_steps = future_periods * samples_per_period
        # Полное окно = текущее + будущее
        self.full_window_raw = window_size + self.future_raw_steps

        # Создаём базовый PrecomputedDataset для извлечения фичей
        # Используем full_window_raw как window_size, чтобы получить и будущие данные
        # Для индексов, у которых нет будущих данных — будет padding
        self._base_dataset = PrecomputedDataset(
            dataframe=dataframe,
            indices=indices,
            window_size=self.full_window_raw,
            feature_mode=feature_mode,
            target_level='base',  # Метки не нужны для SSL, но формат требует
            sampling_strategy='stride',
            downsampling_stride=downsampling_stride,
            target_window_mode='point',
            num_harmonics=num_harmonics,
        )

        # Число временных шагов ПОСЛЕ stride-сэмплинга
        # stride пропускает FFT_WARMUP (32 точки), затем каждый stride-й
        warmup = PrecomputedDataset.FFT_WARMUP
        valid_len = self.full_window_raw - warmup
        self.num_steps_full = max(1, valid_len // downsampling_stride)
        valid_len_current = window_size - warmup
        self.num_steps_current = max(1, valid_len_current // downsampling_stride)

        self.num_channels = len(self._base_dataset.feature_columns)
        self._num_harmonics = num_harmonics

    def get_channel_groups(self) -> list[list[int]] | None:
        """Возвращает группировку каналов по физическим сигналам.

        Используется для нормализации Loss: макс. амплитуда считается
        по ВСЕМ гармоникам одного физ. сигнала (например, IA).

        Returns:
            Список групп (каждая — список индексов каналов) для phase_polar,
            None для неподдерживаемых режимов.
        """
        from osc_tools.ml.losses import build_channel_groups_phase_polar

        # phase_polar: 8 сигналов × num_harmonics гармоник × 2 (mag+angle)
        feature_modes = self._base_dataset.feature_mode
        if 'phase_polar' in feature_modes:
            return build_channel_groups_phase_polar(
                num_signals=8, num_harmonics=self._num_harmonics
            )
        return None

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict с ключами:
                'input': (C, T_current) — маскированный вход для модели
                'target': (C, T_full) — полные целевые признаки
                'mask_input': (C, T_current) — True = маскированные позиции
                'mask_loss': (C, T_full) — True = позиции где Loss НЕ считается
                'current_len': int — число шагов текущего окна
        """
        # Получаем полные данные через базовый dataset
        X_full, _ = self._base_dataset[idx]  # X_full: (C, T_stride)

        T_actual = X_full.shape[1]
        T_current = min(self.num_steps_current, T_actual)
        T_full = T_actual

        # Разделяем на текущее и будущее окна
        X_current = X_full[:, :T_current]  # (C, T_current) — вход модели

        # --- Маскирование (Masked Spectral-Temporal Modeling) ---
        # Маскируем случайные временные шаги в текущем окне
        mask_input = torch.zeros(self.num_channels, T_current, dtype=torch.bool)

        num_to_mask = max(1, int(T_current * self.mask_ratio))
        mask_indices = random.sample(range(T_current), min(num_to_mask, T_current))

        for t_idx in mask_indices:
            mask_input[:, t_idx] = True

        # Маскированный вход: заменяем скрытые шаги на mask_value
        X_masked = X_current.clone()
        X_masked[mask_input] = self.mask_value

        # --- Маска для Loss ---
        # В полном окне: Loss НЕ считается для отсутствующих данных (NaN).
        # Примечание: PrecomputedDataset сейчас заменяет NaN→0 в _preload_data,
        # поэтому mask_loss будет all-False. Если в будущем потребуется подддержка
        # пропущенных каналов, нужно сохранять NaN через PrecomputedDataset.
        mask_loss = torch.isnan(X_full)
        # Если будущих данных не хватило (padding нулями), не считаем Loss и там
        if T_actual < self.num_steps_full:
            pad_mask = torch.ones(
                self.num_channels,
                self.num_steps_full - T_actual,
                dtype=torch.bool,
            )
            mask_loss = torch.cat([mask_loss, pad_mask], dim=1)
            X_full_padded = torch.zeros(self.num_channels, self.num_steps_full)
            X_full_padded[:, :T_actual] = X_full
            X_full = X_full_padded

        return {
            'input': X_masked,           # (C, T_current)
            'target': X_full,            # (C, T_full)
            'mask_input': mask_input,    # (C, T_current)
            'mask_loss': mask_loss,      # (C, T_full)
            'current_len': T_current,
        }
