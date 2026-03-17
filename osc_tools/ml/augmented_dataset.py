"""
AugmentedSpectralDataset — Dataset с аугментацией на сырых данных и on-the-fly FFT.

Pipeline:
1. Загружает сырые 8-канальные данные (IA..UN) из предрассчитанного CSV
2. Берёт расширенное окно (контекст для низших гармоник + основное + будущее)
3. Применяет аугментацию (инверсия, масштабирование, перетасовка фаз)
4. Вычисляет FFT: стандартные гармоники (1–9) + низшие гармоники (периоды 2,4,6,10)
5. Конвертирует в phase_polar формат (амплитуда + относительный угол)
6. Применяет stride прореживание
7. Для SSL: маскирование + предсказание будущего
8. Для classify: возвращает (X, Y)

Используется в Phase 4 для pretrain (SSL) и finetune (classify).

Использование:
    from osc_tools.ml.augmented_dataset import AugmentedSpectralDataset

    dataset = AugmentedSpectralDataset(
        dataframe=df,
        file_boundaries=boundaries,
        indices=train_indices,
        window_size=320,
        augmenter=TimeSeriesAugmenter(),
    )
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from osc_tools.preprocessing.filtering import sliding_window_fft, compute_low_harmonics_fft
from osc_tools.features.polar import calculate_polar_features
from osc_tools.ml.labels import get_target_columns


class AugmentedSpectralDataset(Dataset):
    """Dataset с on-the-fly FFT, низшими гармониками и аугментацией.

    Поддерживает два режима:
    - 'ssl': маскирование + предсказание будущего → dict
    - 'classify': supervised классификация → (X, Y)

    Формат каналов (phase_polar):
        Для каждого из 8 сигналов: [h1_mag, h1_angle, ..., h9_angle, lh2_mag, lh2_angle, ..., lh10_angle]
        = (num_harmonics + num_sub_periods) × 2 каналов на сигнал
    """

    # Константы
    FFT_WINDOW = 32               # Окно FFT для стандартных гармоник (1 период)
    SAMPLES_PER_PERIOD = 32       # Отсчётов в 1 периоде (1600 Гц / 50 Гц)
    RAW_CHANNELS = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
    NUM_RAW_CHANNELS = 8

    def __init__(
        self,
        dataframe: Union[pl.DataFrame, pl.LazyFrame],
        file_boundaries: List[Tuple[int, int]],
        indices: List[int],
        window_size: int = 320,
        num_harmonics: int = 9,
        sub_periods: Optional[List[int]] = None,
        downsampling_stride: int = 16,
        future_periods: int = 2,
        mask_ratio: float = 0.25,
        mask_value: float = 0.0,
        augmenter: Optional[object] = None,
        target_columns: Optional[List[str]] = None,
        target_level: str = 'base',
        mode: str = 'ssl',
        target_window_mode: str = 'any_in_window',
    ) -> None:
        """
        Args:
            dataframe: DataFrame с raw колонками + метки
            file_boundaries: список (file_start, file_length) для каждого файла
            indices: индексы начал окон (абсолютные в DataFrame)
            window_size: размер основного окна в отсчётах (320 = 10 периодов)
            num_harmonics: число стандартных гармоник (9)
            sub_periods: суб-периоды для низших гармоник [2, 4, 6, 10]
            downsampling_stride: шаг прореживания (16 = полпериода)
            future_periods: число будущих периодов для SSL (0 = без предсказания)
            mask_ratio: доля маскируемых шагов (0.0 для val/classify)
            mask_value: значение замены маскированных данных
            augmenter: TimeSeriesAugmenter (None для отключения аугментации)
            target_columns: колонки меток для classify (None → из target_level)
            target_level: 'base' (4 класса)
            mode: 'ssl' или 'classify'
            target_window_mode: 'point' или 'any_in_window'
        """
        super().__init__()

        if isinstance(dataframe, pl.LazyFrame):
            dataframe = dataframe.collect()

        self.window_size = window_size
        self.num_harmonics = num_harmonics
        self.sub_periods = sub_periods if sub_periods is not None else [2, 4, 6, 10]
        self.num_sub_periods = len(self.sub_periods)
        self.downsampling_stride = downsampling_stride
        self.future_periods = future_periods
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.augmenter = augmenter
        self.mode = mode
        self.target_window_mode = target_window_mode

        # Будущие шаги в сырых отсчётах
        self.future_raw_steps = future_periods * self.SAMPLES_PER_PERIOD
        self.full_window_raw = window_size + self.future_raw_steps

        # Контекст для backward-looking низших гармоник
        max_lh_window = max(self.sub_periods) * self.SAMPLES_PER_PERIOD
        self._context_before = max_lh_window - 1  # 319 при sub_period=10

        # Число каналов после FFT + polar conversion
        self._channels_per_signal = (num_harmonics + self.num_sub_periods) * 2
        self.num_output_channels = self.NUM_RAW_CHANNELS * self._channels_per_signal

        # FFT warmup: первые FFT_WINDOW точек forward-looking FFT
        # После применения stride — пропускаем warmup
        warmup = self.FFT_WINDOW
        valid_len_full = self.full_window_raw - warmup
        self.num_steps_full = max(1, valid_len_full // downsampling_stride)
        valid_len_current = window_size - warmup
        self.num_steps_current = max(1, valid_len_current // downsampling_stride)

        # --- Предзагрузка данных ---
        self._raw_np = dataframe.select(self.RAW_CHANNELS).to_numpy().astype(np.float32)

        # Метки
        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = get_target_columns(target_level)
        self._targets_np = dataframe.select(self.target_columns).to_numpy().astype(np.float32)

        # Границы файлов: (start, end) в абсолютных индексах DataFrame
        self._file_boundaries = []
        for fstart, flen in file_boundaries:
            self._file_boundaries.append((fstart, fstart + flen))

        self.indices = indices

        print(f"    [AugmentedSpectralDataset] mode={mode}, indices={len(indices):,}, "
              f"channels={self.num_output_channels}, "
              f"harmonics={num_harmonics}+{self.num_sub_periods}lh, "
              f"augmentation={'ON' if augmenter else 'OFF'}")

    def _find_file_range(self, abs_idx: int) -> Tuple[int, int]:
        """Находит границы файла (start, end), содержащего abs_idx."""
        for fstart, fend in self._file_boundaries:
            if fstart <= abs_idx < fend:
                return fstart, fend
        # Fallback: весь массив
        return 0, len(self._raw_np)

    def _load_context_raw(self, start_idx: int) -> Tuple[np.ndarray, int]:
        """Загружает расширенное окно raw данных с контекстом для низших гармоник.

        Если контекст выходит за начало файла — дублируем первую строку.

        Args:
            start_idx: начало основного (training) окна в DataFrame

        Returns:
            (raw_data, context_offset) — raw_data shape (T_extended, 8),
            context_offset — индекс начала основного окна внутри raw_data
        """
        file_start, file_end = self._find_file_range(start_idx)

        # Границы нужного raw окна
        ctx_start = start_idx - self._context_before
        raw_end = min(start_idx + self.full_window_raw, file_end)
        needed_len = self._context_before + self.full_window_raw

        if ctx_start >= file_start:
            # Контекст полностью внутри файла
            raw_data = self._raw_np[ctx_start:raw_end].copy()
            context_offset = self._context_before
        else:
            # Контекст выходит за начало файла — нужен padding
            actual_start = file_start
            raw_data_part = self._raw_np[actual_start:raw_end].copy()
            pad_len = file_start - ctx_start  # сколько строк не хватает

            # Дублируем первую строку файла
            pad_row = self._raw_np[file_start:file_start + 1]  # (1, 8)
            pad_block = np.repeat(pad_row, pad_len, axis=0)     # (pad_len, 8)
            raw_data = np.concatenate([pad_block, raw_data_part], axis=0)
            context_offset = self._context_before

        # Если raw_data короче нужного — дополняем нулями
        if len(raw_data) < needed_len:
            pad = np.zeros((needed_len - len(raw_data), self.NUM_RAW_CHANNELS), dtype=np.float32)
            raw_data = np.concatenate([raw_data, pad], axis=0)

        return raw_data, context_offset

    def _compute_spectral_features(self, raw: np.ndarray, context_offset: int) -> np.ndarray:
        """Вычисляет спектральные признаки (phase_polar) из сырых данных.

        Args:
            raw: (T_extended, 8) — сырые аналоговые каналы с контекстом
            context_offset: индекс начала основного окна в raw

        Returns:
            features: (T_feat, num_output_channels) — phase_polar признаки
                      для основного + будущего окна
        """
        n_raw = len(raw)

        # --- Стандартные гармоники (forward-looking, окно = FFT_WINDOW) ---
        std_phasors = []  # Список (T_extended, num_harmonics) для каждого канала
        for ch in range(self.NUM_RAW_CHANNELS):
            p = sliding_window_fft(raw[:, ch], self.FFT_WINDOW, self.num_harmonics)
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)
            std_phasors.append(p)  # (T_extended, num_harmonics)

        # --- Низшие гармоники (backward-looking, окна = [2,4,6,10] × SAMPLES_PER_PERIOD) ---
        low_phasors = []  # Список (T_extended, num_sub_periods) для каждого канала
        for ch in range(self.NUM_RAW_CHANNELS):
            lh = compute_low_harmonics_fft(
                raw[:, ch],
                samples_per_period=self.SAMPLES_PER_PERIOD,
                sub_periods=self.sub_periods,
            )
            lh = np.nan_to_num(lh, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)
            low_phasors.append(lh)  # (T_extended, num_sub_periods)

        # --- Обрезаем до нужного окна: [context_offset : context_offset + full_window_raw] ---
        feat_start = context_offset
        feat_end = context_offset + self.full_window_raw
        feat_end = min(feat_end, n_raw)
        feat_len = feat_end - feat_start

        # Объединяем: per signal [std_h1..h9, lh2..lh10] → complex_flat (T, 8 * (H + LH))
        total_harmonics = self.num_harmonics + self.num_sub_periods
        complex_all = np.zeros((feat_len, self.NUM_RAW_CHANNELS, total_harmonics), dtype=np.complex64)

        for ch in range(self.NUM_RAW_CHANNELS):
            complex_all[:, ch, :self.num_harmonics] = std_phasors[ch][feat_start:feat_end]
            complex_all[:, ch, self.num_harmonics:] = low_phasors[ch][feat_start:feat_end]

        # Reshape: (T, 8, H+LH) → (T, 8*(H+LH))
        complex_flat = complex_all.reshape(feat_len, self.NUM_RAW_CHANNELS * total_harmonics)

        # --- Опорный фазор: UA h1 (индекс 4 из 8 каналов, гармоника 0 = h1) ---
        ua_h1 = complex_all[:, 4, 0]  # (T,) complex
        ia_h1 = complex_all[:, 0, 0]  # (T,) complex
        ua_mag = np.nanmean(np.abs(ua_h1))

        if ua_mag > 1e-4:
            ref_phasor = ua_h1
        else:
            ia_mag = np.nanmean(np.abs(ia_h1))
            ref_phasor = ia_h1 if ia_mag > 1e-4 else None

        # --- Polar conversion ---
        polar = calculate_polar_features(complex_flat, ref_phasor)  # (T, 8*(H+LH)*2)
        polar = np.nan_to_num(polar, nan=0.0, posinf=0.0, neginf=0.0)

        return polar  # (feat_len, num_output_channels)

    def _apply_stride(self, features: np.ndarray) -> np.ndarray:
        """Применяет warmup truncation и stride к features.

        Args:
            features: (T_raw, C) — полное окно признаков

        Returns:
            (T_strided, C) — прореженные признаки
        """
        # Пропускаем warmup (первые FFT_WINDOW точек)
        warmup = self.FFT_WINDOW
        if features.shape[0] > warmup:
            features = features[warmup:]
        # Stride
        features = features[::self.downsampling_stride]
        return features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Union[dict, Tuple[torch.Tensor, torch.Tensor]]:
        start_idx = self.indices[idx]

        # 1. Загрузить raw данные с контекстом
        raw_data, context_offset = self._load_context_raw(start_idx)

        # 2. Аугментация (на сырых данных, до FFT)
        if self.augmenter is not None:
            raw_data = self.augmenter(raw_data)  # (T_extended, 8)
            if isinstance(raw_data, torch.Tensor):
                raw_data = raw_data.numpy()
            raw_data = raw_data.astype(np.float32)

        # 3. Вычислить спектральные признаки
        polar_features = self._compute_spectral_features(raw_data, context_offset)
        # polar_features: (full_window_raw, num_output_channels)

        # 4. Stride прореживание
        strided = self._apply_stride(polar_features)
        # strided: (T_strided, C)

        # 5. Конвертация в тензор (C, T)
        X_full = torch.from_numpy(strided.T.copy())  # (C, T_strided)
        T_actual = X_full.shape[1]

        if self.mode == 'ssl':
            return self._format_ssl(X_full, T_actual)
        else:
            return self._format_classify(X_full, start_idx)

    def _format_ssl(
        self, X_full: torch.Tensor, T_actual: int
    ) -> dict:
        """Форматирует данные для SSL (маскирование + будущее предсказание).

        Args:
            X_full: (C, T_strided) полные данные
            T_actual: число реальных шагов

        Returns:
            dict совместимый с SSLSpectralDataset
        """
        C = X_full.shape[0]
        T_current = min(self.num_steps_current, T_actual)
        T_full = T_actual

        X_current = X_full[:, :T_current]

        # --- Маскирование ---
        mask_input = torch.zeros(C, T_current, dtype=torch.bool)
        num_to_mask = max(1, int(T_current * self.mask_ratio))
        mask_indices = random.sample(range(T_current), min(num_to_mask, T_current))
        for t_idx in mask_indices:
            mask_input[:, t_idx] = True

        X_masked = X_current.clone()
        X_masked[mask_input] = self.mask_value

        # --- Маска Loss ---
        mask_loss = torch.isnan(X_full)
        # Padding если T_actual < num_steps_full
        if T_actual < self.num_steps_full:
            pad_mask = torch.ones(C, self.num_steps_full - T_actual, dtype=torch.bool)
            mask_loss = torch.cat([mask_loss, pad_mask], dim=1)
            X_full_padded = torch.zeros(C, self.num_steps_full)
            X_full_padded[:, :T_actual] = X_full
            X_full = X_full_padded

        return {
            'input': X_masked,
            'target': X_full,
            'mask_input': mask_input,
            'mask_loss': mask_loss,
            'current_len': T_current,
        }

    def _format_classify(
        self, X_full: torch.Tensor, start_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Форматирует данные для supervised classification.

        Args:
            X_full: (C, T_strided) — признаки
            start_idx: начало окна в DataFrame (для меток)

        Returns:
            (X, Y)
        """
        T_current = min(self.num_steps_current, X_full.shape[1])
        X = X_full[:, :T_current]

        # Метка: max по окну (any_in_window)
        end_idx = min(start_idx + self.window_size, len(self._targets_np))
        if self.target_window_mode == 'any_in_window':
            y_window = np.max(self._targets_np[start_idx:end_idx], axis=0)
        else:
            target_pos = min(end_idx - 1, len(self._targets_np) - 1)
            y_window = self._targets_np[target_pos]

        Y = torch.from_numpy(y_window.astype(np.float32))
        return X, Y

    # ----- Утилиты для интеграции с Loss -----

    def get_channel_groups(self, separated: bool = True) -> list[list[int]]:
        """Возвращает группировку каналов по физическим сигналам.

        Совместимо с build_channel_groups_phase_polar, но учитывает низшие гармоники.

        Args:
            separated: если True — индексы для amp-only тензора (после split amp/angle)

        Returns:
            Список из 8 групп, каждая — индексы каналов амплитуд одного сигнала
        """
        total_h = self.num_harmonics + self.num_sub_periods  # 9 + 4 = 13

        if separated:
            groups = []
            for sig_idx in range(self.NUM_RAW_CHANNELS):
                base = sig_idx * total_h
                groups.append(list(range(base, base + total_h)))
            return groups
        else:
            channels_per_signal = total_h * 2
            groups = []
            for sig_idx in range(self.NUM_RAW_CHANNELS):
                base = sig_idx * channels_per_signal
                amp_indices = [base + h * 2 for h in range(total_h)]
                groups.append(amp_indices)
            return groups

    @staticmethod
    def compute_file_boundaries(df: pl.DataFrame) -> List[Tuple[int, int]]:
        """Вычисляет границы файлов в DataFrame.

        Args:
            df: DataFrame с колонкой 'file_name'

        Returns:
            Список (start_idx, length) для каждого файла
        """
        if 'row_nr' not in df.columns:
            df = df.with_row_index("row_nr")

        stats = df.group_by("file_name").agg([
            pl.col("row_nr").min().alias("start_idx"),
            pl.len().alias("length"),
        ]).sort("start_idx")

        return list(zip(
            stats["start_idx"].to_list(),
            stats["length"].to_list(),
        ))
