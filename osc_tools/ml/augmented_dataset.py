"""
AugmentedSpectralDataset — Dataset с аугментацией на сырых данных и on-the-fly FFT.

Pipeline (простой и прямолинейный):
1. Берёт окно сырых 8-канальных данных IA..UN (с контекстом слева, если есть; иначе NaN)
2. Аугментация на сырых данных (через TimeSeriesAugmenter)
3. FFT только в нужных точках → polar + symmetric только в них же
4. Возврат: SSL (dict) или classify (X, Y с позонной агрегацией меток)

Stride задаётся как доля периода (напр. 1/2 = 16, 1/4 = 8 при 1600/50=32).
Контекст для низших гармоник: если есть данные слева — берём;
если нет — заполняем NaN (модель научится не смотреть на них).

Каналы:
  - 8 сигналов × (9 гармоник + 4 низших) × 2 (mag+angle) = 208  (phase_polar)
  - 6 симм. составляющих (I1,I2,I0,U1,U2,U0) × 2 (mag+angle) = 12 (symmetric_polar)
  - Итого: 220 каналов

Использование:
    from osc_tools.ml.augmented_dataset import AugmentedSpectralDataset

    dataset = AugmentedSpectralDataset(
        dataframe=df,
        file_boundaries=boundaries,
        indices=train_indices,
        window_size=320,
    )
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from osc_tools.features.polar import calculate_polar_features
from osc_tools.features.phasor import calculate_symmetrical_components
from osc_tools.ml.labels import get_target_columns


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

RAW_CHANNELS = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
NUM_RAW = 8
FFT_WINDOW = 32           # 1 период (1600 Гц / 50 Гц)
SAMPLES_PER_PERIOD = 32
DEFAULT_SUB_PERIODS = [2, 4, 6, 10]
NUM_SYMMETRIC = 6          # I1, I2, I0, U1, U2, U0


# ---------------------------------------------------------------------------
# Ресэмплинг сырых сигналов к заданному target_spp (линейная интерполяция)
# ---------------------------------------------------------------------------

def resample_to_spp(
    raw: np.ndarray,
    source_spp: int,
    target_spp: int = SAMPLES_PER_PERIOD,
    f_network: float = 50.0,
) -> np.ndarray:
    """Передискретизация многоканального сигнала к целевому числу отсчётов на период.

    Алгоритм — линейная интерполяция по каждому каналу:
    - Если source_spp == target_spp → возвращает raw без копирования.
    - Если source_spp > target_spp → прореживание с линейным уточнением
      (если частоты кратны — берётся точка напрямую).
    - Если source_spp < target_spp → «дорисовывание» промежуточных точек
      линейной интерполяцией из двух соседних.

    Проход ведётся по целевому массиву: для каждой целевой позиции t находится
    соответствующая дробная позиция в исходном сигнале (линейное отображение),
    а затем вычисляется значение линейной интерполяцией между соседними отсчётами.

    Args:
        raw: (N_source, C) многоканальный сигнал (обычно 8 каналов IA..UN)
        source_spp: отсчётов на период в исходном сигнале (напр. round(Fs / f_network))
        target_spp: желаемое число отсчётов на период (по умолчанию 32)
        f_network: промышленная частота (50 Гц), не используется напрямую (для документации)

    Returns:
        (N_target, C) float32 массив, где N_target = round(N_source * target_spp / source_spp).
    """
    if source_spp == target_spp:
        return raw

    N_source = raw.shape[0]
    n_channels = raw.shape[1] if raw.ndim > 1 else 1
    # Определяем длину целевого массива пропорционально
    N_target = round(N_source * target_spp / source_spp)
    if N_target <= 0:
        return np.empty((0, n_channels), dtype=np.float32)

    # Дробные позиции в исходном массиве для каждой целевой точки
    ratio = source_spp / target_spp
    target_indices = np.arange(N_target, dtype=np.float64) * ratio

    # Целые индексы и веса для линейной интерполяции
    idx_lo = np.floor(target_indices).astype(np.int64)
    frac = (target_indices - idx_lo).astype(np.float32)

    # Клиппинг на границах
    idx_hi = np.minimum(idx_lo + 1, N_source - 1)
    idx_lo = np.minimum(idx_lo, N_source - 1)

    if raw.ndim == 1:
        result = raw[idx_lo] * (1 - frac) + raw[idx_hi] * frac
        return result.astype(np.float32)

    # Линейная интерполяция: vectorized по всем каналам
    result = raw[idx_lo] * (1 - frac[:, np.newaxis]) + raw[idx_hi] * frac[:, np.newaxis]
    return result.astype(np.float32)


def standardize_voltage_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Переименовывает колонки напряжений 'UA BB'→'UA' и т.д. для совместимости с RAW_CHANNELS."""
    available = set(df.columns)
    rename_map = {}
    if 'UA BB' in available and 'UA' not in available:
        for ph in ('UA', 'UB', 'UC', 'UN'):
            rename_map[f'{ph} BB'] = ph
    elif 'UA CL' in available and 'UA' not in available:
        for ph in ('UA', 'UB', 'UC', 'UN'):
            rename_map[f'{ph} CL'] = ph
    if rename_map:
        df = df.rename(rename_map)
    return df


def compute_stride(samples_per_period: int = SAMPLES_PER_PERIOD,
                   period_fraction: int = 2) -> int:
    """Вычисляет stride как долю периода.

    Args:
        samples_per_period: отсчётов в периоде (32 при 1600/50)
        period_fraction: делитель (2 = полпериода, 4 = четверть)

    Returns:
        stride в отсчётах (минимум 1)
    """
    raw = samples_per_period / period_fraction
    stride = int(round(raw))
    return max(1, stride)


def compute_num_channels(num_harmonics: int = 9,
                         num_sub_periods: int = 4,
                         include_symmetric: bool = True) -> int:
    """Вычисляет итоговое число каналов.

    phase_polar: 8 × (H + LH) × 2
    symmetric_polar: 6 × 2 (только h1)
    """
    ch = NUM_RAW * (num_harmonics + num_sub_periods) * 2
    if include_symmetric:
        ch += NUM_SYMMETRIC * 2
    return ch


def _extract_fft_harmonics(windows: np.ndarray, num_harmonics: int) -> np.ndarray:
    """Извлекает 1..num_harmonics из набора окон FFT."""
    out = np.full((windows.shape[0], num_harmonics), np.nan + 1j * np.nan, dtype=np.complex64)
    if windows.shape[0] == 0:
        return out

    fft_coeffs = np.fft.fft(windows, axis=1) / windows.shape[1]
    max_h = min(num_harmonics, fft_coeffs.shape[1] - 1)
    if max_h > 0:
        out[:, :max_h] = (fft_coeffs[:, 1:max_h + 1] * 2).astype(np.complex64)
    return out


def _compute_fft_selected(
    signal: np.ndarray,
    positions: np.ndarray,
    fft_window_size: int,
    num_harmonics: int,
) -> np.ndarray:
    """Считает FFT только для выбранных стартовых позиций окна."""
    pos = np.asarray(positions, dtype=np.int64)
    out = np.full((len(pos), num_harmonics), np.nan + 1j * np.nan, dtype=np.complex64)

    if len(pos) == 0 or len(signal) < fft_window_size:
        return out

    valid = (pos >= 0) & (pos + fft_window_size <= len(signal))
    if not np.any(valid):
        return out

    windows = np.lib.stride_tricks.sliding_window_view(signal, fft_window_size)
    out[valid] = _extract_fft_harmonics(windows[pos[valid]], num_harmonics)
    return out


def _compute_low_harmonics_selected(
    signal: np.ndarray,
    positions: np.ndarray,
    samples_per_period: int,
    sub_periods: List[int],
) -> np.ndarray:
    """Считает низшие гармоники только в выбранных backward-looking точках."""
    pos = np.asarray(positions, dtype=np.int64)
    out = np.full((len(pos), len(sub_periods)), np.nan + 1j * np.nan, dtype=np.complex64)

    if len(pos) == 0:
        return out

    for j, sub_period in enumerate(sub_periods):
        window_size = sub_period * samples_per_period
        if len(signal) < window_size:
            continue

        start_pos = pos - window_size + 1
        valid = (start_pos >= 0) & (pos < len(signal))

        first_valid = _extract_fft_harmonics(signal[:window_size][None, :], 1)[0, 0]
        if np.isfinite(first_valid.real) and np.isfinite(first_valid.imag):
            out[~valid, j] = first_valid

        if not np.any(valid):
            continue

        windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
        out[valid, j] = _extract_fft_harmonics(windows[start_pos[valid]], 1)[:, 0]

    return out


# ---------------------------------------------------------------------------
# Спектральная обработка одного окна сырых данных
# ---------------------------------------------------------------------------

def compute_spectral_from_raw(
    raw: np.ndarray,
    num_harmonics: int = 9,
    sub_periods: Optional[List[int]] = None,
    include_symmetric: bool = True,
    stride: Optional[int] = None,
    warmup: int = 0,
    fft_window: Optional[int] = None,
    samples_per_period: Optional[int] = None,
) -> np.ndarray:
    """Вычисляет спектральные признаки из сырых 8-канальных данных.

    Алгоритм:
    1. Выбираем нужные позиции по stride
    2. Считаем FFT только для этих позиций
    3. Polar conversion + симметричные только на этих позициях

    Args:
        raw: (T, 8) сырые данные
        num_harmonics: число стандартных гармоник (9)
        sub_periods: суб-периоды низших гармоник [2,4,6,10]
        include_symmetric: добавить ли симметричные составляющие
        stride: если задан — FFT/polar/symmetric считаются только в позициях
            [warmup, warmup+stride, warmup+2*stride, ...]
        warmup: пропускаемые начальные позиции (FFT warmup)
        fft_window: размер окна FFT в отсчётах (по умолчанию FFT_WINDOW = 1 период
            при 1600 Гц). Для других Fs передайте round(Fs / f_network).
        samples_per_period: отсчётов на период (по умолчанию SAMPLES_PER_PERIOD = 32).
            Для других Fs передайте round(Fs / f_network).

    Returns:
        (T_out, C) float32 массив. T_out = T если stride=None,
        иначе len(arange(warmup, T, stride)).
    """
    if sub_periods is None:
        sub_periods = DEFAULT_SUB_PERIODS
    if fft_window is None:
        fft_window = FFT_WINDOW
    if samples_per_period is None:
        samples_per_period = SAMPLES_PER_PERIOD

    n_time = raw.shape[0]

    # --- 1. Отбор позиций (stride) ---
    if stride is not None:
        sel = np.arange(warmup, n_time, stride)
    else:
        sel = np.arange(n_time)
    n_out = len(sel)

    # --- 2. FFT только в нужных позициях ---
    total_h = num_harmonics + len(sub_periods)
    complex_all = np.zeros((n_out, NUM_RAW, total_h), dtype=np.complex64)
    for ch in range(NUM_RAW):
        signal = raw[:, ch]
        phase_phasors = _compute_fft_selected(signal, sel, fft_window, num_harmonics)
        phase_phasors = np.nan_to_num(phase_phasors, nan=0.0, posinf=0.0, neginf=0.0)
        complex_all[:, ch, :num_harmonics] = phase_phasors

        low_phasors = _compute_low_harmonics_selected(signal, sel, samples_per_period, sub_periods)
        low_phasors = np.nan_to_num(low_phasors, nan=0.0, posinf=0.0, neginf=0.0)
        complex_all[:, ch, num_harmonics:] = low_phasors

    # --- 3. Polar + symmetric только на отобранных позициях ---
    complex_flat = complex_all.reshape(n_out, NUM_RAW * total_h)

    # Опорный фазор: UA h1 на отобранных позициях
    ua_h1 = complex_all[:, 4, 0]
    ia_h1 = complex_all[:, 0, 0]
    ua_mag = np.nanmean(np.abs(ua_h1))
    if ua_mag > 1e-4:
        ref_phasor = ua_h1
    else:
        ia_mag = np.nanmean(np.abs(ia_h1))
        ref_phasor = ia_h1 if ia_mag > 1e-4 else None

    polar = calculate_polar_features(complex_flat, ref_phasor)
    polar = np.nan_to_num(polar, nan=0.0, posinf=0.0, neginf=0.0)

    if not include_symmetric:
        return polar.astype(np.float32)

    # Симметричные составляющие (h1, уже на отобранных позициях)
    i1, i2, i0 = calculate_symmetrical_components(
        complex_all[:, 0, 0], complex_all[:, 1, 0], complex_all[:, 2, 0],
    )
    u1, u2, u0 = calculate_symmetrical_components(
        complex_all[:, 4, 0], complex_all[:, 5, 0], complex_all[:, 6, 0],
    )
    sym_complex = np.stack([i1, i2, i0, u1, u2, u0], axis=1)
    sym_polar = calculate_polar_features(sym_complex, ref_phasor)
    sym_polar = np.nan_to_num(sym_polar, nan=0.0, posinf=0.0, neginf=0.0)

    result = np.concatenate([polar, sym_polar], axis=1)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AugmentedSpectralDataset(Dataset):
    """Dataset с on-the-fly аугментацией и FFT.

    Режимы:
    - 'ssl': маскирование + предсказание будущего → dict
    - 'classify': supervised → (X, Y), включая будущие шаги
    """

    def __init__(
        self,
        dataframe: Union[pl.DataFrame, pl.LazyFrame],
        file_boundaries: List[Tuple[int, int]],
        indices: List[int],
        window_size: int = 320,
        num_harmonics: int = 9,
        sub_periods: Optional[List[int]] = None,
        include_symmetric: bool = True,
        downsampling_stride: Optional[int] = None,
        stride_fraction: int = 2,
        future_zones: int = 0,
        mask_ratio: float = 0.25,
        mask_value: float = 0.0,
        augmenter: Optional[object] = None,
        target_columns: Optional[List[str]] = None,
        target_level: str = 'base',
        mode: str = 'ssl',
        target_window_mode: str = 'any_in_window',
        zone_target_aggregation: str = 'max',
    ) -> None:
        """
        Args:
            dataframe: DataFrame с raw колонками + метки
            file_boundaries: (file_start, file_length) для каждого файла
            indices: абсолютные индексы начал окон в DataFrame
            window_size: основное окно (320 = 10 периодов)
            num_harmonics: стандартных гармоник (9)
            sub_periods: суб-периоды низших гармоник [2,4,6,10]
            include_symmetric: добавить симметричные составляющие (I1,I2,I0,U1,U2,U0)
            downsampling_stride: явный stride в отсчётах (если задан — игнорирует stride_fraction)
            stride_fraction: доля периода (2 = полпериода=16, 4 = четверть=8)
            future_zones: число выходных зон модели для предсказания вперёд.
                Не зависит от частоты дискретизации — оперирует шагами модели.
                future_raw_steps = future_zones * downsampling_stride
            mask_ratio: доля маскируемых шагов (0.0 для val)
            mask_value: значение замены
            augmenter: TimeSeriesAugmenter или None
            target_columns: колонки меток (None → из target_level)
            target_level: 'base' (4 класса)
            mode: 'ssl' или 'classify'
            target_window_mode: 'point' или 'any_in_window'
            zone_target_aggregation: агрегация меток внутри зоны ('max' или 'mean')
        """
        super().__init__()

        if isinstance(dataframe, pl.LazyFrame):
            dataframe = dataframe.collect()

        self.window_size = window_size
        self.num_harmonics = num_harmonics
        self.sub_periods = sub_periods if sub_periods is not None else DEFAULT_SUB_PERIODS
        self.include_symmetric = include_symmetric
        self.future_zones = future_zones
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.augmenter = augmenter
        self.mode = mode
        self.target_window_mode = target_window_mode
        self.zone_target_aggregation = zone_target_aggregation.lower()
        if self.zone_target_aggregation not in {'max', 'mean'}:
            raise ValueError(
                f"zone_target_aggregation must be 'max' or 'mean', got {zone_target_aggregation!r}",
            )

        # Stride: явный или из доли периода
        if downsampling_stride is not None:
            self.downsampling_stride = downsampling_stride
        else:
            self.downsampling_stride = compute_stride(SAMPLES_PER_PERIOD, stride_fraction)

        # Будущие шаги: определяются числом зон × stride (не зависят от SAMPLES_PER_PERIOD)
        self.future_raw_steps = future_zones * self.downsampling_stride
        self.full_window_raw = window_size + self.future_raw_steps

        # Контекст слева для backward-looking низших гармоник
        max_lh_window = max(self.sub_periods) * SAMPLES_PER_PERIOD
        self._context_before = max_lh_window  # 320 при период=10

        # Число выходных каналов
        self.num_output_channels = compute_num_channels(
            num_harmonics, len(self.sub_periods), include_symmetric,
        )

        # Число шагов после stride (warmup = FFT_WINDOW пропускается)
        warmup = FFT_WINDOW
        self.num_steps_current = max(1, (window_size - warmup) // self.downsampling_stride)
        self.num_steps_full = self.num_steps_current + future_zones

        # --- Предзагрузка ---
        dataframe = standardize_voltage_columns(dataframe)
        self._raw_np = dataframe.select(RAW_CHANNELS).to_numpy().astype(np.float32)

        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = get_target_columns(target_level)
        self._targets_np = dataframe.select(self.target_columns).to_numpy().astype(np.float32)

        # Границы файлов: (start, end)
        self._file_bounds = [(s, s + l) for s, l in file_boundaries]
        self.indices = indices

        print(f"    [AugmentedSpectralDataset] mode={mode}, "
              f"indices={len(indices):,}, channels={self.num_output_channels}, "
              f"stride={self.downsampling_stride} (1/{SAMPLES_PER_PERIOD // self.downsampling_stride} периода), "
              f"h={num_harmonics}+{len(self.sub_periods)}lh"
              f"{'+sym' if include_symmetric else ''}, "
              f"yagg={self.zone_target_aggregation}, "
              f"aug={'ON' if augmenter else 'OFF'}, "
              f"future_zones={future_zones}, ctx={self._context_before}")

    # ----- Загрузка сырых данных -----

    def _find_file_range(self, abs_idx: int) -> Tuple[int, int]:
        """Находит (start, end) файла, содержащего abs_idx."""
        for fstart, fend in self._file_bounds:
            if fstart <= abs_idx < fend:
                return fstart, fend
        return 0, len(self._raw_np)

    def _load_raw_window(self, start_idx: int) -> np.ndarray:
        """Загружает raw окно с контекстом слева для низших гармоник.

        Если данные слева есть — берём.
        Если нет (начало файла) — заполняем NaN,
        чтобы модель научилась не смотреть на них.

        Args:
            start_idx: начало основного окна в DataFrame

        Returns:
            (context + full_window_raw, 8) — сырые данные с контекстом
        """
        file_start, file_end = self._find_file_range(start_idx)

        # Контекст слева для backward-looking низших гармоник
        ctx = self._context_before
        ctx_start = start_idx - ctx
        raw_end = min(start_idx + self.full_window_raw, file_end)

        if ctx_start >= file_start:
            # Контекст полностью внутри файла
            raw = self._raw_np[ctx_start:raw_end].copy()
        else:
            # Не хватает данных слева — заполняем NaN
            nan_len = file_start - ctx_start
            raw_part = self._raw_np[file_start:raw_end].copy()
            nan_pad = np.full((nan_len, NUM_RAW), np.nan, dtype=np.float32)
            raw = np.concatenate([nan_pad, raw_part], axis=0)

        # Дополняем NaN справа, если короче
        needed = ctx + self.full_window_raw
        if len(raw) < needed:
            raw = np.concatenate([
                raw,
                np.full((needed - len(raw), NUM_RAW), np.nan, dtype=np.float32),
            ], axis=0)

        return raw

    # ----- Dataset протокол -----

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Union[dict, Tuple[torch.Tensor, torch.Tensor]]:
        start_idx = self.indices[idx]

        # 1. Загрузить raw окно (с контекстом слева / NaN если нет)
        raw = self._load_raw_window(start_idx)

        # 2. Аугментация на сырых данных (до FFT)
        if self.augmenter is not None:
            raw = self.augmenter(raw)
            if isinstance(raw, torch.Tensor):
                raw = raw.numpy()
            raw = np.asarray(raw, dtype=np.float32)

        # 3. FFT только в нужных точках → polar/symmetric только в них же
        # warmup смещён на контекст: первый stride-отсчёт начинается
        # с позиции _context_before + FFT_WINDOW в расширенном окне
        spectral = compute_spectral_from_raw(
            raw, self.num_harmonics, self.sub_periods, self.include_symmetric,
            stride=self.downsampling_stride,
            warmup=self._context_before + FFT_WINDOW,
        )  # (T_strided, C) — уже прорежено

        # 4. Тензор (C, T)
        X_full = torch.from_numpy(spectral.T.copy())
        T_actual = X_full.shape[1]

        if self.mode == 'ssl':
            return self._format_ssl(X_full, T_actual)
        else:
            return self._format_classify(X_full, start_idx)

    # ----- SSL формат -----

    def _format_ssl(self, X_full: torch.Tensor, T_actual: int) -> dict:
        C = X_full.shape[0]
        T_current = min(self.num_steps_current, T_actual)

        X_current = X_full[:, :T_current]

        # Маскирование
        mask_input = torch.zeros(C, T_current, dtype=torch.bool)
        n_mask = max(1, int(T_current * self.mask_ratio))
        for t in random.sample(range(T_current), min(n_mask, T_current)):
            mask_input[:, t] = True

        X_masked = X_current.clone()
        X_masked[mask_input] = self.mask_value

        # Маска Loss
        mask_loss = torch.isnan(X_full)
        if T_actual < self.num_steps_full:
            pad = torch.ones(C, self.num_steps_full - T_actual, dtype=torch.bool)
            mask_loss = torch.cat([mask_loss, pad], dim=1)
            X_padded = torch.zeros(C, self.num_steps_full)
            X_padded[:, :T_actual] = X_full
            X_full = X_padded

        return {
            'input': X_masked,
            'target': X_full,
            'mask_input': mask_input,
            'mask_loss': mask_loss,
            'current_len': T_current,
        }

    # ----- Classify формат -----

    def _format_classify(
        self, X_full: torch.Tensor, start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (X, Y) с позонной агрегацией меток.

        Каждая зона = downsampling_stride raw отсчётов.
        Y[z, c] агрегируется по точкам внутри зоны согласно zone_target_aggregation.

        ВАЖНО: X содержит только текущие шаги (num_steps_current),
        модель НЕ видит данные из будущего — это критично для РЗА.
        Y содержит num_steps_full зон (текущие + будущие) — модель
        предсказывает будущие зоны через FuturePredictionHead (cross-attention
        к encoder output), не имея доступа к будущим признакам.
        """
        # Модель видит только текущие данные, не будущие
        T_current = min(self.num_steps_current, X_full.shape[1])
        X = X_full[:, :T_current]

        # Y покрывает все зоны: текущие + будущие
        T_full = min(self.num_steps_full, X_full.shape[1])
        # Если future_zones=0, T_full == T_current
        T_y = max(T_full, T_current)

        num_targets = len(self.target_columns)
        Y = np.zeros((T_y, num_targets), dtype=np.float32)

        for z in range(T_y):
            # Зона z → raw позиции: [warmup + z*stride, warmup + (z+1)*stride)
            z_raw_start = start_idx + FFT_WINDOW + z * self.downsampling_stride
            z_raw_end = z_raw_start + self.downsampling_stride
            z_raw_end = min(z_raw_end, len(self._targets_np))
            z_raw_start = min(z_raw_start, len(self._targets_np) - 1)
            if z_raw_start < z_raw_end:
                zone_targets = self._targets_np[z_raw_start:z_raw_end]
                if self.zone_target_aggregation == 'mean':
                    Y[z] = np.mean(zone_targets, axis=0)
                else:
                    Y[z] = np.max(zone_targets, axis=0)
            else:
                Y[z] = self._targets_np[z_raw_start]

        return X, torch.from_numpy(Y)

    # ----- Утилиты -----

    def get_channel_groups(self, separated: bool = True) -> list[list[int]]:
        """Группировка каналов по физическим сигналам (для нормализации Loss).

        При separated=True — индексы для amp-only тензора (после 0::2 split).
        Симметричные составляющие НЕ включаются в группы (нормализуются отдельно).
        """
        total_h = self.num_harmonics + len(self.sub_periods)

        if separated:
            groups = []
            for sig in range(NUM_RAW):
                base = sig * total_h
                groups.append(list(range(base, base + total_h)))
            return groups
        else:
            cps = total_h * 2
            groups = []
            for sig in range(NUM_RAW):
                base = sig * cps
                groups.append([base + h * 2 for h in range(total_h)])
            return groups

    @staticmethod
    def compute_file_boundaries(df: pl.DataFrame) -> List[Tuple[int, int]]:
        """Вычисляет (start, length) для каждого файла в DataFrame."""
        if 'row_nr' not in df.columns:
            df = df.with_row_index("row_nr")
        stats = df.group_by("file_name").agg([
            pl.col("row_nr").min().alias("start_idx"),
            pl.len().alias("length"),
        ]).sort("start_idx")
        return list(zip(stats["start_idx"].to_list(), stats["length"].to_list()))
