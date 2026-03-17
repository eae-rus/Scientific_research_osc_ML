"""
AugmentedSpectralDataset — Dataset с аугментацией на сырых данных и on-the-fly FFT.

Pipeline (простой и прямолинейный):
1. Берёт окно сырых 8-канальных данных IA..UN (с контекстом для низших гармоник)
2. Аугментация на сырых данных (через TimeSeriesAugmenter)
3. Вычисляет FFT → phase_polar + низшие гармоники + симметричные составляющие
4. Stride прореживание
5. Возврат в формате SSL (dict) или classify (X, Y)

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

from osc_tools.preprocessing.filtering import sliding_window_fft, compute_low_harmonics_fft
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


# ---------------------------------------------------------------------------
# Спектральная обработка одного окна сырых данных
# ---------------------------------------------------------------------------

def compute_spectral_from_raw(
    raw: np.ndarray,
    num_harmonics: int = 9,
    sub_periods: Optional[List[int]] = None,
    include_symmetric: bool = True,
) -> np.ndarray:
    """Вычисляет все спектральные признаки из сырых 8-канальных данных.

    Вход: (T, 8) — сырые отсчёты (IA, IB, IC, IN, UA, UB, UC, UN).
    Выход: (T, C) — phase_polar + symmetric_polar.

    Эту функцию можно вызывать откуда угодно — она не зависит от Dataset.

    Args:
        raw: (T, 8) сырые данные
        num_harmonics: число стандартных гармоник (9)
        sub_periods: суб-периоды низших гармоник [2,4,6,10]
        include_symmetric: добавить ли симметричные составляющие

    Returns:
        (T, C) float32 массив в формате phase_polar (+ symmetric_polar)
    """
    if sub_periods is None:
        sub_periods = DEFAULT_SUB_PERIODS

    n_time = raw.shape[0]

    # --- Стандартные гармоники (forward-looking) ---
    phasors = []  # 8 массивов (T, H) complex
    for ch in range(NUM_RAW):
        p = sliding_window_fft(raw[:, ch], FFT_WINDOW, num_harmonics)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)
        phasors.append(p)

    # --- Низшие гармоники (backward-looking) ---
    low_phasors = []  # 8 массивов (T, LH) complex
    for ch in range(NUM_RAW):
        lh = compute_low_harmonics_fft(raw[:, ch], SAMPLES_PER_PERIOD, sub_periods)
        lh = np.nan_to_num(lh, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)
        low_phasors.append(lh)

    # Объединяем per-signal: [h1..h9, lh2..lh10] → (T, 8, H+LH)
    total_h = num_harmonics + len(sub_periods)
    complex_all = np.zeros((n_time, NUM_RAW, total_h), dtype=np.complex64)
    for ch in range(NUM_RAW):
        complex_all[:, ch, :num_harmonics] = phasors[ch]
        complex_all[:, ch, num_harmonics:] = low_phasors[ch]

    # Flatten: (T, 8*(H+LH))
    complex_flat = complex_all.reshape(n_time, NUM_RAW * total_h)

    # Опорный фазор: UA h1
    ua_h1 = phasors[4][:, 0]
    ia_h1 = phasors[0][:, 0]
    ua_mag = np.nanmean(np.abs(ua_h1))
    if ua_mag > 1e-4:
        ref_phasor = ua_h1
    else:
        ia_mag = np.nanmean(np.abs(ia_h1))
        ref_phasor = ia_h1 if ia_mag > 1e-4 else None

    # Phase polar: (T, 8*(H+LH)*2), чередование [mag, angle, mag, angle, ...]
    polar = calculate_polar_features(complex_flat, ref_phasor)
    polar = np.nan_to_num(polar, nan=0.0, posinf=0.0, neginf=0.0)

    if not include_symmetric:
        return polar

    # --- Симметричные составляющие (только h1) ---
    # Токи: IA(h1), IB(h1), IC(h1) → I1, I2, I0
    i1, i2, i0 = calculate_symmetrical_components(
        phasors[0][:, 0], phasors[1][:, 0], phasors[2][:, 0],
    )
    # Напряжения: UA(h1), UB(h1), UC(h1) → U1, U2, U0
    u1, u2, u0 = calculate_symmetrical_components(
        phasors[4][:, 0], phasors[5][:, 0], phasors[6][:, 0],
    )

    # Собираем (T, 6) complex и конвертируем в polar
    sym_complex = np.stack([i1, i2, i0, u1, u2, u0], axis=1)  # (T, 6)
    sym_polar = calculate_polar_features(sym_complex, ref_phasor)
    sym_polar = np.nan_to_num(sym_polar, nan=0.0, posinf=0.0, neginf=0.0)

    # Конкатенация: [phase_polar | symmetric_polar]
    result = np.concatenate([polar, sym_polar], axis=1)  # (T, 208+12=220)
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
            file_boundaries: (file_start, file_length) для каждого файла
            indices: абсолютные индексы начал окон в DataFrame
            window_size: основное окно (320 = 10 периодов)
            num_harmonics: стандартных гармоник (9)
            sub_periods: суб-периоды низших гармоник [2,4,6,10]
            include_symmetric: добавить симметричные составляющие (I1,I2,I0,U1,U2,U0)
            downsampling_stride: шаг прореживания (16 = полпериода)
            future_periods: будущих периодов (2 для SSL, >0 для расширенного classify)
            mask_ratio: доля маскируемых шагов (0.0 для val)
            mask_value: значение замены
            augmenter: TimeSeriesAugmenter или None
            target_columns: колонки меток (None → из target_level)
            target_level: 'base' (4 класса)
            mode: 'ssl' или 'classify'
            target_window_mode: 'point' или 'any_in_window'
        """
        super().__init__()

        if isinstance(dataframe, pl.LazyFrame):
            dataframe = dataframe.collect()

        self.window_size = window_size
        self.num_harmonics = num_harmonics
        self.sub_periods = sub_periods if sub_periods is not None else DEFAULT_SUB_PERIODS
        self.include_symmetric = include_symmetric
        self.downsampling_stride = downsampling_stride
        self.future_periods = future_periods
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.augmenter = augmenter
        self.mode = mode
        self.target_window_mode = target_window_mode

        # Будущие шаги
        self.future_raw_steps = future_periods * SAMPLES_PER_PERIOD
        self.full_window_raw = window_size + self.future_raw_steps

        # Контекст для backward-looking низших гармоник
        max_lh_window = max(self.sub_periods) * SAMPLES_PER_PERIOD
        self._context_before = max_lh_window - 1  # 319 при период=10

        # Число выходных каналов
        self.num_output_channels = compute_num_channels(
            num_harmonics, len(self.sub_periods), include_symmetric,
        )

        # Число шагов после stride
        warmup = FFT_WINDOW
        self.num_steps_full = max(1, (self.full_window_raw - warmup) // downsampling_stride)
        self.num_steps_current = max(1, (window_size - warmup) // downsampling_stride)

        # --- Предзагрузка ---
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
              f"h={num_harmonics}+{len(self.sub_periods)}lh"
              f"{'+sym' if include_symmetric else ''}, "
              f"aug={'ON' if augmenter else 'OFF'}, "
              f"future={future_periods}p")

    # ----- Загрузка сырых данных -----

    def _find_file_range(self, abs_idx: int) -> Tuple[int, int]:
        """Находит (start, end) файла, содержащего abs_idx."""
        for fstart, fend in self._file_bounds:
            if fstart <= abs_idx < fend:
                return fstart, fend
        return 0, len(self._raw_np)

    def _load_raw_window(self, start_idx: int) -> Tuple[np.ndarray, int]:
        """Загружает raw окно с контекстом для низших гармоник.

        Args:
            start_idx: начало основного окна в DataFrame

        Returns:
            (raw_data, context_offset) — (T_ext, 8), смещение основного окна
        """
        file_start, file_end = self._find_file_range(start_idx)
        needed = self._context_before + self.full_window_raw

        ctx_start = start_idx - self._context_before
        raw_end = min(start_idx + self.full_window_raw, file_end)

        if ctx_start >= file_start:
            raw = self._raw_np[ctx_start:raw_end].copy()
        else:
            # Padding — дублируем первую строку файла
            pad_len = file_start - ctx_start
            raw_part = self._raw_np[file_start:raw_end].copy()
            pad = np.repeat(self._raw_np[file_start:file_start + 1], pad_len, axis=0)
            raw = np.concatenate([pad, raw_part], axis=0)

        # Если короче — дополняем нулями
        if len(raw) < needed:
            raw = np.concatenate([
                raw,
                np.zeros((needed - len(raw), NUM_RAW), dtype=np.float32),
            ], axis=0)

        return raw, self._context_before

    # ----- Dataset протокол -----

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Union[dict, Tuple[torch.Tensor, torch.Tensor]]:
        start_idx = self.indices[idx]

        # 1. Загрузить raw с контекстом
        raw_ext, ctx_offset = self._load_raw_window(start_idx)

        # 2. Аугментация на сырых данных (до FFT)
        if self.augmenter is not None:
            raw_ext = self.augmenter(raw_ext)
            if isinstance(raw_ext, torch.Tensor):
                raw_ext = raw_ext.numpy()
            raw_ext = np.asarray(raw_ext, dtype=np.float32)

        # 3. Спектральная обработка (всё через compute_spectral_from_raw)
        spectral = compute_spectral_from_raw(
            raw_ext, self.num_harmonics, self.sub_periods, self.include_symmetric,
        )

        # 4. Обрезаем до нужного окна (убираем контекст)
        spectral = spectral[ctx_offset: ctx_offset + self.full_window_raw]

        # 5. Stride: пропустить warmup + прореживание
        if spectral.shape[0] > FFT_WINDOW:
            spectral = spectral[FFT_WINDOW:]
        spectral = spectral[::self.downsampling_stride]

        # 6. Тензор (C, T)
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
        """Возвращает (X, Y).

        X включает и основное окно, и будущие периоды (если future_periods > 0).
        Метка берётся по ПОЛНОМУ окну (основное + будущее).
        """
        # Используем все шаги (текущее + будущее)
        T_use = min(self.num_steps_full, X_full.shape[1])
        X = X_full[:, :T_use]

        # Метка: max по полному окну (основное + будущее)
        label_end = min(start_idx + self.full_window_raw, len(self._targets_np))
        if self.target_window_mode == 'any_in_window':
            y = np.max(self._targets_np[start_idx:label_end], axis=0)
        else:
            pos = min(label_end - 1, len(self._targets_np) - 1)
            y = self._targets_np[pos]

        return X, torch.from_numpy(y.astype(np.float32))

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
