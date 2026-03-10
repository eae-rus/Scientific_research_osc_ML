"""
Детерминированный (физический) алгоритм детектирования ОЗЗ и ДПОЗЗ.

Реализует классические электротехнические критерии для классификации
однофазных замыканий на землю (ОЗЗ) в сетях с изолированной/компенсированной
нейтралью.

Целевые классы (НЕ взаимоисключающие — ДПОЗЗ и затухающее тоже являются ОЗЗ):
    0 — ОЗЗ (любое однофазное замыкание на землю, включая затухающее и ДПОЗЗ)
    1 — Затухающее ОЗЗ (пробой → экспоненциальный спад)
    2 — ДПОЗЗ (дуговые перемежающиеся замыкания, ступенчатая 3U0)
    None — Нормальный режим (3U0 ниже порога)

Важно: если классификатор определяет ДПОЗЗ (класс 2) или затухающее ОЗЗ (класс 1),
то ОЗЗ (класс 0) ТОЖЕ активен. В отличие от одноклассовой классификации,
здесь используется multi-label подход.

Пороги рассчитаны для НОРМАЛИЗОВАННЫХ данных:
    - Напряжения нормализованы на 3 * Ub_base (типично 300В или 1200В)
    - После нормализации: σ(UA) ≈ 0.146, диапазон ~[-0.44, +0.44]
    - 3U0 в нормальном режиме ≈ 0 (симметричная система)
    - 3U0 при ОЗЗ ≈ 0.3—1.5 (в нормализованных единицах)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Dict, Any, Tuple, Set
from scipy.signal import hilbert, find_peaks


DEFAULT_U0_RAW_THRESHOLD_VOLTS = 10.0
DEFAULT_UB_BASE_SECONDARY_VOLTS = 100.0


def u0_threshold_raw_to_normalized(
    raw_threshold_volts: float = DEFAULT_U0_RAW_THRESHOLD_VOLTS,
    ub_base_secondary_volts: float = DEFAULT_UB_BASE_SECONDARY_VOLTS,
) -> float:
    """Переводит уставку 3U0 из вторичных в нормализованные единицы.

    Нормализация в проекте: U_norm = U_raw / (3 * Ub_base).
    Следовательно, уставка в нормализованных единицах:
        u0_thr_norm = U_thr_raw / (3 * Ub_base)
    """
    denom = 3.0 * float(ub_base_secondary_volts)
    if denom <= 1e-9:
        return 0.0
    return float(raw_threshold_volts) / denom


DEFAULT_U0_THRESHOLD_NORM = u0_threshold_raw_to_normalized()


def _rms_fundamental_sliding(signal: np.ndarray, fs: int = 1600, f0: float = 50.0) -> np.ndarray:
    """
    Массив RMS первой гармоники (f0) через скользящее БПФ с окном в 1 период.

    Скользит окном N_period = round(fs/f0) точек по сигналу.
    Для каждой позиции i вычисляет БПФ окна signal[i:i+N_period] и
    извлекает амплитуду первой гармоники (бин 1).

    При окне длиной N_period = fs/f0 разрешение по частоте = f0,
    поэтому первая гармоника (f0) попадает ровно в бин с индексом 1.

    Args:
        signal: временной ряд длины T
        fs: частота дискретизации, Гц
        f0: номинальная частота, Гц

    Returns:
        Массив RMS значений длины (T - N_period + 1).
        rms[i] соответствует окну signal[i:i+N_period].
    """
    n_period = int(round(fs / f0))  # 32 отсчёта для 1600/50
    T = len(signal)
    if T < n_period or n_period < 2:
        return np.array([], dtype=np.float64)

    # sliding_window_view создаёт view без копирования данных
    windows = sliding_window_view(signal, n_period)  # (T - n_period + 1, n_period)

    # Батчевое БПФ по всем окнам
    spectra = np.fft.rfft(windows, axis=1)  # (n_windows, n_period//2 + 1)

    # Бин 1 = частота f0 (разрешение = fs/n_period = f0)
    magnitudes = np.abs(spectra[:, 1]) * 2.0 / n_period  # амплитуды
    rms_arr = magnitudes / np.sqrt(2)  # RMS = A / sqrt(2)

    return rms_arr


def _envelope(signal: np.ndarray) -> np.ndarray:
    """
    Огибающая амплитуды сигнала через преобразование Гильберта.

    Returns:
        Массив огибающей той же длины, что и сигнал
    """
    if len(signal) < 4:
        return np.abs(signal)
    analytic = hilbert(signal)
    return np.abs(analytic).astype(np.float64)


# ---------------------------------------------------------------------------
#  Предрассчёт фич для всего файла (один раз)
# ---------------------------------------------------------------------------

class OzzPrecomputedFeatures:
    """Предрассчитанные фичи для физического алгоритма ОЗЗ на весь файл.

    Позволяет избежать повторных вычислений при скольжении окна 320 точек.

    Attributes:
        u0_3: Массив 3U0(t) длины T.
        u0_rms_arr: Массив RMS первой гармоники за каждый период.
                    Длина (T - n_period + 1). rms[i] → окно [i, i+n_period).
        du0: Производная 3U0 длины (T - 1).
        envelope: Огибающая 3U0 длины T.
        n_period: Число отсчётов в 1 периоде (fs/f0).
        fs: Частота дискретизации.
    """

    __slots__ = ('u0_3', 'u0_rms_arr', 'du0', 'envelope', 'n_period', 'fs')

    def __init__(self, u0_3: np.ndarray, u0_rms_arr: np.ndarray,
                 du0: np.ndarray, envelope: np.ndarray,
                 n_period: int, fs: int):
        self.u0_3 = u0_3
        self.u0_rms_arr = u0_rms_arr
        self.du0 = du0
        self.envelope = envelope
        self.n_period = n_period
        self.fs = fs


def precompute_ozz_features(
    signals: np.ndarray,
    fs: int = 1600,
    f0: float = 50.0,
) -> OzzPrecomputedFeatures:
    """Предрассчёт всех фич для файла целиком (один раз).

    Args:
        signals: Массив формы (T, 8) — [IA, IB, IC, IN, UA, UB, UC, UN].
                 Или (8, T) — автоматически транспонируется.
        fs: Частота дискретизации, Гц.
        f0: Номинальная частота сети, Гц.

    Returns:
        OzzPrecomputedFeatures с готовыми массивами.
    """
    data = np.asarray(signals, dtype=np.float64)
    # Если форма (8, T) — транспонируем в (T, 8)
    if data.ndim == 2 and data.shape[0] == 8 and data.shape[1] != 8:
        data = data.T

    ua = data[:, 4]
    ub = data[:, 5]
    uc = data[:, 6]
    u0_3 = ua + ub + uc

    u0_rms_arr = _rms_fundamental_sliding(u0_3, fs=fs, f0=f0)

    dt = 1.0 / fs
    du0 = np.diff(u0_3) / dt

    env = _envelope(u0_3)

    n_period = int(round(fs / f0))

    return OzzPrecomputedFeatures(
        u0_3=u0_3,
        u0_rms_arr=u0_rms_arr,
        du0=du0,
        envelope=env,
        n_period=n_period,
        fs=fs,
    )


def _max_consecutive_run(mask: np.ndarray) -> int:
    """Максимальная длина подряд идущих True в булевом массиве."""
    if len(mask) == 0:
        return 0
    # diff между 0 и 1: группируем серии единиц
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def classify_window_from_features(
    features: OzzPrecomputedFeatures,
    start: int,
    end: int,
    u0_threshold: float = DEFAULT_U0_THRESHOLD_NORM,
    deriv_peak_sigma: float = 5.0,
    min_dpozz_peaks: int = 3,
    decay_ratio: float = 0.30,
    envelope_tail_fraction: float = 0.25,
    operate_delay_samples: int = 32,
    operate_delay_periods: float = 1.0,
) -> Optional[Set[int]]:
    """Классифицирует окно [start, end) по предрассчитанным фичам.

    Возвращает НАБОР активных классов (multi-label), т.к. ДПОЗЗ и затухающее
    ОЗЗ могут наблюдаться одновременно в одном окне 320 точек.
    Без базового факта ОЗЗ (класс 0) остальные классы не проверяются.

    Args:
        features: Предрассчитанные фичи файла (OzzPrecomputedFeatures).
        start: Начало окна (индекс отсчёта, включительно).
        end: Конец окна (индекс отсчёта, исключительно). end - start = window_size.
        u0_threshold: Порог RMS первой гармоники 3U0. Ниже — нормальный режим.
        deriv_peak_sigma: Порог пиков производной в единицах СКО.
        min_dpozz_peaks: Минимальное число значимых пиков производной для ДПОЗЗ.
        decay_ratio: Если огибающая в конце окна < decay_ratio * max(огибающая),
                     считаем режим затухающим.
        envelope_tail_fraction: Доля конца окна для оценки затухания (0..1).
        operate_delay_samples: Выдержка времени: число ПОДРЯД идущих отсчётов
            с RMS выше порога. Используется если operate_delay_periods <= 0.
        operate_delay_periods: Выдержка времени в периодах сети 50 Гц.
            Преобразуется в отсчёты: periods * fs / 50.
            Если > 0, имеет приоритет над operate_delay_samples.

    Returns:
        set({0})       — устойчивое ОЗЗ
        set({0, 1})    — затухающее ОЗЗ
        set({0, 2})    — ДПОЗЗ
        set({0, 1, 2}) — ДПОЗЗ + затухающее одновременно
        None           — нормальный режим (нет ОЗЗ)
    """
    T = end - start
    if T < 2:
        return None
    fs = features.fs
    n_period = features.n_period

    # --- Шаг 2: Срез скользящего RMS для текущего окна ---
    # u0_rms_arr[i] соответствует периоду signal[i:i+n_period].
    # Для окна [start, end) нужны позиции от start до (end - n_period).
    rms_start = start
    rms_end = end - n_period + 1
    if rms_end <= rms_start or rms_end > len(features.u0_rms_arr):
        return None
    rms_slice = features.u0_rms_arr[rms_start:rms_end]
    if len(rms_slice) == 0 or np.max(rms_slice) < u0_threshold:
        return None  # Нормальный режим

    # --- Шаг 2.1: Выдержка времени (подряд идущие отсчёты выше порога) ---
    if operate_delay_periods > 0:
        delay_samples = int(round(operate_delay_periods * fs / 50.0))
    else:
        delay_samples = int(max(0, operate_delay_samples))

    if delay_samples > 0:
        above_mask = rms_slice >= u0_threshold
        max_run = _max_consecutive_run(above_mask)
        if max_run < delay_samples:
            return None

    # --- Базовый ОЗЗ подтверждён, собираем набор активных классов ---
    active_classes: Set[int] = {0}

    # --- Шаг 3-4: Производная 3U0 → критерий ДПОЗЗ ---
    # du0 имеет длину (T_total - 1). Для окна [start, end) берём [start, end-1).
    du0_slice = features.du0[start:end - 1]
    du0_std = np.std(du0_slice)

    env_slice = features.envelope[start:end]
    tail_len = max(int(T * envelope_tail_fraction), 1)
    env_tail_mean = np.mean(env_slice[-tail_len:])
    env_max = np.max(env_slice)

    if du0_std > 1e-9:
        threshold_abs = deriv_peak_sigma * du0_std
        min_distance = max(int(fs / 50.0 / 2), 1)
        peaks, _ = find_peaks(np.abs(du0_slice), height=threshold_abs, distance=min_distance)

        has_trapped_charge = (env_max > 1e-9) and (env_tail_mean > decay_ratio * env_max)

        if len(peaks) >= min_dpozz_peaks and has_trapped_charge:
            active_classes.add(2)  # ДПОЗЗ

    # --- Шаг 5: Критерий затухающего ОЗЗ ---
    if env_max > 1e-9:
        if env_tail_mean < decay_ratio * env_max:
            peak_pos = np.argmax(env_slice)
            if peak_pos < T * 0.7:
                active_classes.add(1)  # Затухающее ОЗЗ

    return active_classes


def predict_ozz_physics_batch(
    windows: np.ndarray,
    fs: int = 1600,
    **kwargs
) -> np.ndarray:
    """
    Батчевый вариант классификации ОЗЗ (multi-label).

    Args:
        windows: Массив формы (B, T, 8) или (B, 8, T).
        fs: Частота дискретизации.
        **kwargs: Дополнительные параметры для классификации.

    Returns:
        Массив предсказаний (B, 3) dtype=int8.
        Колонки: [OZZ, Decay, DPOZZ]. 1 = активен, 0 = нет.
    """
    B = windows.shape[0]
    results = np.zeros((B, 3), dtype=np.int8)
    for i in range(B):
        data = np.asarray(windows[i], dtype=np.float64)
        if data.ndim == 2:
            if data.shape[0] == 8 and data.shape[1] != 8:
                data = data.T
            features = precompute_ozz_features(data, fs=fs)
            pred = classify_window_from_features(features, 0, data.shape[0], **kwargs)
            if pred is not None:
                results[i, 0] = 1  # OZZ
                if 1 in pred:
                    results[i, 1] = 1  # Decay
                if 2 in pred:
                    results[i, 2] = 1  # DPOZZ
    return results


def evaluate_physics_baseline(
    dataframe,
    window_size: int = 320,
    stride: int = 1,
    target_cols=None,
    fs: int = 1600,
    **kwargs
) -> Dict[str, Any]:
    """
    Оценивает физическую модель на DataFrame с осциллограммами.

    Проходит по DataFrame скользящим окном, выполняет классификацию
    и сравнивает с реальными метками.

    Args:
        dataframe: polars DataFrame с колонками IA..UN и целевыми метками.
        window_size: Размер окна.
        stride: Шаг скользящего окна.
        target_cols: Список целевых колонок (по умолчанию OZZ-колонки).
        fs: Частота дискретизации.

    Returns:
        Словарь с метриками: predictions, targets, per_class_f1, macro_f1.
    """
    import polars as pl

    if target_cols is None:
        target_cols = ['Target_OZZ', 'Target_OZZ_decay', 'Target_OZZ_dpozz']

    # Определяем колонки сигналов
    raw_cols = ['IA', 'IB', 'IC', 'IN']
    voltage_candidates = [
        ['UA BB', 'UB BB', 'UC BB', 'UN BB'],
        ['UA CL', 'UB CL', 'UC CL', 'UN CL'],
        ['UA', 'UB', 'UC', 'UN'],
    ]
    available = set(dataframe.columns)
    u_cols = None
    for candidates in voltage_candidates:
        if all(c in available for c in candidates):
            u_cols = candidates
            break
    if u_cols is None:
        raise ValueError("Не найдены колонки напряжений в DataFrame")

    signal_cols = raw_cols + u_cols

    # Извлекаем массивы
    signal_data = dataframe.select(signal_cols).to_numpy().astype(np.float64)

    # Предрассчёт фич один раз на весь массив
    features = precompute_ozz_features(signal_data, fs=fs)

    all_preds = []
    all_targets = []

    n_samples = len(dataframe)
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        pred_result = classify_window_from_features(
            features, start=start, end=end, **kwargs,
        )

        # Формируем вектор предсказания (multi-label):
        # classify_window_from_features возвращает set или None.
        pred_vec = np.zeros(len(target_cols), dtype=np.int8)
        if pred_result is not None:
            pred_vec[0] = 1  # Target_OZZ (класс 0 всегда в set)
            if 1 in pred_result and len(target_cols) > 1:
                pred_vec[1] = 1  # Target_OZZ_decay
            if 2 in pred_result and len(target_cols) > 2:
                pred_vec[2] = 1  # Target_OZZ_dpozz

        # Реальные метки (берём по последней точке окна — point mode)
        target_idx = end - 1
        target_vec = np.array(
            [dataframe[col][target_idx] for col in target_cols], dtype=np.int8
        )

        all_preds.append(pred_vec)
        all_targets.append(target_vec)

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    # Расчёт F1 per-class
    from sklearn.metrics import f1_score, classification_report

    # Конвертируем one-hot в мультилейбл формат для f1_score
    f1_per_class = f1_score(targets_arr, preds_arr, average=None, zero_division=0)
    f1_macro = f1_score(targets_arr, preds_arr, average='macro', zero_division=0)

    return {
        'predictions': preds_arr,
        'targets': targets_arr,
        'f1_per_class': dict(zip(target_cols, f1_per_class.tolist())),
        'f1_macro': float(f1_macro),
        'classification_report': classification_report(
            targets_arr, preds_arr,
            target_names=target_cols,
            zero_division=0
        )
    }
