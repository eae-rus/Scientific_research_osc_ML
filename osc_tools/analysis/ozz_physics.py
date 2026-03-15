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
from scipy.signal import find_peaks


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


def _rms_fundamental_sliding(
    signal: np.ndarray, fs: int = 1600, f0: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Скользящее БПФ: RMS первой гармоники, RMS высших гармоник и THD.

    Скользит окном N_period = round(fs/f0) точек по сигналу.
    Для каждой позиции i вычисляет БПФ окна signal[i:i+N_period].

    При окне длиной N_period = fs/f0 разрешение по частоте = f0,
    поэтому первая гармоника (f0) попадает ровно в бин с индексом 1.

    Args:
        signal: временной ряд длины T
        fs: частота дискретизации, Гц
        f0: номинальная частота, Гц

    Returns:
        (rms_fundamental, rms_harmonics, thd_arr) — три массива длины
        (T - N_period + 1).
        - rms_fundamental: RMS первой гармоники (бин 1).
        - rms_harmonics: суммарное RMS высших гармоник (бины 2..Nyquist).
        - thd_arr: THD = rms_harmonics / (rms_fundamental + ε).
    """
    n_period = int(round(fs / f0))  # 32 отсчёта для 1600/50
    T = len(signal)
    empty = np.array([], dtype=np.float64)
    if T < n_period or n_period < 2:
        return empty, empty, empty

    # sliding_window_view создаёт view без копирования данных
    windows = sliding_window_view(signal, n_period)  # (n_win, n_period)

    # Батчевое БПФ по всем окнам
    spectra = np.fft.rfft(windows, axis=1)  # (n_win, n_period//2 + 1)

    # Амплитуды каждого бина (двусторонний спектр → *2/N)
    all_mag = np.abs(spectra) * 2.0 / n_period  # (n_win, n_bins)

    # Бин 0 — постоянная составляющая, бин 1 — f0, бины 2+ — высшие
    mag_fund = all_mag[:, 1]                          # первая гармоника
    # Суммарная амплитуда высших гармоник (корень из суммы квадратов)
    mag_harmonics = np.sqrt(np.sum(all_mag[:, 2:] ** 2, axis=1))

    rms_fund = mag_fund / np.sqrt(2)
    rms_harm = mag_harmonics / np.sqrt(2)
    thd_arr = rms_harm / (rms_fund + 1e-9)

    return rms_fund, rms_harm, thd_arr


# _envelope удалена: преобразование Гильберта заменено на анализ тренда
# скользящего RMS (более устойчив на краях окна и при скачках).


# ---------------------------------------------------------------------------
#  Предрассчёт фич для всего файла (один раз)
# ---------------------------------------------------------------------------

class OzzPrecomputedFeatures:
    """Предрассчитанные фичи для физического алгоритма ОЗЗ на весь файл.

    Позволяет избежать повторных вычислений при скольжении окна 320 точек.

    Attributes:
        u0_3: Массив 3U0(t) длины T.
        u0_rms_arr: Массив RMS первой гармоники 3U0 за каждый период.
                    Длина (T - n_period + 1). rms[i] → окно [i, i+n_period).
        thd_arr: Массив THD (Total Harmonic Distortion) 3U0.
                 THD[i] = RMS_harmonics[i] / (RMS_fundamental[i] + ε).
                 Та же длина, что и u0_rms_arr.
        du0: Производная 3U0 длины (T - 1).
        n_period: Число отсчётов в 1 периоде (fs/f0).
        fs: Частота дискретизации.
    """

    __slots__ = ('u0_3', 'u0_rms_arr', 'thd_arr', 'du0', 'n_period', 'fs')

    def __init__(self, u0_3: np.ndarray, u0_rms_arr: np.ndarray,
                 thd_arr: np.ndarray, du0: np.ndarray,
                 n_period: int, fs: int):
        self.u0_3 = u0_3
        self.u0_rms_arr = u0_rms_arr
        self.thd_arr = thd_arr
        self.du0 = du0
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

    rms_fund, _rms_harm, thd_arr = _rms_fundamental_sliding(u0_3, fs=fs, f0=f0)

    dt = 1.0 / fs
    du0 = np.diff(u0_3) / dt

    n_period = int(round(fs / f0))

    return OzzPrecomputedFeatures(
        u0_3=u0_3,
        u0_rms_arr=rms_fund,
        thd_arr=thd_arr,
        du0=du0,
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
    thd_threshold: float = 0.15,
    decay_ratio: float = 0.30,
    rms_tail_fraction: float = 0.20,
    operate_delay_samples: int = 32,
    operate_delay_periods: float = 1.0,
) -> Optional[Set[int]]:
    """Классифицирует окно [start, end) по предрассчитанным фичам.

    Возвращает НАБОР активных классов (multi-label), т.к. ДПОЗЗ и затухающее
    ОЗЗ могут наблюдаться одновременно в одном окне 320 точек.
    Без базового факта ОЗЗ (класс 0) остальные классы не проверяются.

    Критерий ДПОЗЗ (класс 2):
        Дуговые перемежающиеся замыкания порождают ступенчатую 3U0 с огромным
        содержанием высших гармоник. Используются два условия (логическое ИЛИ):
        a) THD(3U0) на активном участке > thd_threshold (спектральный критерий);
        b) >= min_dpozz_peaks резких скачков dU0/dt > deriv_peak_sigma * σ,
           при условии что RMS в хвосте окна НЕ спадает (запертый заряд).

    Критерий затухающего ОЗЗ (класс 1):
        Амплитуда 3U0 экспоненциально спадает. Оценивается по тренду
        скользящего RMS первой гармоники: если RMS в хвосте окна
        < decay_ratio * max(RMS), то режим затухающий.

    Args:
        features: Предрассчитанные фичи файла (OzzPrecomputedFeatures).
        start: Начало окна (индекс отсчёта, включительно).
        end: Конец окна (индекс отсчёта, исключительно). end - start = window_size.
        u0_threshold: Порог RMS первой гармоники 3U0. Ниже — нормальный режим.
        deriv_peak_sigma: Порог пиков производной в единицах СКО (fallback ДПОЗЗ).
        min_dpozz_peaks: Минимальное число пиков производной для ДПОЗЗ (fallback).
        thd_threshold: Порог THD для детектирования ДПОЗЗ (0.15 = 15%).
        decay_ratio: Если RMS в хвосте окна < decay_ratio * max(RMS),
                     режим считается затухающим.
        rms_tail_fraction: Доля конца окна для оценки хвоста RMS (0..1).
        operate_delay_samples: Выдержка времени: число ПОДРЯД идущих отсчётов
            с RMS выше порога. Используется если operate_delay_periods <= 0.
        operate_delay_periods: Выдержка времени в периодах сети 50 Гц.
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

    # --- Шаг 1: Срез скользящего RMS для текущего окна ---
    rms_start = start
    rms_end = end - n_period + 1
    if rms_end <= rms_start or rms_end > len(features.u0_rms_arr):
        return None
    rms_slice = features.u0_rms_arr[rms_start:rms_end]
    if len(rms_slice) == 0 or np.max(rms_slice) < u0_threshold:
        return None  # Нормальный режим

    # --- Шаг 1.1: Выдержка времени (подряд идущие отсчёты выше порога) ---
    if operate_delay_periods > 0:
        delay_samples = int(round(operate_delay_periods * fs / 50.0))
    else:
        delay_samples = int(max(0, operate_delay_samples))

    above_mask = rms_slice >= u0_threshold
    if delay_samples > 0:
        max_run = _max_consecutive_run(above_mask)
        if max_run < delay_samples:
            return None

    # --- Базовый ОЗЗ подтверждён ---
    active_classes: Set[int] = {0}

    # --- Шаг 2: Критерий ДПОЗЗ (THD + fallback на пики производной) ---
    thd_slice = features.thd_arr[rms_start:rms_end]

    # Условие (a): THD на активном участке (где RMS выше порога)
    thd_on_active = thd_slice[above_mask]
    thd_detected = False
    if len(thd_on_active) > 0:
        # 90-й перцентиль THD на активных точках
        thd_p90 = np.percentile(thd_on_active, 90)
        if thd_p90 > thd_threshold:
            thd_detected = True

    # Условие (b) — fallback: пики производной + запертый заряд
    peaks_detected = False
    du0_slice = features.du0[start:end - 1]
    du0_std = np.std(du0_slice)
    if du0_std > 1e-9:
        threshold_abs = deriv_peak_sigma * du0_std
        min_distance = max(int(fs / 50.0 / 2), 1)
        peaks, _ = find_peaks(np.abs(du0_slice), height=threshold_abs, distance=min_distance)

        # Запертый заряд: RMS в хвосте окна не спадает
        tail_len = max(int(len(rms_slice) * rms_tail_fraction), 1)
        tail_rms = np.mean(rms_slice[-tail_len:])
        rms_max = np.max(rms_slice)
        has_trapped_charge = (rms_max > 1e-9) and (tail_rms > decay_ratio * rms_max)

        if len(peaks) >= min_dpozz_peaks and has_trapped_charge:
            peaks_detected = True

    if thd_detected or peaks_detected:
        active_classes.add(2)  # ДПОЗЗ

    # --- Шаг 3: Критерий затухающего ОЗЗ (тренд RMS) ---
    rms_max = np.max(rms_slice)
    idx_max = int(np.argmax(rms_slice))
    tail_len = max(int(len(rms_slice) * rms_tail_fraction), 1)
    tail_rms = np.mean(rms_slice[-tail_len:])

    if rms_max > 1e-9:
        # Пик не в самом конце окна — есть время оценить затухание
        if idx_max < 0.7 * len(rms_slice):
            if tail_rms < decay_ratio * rms_max:
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
