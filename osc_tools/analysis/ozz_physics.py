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
from typing import Optional, Dict, Any
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


def _rms(signal: np.ndarray) -> float:
    """Действующее значение (RMS) сигнала."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def _rms_fundamental(signal: np.ndarray, fs: int = 1600, f0: float = 50.0) -> float:
    """
    RMS первой гармоники (f0) через БПФ.

    Args:
        signal: временной ряд
        fs: частота дискретизации, Гц
        f0: номинальная частота, Гц

    Returns:
        Действующее значение первой гармоники
    """
    n = len(signal)
    if n < 2:
        return 0.0
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Ищем индекс ближайший к f0
    idx = int(np.argmin(np.abs(freqs - f0)))
    magnitude = np.abs(spectrum[idx]) * 2.0 / n  # амплитуда
    return float(magnitude / np.sqrt(2))  # RMS = A / sqrt(2)


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


def predict_ozz_physics(
    window_data: np.ndarray,
    fs: int = 1600,
    u0_threshold: float = DEFAULT_U0_THRESHOLD_NORM,
    deriv_peak_sigma: float = 5.0,
    min_dpozz_peaks: int = 3,
    decay_ratio: float = 0.30,
    envelope_tail_fraction: float = 0.25,
    operate_delay_samples: int = 32,
    operate_delay_periods: float = 0.0,
) -> Optional[int]:
    """
    Физический (детерминированный) алгоритм классификации ОЗЗ по окну данных.

    Алгоритм работает по напряжениям фаз: 3U0 = UA + UB + UC.
    Последовательно проверяет критерии ДПОЗЗ, затухающего ОЗЗ и устойчивого ОЗЗ.

    Args:
        window_data: Массив формы (T, 8) — 8 нормализованных каналов
                     [IA, IB, IC, IN, UA, UB, UC, UN].
                     Или (8, T) — автоматически транспонируется.
        fs: Частота дискретизации, Гц (по умолчанию 1600).
        u0_threshold: Порог RMS первой гармоники 3U0 (в нормализованных единицах).
                  По умолчанию соответствует 10В вторичных при Ub_base=100В:
                  u0_threshold = 10 / (3*100) ≈ 0.0333.
                      Ниже этого значения — нормальный режим.
        deriv_peak_sigma: Порог пиков производной в единицах СКО.
                          Пик считается значимым, если |dU0/dt| > deriv_peak_sigma * std(dU0).
        min_dpozz_peaks: Минимальное число значимых пиков производной для ДПОЗЗ.
        decay_ratio: Если огибающая в конце окна < decay_ratio * max(огибающая),
                     считаем режим затухающим.
        envelope_tail_fraction: Доля конца окна для оценки затухания (0..1).
        operate_delay_samples: Выдержка времени в отсчётах.
            Если событие обнаружено слишком близко к концу окна и не успевает
            отработать заданную выдержку, возвращается None.
        operate_delay_periods: Выдержка времени в периодах сети 50 Гц.
            Преобразуется в отсчёты: periods * fs / 50.
            Если > 0, имеет приоритет над operate_delay_samples.

    Returns:
        0 — ОЗЗ (любое, включая подтипы)
        1 — Затухающее ОЗЗ (подтип → ОЗЗ тоже = 1)
        2 — ДПОЗЗ (подтип → ОЗЗ тоже = 1)
        None — Нормальный режим (нет ОЗЗ)

    Примечание:
        Функция возвращает НАИБОЛЕЕ СПЕЦИФИЧНЫЙ класс.
        При формировании multi-label вектора (Target_OZZ, Target_OZZ_decay,
        Target_OZZ_dpozz) вызывающий код должен учитывать, что классы 1 и 2
        являются подтипами класса 0 (ОЗЗ).
    """
    # --- Валидация и нормализация формы ---
    data = np.asarray(window_data, dtype=np.float64)
    if data.ndim != 2:
        return None
    # Если форма (8, T) — транспонируем в (T, 8)
    if data.shape[0] == 8 and data.shape[1] != 8:
        data = data.T
    if data.shape[1] < 8:
        return None

    T = data.shape[0]
    if T < 2:
        return None

    # --- Шаг 1: Вычисление 3U0 ---
    # Каналы: [IA(0), IB(1), IC(2), IN(3), UA(4), UB(5), UC(6), UN(7)]
    ua = data[:, 4]
    ub = data[:, 5]
    uc = data[:, 6]
    u0_3 = ua + ub + uc  # 3U0(t)

    # --- Шаг 2: Базовый критерий — RMS первой гармоники 3U0 ---
    u0_rms = _rms_fundamental(u0_3, fs=fs)
    if u0_rms < u0_threshold:
        return None  # Нормальный режим

    # --- Шаг 2.1: Выдержка времени (не-мгновенная отстройка) ---
    if operate_delay_periods > 0:
        delay_samples = int(round(operate_delay_periods * fs / 50.0))
    else:
        delay_samples = int(max(0, operate_delay_samples))

    if delay_samples > 0:
        amp_threshold = float(u0_threshold) * np.sqrt(2.0)
        above_idx = np.where(np.abs(u0_3) >= amp_threshold)[0]
        if len(above_idx) == 0:
            return None
        first_event_idx = int(above_idx[0])
        # Если до конца окна меньше выдержки — считаем, что уставка не отработала.
        if (T - 1 - first_event_idx) < delay_samples:
            return None

    # --- Шаг 3: Производная 3U0 ---
    dt = 1.0 / fs
    du0 = np.diff(u0_3) / dt
    du0_std = np.std(du0)

    # --- Шаг 4: Критерий ДПОЗЗ (Класс 2) ---
    # ДПОЗЗ: ступенчатая 3U0, множество резких всплесков dU0/dt
    if du0_std > 1e-9:
        threshold_abs = deriv_peak_sigma * du0_std
        # Минимальное расстояние между пиками — примерно полпериода (fs/f0/2)
        min_distance = max(int(fs / 50.0 / 2), 1)
        peaks, _ = find_peaks(np.abs(du0), height=threshold_abs, distance=min_distance)

        # Проверяем наличие «запертого заряда»: 3U0 не спадает до нуля
        env = _envelope(u0_3)
        tail_len = max(int(T * envelope_tail_fraction), 1)
        env_tail_mean = np.mean(env[-tail_len:])
        env_max = np.max(env)
        has_trapped_charge = (env_max > 1e-9) and (env_tail_mean > decay_ratio * env_max)

        if len(peaks) >= min_dpozz_peaks and has_trapped_charge:
            return 2  # ДПОЗЗ

    # --- Шаг 5: Критерий затухающего ОЗЗ (Класс 1) ---
    env = _envelope(u0_3)
    env_max = np.max(env)
    if env_max > 1e-9:
        tail_len = max(int(T * envelope_tail_fraction), 1)
        env_tail_mean = np.mean(env[-tail_len:])
        # Если конец окна значительно ниже пика — затухание
        if env_tail_mean < decay_ratio * env_max:
            # Дополнительно: проверяем, что максимум ближе к началу (пробой в начале)
            peak_pos = np.argmax(env)
            if peak_pos < T * 0.7:
                return 1  # Затухающее ОЗЗ

    # --- Шаг 6: Устойчивое ОЗЗ (Класс 0) ---
    # Если 3U0 выше порога, не ДПОЗЗ и не затухает — устойчивое
    return 0


def predict_ozz_physics_batch(
    windows: np.ndarray,
    fs: int = 1600,
    **kwargs
) -> np.ndarray:
    """
    Батчевый вариант predict_ozz_physics.

    Args:
        windows: Массив формы (B, T, 8) или (B, 8, T).
        fs: Частота дискретизации.
        **kwargs: Дополнительные параметры для predict_ozz_physics.

    Returns:
        Массив предсказаний (B,) dtype=int.
        -1 означает «Норма» (None в одиночном варианте).
    """
    B = windows.shape[0]
    results = np.full(B, -1, dtype=np.int32)
    for i in range(B):
        pred = predict_ozz_physics(windows[i], fs=fs, **kwargs)
        if pred is not None:
            results[i] = pred
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

    Проходит по DataFrame скользящим окном, вызывает predict_ozz_physics
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

    all_preds = []
    all_targets = []

    n_samples = len(dataframe)
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        window = signal_data[start:end, :]  # (T, 8)
        pred_class = predict_ozz_physics(window, fs=fs, **kwargs)

        # Формируем вектор предсказания (multi-label):
        # ДПОЗЗ/затухающее также активируют общий класс ОЗЗ.
        pred_vec = np.zeros(len(target_cols), dtype=np.int8)
        if pred_class is not None:
            pred_vec[0] = 1  # Target_OZZ
            if pred_class == 1 and len(target_cols) > 1:
                pred_vec[1] = 1
            elif pred_class == 2 and len(target_cols) > 2:
                pred_vec[2] = 1

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
