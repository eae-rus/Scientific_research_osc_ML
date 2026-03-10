"""
Тесты для физической модели ОЗЗ/ДПОЗЗ и утилит разбиения.

Покрытие:
- predict_ozz_physics: синтетические сигналы для каждого класса
- add_ozz_target_columns: корректность формирования целевых колонок
- stratified_ozz_split: гарантия представительства классов
"""

import numpy as np
import polars as pl
import pytest

from osc_tools.analysis.ozz_physics import (
    predict_ozz_physics,
    _rms_fundamental,
    _envelope,
    u0_threshold_raw_to_normalized,
)
from osc_tools.data_management.ozz_split import (
    add_ozz_target_columns,
    classify_file_ozz,
    stratified_ozz_split,
    OZZ_TARGET_COLS
)
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment


# === Фикстуры для синтетических данных ===

def _make_window(fs=1600, duration_ms=200, ua=None, ub=None, uc=None):
    """Создаёт окно (T, 8) с заданными напряжениями фаз. Токи = 0."""
    T = int(fs * duration_ms / 1000)
    t = np.arange(T) / fs
    data = np.zeros((T, 8), dtype=np.float64)
    if ua is not None:
        data[:, 4] = ua(t) if callable(ua) else ua
    if ub is not None:
        data[:, 5] = ub(t) if callable(ub) else ub
    if uc is not None:
        data[:, 6] = uc(t) if callable(uc) else uc
    return data


class TestPredictOzzPhysics:
    """Тесты для predict_ozz_physics."""

    def test_normal_mode_low_u0(self):
        """Если 3U0 ~ 0, должен вернуть None (норма)."""
        # Симметричные напряжения → 3U0 = 0
        f = 50
        window = _make_window(
            ua=lambda t: 100 * np.sin(2 * np.pi * f * t),
            ub=lambda t: 100 * np.sin(2 * np.pi * f * t - 2 * np.pi / 3),
            uc=lambda t: 100 * np.sin(2 * np.pi * f * t + 2 * np.pi / 3),
        )
        result = predict_ozz_physics(window, fs=1600, u0_threshold=3.0)
        assert result is None, f"Ожидалось None (норма), получено {result}"

    def test_stable_ozz(self):
        """Устойчивое ОЗЗ: стабильная синусоида 3U0 выше порога."""
        f = 50
        # Все три фазы в фазе → 3U0 = 3 * A * sin(...)
        A = 10.0  # Амплитуда → RMS ≈ 10 * 3 / sqrt(2) ≈ 21В >> 3В
        window = _make_window(
            ua=lambda t: A * np.sin(2 * np.pi * f * t),
            ub=lambda t: A * np.sin(2 * np.pi * f * t),
            uc=lambda t: A * np.sin(2 * np.pi * f * t),
        )
        result = predict_ozz_physics(window, fs=1600)
        assert result == 0, f"Ожидалось 0 (устойчивое ОЗЗ), получено {result}"

    def test_decaying_ozz(self):
        """Затухающее ОЗЗ: экспоненциальный спад 3U0."""
        f = 50
        # Затухающий сигнал: большая амплитуда в начале → спадает до 0
        window = _make_window(
            ua=lambda t: 20.0 * np.exp(-15 * t) * np.sin(2 * np.pi * f * t),
            ub=lambda t: 20.0 * np.exp(-15 * t) * np.sin(2 * np.pi * f * t),
            uc=lambda t: 20.0 * np.exp(-15 * t) * np.sin(2 * np.pi * f * t),
        )
        result = predict_ozz_physics(window, fs=1600)
        assert result == 1, f"Ожидалось 1 (затухающее ОЗЗ), получено {result}"

    def test_dpozz(self):
        """ДПОЗЗ: ступенчатое нарастание 3U0 с «запертым зарядом».

        Физика ДПОЗЗ: каждый дуговой пробой добавляет DC-смещение (запертый заряд).
        Алгоритм требует:
        1) RMS(3U0) > u0_threshold
        2) >= min_dpozz_peaks (3) резких скачков dU0/dt > 5σ
        3) огибающая в конце окна НЕ спадает (trapped charge)
        """
        T = 320
        fs = 1600
        t = np.arange(T) / fs
        f = 50

        # Базовая синусоида 3U0
        u0 = 5.0 * np.sin(2 * np.pi * f * t)

        # Ступенчатые DC-смещения (имитация запертого заряда при дуговых пробоях)
        # Каждый «пробой» добавляет постоянное смещение → резкий скачок dU0/dt
        step_positions = [40, 80, 120, 160, 200]
        for pos in step_positions:
            u0[pos:] += 8.0  # +8 В за каждый пробой

        # На позиции 200+: DC = 40, синусоида ±5 → u0 ∈ [35, 45]
        # Огибающая в хвосте ≈ 45, env_max ≈ 45 → ratio ≈ 1.0 (trapped charge)
        # Резкие скачки: delta = 8В за 1 отсчёт → dU₀/dt = 8*1600 = 12800 В/с

        data = np.zeros((T, 8), dtype=np.float64)
        data[:, 4] = u0 / 3
        data[:, 5] = u0 / 3
        data[:, 6] = u0 / 3

        result = predict_ozz_physics(data, fs=fs)
        assert result == 2, f"Ожидалось 2 (ДПОЗЗ), получено {result}"

    def test_input_shape_transpose(self):
        """Проверка, что (8, T) автоматически транспонируется в (T, 8)."""
        f = 50
        A = 10.0
        window = _make_window(
            ua=lambda t: A * np.sin(2 * np.pi * f * t),
            ub=lambda t: A * np.sin(2 * np.pi * f * t),
            uc=lambda t: A * np.sin(2 * np.pi * f * t),
        )
        # Транспонируем в (8, T)
        result = predict_ozz_physics(window.T, fs=1600)
        assert result == 0, f"Ожидалось 0 после авто-транспонирования, получено {result}"

    def test_empty_or_short_input(self):
        """Короткий или пустой вход → None."""
        assert predict_ozz_physics(np.zeros((1, 8))) is None
        assert predict_ozz_physics(np.zeros((0, 8))) is None
        assert predict_ozz_physics(np.array([[1, 2, 3]])) is None  # < 8 каналов

    def test_raw_threshold_conversion_10v(self):
        """Проверка пересчёта уставки 10В в нормализованные единицы."""
        thr = u0_threshold_raw_to_normalized(10.0, 100.0)
        assert abs(thr - (10.0 / 300.0)) < 1e-9

    def test_operate_delay_samples_blocks_short_event(self):
        """Выдержка в отсчётах должна блокировать событие в коротком окне."""
        f = 50
        window = _make_window(
            ua=lambda t: 10.0 * np.sin(2 * np.pi * f * t),
            ub=lambda t: 10.0 * np.sin(2 * np.pi * f * t),
            uc=lambda t: 10.0 * np.sin(2 * np.pi * f * t),
        )

        # Без выдержки — ОЗЗ детектируется.
        assert predict_ozz_physics(window, fs=1600) == 0

        # С выдержкой 500 отсчётов (> длины окна 320) — не успевает отработать.
        assert predict_ozz_physics(window, fs=1600, operate_delay_samples=500) is None

    def test_operate_delay_periods(self):
        """Выдержка в периодах эквивалентна выдержке в отсчётах."""
        f = 50
        window = _make_window(
            ua=lambda t: 10.0 * np.sin(2 * np.pi * f * t),
            ub=lambda t: 10.0 * np.sin(2 * np.pi * f * t),
            uc=lambda t: 10.0 * np.sin(2 * np.pi * f * t),
        )

        # 1 период при fs=1600 и 50Гц -> 32 отсчёта, событие должно пройти.
        assert predict_ozz_physics(window, fs=1600, operate_delay_periods=1.0) == 0


class TestRmsAndEnvelope:
    """Тесты вспомогательных функций."""

    def test_rms_fundamental(self):
        """RMS синусоиды 50 Гц должно быть ~ A/sqrt(2)."""
        fs = 1600
        A = 10.0
        t = np.arange(320) / fs
        signal = A * np.sin(2 * np.pi * 50 * t)
        rms = _rms_fundamental(signal, fs=fs)
        expected = A / np.sqrt(2)
        assert abs(rms - expected) < 0.5, f"RMS = {rms:.2f}, ожидалось ~{expected:.2f}"

    def test_envelope_constant(self):
        """Огибающая синусоиды должна быть ~A."""
        fs = 1600
        A = 10.0
        t = np.arange(320) / fs
        signal = A * np.sin(2 * np.pi * 50 * t)
        env = _envelope(signal)
        # Огибающая должна быть примерно A (с точностью до ~10%)
        assert np.mean(env[32:]) > A * 0.85


class TestAddOzzTargetColumns:
    """Тесты для add_ozz_target_columns."""

    def _make_df(self, n=10, ml_2_1=0, ml_2_1_1=0, ml_2_1_2=0, ml_2_1_3=0):
        return pl.DataFrame({
            'ML_2_1': [ml_2_1] * n,
            'ML_2_1_1': [ml_2_1_1] * n,
            'ML_2_1_2': [ml_2_1_2] * n,
            'ML_2_1_3': [ml_2_1_3] * n,
        }).cast({col: pl.Int8 for col in ['ML_2_1', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3']})

    def test_no_ozz(self):
        df = add_ozz_target_columns(self._make_df())
        assert df['Target_OZZ'][0] == 0
        assert df['Target_OZZ_decay'][0] == 0
        assert df['Target_OZZ_dpozz'][0] == 0

    def test_stable_ozz(self):
        df = add_ozz_target_columns(self._make_df(ml_2_1=1))
        assert df['Target_OZZ'][0] == 1
        assert df['Target_OZZ_decay'][0] == 0
        assert df['Target_OZZ_dpozz'][0] == 0

    def test_decay_also_ozz(self):
        """Затухающее ОЗЗ — это тоже ОЗЗ (Target_OZZ = 1)."""
        df = add_ozz_target_columns(self._make_df(ml_2_1_2=1))
        assert df['Target_OZZ'][0] == 1   # ОЗЗ = 1 (подтип!)
        assert df['Target_OZZ_decay'][0] == 1

    def test_dpozz_also_ozz(self):
        """ДПОЗЗ — это тоже ОЗЗ (Target_OZZ = 1)."""
        df = add_ozz_target_columns(self._make_df(ml_2_1_3=1))
        assert df['Target_OZZ'][0] == 1   # ОЗЗ = 1
        assert df['Target_OZZ_dpozz'][0] == 1


class TestStratifiedOzzSplit:
    """Тесты для стратифицированного разбиения."""

    def _make_dataset(self):
        """Создаёт минимальный датасет с файлами разных классов."""
        rows = []
        # 5 файлов: 2 no_ozz, 1 stable, 1 decay, 1 dpozz
        for i, (fname, ml21, ml211, ml212, ml213) in enumerate([
            ('file_no_ozz_1.cfg', 0, 0, 0, 0),
            ('file_no_ozz_2.cfg', 0, 0, 0, 0),
            ('file_stable.cfg',    1, 1, 0, 0),
            ('file_decay.cfg',     0, 0, 1, 0),
            ('file_dpozz.cfg',     0, 0, 0, 1),
        ]):
            for j in range(100):
                rows.append({
                    'file_name': fname,
                    'ML_2_1': ml21,
                    'ML_2_1_1': ml211,
                    'ML_2_1_2': ml212,
                    'ML_2_1_3': ml213,
                })

        return pl.DataFrame(rows).cast({
            'ML_2_1': pl.Int8, 'ML_2_1_1': pl.Int8,
            'ML_2_1_2': pl.Int8, 'ML_2_1_3': pl.Int8,
        })

    def test_split_guarantees_class_coverage(self):
        df = self._make_dataset()
        train_files, test_files, stats = stratified_ozz_split(df, test_size=0.4, min_test_per_class=1)

        # Все файлы покрыты
        assert len(train_files) + len(test_files) == 5

        # Каждый класс ОЗЗ должен быть в test
        assert stats['test']['dpozz'] >= 1
        assert stats['test']['decay'] >= 1
        assert stats['test']['stable'] >= 1

    def test_classify_file_ozz(self):
        """Проверяем иерархию приоритетов."""
        dpozz_df = pl.DataFrame({
            'ML_2_1': [1, 0], 'ML_2_1_1': [0, 0],
            'ML_2_1_2': [0, 0], 'ML_2_1_3': [0, 1]
        }).cast({c: pl.Int8 for c in ['ML_2_1', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3']})
        assert classify_file_ozz(dpozz_df) == 'dpozz'

        no_ozz_df = pl.DataFrame({
            'ML_2_1': [0, 0], 'ML_2_1_1': [0, 0],
            'ML_2_1_2': [0, 0], 'ML_2_1_3': [0, 0]
        }).cast({c: pl.Int8 for c in ['ML_2_1', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3']})
        assert classify_file_ozz(no_ozz_df) == 'no_ozz'


class TestLabelsIntegration:
    """Тесты интеграции с системой меток."""

    def test_get_target_columns_ozz(self):
        cols = get_target_columns('ozz')
        assert cols == ['Target_OZZ', 'Target_OZZ_decay', 'Target_OZZ_dpozz']

    def test_prepare_labels_adds_columns(self):
        df = pl.DataFrame({
            'ML_2_1': [1, 0, 0],
            'ML_2_1_1': [0, 0, 0],
            'ML_2_1_2': [0, 1, 0],
            'ML_2_1_3': [0, 0, 1],
        }).cast({c: pl.Int8 for c in ['ML_2_1', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3']})

        result = prepare_labels_for_experiment(df, 'ozz')
        assert 'Target_OZZ' in result.columns
        assert 'Target_OZZ_decay' in result.columns
        assert 'Target_OZZ_dpozz' in result.columns

        # Проверяем значения (multi-label: ОЗЗ включает подтипы)
        assert result['Target_OZZ'][0] == 1       # ML_2_1=1 → ОЗЗ
        assert result['Target_OZZ_decay'][1] == 1  # ML_2_1_2=1
        assert result['Target_OZZ'][1] == 1        # затухающее → тоже ОЗЗ
        assert result['Target_OZZ_dpozz'][2] == 1  # ML_2_1_3=1
        assert result['Target_OZZ'][2] == 1        # ДПОЗЗ → тоже ОЗЗ
