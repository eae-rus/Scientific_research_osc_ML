"""
Unit-тесты для модуля osc_tools.features.pdr_calculator

Тестируем вспомогательные функции расчета признаков PDR.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.features.pdr_calculator import (
    sliding_window_fft,
    calculate_symmetrical_components,
    calculate_impedance,
    calculate_power,
    calculate_linear_voltages,
    format_complex_to_mag_angle,
    cols_exist_and_not_all_nan
)

class TestPDRHelpers:
    """Тесты для вспомогательных функций pdr_calculator."""

    def test_sliding_window_fft_basic(self):
        """Тест БПФ в скользящем окне."""
        window_size = 10
        num_harmonics = 1
        # Синусоида: 10 точек на период
        t = np.arange(30)
        signal = np.sin(2 * np.pi * t / window_size)
        
        result = sliding_window_fft(signal, window_size, num_harmonics)
        
        assert result.shape == (30, 1)
        # Первые window_size точек должны быть NaN (согласно реализации, результат пишется в i + window_size)
        # Подождите, в реализации: center_index = i + fft_start_offset, где fft_start_offset = window_size
        assert np.all(np.isnan(result[:window_size]))
        # Проверяем наличие данных после окна
        assert np.any(~np.isnan(result[window_size:]))

    def test_calculate_symmetrical_components(self):
        """Тест расчета симметричных составляющих."""
        # Сбалансированная система: только прямая последовательность
        ua = 1.0 + 0j
        ub = np.exp(-1j * 2 * np.pi / 3)
        uc = np.exp(1j * 2 * np.pi / 3)
        
        v1, v2, v0 = calculate_symmetrical_components(ua, ub, uc)
        
        assert np.abs(v1) == pytest.approx(1.0)
        assert np.abs(v2) == pytest.approx(0.0)
        assert np.abs(v0) == pytest.approx(0.0)

    def test_calculate_impedance(self):
        """Тест расчета сопротивления."""
        v = np.array([100.0 + 0j, 100.0 + 0j])
        i = np.array([10.0 + 0j, 0.0]) # Один нулевой ток
        
        z = calculate_impedance(v, i, min_current_threshold=1e-6)
        
        assert z[0] == 10.0 + 0j
        # Для нулевого тока должно быть большое значение (1/threshold + 1/threshold*j)
        # Модуль будет sqrt(2) * 1/threshold
        assert np.abs(z[1]) == pytest.approx(np.sqrt(2) * 1e6, rel=1e-3)

    def test_calculate_power(self):
        """Тест расчета мощности."""
        v = np.array([100.0 + 0j])
        i = np.array([10.0 - 5j]) # Ток отстает (индуктивный)
        
        s, p, q = calculate_power(v, i)
        
        # S = V * conj(I) = 100 * (10 + 5j) = 1000 + 500j
        assert s[0] == 1000 + 500j
        assert p[0] == 1000.0
        assert q[0] == 500.0

    def test_calculate_linear_voltages(self):
        """Тест расчета линейных напряжений."""
        ua = np.array([100.0])
        ub = np.array([0.0])
        uc = np.array([-50.0])
        
        uab, ubc, uca = calculate_linear_voltages(ua, ub, uc)
        
        assert uab[0] == 100.0
        assert ubc[0] == 50.0
        assert uca[0] == -150.0

    def test_format_complex_to_mag_angle(self):
        """Тест форматирования комплексных чисел."""
        df = pd.DataFrame({
            'Z': [3 + 4j, np.nan]
        })
        
        result = format_complex_to_mag_angle(df, ['Z'])
        
        assert 'Z_mag' in result.columns
        assert 'Z_angle' in result.columns
        assert 'Z' not in result.columns
        assert result['Z_mag'][0] == 5.0
        assert result['Z_angle'][0] == pytest.approx(np.arctan2(4, 3))
        assert np.isnan(result['Z_mag'][1])

    def test_cols_exist_and_not_all_nan(self):
        """Тест проверки существования колонок."""
        df = pd.DataFrame({
            'A': [1, 2],
            'B': [np.nan, np.nan],
            'C': [1, np.nan]
        })
        
        assert cols_exist_and_not_all_nan(df, ['A', 'C']) is True
        assert cols_exist_and_not_all_nan(df, ['A', 'B']) is False # B - все NaN
        assert cols_exist_and_not_all_nan(df, ['A', 'D']) is False # D - нет
