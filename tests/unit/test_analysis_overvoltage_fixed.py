"""
Тесты для osc_tools/analysis/overvoltage.py

Модуль OvervoltageAnalyzer анализирует осциллограммы на предмет максимальных перенапряжений при ОЗЗ.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Tuple

# Добавляем PROJECT_ROOT в sys.path для правильных импортов
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.analysis.overvoltage import OvervoltageAnalyzer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_norm_coef_df() -> pd.DataFrame:
    """DataFrame с коэффициентами нормализации."""
    return pd.DataFrame({
        'name': ['file_1', 'file_2', 'file_3'],
        '1Ub_base': [400.0, 10000.0, 400.0],
        '1Uc_base': [400.0, 10000.0, 400.0],
        '2Ub_base': [400.0, np.nan, np.nan],
        '2Uc_base': [400.0, np.nan, np.nan],
    })


@pytest.fixture
def sample_normalized_dataframe() -> pd.DataFrame:
    """Синтезированный нормализованный DataFrame с фазными напряжениями."""
    fs = 1600
    f = 50
    duration = 0.5
    
    t = np.arange(0, duration, 1/fs)
    
    # Фазные напряжения (нормализованные)
    ua_bb = 0.5 * np.sin(2 * np.pi * f * t)
    ub_bb = 0.5 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc_bb = 0.5 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    un_bb = 0.01 * np.sin(2 * np.pi * f * t)
    ua_cl = 0.5 * np.sin(2 * np.pi * f * t)
    ub_cl = 0.5 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc_cl = 0.5 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    return pd.DataFrame({
        'UA BB': ua_bb,
        'UB BB': ub_bb,
        'UC BB': uc_bb,
        'UN BB': un_bb,
        'UA CL': ua_cl,
        'UB CL': ub_cl,
        'UC CL': uc_cl,
    })


@pytest.fixture
def sample_spef_dataframe() -> pd.DataFrame:
    """DataFrame с ОЗЗ событием (повышенные нулевое напряжение)."""
    fs = 1600
    f = 50
    duration = 0.5
    
    t = np.arange(0, duration, 1/fs)
    
    # Нормальные фазные напряжения
    ua_bb = 0.5 * np.sin(2 * np.pi * f * t)
    ub_bb = 0.5 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc_bb = 0.5 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    # Повышенное нулевое напряжение (ОЗЗ событие)
    un_bb = np.zeros_like(t)
    un_bb[int(0.1*fs):int(0.4*fs)] = 0.15
    
    # КЛ сигналы
    ua_cl = 0.5 * np.sin(2 * np.pi * f * t)
    ub_cl = 0.5 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc_cl = 0.5 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    return pd.DataFrame({
        'UA BB': ua_bb,
        'UB BB': ub_bb,
        'UC BB': uc_bb,
        'UN BB': un_bb,
        'UA CL': ua_cl,
        'UB CL': ub_cl,
        'UC CL': uc_cl,
    })


# ============================================================================
# ТЕСТЫ ИНИЦИАЛИЗАЦИИ
# ============================================================================

@pytest.mark.unit
class TestOvervoltageAnalyzerInit:
    """Тесты инициализации OvervoltageAnalyzer."""
    
    def test_init_basic(self):
        """Базовая инициализация."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data/osc',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert analyzer.osc_folder_path == '/data/osc'
        assert analyzer.norm_coef_path == '/data/norm.csv'
        assert analyzer.output_path == '/output/result.csv'
        assert analyzer.log_path == '/output/error.log'
        assert analyzer.results == []
        assert analyzer.error_files == []
    
    def test_init_valid_nominal_voltages(self):
        """Проверяет валидные номиналы напряжений."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert 400.0 in analyzer.VALID_NOMINAL_VOLTAGES
        assert 10000.0 in analyzer.VALID_NOMINAL_VOLTAGES
        assert 0.221 in analyzer.VALID_NOMINAL_VOLTAGES
        assert 35000.0 in analyzer.VALID_NOMINAL_VOLTAGES
    
    def test_init_thresholds_positive(self):
        """Проверяет что пороги положительны."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert analyzer.SPEF_THRESHOLD_U0 > 0
        assert analyzer.SPEF_THRESHOLD_Un > 0


# ============================================================================
# ТЕСТЫ МЕТОДОВ (с минимальным функционалом)
# ============================================================================

@pytest.mark.unit
class TestFindSpefZones:
    """Тесты для поиска ОЗЗ зон."""
    
    def test_find_spef_zones_empty_dataframe(self):
        """Пустой DataFrame - нет зон."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        empty_df = pd.DataFrame()
        zones = analyzer._find_spef_zones(empty_df, 'BB', samples_per_period=32)
        
        assert zones == []
    
    def test_find_spef_zones_missing_columns(self):
        """DataFrame без нужных столбцов."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        df = pd.DataFrame({'other': [1, 2, 3]})
        zones = analyzer._find_spef_zones(df, 'BB', samples_per_period=32)
        
        assert zones == []
    
    def test_find_spef_zones_returns_list(self, sample_normalized_dataframe):
        """Результат всегда список."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        zones = analyzer._find_spef_zones(sample_normalized_dataframe, 'BB', samples_per_period=32)
        
        assert isinstance(zones, list)


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.unit
class TestOvervoltageAnalyzerEdgeCases:
    """Edge cases для OvervoltageAnalyzer."""
    
    def test_valid_nominal_voltages_list(self):
        """Валидные номиналы - это конкретный список."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Проверяем что это не пустой список
        assert len(analyzer.VALID_NOMINAL_VOLTAGES) > 0
        # Проверяем что это список чисел
        assert all(isinstance(v, (int, float)) for v in analyzer.VALID_NOMINAL_VOLTAGES)
    
    def test_similar_amplitudes_filter_exists(self):
        """Порог фильтра похожести амплитуд существует."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Порог должен быть между 0 и 1
        assert 0 < analyzer.SIMILAR_AMPLITUDES_MAX_RELATIVE_DIFFERENCE < 1
    
    def test_max_bus_count_reasonable(self):
        """Максимальное количество шин разумно."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Максимум шин должен быть от 1 до 20
        assert 1 <= analyzer.MAX_BUS_COUNT <= 20
    
    def test_min_spef_duration_periods(self):
        """Минимальная длительность ОЗЗ события."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Минимум должен быть хотя бы 1 период
        assert analyzer.SPEF_MIN_DURATION_PERIODS >= 1
    
    def test_zero_nominal_not_valid(self):
        """Нулевой номинал не валидный."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert 0.0 not in analyzer.VALID_NOMINAL_VOLTAGES


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (с полной инициализацией)
# ============================================================================

@pytest.mark.unit
class TestRunAnalysisIntegration:
    """Интеграционные тесты для полного анализа."""
    
    def test_analyzer_can_be_created(self):
        """Анализатор может быть создан без ошибок."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert analyzer is not None
        assert hasattr(analyzer, 'results')
        assert hasattr(analyzer, 'error_files')
    
    def test_results_initialized_empty(self):
        """Результаты инициализируются пустыми."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert analyzer.results == []
        assert analyzer.error_files == []
    
    def test_results_can_be_appended(self):
        """К результатам можно добавлять данные."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        test_result = {
            'filename': 'test.dat',
            'overvoltage': 1.5,
            'bus': 1,
            'group': 'BB'
        }
        
        analyzer.results.append(test_result)
        
        assert len(analyzer.results) == 1
        assert analyzer.results[0]['filename'] == 'test.dat'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
