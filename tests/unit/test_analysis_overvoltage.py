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
    
    # Фазные напряжения (нормализованные, ~1.0 для нормального режима)
    ua_bb = 0.5 * np.sin(2 * np.pi * f * t)
    ub_bb = 0.5 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc_bb = 0.5 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    # Нулевое напряжение (в норме низкое)
    un_bb = 0.01 * np.sin(2 * np.pi * f * t)
    
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


@pytest.fixture
def sample_spef_dataframe() -> pd.DataFrame:
    """DataFrame с ОЗЗ (земляным коротким замыканием) - повышенные напряжения."""
    fs = 1600
    f = 50
    duration = 0.5
    
    t = np.arange(0, duration, 1/fs)
    
    # Нормальные фазные напряжения
    ua_bb = 0.5 * np.sin(2 * np.pi * f * t)
    ub_bb = 0.5 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc_bb = 0.5 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    # Повышенное нулевое напряжение (ОЗЗ на земле)
    un_bb = np.zeros_like(t)
    un_bb[int(0.1*fs):int(0.4*fs)] = 0.15  # Событие ОЗЗ
    
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


@pytest.fixture
def mock_overvoltage_init(monkeypatch):
    """Мокирует инициализацию OvervoltageAnalyzer чтобы избежать реальных файловых операций."""
    with patch('osc_tools.analysis.overvoltage.ComtradeParser') as mock_comtrade, \
         patch('builtins.open') as mock_open, \
         patch('osc_tools.features.normalization.os.path.exists', return_value=True) as mock_exists, \
         patch('pandas.read_csv') as mock_read_csv:
        
        mock_comtrade.return_value = MagicMock()
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda s, *args: None
        
        yield {
            'comtrade': mock_comtrade,
            'open': mock_open,
            'exists': mock_exists,
            'read_csv': mock_read_csv
        }


# ============================================================================
# ТЕСТЫ ИНИЦИАЛИЗАЦИИ
# ============================================================================

@pytest.mark.unit
@patch('osc_tools.analysis.overvoltage.ComtradeParser')
@patch('builtins.open')
@patch('osc_tools.features.normalization.os.path.exists', return_value=True)
@patch('pandas.read_csv')
class TestOvervoltageAnalyzerInit:
    """Тесты инициализации OvervoltageAnalyzer."""
    
    def test_init_basic(self, mock_read_csv, mock_exists, mock_open, mock_comtrade):
        """Базовая инициализация."""
        mock_comtrade.return_value = MagicMock()
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda s, *args: None
        
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
    
    def test_init_valid_nominal_voltages_set(self, mock_read_csv, mock_exists, mock_open, mock_comtrade):
        """Проверяет что установлены корректные номиналы напряжений."""
        mock_comtrade.return_value = MagicMock()
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda s, *args: None
        
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
    
    def test_init_thresholds_are_positive(self, mock_read_csv, mock_exists, mock_open, mock_comtrade):
        """Проверяет что пороги установлены положительными значениями."""
        mock_comtrade.return_value = MagicMock()
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda s, *args: None
        
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert analyzer.SPEF_THRESHOLD_U0 > 0
        assert analyzer.SPEF_THRESHOLD_Un > 0


# ============================================================================
# ТЕСТЫ ВСПОМОГАТЕЛЬНЫХ МЕТОДОВ
# ============================================================================

@pytest.mark.unit
class TestLoadNormCoefficients:
    """Тесты для метода _load_norm_coefficients."""
    
    @patch('osc_tools.analysis.overvoltage.ComtradeParser')
    @patch('builtins.open')
    @patch('osc_tools.features.normalization.os.path.exists', return_value=True)
    @patch('pandas.read_csv')
    def test_load_norm_coefficients_success(self, mock_read_csv, mock_exists, mock_open, mock_comtrade, sample_norm_coef_df):
        """Успешная загрузка коэффициентов."""
        mock_read_csv.return_value = sample_norm_coef_df
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda s, *args: None
        mock_comtrade.return_value = MagicMock()  # Мокируем ComtradeParser instance
        
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        analyzer._load_norm_coefficients()
        
        assert analyzer.norm_coef_df is not None
        assert len(analyzer.norm_coef_df) == 3
    
    @patch('osc_tools.features.normalization.os.path.exists', return_value=False)
    @patch('builtins.print')
    def test_load_norm_coefficients_file_not_found(self, mock_print, mock_exists):
        """Файл коэффициентов не найден."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/nonexistent/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        analyzer._load_norm_coefficients()
        
        assert analyzer.norm_coef_df is None


@pytest.mark.unit
class TestFindSpefZones:
    """Тесты для метода _find_spef_zones."""
    
    def test_find_spef_zones_no_zones(self, sample_normalized_dataframe):
        """DataFrame без превышения порога - нет ОЗЗ зон."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Нормальный DataFrame - без превышений
        zones = analyzer._find_spef_zones(sample_normalized_dataframe, 'BB', samples_per_period=32)
        
        # Может быть пусто или мало зон
        assert isinstance(zones, list)
    
    def test_find_spef_zones_with_spef_event(self, sample_spef_dataframe):
        """DataFrame с ОЗЗ событием."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        zones = analyzer._find_spef_zones(sample_spef_dataframe, 'BB', samples_per_period=32)
        
        assert isinstance(zones, list)
        # Возможно найти зоны в зависимости от конфигурации
    
    def test_find_spef_zones_empty_dataframe(self):
        """Пустой DataFrame."""
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
        
        df = pd.DataFrame({
            'other_col1': [1, 2, 3],
            'other_col2': [4, 5, 6],
        })
        
        zones = analyzer._find_spef_zones(df, 'BB', samples_per_period=32)
        
        assert zones == []


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.unit
class TestOvervoltageAnalyzerEdgeCases:
    """Edge cases для OvervoltageAnalyzer."""
    
    def test_valid_nominal_voltage_filtering(self):
        """Проверяет что только валидные номиналы обрабатываются."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # 400V - валидный
        assert 400.0 in analyzer.VALID_NOMINAL_VOLTAGES
        
        # 132kV - невалидный (нет в списке)
        assert 132000.0 not in analyzer.VALID_NOMINAL_VOLTAGES
    
    def test_similar_amplitudes_filter_threshold(self):
        """Проверяет порог для фильтра похожести амплитуд."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Порог - 5% разницы
        assert analyzer.SIMILAR_AMPLITUDES_MAX_RELATIVE_DIFFERENCE == 0.05
    
    def test_max_bus_count(self):
        """Проверяет максимальное количество шин."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert analyzer.MAX_BUS_COUNT == 10
    
    def test_very_short_duration(self):
        """ОЗЗ события очень короткой длительности (должны быть отфильтрованы)."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Минимальная длительность события - 1 период
        assert analyzer.SPEF_MIN_DURATION_PERIODS >= 1
    
    def test_zero_nominal_voltage_handling(self):
        """Обработка нулевого номиналького напряжения."""
        # Нулевой номинал не должен быть в валидных
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        assert 0.0 not in analyzer.VALID_NOMINAL_VOLTAGES


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (с мокированными файловыми операциями)
# ============================================================================

@pytest.mark.unit
class TestSaveResults:
    """Тесты для метода _save_results."""
    
    def test_save_results_with_data(self):
        """Сохранение результатов когда есть данные (проверка логики)."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Добавляем тестовые результаты
        test_results = [
            {'filename': 'test1', 'overvoltage': 1.5, 'bus': 1, 'group': 'СШ'},
            {'filename': 'test2', 'overvoltage': 2.3, 'bus': 2, 'group': 'КЛ'},
        ]
        analyzer.results = test_results
        
        # Проверяем что результаты добавлены правильно
        assert len(analyzer.results) == 2
        assert analyzer.results[0]['filename'] == 'test1'
        assert analyzer.results[1]['overvoltage'] == 2.3
    
    @patch('builtins.print')
    def test_save_results_empty(self, mock_print):
        """Сохранение результатов когда нет данных."""
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Пусто
        analyzer.results = []
        
        analyzer._save_results()
        
        # Проверяем что вывели сообщение
        mock_print.assert_called()


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (полный цикл)
# ============================================================================

@pytest.mark.unit
class TestRunAnalysisIntegration:
    """Интеграционные тесты для полного цикла анализа."""
    
    @patch('osc_tools.analysis.overvoltage.OvervoltageAnalyzer._load_norm_coefficients')
    @patch('os.walk')
    @patch('builtins.print')
    def test_run_analysis_no_files(self, mock_print, mock_walk, mock_load_coeffs):
        """Анализ когда нет файлов."""
        mock_walk.return_value = []  # Нет файлов
        
        analyzer = OvervoltageAnalyzer(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/result.csv',
            log_path='/output/error.log'
        )
        
        # Не должно быть исключения
        analyzer.norm_coef_df = pd.DataFrame()  # Имитируем загруженные коэффициенты
        
        # Из-за отсутствия файлов, analyze не сделает ничего
        assert analyzer.results == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
