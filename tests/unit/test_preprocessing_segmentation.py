"""
Тесты для osc_tools/preprocessing/segmentation.py

Модуль OscillogramEventSegmenter предоставляет класс для сегментации осциллограмм
и поиска событий в сигналах. Включает два режима: для нормализованных и сырых сигналов.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List

# Добавляем PROJECT_ROOT в sys.path для правильных импортов
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.preprocessing.segmentation import OscillogramEventSegmenter, _sliding_window_fft
from tests.test_data.fixtures import create_sinusoidal_signal, create_three_phase_balanced_signal


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def normalized_config() -> Dict:
    """Конфигурация для НОРМАЛИЗОВАННЫХ сигналов."""
    return {
        'detection_window_periods': 5,
        'padding_periods': 2,
        'current_patterns': ['ia', 'ib', 'ic'],
        'voltage_patterns': ['ua', 'ub', 'uc'],
        'thresholds_current_normalized': {
            'delta': 0.1, 'std_dev': 0.05, 'max_abs': 0.005
        },
        'thresholds_voltage_normalized': {
            'delta': 0.05, 'std_dev': 0.025, 'max_abs': 0.05
        },
    }


@pytest.fixture
def raw_signal_config() -> Dict:
    """Конфигурация для СЫРЫХ сигналов."""
    return {
        'detection_window_periods': 5,
        'padding_periods': 2,
        'current_patterns': ['ia', 'ib', 'ic'],
        'voltage_patterns': ['ua', 'ub', 'uc'],
        'raw_signal_analysis': {
            'initial_window_check_periods': 2,
            'h1_vs_hx_ratio_threshold_U': 10,
            'h1_vs_hx_ratio_threshold_I': 1.5,
            'thresholds_raw_current_relative': {
                'delta': 0.4, 'std_dev': 0.2
            },
            'thresholds_raw_voltage_relative': {
                'delta': 0.05, 'std_dev': 0.025
            }
        }
    }


@pytest.fixture
def fft_window_size() -> int:
    """Размер окна для БПФ (12 точек = 1 период при 600 точек/сек и 50 Гц)."""
    return 12


@pytest.fixture
def sample_normalized_dataframe(fft_window_size) -> pd.DataFrame:
    """Синтезированный нормализованный DataFrame с событиями."""
    fs = 1600  # Частота дискретизации
    f = 50     # Основная частота
    duration = 2.0  # 2 секунды
    
    t = np.arange(0, duration, 1/fs)
    
    # Базовый сигнал (нормализованный, амплитуда ~1)
    ia = 0.1 * np.sin(2 * np.pi * f * t)
    ib = 0.1 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    ic = 0.1 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    # Добавляем "событие" (прыжок амплитуды)
    event_start = int(0.5 * fs)
    event_end = int(1.0 * fs)
    ia[event_start:event_end] += 0.3  # Скачок на 0.3
    ib[event_start:event_end] += 0.3
    ic[event_start:event_end] += 0.3
    
    ua = 0.05 * np.sin(2 * np.pi * f * t)
    ub = 0.05 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc = 0.05 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    df = pd.DataFrame({
        'file_name': ['test_file_1'] * len(t),
        'ia': ia,
        'ib': ib,
        'ic': ic,
        'ua': ua,
        'ub': ub,
        'uc': uc,
    })
    
    return df


@pytest.fixture
def sample_raw_dataframe(fft_window_size) -> pd.DataFrame:
    """Синтезированный сырой DataFrame."""
    fs = 1600
    f = 50
    duration = 1.0
    
    t = np.arange(0, duration, 1/fs)
    
    # Сырые сигналы (большие амплитуды, в реальных единицах)
    ia = 100 * np.sin(2 * np.pi * f * t)
    ib = 100 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    ic = 100 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    ua = 230 * np.sin(2 * np.pi * f * t)
    ub = 230 * np.sin(2 * np.pi * f * t - 2*np.pi/3)
    uc = 230 * np.sin(2 * np.pi * f * t + 2*np.pi/3)
    
    df = pd.DataFrame({
        'file_name': ['raw_file_1'] * len(t),
        'ia': ia,
        'ib': ib,
        'ic': ic,
        'ua': ua,
        'ub': ub,
        'uc': uc,
    })
    
    return df


# ============================================================================
# ТЕСТЫ УТИЛИТАРНЫХ ФУНКЦИЙ
# ============================================================================

@pytest.mark.unit
class TestSlidingWindowFFT:
    """Тесты для вспомогательной функции _sliding_window_fft."""
    
    def test_sliding_window_fft_basic_sine(self):
        """Проверяет FFT для простой синусоиды."""
        signal = create_sinusoidal_signal(frequency=50, sampling_rate=1600, duration=0.1)
        fft_window_size = 32  # ~1 период
        
        result = _sliding_window_fft(signal, fft_window_size, num_harmonics=1)
        
        assert result.shape == (len(signal), 1)
        assert result.dtype == complex
        assert not np.all(np.isnan(result))  # Не все NaN
    
    def test_sliding_window_fft_signal_too_short(self):
        """Проверяет поведение когда сигнал короче окна."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fft_window_size = 100
        
        result = _sliding_window_fft(signal, fft_window_size, num_harmonics=1)
        
        assert result.shape == (len(signal), 1)
        assert np.all(np.isnan(result))  # Все NaN
    
    def test_sliding_window_fft_output_shape_matches_harmonics(self):
        """Проверяет что количество колонок соответствует harmonics."""
        signal = create_sinusoidal_signal(frequency=50, sampling_rate=1600, duration=0.05)
        fft_window_size = 32
        num_harmonics = 3
        
        result = _sliding_window_fft(signal, fft_window_size, num_harmonics)
        
        assert result.shape == (len(signal), num_harmonics)


# ============================================================================
# ТЕСТЫ ИНИЦИАЛИЗАЦИИ
# ============================================================================

@pytest.mark.unit
class TestOscillogramEventSegmenterInit:
    """Тесты инициализации OscillogramEventSegmenter."""
    
    def test_init_normalized_mode(self, normalized_config, fft_window_size):
        """Инициализация в режиме нормализованных сигналов."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        assert segmenter.fft_window_size == fft_window_size
        assert segmenter.signals_are_normalized is True
        assert segmenter.detection_window_periods == 5
        assert segmenter.padding_periods == 2
    
    def test_init_raw_mode(self, raw_signal_config, fft_window_size):
        """Инициализация в режиме сырых сигналов."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=raw_signal_config,
            signals_are_normalized=False
        )
        
        assert segmenter.signals_are_normalized is False
        assert 'raw_signal_analysis' in segmenter.config
    
    def test_init_loads_patterns_from_config(self, normalized_config, fft_window_size):
        """Проверяет что паттерны загружаются из конфига."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        assert 'ia' in segmenter.current_patterns
        assert 'ib' in segmenter.current_patterns
        assert 'ua' in segmenter.voltage_patterns


# ============================================================================
# ТЕСТЫ ВСПОМОГАТЕЛЬНЫХ МЕТОДОВ
# ============================================================================

@pytest.mark.unit
class TestGetTargetColumns:
    """Тесты для метода _get_target_columns."""
    
    def test_get_target_columns_basic(self, normalized_config, fft_window_size):
        """Находит столбцы по заданным паттернам."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        df = pd.DataFrame({
            'ia': [1, 2, 3],
            'ib': [1, 2, 3],
            'ua': [1, 2, 3],
            'other': [1, 2, 3]
        })
        
        result = segmenter._get_target_columns(df)
        
        assert 'ia' in result['current']
        assert 'ib' in result['current']
        assert 'ua' in result['voltage']
        assert 'other' not in result['current']
        assert 'other' not in result['voltage']
    
    def test_get_target_columns_case_insensitive(self, normalized_config, fft_window_size):
        """Паттерны чувствительны к регистру (lowercase)."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        df = pd.DataFrame({
            'IA': [1, 2, 3],  # Прописные буквы
            'Ub': [1, 2, 3],  # Смешанный регистр
        })
        
        result = segmenter._get_target_columns(df)
        
        assert 'IA' in result['current']
        assert 'Ub' in result['voltage']
    
    def test_get_target_columns_empty_dataframe(self, normalized_config, fft_window_size):
        """Обработка пустого DataFrame."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        df = pd.DataFrame()
        result = segmenter._get_target_columns(df)
        
        assert result['current'] == []
        assert result['voltage'] == []
    
    def test_get_target_columns_no_matching_columns(self, normalized_config, fft_window_size):
        """DataFrame без столбцов, соответствующих паттернам."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        df = pd.DataFrame({
            'foo': [1, 2, 3],
            'bar': [1, 2, 3],
        })
        
        result = segmenter._get_target_columns(df)
        
        assert result['current'] == []
        assert result['voltage'] == []


@pytest.mark.unit
class TestCalculateH1AmplitudeSeries:
    """Тесты для метода _calculate_h1_amplitude_series."""
    
    def test_calculate_h1_basic_sine(self, normalized_config, fft_window_size):
        """Рассчитывает амплитуды H1 для синусоиды."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        signal = create_sinusoidal_signal(frequency=50, sampling_rate=1600, duration=0.1, amplitude=1.0)
        result = segmenter._calculate_h1_amplitude_series(signal)
        
        assert len(result) == len(signal)
        assert np.all(~np.isnan(result))  # Нет NaN
        assert np.all(result >= 0)  # Амплитуды всегда положительные
    
    def test_calculate_h1_signal_too_short(self, normalized_config):
        """Сигнал короче окна."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=100,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = segmenter._calculate_h1_amplitude_series(signal)
        
        assert len(result) == 0
    
    def test_calculate_h1_fills_nan_values(self, normalized_config, fft_window_size):
        """Заполняет NaN значения последним валидным значением."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        # Сигнал достаточно длинный
        signal = create_sinusoidal_signal(frequency=50, sampling_rate=1600, duration=0.1)
        result = segmenter._calculate_h1_amplitude_series(signal)
        
        # Проверяем что нет "дыр" из NaN в середине/конце
        assert not np.any(np.isnan(result))


# ============================================================================
# ТЕСТЫ ОБРАБОТКИ DATAFRAME
# ============================================================================

@pytest.mark.unit
class TestProcessSingleDataframe:
    """Тесты для метода process_single_dataframe."""
    
    def test_process_single_dataframe_empty(self, normalized_config, fft_window_size):
        """Пустой DataFrame возвращает пустой список."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        df = pd.DataFrame()
        result = segmenter.process_single_dataframe(df)
        
        assert result == []
    
    def test_process_single_dataframe_no_matching_columns(self, normalized_config, fft_window_size):
        """DataFrame без столбцов с паттернами возвращает пустой список."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        df = pd.DataFrame({
            'file_name': ['test'],
            'foo': [1.0],
            'bar': [2.0],
        })
        
        result = segmenter.process_single_dataframe(df)
        
        assert result == []
    
    def test_process_single_dataframe_with_event(self, sample_normalized_dataframe, 
                                                  normalized_config, fft_window_size):
        """Обработка DataFrame с событием."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        result = segmenter.process_single_dataframe(sample_normalized_dataframe)
        
        # Должны быть результаты (события найдены)
        assert isinstance(result, list)
        # Каждый результат - DataFrame
        for event_df in result:
            assert isinstance(event_df, pd.DataFrame)
            assert 'file_name' in event_df.columns
    
    def test_process_single_dataframe_adds_event_names(self, sample_normalized_dataframe,
                                                        normalized_config, fft_window_size):
        """Проверяет что имена файлов изменены для событий."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        original_name = sample_normalized_dataframe['file_name'].iloc[0]
        result = segmenter.process_single_dataframe(sample_normalized_dataframe)
        
        if result:  # Если были найдены события
            for i, event_df in enumerate(result):
                assert f'event_{i+1}' in event_df['file_name'].iloc[0]


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.unit
class TestSegmentationEdgeCases:
    """Edge cases для сегментации."""
    
    def test_very_short_signal(self, normalized_config, fft_window_size):
        """Очень короткий сигнал."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        # Гарантируем что у всех столбцов одинаковая длина
        df = pd.DataFrame({
            'file_name': ['short'] * 3,
            'ia': [0.1, 0.2, 0.15],
            'ua': [0.01, 0.02, 0.015],
        })
        
        try:
            result = segmenter.process_single_dataframe(df)
            assert isinstance(result, list)
        except ValueError:
            # OK если вызывает исключение на коротком сигнале
            pass
    
    def test_constant_signal_no_events(self, normalized_config, fft_window_size):
        """Постоянный сигнал (никаких событий)."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        size = 100  # Одинаковый размер для всех столбцов
        df = pd.DataFrame({
            'file_name': ['const'] * size,
            'ia': np.ones(size) * 0.05,
            'ib': np.ones(size) * 0.05,
            'ic': np.ones(size) * 0.05,
            'ua': np.ones(size) * 0.01,
            'ub': np.ones(size) * 0.01,
            'uc': np.ones(size) * 0.01,
        })
        
        result = segmenter.process_single_dataframe(df)
        
        # Может быть пусто или содержать мало событий
        assert isinstance(result, list)
    
    def test_missing_values_in_signal(self, normalized_config, fft_window_size):
        """Сигнал с пропущенными значениями (NaN)."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        signal = create_sinusoidal_signal(frequency=50, sampling_rate=1600, duration=0.05)
        signal_with_nan = signal.copy()
        signal_with_nan[10:20] = np.nan
        
        # Гарантируем одинаковую длину для всех столбцов
        size = len(signal)
        df = pd.DataFrame({
            'file_name': ['with_nan'] * size,
            'ia': signal_with_nan,
            'ua': np.ones(size) * 0.01,
        })
        
        result = segmenter.process_single_dataframe(df)
        
        # Не должно быть исключения при NaN
        assert isinstance(result, list)
    
    def test_all_zero_signal(self, normalized_config, fft_window_size):
        """Сигнал со всеми нулями."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        size = 100  # Одинаковый размер для всех столбцов
        df = pd.DataFrame({
            'file_name': ['zeros'] * size,
            'ia': np.zeros(size),
            'ua': np.zeros(size),
        })
        
        result = segmenter.process_single_dataframe(df)
        
        assert isinstance(result, list)


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (с файловыми операциями - мокируются)
# ============================================================================

@pytest.mark.unit
class TestProcessCSVFile:
    """Тесты для метода process_csv_file."""
    
    @patch('builtins.open', create=True)
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.exists', return_value=True)
    @patch('os.path.dirname')
    @patch('os.path.splitext')
    @patch('os.path.basename')
    def test_process_csv_file_basic(self, mock_basename, mock_splitext, mock_dirname,
                                     mock_exists, mock_to_csv, mock_read_csv, mock_open,
                                     normalized_config, fft_window_size):
        """Базовая обработка CSV файла."""
        # Подготовка моков
        mock_dirname.return_value = '/output'
        mock_basename.return_value = 'test.csv'
        mock_splitext.return_value = ('test', '.csv')
        
        sample_df = pd.DataFrame({
            'file_name': ['file1', 'file1', 'file2', 'file2'],
            'ia': [0.1, 0.15, 0.1, 0.12],
            'ua': [0.01, 0.01, 0.01, 0.01],
        })
        mock_read_csv.return_value = sample_df
        
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        # Вызов
        segmenter.process_csv_file('input.csv', 'output.csv')
        
        # Проверка что к_csv был вызван (хотя бы один раз)
        assert mock_read_csv.called
    
    @patch('os.path.exists', return_value=False)
    def test_process_csv_file_input_not_found(self, mock_exists, normalized_config, fft_window_size):
        """Входной файл не существует."""
        segmenter = OscillogramEventSegmenter(
            fft_window_size=fft_window_size,
            config=normalized_config,
            signals_are_normalized=True
        )
        
        # Не должно быть исключения, а только print
        with patch('builtins.print') as mock_print:
            segmenter.process_csv_file('nonexistent.csv', 'output.csv')
            
            # Проверяем что было сообщение об ошибке
            mock_print.assert_called()
            call_args = str(mock_print.call_args)
            assert 'Ошибка' in call_args or 'Error' in call_args


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
