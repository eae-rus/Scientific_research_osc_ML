"""
Тесты для osc_tools/analysis/detect_motor_starts.py и spef_finder.py

Модули для анализа пусков двигателей и поиска ОЗЗ событий.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict

# Добавляем PROJECT_ROOT в sys.path для правильных импортов
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.analysis.detect_motor_starts import MotorStartDetector


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_norm_coef_for_motor() -> pd.DataFrame:
    """DataFrame с коэффициентами нормализации для MotorStartDetector."""
    return pd.DataFrame({
        'name': ['motor_file_1', 'motor_file_2'],
        '1I_base': [100.0, 50.0],
        '2I_base': [100.0, 50.0],
    })


# ============================================================================
# ТЕСТЫ ИНИЦИАЛИЗАЦИИ MotorStartDetector
# ============================================================================

@pytest.mark.unit
class TestMotorStartDetectorInit:
    """Тесты инициализации MotorStartDetector."""
    
    def test_init_basic(self):
        """Базовая инициализация."""
        detector = MotorStartDetector(
            osc_folder_path='/data/osc',
            norm_coef_path='/data/norm.csv',
            output_path='/output/motors',
            log_path='/output/motor_errors.log'
        )
        
        assert detector.osc_folder_path == '/data/osc'
        assert detector.norm_coef_path == '/data/norm.csv'
        assert detector.output_path == '/output/motors'
        assert detector.log_path == '/output/motor_errors.log'
        assert detector.results == []
        assert detector.error_files == []
    
    def test_init_all_factors_enabled(self):
        """Все факторы включены по умолчанию."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        assert detector.enable_factor_1 is True
        assert detector.enable_factor_2 is True
        assert detector.enable_factor_3 is True
        assert detector.enable_factor_4 is True
    
    def test_init_disable_specific_factors(self):
        """Возможность отключить отдельные факторы."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log',
            enable_factor_1=False,
            enable_factor_3=False
        )
        
        assert detector.enable_factor_1 is False
        assert detector.enable_factor_2 is True
        assert detector.enable_factor_3 is False
        assert detector.enable_factor_4 is True
    
    def test_init_hyperparameters_positive(self):
        """Все гиперпараметры установлены положительными значениями."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        assert detector.NOISE_THRESHOLD_PU > 0
        assert detector.LOADED_BUS_THRESHOLD_PU > 0
        assert detector.PLATEAU_DURATION_MS > 0
        assert detector.PLATEAU_DROP_RATIO > 1.0
        assert detector.FINAL_DROP_RATIO > 1.0
        assert detector.INRUSH_JUMP_RATIO > 1.0


# ============================================================================
# ТЕСТЫ ВСПОМОГАТЕЛЬНЫХ МЕТОДОВ MotorStartDetector
# ============================================================================

@pytest.mark.unit
class TestMotorStartDetectorMethods:
    """Тесты методов MotorStartDetector."""
    
    @patch('pandas.read_csv')
    def test_load_norm_coefficients_success(self, mock_read_csv, sample_norm_coef_for_motor):
        """Успешная загрузка коэффициентов."""
        mock_read_csv.return_value = sample_norm_coef_for_motor
        
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        # Mock sys.exit чтобы он не прерывал тест
        with patch('sys.exit'):
            detector._load_norm_coefficients()
        
        assert detector.norm_coef_df is not None
        assert len(detector.norm_coef_df) == 2
    
    @patch('os.path.exists', return_value=False)
    @patch('sys.exit')
    def test_load_norm_coefficients_not_found(self, mock_exit, mock_exists):
        """Файл коэффициентов не найден."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/nonexistent/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        detector._load_norm_coefficients()
        
        # sys.exit должен был быть вызван
        mock_exit.assert_called()
    
    @patch('os.walk')
    def test_build_source_file_map(self, mock_walk):
        """Построение маппинга исходных файлов."""
        mock_walk.return_value = [
            ('/data', [], ['file1.cfg', 'file1.dat', 'file2.cfg', 'file2.dat'])
        ]
        
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        detector._build_source_file_map()
        
        assert detector._source_file_map is not None
        assert 'file1' in detector._source_file_map
        assert 'file2' in detector._source_file_map


# ============================================================================
# EDGE CASES MotorStartDetector
# ============================================================================

@pytest.mark.unit
class TestMotorStartDetectorEdgeCases:
    """Edge cases для MotorStartDetector."""
    
    def test_threshold_values_make_sense(self):
        """Пороги имеют логичные значения."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        # Шум должен быть меньше загруженной шины
        assert detector.NOISE_THRESHOLD_PU < detector.LOADED_BUS_THRESHOLD_PU
        
        # Бросок должен быть больше платё
        assert detector.INRUSH_JUMP_RATIO > 1.0
        assert detector.PLATEAU_DROP_RATIO > 1.0
    
    def test_plateau_duration_milliseconds(self):
        """Длительность плато в миллисекундах."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        # Должно быть в разумном диапазоне (100-500 ms)
        assert 100 <= detector.PLATEAU_DURATION_MS <= 500
    
    def test_empty_processed_files_set(self):
        """В начале множество обработанных файлов пусто."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output',
            log_path='/output/error.log'
        )
        
        assert len(detector.processed_files) == 0


# ============================================================================
# ТЕСТЫ SPEF FINDER FUNCTIONS
# ============================================================================

@pytest.mark.unit
class TestSpefFinderFunctions:
    """Тесты для функций из spef_finder.py."""
    
    def test_spef_finder_module_imports(self):
        """Проверяет что модуль успешно импортируется."""
        try:
            import osc_tools.analysis.spef_finder as spef_finder
            # Просто проверяем что модуль загрузился
            assert hasattr(spef_finder, '_get_signals_from_prepared_cfg')
        except ImportError as e:
            pytest.fail(f"Не удалось импортировать spef_finder: {e}")


@pytest.mark.unit
class TestGetSignalsFromPreparedCfg:
    """Тесты для функции _get_signals_from_prepared_cfg."""
    
    def test_invalid_cfg_file(self):
        """Попытка чтения несуществующего файла."""
        try:
            from osc_tools.analysis.spef_finder import _get_signals_from_prepared_cfg
            result = _get_signals_from_prepared_cfg('/nonexistent/file.cfg', 'utf-8')
            # Функция должна вернуть пустой список или None при ошибке
            assert result is None or isinstance(result, list)
        except ImportError:
            pytest.skip("spef_finder не импортируется")
    
    def test_cfg_parser_returns_list_or_none(self):
        """Проверяет что функция возвращает список или None."""
        try:
            from osc_tools.analysis.spef_finder import _get_signals_from_prepared_cfg
            # При любых условиях должна вернуться либо list, либо None
            result = _get_signals_from_prepared_cfg('/invalid/path.cfg', 'utf-8')
            assert result is None or isinstance(result, list)
        except ImportError:
            pytest.skip("spef_finder не импортируется")


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

@pytest.mark.unit
class TestMotorStartDetectorIntegration:
    """Интеграционные тесты для MotorStartDetector."""
    
    @patch('os.makedirs')
    def test_output_folder_creation(self, mock_makedirs):
        """Проверяет что создаётся выходная папка."""
        detector = MotorStartDetector(
            osc_folder_path='/data',
            norm_coef_path='/data/norm.csv',
            output_path='/output/motors',
            log_path='/output/error.log'
        )
        
        # makedirs должен был быть вызван
        mock_makedirs.assert_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
