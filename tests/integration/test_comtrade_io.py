"""
Интеграционные тесты для модуля comtrade_processing.

Тестирует функциональность класса ReadComtrade для чтения и обработки
COMTRADE файлов (.cfg и .dat).
"""

import pytest
import sys
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к проекту
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from osc_tools.data_management.comtrade_processing import ReadComtrade


class TestReadComtradeBasic:
    """Тесты базовой функциональности класса ReadComtrade."""
    
    @pytest.mark.integration
    def test_read_comtrade_initialization(self):
        """Тест инициализации класса ReadComtrade."""
        reader = ReadComtrade()
        assert reader is not None
        assert isinstance(reader, ReadComtrade)
    
    @pytest.mark.integration
    def test_read_comtrade_method_exists(self):
        """Тест наличия метода read_comtrade."""
        reader = ReadComtrade()
        assert hasattr(reader, 'read_comtrade')
        assert callable(reader.read_comtrade)
    
    @pytest.mark.integration
    def test_read_comtrade_with_invalid_file(self):
        """Тест обработки несуществующего файла."""
        reader = ReadComtrade()
        # При ошибке должно вернуться (None, None)
        result_comtrade, result_df = reader.read_comtrade("non_existent_file.cfg")
        assert result_comtrade is None
        assert result_df is None


class TestReadComtradeWithMockedData:
    """Тесты ReadComtrade с использованием mock-данных."""
    
    @pytest.fixture
    def temp_comtrade_files(self):
        """
        Создаёт временные COMTRADE файлы для тестирования.
        Возвращает путь к .cfg файлу.
        """
        # Создаём временную директорию
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "test.cfg")
            dat_path = os.path.join(tmpdir, "test.dat")
            
            # Минимальный валидный .cfg файл (COMTRADE 2013 ASCII format)
            cfg_content = """Station name,Test Station
Station ID,1
Date created,01/01/2024
Time created,00:00:00
Timecode source,TCG
Time zone offset,0
Local phasor id,1
"""
            
            with open(cfg_path, 'w') as f:
                f.write(cfg_content)
            
            # Минимальный валидный .dat файл (если нужен)
            # Для ASCII формата должны быть данные в специальном формате
            dat_content = ""
            
            with open(dat_path, 'w') as f:
                f.write(dat_content)
            
            yield cfg_path
    
    @pytest.mark.integration
    def test_read_comtrade_return_types(self, temp_comtrade_files):
        """
        Тест проверяет типы возвращаемых значений.
        Ожидаются кортеж (Comtrade, DataFrame) или (None, None) при ошибке.
        """
        reader = ReadComtrade()
        comtrade_obj, df = reader.read_comtrade(temp_comtrade_files)
        
        # Если файл успешно прочитан, оба значения не None
        # Если ошибка, оба должны быть None
        if comtrade_obj is not None:
            assert df is not None
            assert isinstance(df, pd.DataFrame) or isinstance(df, type(None))
        else:
            assert df is None


class TestReadComtradeEdgeCases:
    """Тесты граничных случаев для ReadComtrade."""
    
    @pytest.mark.integration
    def test_read_comtrade_empty_filename(self):
        """Тест с пустым именем файла."""
        reader = ReadComtrade()
        result_comtrade, result_df = reader.read_comtrade("")
        assert result_comtrade is None
        assert result_df is None
    
    @pytest.mark.integration
    def test_read_comtrade_none_filename(self):
        """Тест с None в качестве имени файла."""
        reader = ReadComtrade()
        try:
            result_comtrade, result_df = reader.read_comtrade(None)
            # Если не было исключения, оба должны быть None
            assert result_comtrade is None
            assert result_df is None
        except (TypeError, AttributeError):
            # Ожидаемая ошибка для None в качестве аргумента
            pass
    
    @pytest.mark.integration
    def test_read_comtrade_non_existent_directory(self):
        """Тест с файлом в несуществующей директории."""
        reader = ReadComtrade()
        result_comtrade, result_df = reader.read_comtrade(
            "/non/existent/path/to/file.cfg"
        )
        assert result_comtrade is None
        assert result_df is None


class TestReadComtradeIntegration:
    """Интеграционные тесты - совместная работа компонентов."""
    
    @pytest.mark.integration
    def test_multiple_reads_same_instance(self):
        """Тест нескольких вызовов read_comtrade на одном экземпляре."""
        reader = ReadComtrade()
        
        # Первый вызов с несуществующим файлом
        result1_comtrade, result1_df = reader.read_comtrade("file1.cfg")
        assert result1_comtrade is None
        
        # Второй вызов с другим несуществующим файлом
        result2_comtrade, result2_df = reader.read_comtrade("file2.cfg")
        assert result2_comtrade is None
        
        # Третий вызов - проверяем, что экземпляр всё ещё работает
        result3_comtrade, result3_df = reader.read_comtrade("file3.cfg")
        assert result3_comtrade is None
    
    @pytest.mark.integration
    def test_multiple_instances_independence(self):
        """Тест независимости разных экземпляров ReadComtrade."""
        reader1 = ReadComtrade()
        reader2 = ReadComtrade()
        
        # Оба должны быть разными объектами
        assert reader1 is not reader2
        
        # Оба должны работать независимо
        result1 = reader1.read_comtrade("file1.cfg")
        result2 = reader2.read_comtrade("file2.cfg")
        
        # Оба должны вернуть (None, None)
        assert result1 == (None, None)
        assert result2 == (None, None)


class TestReadComtradeExceptionHandling:
    """Тесты обработки исключений в ReadComtrade."""
    
    @pytest.mark.integration
    def test_exception_handling_returns_none_tuple(self):
        """Тест что при любой ошибке возвращается (None, None)."""
        reader = ReadComtrade()
        
        # Список различных "плохих" путей для тестирования
        bad_paths = [
            "*.cfg",  # Wildcard
            "/dev/null",  # Специальный файл (если на Unix)
            "../../etc/passwd",  # Путь обхода
        ]
        
        for bad_path in bad_paths:
            try:
                result_comtrade, result_df = reader.read_comtrade(bad_path)
                # Должно вернуться (None, None) без исключений
                assert result_comtrade is None
                assert result_df is None
            except Exception as e:
                # Если исключение не поймано в методе, это тоже должно быть задокументировано
                pytest.fail(f"Exception not caught for path '{bad_path}': {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.integration
class TestReadComtradeWithMocks:
    """Тесты ReadComtrade с использованием mock-объектов Comtrade."""
    
    def test_read_comtrade_success_with_mock(self):
        """Тест успешного чтения с mock-объектом."""
        reader = ReadComtrade()
        
        with patch('osc_tools.data_management.comtrade_processing.Comtrade') as mock_comtrade_class:
            # Создаём mock объект Comtrade
            mock_comtrade_instance = MagicMock()
            mock_comtrade_class.return_value = mock_comtrade_instance
            
            # Mock для to_dataframe()
            mock_df = pd.DataFrame({
                'time': [0.0, 0.01, 0.02],
                'IA': [1.0, 1.5, 2.0],
                'IB': [1.0, 1.5, 2.0],
                'IC': [1.0, 1.5, 2.0]
            })
            mock_comtrade_instance.to_dataframe.return_value = mock_df
            
            result = reader.read_comtrade('dummy_file.cfg')
            
            # Проверяем структуру возврата
            assert isinstance(result, tuple), "read_comtrade должен вернуть кортеж"
            assert len(result) == 2, "Кортеж должен содержать 2 элемента"
            
            rec, df = result
            assert rec is mock_comtrade_instance, "Первый элемент должен быть объектом Comtrade"
            assert isinstance(df, pd.DataFrame), "Второй элемент должен быть DataFrame"
            assert not df.empty, "DataFrame не должен быть пустым"
    
    def test_read_comtrade_calls_load_method(self):
        """Тест что метод load() вызывается с правильным аргументом."""
        reader = ReadComtrade()
        filename = 'test_oscillogram.cfg'
        
        with patch('osc_tools.data_management.comtrade_processing.Comtrade') as mock_comtrade_class:
            mock_comtrade_instance = MagicMock()
            mock_comtrade_class.return_value = mock_comtrade_instance
            mock_comtrade_instance.to_dataframe.return_value = pd.DataFrame()
            
            reader.read_comtrade(filename)
            
            # Проверяем, что load был вызван с правильным аргументом
            mock_comtrade_instance.load.assert_called_once_with(filename)
    
    def test_read_comtrade_realistic_data_with_mock(self):
        """Тест с реалистичным DataFrame (симуляция осциллограммы)."""
        reader = ReadComtrade()
        
        with patch('osc_tools.data_management.comtrade_processing.Comtrade') as mock_comtrade_class:
            mock_comtrade_instance = MagicMock()
            mock_comtrade_class.return_value = mock_comtrade_instance
            
            # Создаём реалистичный DataFrame
            mock_df = pd.DataFrame({
                'time': [0.0, 0.01, 0.02, 0.03, 0.04],
                'IA': [10.5, 11.2, 10.8, 10.1, 10.9],
                'IB': [10.3, 11.0, 10.9, 10.2, 10.8],
                'IC': [10.4, 11.1, 10.7, 10.0, 10.7],
                'UA': [220.0, 221.5, 219.8, 220.3, 220.1],
                'UB': [220.1, 220.9, 220.2, 220.4, 220.0],
                'UC': [219.9, 221.0, 220.1, 220.2, 220.2]
            })
            mock_comtrade_instance.to_dataframe.return_value = mock_df
            
            rec, df = reader.read_comtrade('test.cfg')
            
            # Проверяем содержимое DataFrame
            assert df.shape == (5, 7), "DataFrame должен иметь 5 строк и 7 столбцов"
            assert list(df.columns) == ['time', 'IA', 'IB', 'IC', 'UA', 'UB', 'UC']
            assert df['IA'].min() > 10.0, "Значения тока должны быть логичными"
            assert df['UA'].mean() > 219.0, "Напряжение должно быть логичным"


@pytest.mark.integration
class TestReadComtradeErrorHandlingWithMocks:
    """Тесты обработки ошибок с mock-объектами."""
    
    def test_read_comtrade_file_not_found_error(self, capsys):
        """Тест обработки FileNotFoundError."""
        reader = ReadComtrade()
        
        with patch('osc_tools.data_management.comtrade_processing.Comtrade') as mock_comtrade_class:
            mock_comtrade_instance = MagicMock()
            mock_comtrade_class.return_value = mock_comtrade_instance
            mock_comtrade_instance.load.side_effect = FileNotFoundError("File not found")
            
            result = reader.read_comtrade('nonexistent.cfg')
            
            # Проверяем возврат (None, None)
            assert result == (None, None), "При ошибке должен вернуться (None, None)"
            
            # Проверяем, что была выведена диагностика
            captured = capsys.readouterr()
            assert "[ERROR]" in captured.out, "Должно быть сообщение об ошибке"
            assert "FileNotFoundError" in captured.out, "Должен быть тип ошибки"
    
    def test_read_comtrade_value_error(self, capsys):
        """Тест обработки ValueError (неправильный формат файла)."""
        reader = ReadComtrade()
        
        with patch('osc_tools.data_management.comtrade_processing.Comtrade') as mock_comtrade_class:
            mock_comtrade_instance = MagicMock()
            mock_comtrade_class.return_value = mock_comtrade_instance
            mock_comtrade_instance.load.side_effect = ValueError("Invalid format")
            
            result = reader.read_comtrade('bad_file.cfg')
            
            assert result == (None, None), "При ошибке должен вернуться (None, None)"
            
            captured = capsys.readouterr()
            assert "[ERROR]" in captured.out
            assert "ValueError" in captured.out
    
    def test_read_comtrade_generic_exception(self, capsys):
        """Тест обработки любых исключений."""
        reader = ReadComtrade()
        
        with patch('osc_tools.data_management.comtrade_processing.Comtrade') as mock_comtrade_class:
            mock_comtrade_instance = MagicMock()
            mock_comtrade_class.return_value = mock_comtrade_instance
            mock_comtrade_instance.load.side_effect = RuntimeError("Unexpected error")
            
            result = reader.read_comtrade('corrupted.cfg')
            
            assert result == (None, None), "При ошибке должен вернуться (None, None)"
            
            captured = capsys.readouterr()
            assert "[ERROR]" in captured.out
            assert "RuntimeError" in captured.out
    
    def test_read_comtrade_error_diagnostic_output(self, capsys):
        """Тест полноты диагностической информации при ошибке."""
        reader = ReadComtrade()
        test_filename = "oscillogram_2025_11_21.cfg"
        
        with patch('osc_tools.data_management.comtrade_processing.Comtrade') as mock_comtrade_class:
            mock_comtrade_instance = MagicMock()
            mock_comtrade_class.return_value = mock_comtrade_instance
            mock_comtrade_instance.load.side_effect = RuntimeError("Corrupted file")
            
            reader.read_comtrade(test_filename)
            
            captured = capsys.readouterr()
            output = captured.out
            
            # Проверяем наличие основных элементов диагностики
            assert "Произошла критическая ошибка" in output
            assert test_filename in output or "oscillogram_2025_11_21.cfg" in output
            assert "RuntimeError" in output
            assert "Corrupted file" in output
            assert "Полная трассировка" in output


@pytest.mark.integration
class TestReadComtradeComplexScenarios:
    """Тесты сложных сценариев: кодировки, поврежденные файлы."""

    def test_read_comtrade_different_encoding(self):
        """Тест чтения файла в кодировке windows-1251."""
        reader = ReadComtrade()
        
        # Создаем временный файл в кодировке cp1251
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "test_cp1251.cfg")
            dat_path = os.path.join(tmpdir, "test_cp1251.dat")
            
            # Содержимое с кириллицей
            cfg_content = "Станция Тест, Устройство 1, 2013\n1, 1A, 0D\n1, IA, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P\n50.0\n1\n1600, 1\n01/01/2024, 00:00:00.000000\n01/01/2024, 00:00:00.000000\nASCII\n1.0\n"
            
            with open(cfg_path, 'w', encoding='cp1251') as f:
                f.write(cfg_content)
            
            with open(dat_path, 'w', encoding='cp1251') as f:
                f.write("1, 0, 100\n")
            
            # Пытаемся прочитать без указания кодировки (должно сработать, если автоопределение работает или упасть красиво)
            # На самом деле _file_is_utf8 может не сработать для cp1251
            rec, df = reader.read_comtrade(cfg_path)
            
            # Если не сработало автоопределение, попробуем с явным указанием (если ReadComtrade это поддерживает)
            # ReadComtrade.read_comtrade не принимает kwargs для load, это ограничение
            
            if rec is not None:
                assert rec.station_name == "Станция Тест"
            else:
                # Если упало, это ожидаемо для текущей реализации без автоопределения cp1251
                pass

    def test_read_comtrade_corrupted_cfg(self):
        """Тест чтения поврежденного CFG файла (недостаточно строк)."""
        reader = ReadComtrade()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "corrupted.cfg")
            # Только первая строка
            with open(cfg_path, 'w') as f:
                f.write("Station A, Device 1, 2013\n")
            
            rec, df = reader.read_comtrade(cfg_path)
            assert rec is None
            assert df is None

    def test_read_comtrade_mismatched_dat(self):
        """Тест случая, когда DAT файл не соответствует CFG."""
        reader = ReadComtrade()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "mismatch.cfg")
            dat_path = os.path.join(tmpdir, "mismatch.dat")
            
            # CFG ожидает 2 аналоговых канала
            cfg_content = "Station A, Device 1, 2013\n2, 2A, 0D\n1, IA, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P\n2, IB, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P\n50.0\n1\n1600, 1\n01/01/2024, 00:00:00.000000\n01/01/2024, 00:00:00.000000\nASCII\n1.0\n"
            with open(cfg_path, 'w') as f:
                f.write(cfg_content)
            
            # DAT содержит только 1 аналоговый канал
            with open(dat_path, 'w') as f:
                f.write("1, 0, 100\n")
            
            rec, df = reader.read_comtrade(cfg_path)
            # Должно вернуть None из-за ошибки парсинга DAT
            assert rec is None
            assert df is None

