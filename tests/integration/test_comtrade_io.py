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
