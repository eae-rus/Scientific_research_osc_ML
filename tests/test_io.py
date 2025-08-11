import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock

# Это необходимо, чтобы pytest мог найти модуль osc_tools
sys.path.insert(0, '.')

from osc_tools.io.comtrade_parser import ComtradeParser

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_comtrade_parser(monkeypatch):
    """
    Фикстура, которая создает экземпляр ComtradeParser,
    но с "замоканными" (подмененными) зависимостями.
    """
    # 1. Мокаем зависимость от dataflow.comtrade_processing.ReadComtrade
    # Создаем фиктивный класс и фиктивный метод read_comtrade
    mock_read_comtrade_instance = MagicMock()

    # Настраиваем фиктивный метод read_comtrade, чтобы он возвращал нужные нам данные
    # Он должен возвращать кортеж: (объект_с_атрибутом_cfg, DataFrame)
    mock_cfg_object = MagicMock()
    mock_cfg_object.frequency = 50.0
    mock_cfg_object.sample_rates = [(1600.0, 1000)]

    # Создаем DataFrame, который якобы был прочитан из файла
    test_df = pd.DataFrame({
        'U | BusBar-1 | phase: A': [1.0, 1.1, 1.2],
        'I | Bus-1 | phase: A': [0.1, 0.2, 0.3],
        'U | BusBar-2 | phase: A': [2.0, 2.1, 2.2],
        'I | Bus-2 | phase: A': [0.4, 0.5, 0.6],
        'MLsignal_1_2_1_1': [0, 1, 0] # Неиспользуемый столбец для проверки check_columns
    })
    mock_read_comtrade_instance.read_comtrade.return_value = (mock_cfg_object, test_df)

    # Подменяем настоящий ReadComtrade на наш фиктивный
    monkeypatch.setattr('osc_tools.io.comtrade_parser.ReadComtrade', lambda: mock_read_comtrade_instance)

    # 2. Мокаем проверку на неизвестные столбцы, чтобы она не падала на нашем тестовом ML-сигнале
    monkeypatch.setattr(ComtradeParser, 'check_columns', lambda self, df: None)

    # 3. Создаем и возвращаем экземпляр парсера.
    # Он будет использовать наши фиктивные JSON-файлы, которые мы создали ранее.
    # Указываем raw_path, так как конструктор проверяет его наличие.
    parser = ComtradeParser(raw_path='tests/fixtures/')
    return parser


# --- Tests ---

def test_create_one_df_splits_and_renames_buses(mock_comtrade_parser):
    """
    Тестирует метод create_one_df.
    Проверяет, что:
    1. Данные корректно разделяются на шины (Bus-1, Bus-2).
    2. Для каждой шины создается правильное имя файла.
    3. Столбцы переименовываются согласно правилам.
    """
    # Получаем наш "замоканный" парсер из фикстуры
    parser = mock_comtrade_parser

    # Вызываем тестируемый метод
    # Путь и имя файла здесь не важны, так как чтение замокано,
    # но они используются для формирования имени в итоговом DataFrame.
    result_df = parser.create_one_df(file_path='tests/fixtures/test.cfg', file_name='test.cfg')

    # --- ПРОВЕРКИ ---

    # 1. Проверяем, что результат - это DataFrame
    assert isinstance(result_df, pd.DataFrame)
    # В нашем тестовом DataFrame 3 строки, и мы ожидаем 2 шины, итого 6 строк.
    assert len(result_df) == 6

    # 2. Проверяем, что появились правильные имена файлов
    expected_filenames = [
        'test_Bus-1', 'test_Bus-1', 'test_Bus-1',
        'test_Bus-2', 'test_Bus-2', 'test_Bus-2'
    ]
    assert result_df['file_name'].tolist() == expected_filenames

    # 3. Проверяем, что столбцы правильно переименованы и разделены

    # Для первой шины
    bus1_df = result_df[result_df['file_name'] == 'test_Bus-1']
    assert 'UA BB' in bus1_df.columns
    assert 'IA' in bus1_df.columns
    assert np.allclose(bus1_df['UA BB'], [1.0, 1.1, 1.2])
    assert np.allclose(bus1_df['IA'], [0.1, 0.2, 0.3])
    # Проверяем, что данных от второй шины тут нет
    assert 'UA CL' not in bus1_df.columns # Пример другого типа напряжения
    assert pd.isna(bus1_df['UA BB'].iloc[0]) or 'U | BusBar-2 | phase: A' not in bus1_df.columns

    # Для второй шины
    bus2_df = result_df[result_df['file_name'] == 'test_Bus-2']
    assert 'UA BB' in bus2_df.columns
    assert 'IA' in bus2_df.columns
    assert np.allclose(bus2_df['UA BB'], [2.0, 2.1, 2.2])
    assert np.allclose(bus2_df['IA'], [0.4, 0.5, 0.6])
    assert pd.isna(bus2_df['UA BB'].iloc[0]) or 'U | BusBar-1 | phase: A' not in bus2_df.columns
