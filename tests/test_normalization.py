import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

from osc_tools.features.normalization import NormOsc

@pytest.fixture
def norm_coef_file(tmp_path: Path) -> str:
    """
    Создает временный CSV-файл с коэффициентами нормализации
    и возвращает путь к нему.
    """
    coef_data = {
        'name': ['test_osc_1', 'test_osc_2'],
        'norm': ['YES', 'NO'],
        '1Ip_base': [5.0, 10.0],
        '1Ub_base': [100.0, 100.0],
        '1Iz_base': [1.0, 1.0],
        '1Uc_base': [100.0, 100.0],
    }
    coef_df = pd.DataFrame(coef_data)

    file_path = tmp_path / "norm_coef.csv"
    coef_df.to_csv(file_path, index=False)
    return str(file_path)

def test_norm_osc_initialization(norm_coef_file):
    """Тест проверяет, что класс NormOsc корректно инициализируется."""
    norm_osc = NormOsc(norm_coef_file_path=norm_coef_file)
    assert norm_osc.norm_coef is not None
    assert len(norm_osc.norm_coef) == 2

def test_normalize_bus_signals_applies_correctly(norm_coef_file):
    """
    Тест проверяет, что коэффициенты нормализации правильно применяются
    к данным в DataFrame.
    """
    # 1. Подготовка
    norm_osc = NormOsc(norm_coef_file_path=norm_coef_file)

    # Создаем DataFrame для нормализации
    raw_data = {
        'I | Bus-1 | phase: A': [1000.0],
        'U | BusBar-1 | phase: A': [30000.0],
        'I | Bus-1 | phase: N': [10.0],
        'U | CableLine-1 | phase: A': [30000.0],
        'Other Signal': [999]
    }
    raw_df = pd.DataFrame(raw_data)

    # 2. Выполнение
    normalized_df = norm_osc.normalize_bus_signals(raw_df.copy(), file_name='test_osc_1', yes_prase="YES")

    # 3. Проверка
    assert normalized_df is not None
    # Проверяем ток: 1000 / (20 * 5.0) = 10.0
    assert np.isclose(normalized_df['I | Bus-1 | phase: A'].iloc[0], 10.0)
    # Проверяем напряжение шины: 30000 / (3 * 100.0) = 100.0
    assert np.isclose(normalized_df['U | BusBar-1 | phase: A'].iloc[0], 100.0)
    # Проверяем ток НП: 10.0 / (5 * 1.0) = 2.0
    assert np.isclose(normalized_df['I | Bus-1 | phase: N'].iloc[0], 2.0)
    # Проверяем напряжение КЛ: 30000 / (3 * 100.0) = 100.0
    assert np.isclose(normalized_df['U | CableLine-1 | phase: A'].iloc[0], 100.0)
    # Проверяем, что другой столбец не изменился
    assert normalized_df['Other Signal'].iloc[0] == 999


def test_normalize_bus_signals_returns_none_if_not_found(norm_coef_file):
    """
    Тест проверяет, что метод возвращает None, если имя файла
    не найдено в файле коэффициентов.
    """
    norm_osc = NormOsc(norm_coef_file_path=norm_coef_file)
    raw_df = pd.DataFrame({'I | Bus-1 | phase: A': [100.0]})

    result_df = norm_osc.normalize_bus_signals(raw_df, file_name='non_existent_file')

    assert result_df is None

def test_normalize_bus_signals_returns_none_if_norm_is_no(norm_coef_file):
    """

    Тест проверяет, что метод возвращает None, если для файла
    нормализация запрещена ('norm' != 'YES').
    """
    norm_osc = NormOsc(norm_coef_file_path=norm_coef_file)
    raw_df = pd.DataFrame({'I | Bus-1 | phase: A': [100.0]})

    # 'test_osc_2' имеет norm = 'NO' в нашем фиктивном файле
    result_df = norm_osc.normalize_bus_signals(raw_df, file_name='test_osc_2', yes_prase="YES")

    assert result_df is None
