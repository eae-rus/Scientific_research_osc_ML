import pytest
import polars as pl
import numpy as np
import torch
from osc_tools.ml.dataset import OscillogramDataset

@pytest.fixture
def sample_df_phase():
    """DataFrame с фазными напряжениями."""
    length = 100
    return pl.DataFrame({
        'IA': np.random.randn(length),
        'IB': np.random.randn(length),
        'IC': np.random.randn(length),
        'IN': np.random.randn(length) * 0.1,  # Нулевой ток
        'UA BB': np.random.randn(length),
        'UB BB': np.random.randn(length),
        'UC BB': np.random.randn(length),
        'UN BB': np.random.randn(length) * 0.1,  # Нулевое напряжение
        'target': np.zeros(length)
    })

@pytest.fixture
def sample_df_line():
    """DataFrame только с линейными напряжениями."""
    length = 100
    return pl.DataFrame({
        'IA': np.random.randn(length),
        'IB': np.random.randn(length),
        'IC': np.random.randn(length),
        'IN': np.random.randn(length) * 0.1,
        'UAB': np.random.randn(length),
        'UBC': np.random.randn(length),
        'UCA': np.random.randn(length),
        'UN': np.random.randn(length) * 0.1,
        'target': np.zeros(length)
    })

@pytest.fixture
def sample_df_missing_voltage():
    """DataFrame без напряжений."""
    length = 100
    return pl.DataFrame({
        'IA': np.random.randn(length),
        'IB': np.random.randn(length),
        'IC': np.random.randn(length),
        'IN': np.random.randn(length) * 0.1,
        'target': np.zeros(length)
    })

@pytest.fixture
def sample_df_zeros_in_phase():
    """DataFrame с фазными напряжениями, но они заполнены нулями (имитация отсутствия данных)."""
    length = 100
    return pl.DataFrame({
        'IA': np.random.randn(length),
        'IB': np.random.randn(length),
        'IC': np.random.randn(length),
        'IN': np.random.randn(length) * 0.1,
        'UA BB': np.zeros(length), # Нули
        'UB BB': np.zeros(length),
        'UC BB': np.zeros(length),
        'UN BB': np.zeros(length),
        'UAB': np.random.randn(length), # Линейные есть
        'UBC': np.random.randn(length),
        'UCA': np.random.randn(length),
        'UN': np.random.randn(length) * 0.1,
        'target': np.zeros(length)
    })

@pytest.fixture
def sample_df_missing_ib():
    """DataFrame без тока фазы B (нули), но с IA и IC."""
    length = 100
    ia = np.random.randn(length)
    ic = np.random.randn(length)
    return pl.DataFrame({
        'IA': ia,
        'IB': np.zeros(length), # Нули
        'IC': ic,
        'UA': np.random.randn(length),
        'UB': np.random.randn(length),
        'UC': np.random.randn(length),
        'target': np.zeros(length)
    })

def test_smart_selector_phase(sample_df_phase):
    dataset = OscillogramDataset(
        dataframe=sample_df_phase,
        indices=[0],
        window_size=50,
        feature_mode='raw',
        target_columns='target'
    )
    x, y = dataset[0]
    # Ожидаем 8 каналов: IA, IB, IC, In, UA, UB, UC, Un
    assert x.shape == (8, 50)
    # Проверяем, что напряжения не все нули
    assert not torch.all(x[4:, :] == 0)

def test_smart_selector_line(sample_df_line):
    dataset = OscillogramDataset(
        dataframe=sample_df_line,
        indices=[0],
        window_size=50,
        feature_mode='raw',
        target_columns='target'
    )
    x, y = dataset[0]
    # Ожидаем 8 каналов: IA, IB, IC, In, UA(rec), UB(rec), UC(rec), Un(rec)
    assert x.shape == (8, 50)
    # Проверяем, что напряжения не все нули
    assert not torch.all(x[4:, :] == 0)

def test_smart_selector_missing(sample_df_missing_voltage):
    dataset = OscillogramDataset(
        dataframe=sample_df_missing_voltage,
        indices=[0],
        window_size=50,
        feature_mode='raw',
        target_columns='target'
    )
    x, y = dataset[0]
    # Ожидаем 8 каналов: IA, IB, IC, In, 0, 0, 0, 0
    assert x.shape == (8, 50)
    # Проверяем, что напряжения ВСЕ нули
    assert torch.all(x[4:, :] == 0)
    # Проверяем, что токи НЕ все нули
    assert not torch.all(x[:3, :] == 0)

def test_smart_selector_zeros_priority(sample_df_zeros_in_phase):
    """Тест приоритета: если фазные нули, должны взяться линейные."""
    dataset = OscillogramDataset(
        dataframe=sample_df_zeros_in_phase,
        indices=[0],
        window_size=50,
        feature_mode='raw',
        target_columns='target'
    )
    x, y = dataset[0]
    # Если бы взялись фазные (которые нули), то x[3:] были бы нули.
    # Но должны взяться линейные и восстановиться.
    assert not torch.all(x[3:, :] == 0)

def test_smart_selector_reconstruct_ib(sample_df_missing_ib):
    """Тест восстановления IB."""
    dataset = OscillogramDataset(
        dataframe=sample_df_missing_ib,
        indices=[0],
        window_size=50,
        feature_mode='raw',
        target_columns='target'
    )
    x, y = dataset[0]
    
    ia = x[0, :].numpy()
    ib = x[1, :].numpy()
    ic = x[2, :].numpy()
    
    # Проверяем, что IB не нули (восстановлен)
    assert not np.allclose(ib, 0)
    
    # Проверяем закон Кирхгофа: IA + IB + IC = 0 (приближенно)
    assert np.allclose(ia + ib + ic, 0, atol=1e-5)

def test_symmetric_mode_line(sample_df_line):
    dataset = OscillogramDataset(
        dataframe=sample_df_line,
        indices=[0],
        window_size=50,
        feature_mode='symmetric',
        target_columns='target'
    )
    x, y = dataset[0]
    # Ожидаем 12 каналов: 
    # I1(re,im), I2(re,im), I0(re,im) -> 6
    # U1(re,im), U2(re,im), U0(re,im) -> 6
    assert x.shape == (12, 50)
    # U0 теперь рассчитывается из Un, если Un присутствует, так что не ноль
    # U0 находится по индексам 10, 11 (0-based: 6+4, 6+5)
    assert not torch.all(x[10:12, :] == 0)  # U0 не ноль, поскольку Un есть
    # U1, U2 должны быть не нулями
    assert not torch.all(x[6:10, :] == 0)

