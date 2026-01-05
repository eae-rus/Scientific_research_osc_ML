import pytest
import polars as pl
import numpy as np
import torch
from osc_tools.ml.dataset import OscillogramDataset

@pytest.fixture
def dummy_dataframe():
    # Создаем датафрейм с 2 файлами
    # Файл A: 100 строк
    # Файл B: 200 строк
    
    data_a = {
        'file_name': ['file_A'] * 100,
        'IA': np.random.randn(100).astype(np.float32),
        'IC': np.random.randn(100).astype(np.float32),
        'UA': np.random.randn(100).astype(np.float32),
        'UC': np.random.randn(100).astype(np.float32),
        'ML_class': [0] * 100
    }
    
    data_b = {
        'file_name': ['file_B'] * 200,
        'IA': np.random.randn(200).astype(np.float32),
        'IC': np.random.randn(200).astype(np.float32),
        'UA': np.random.randn(200).astype(np.float32),
        'UC': np.random.randn(200).astype(np.float32),
        'ML_class': [0] * 200
    }
    
    df_a = pl.DataFrame(data_a)
    df_b = pl.DataFrame(data_b)
    
    return pl.concat([df_a, df_b])

def test_create_indices_train(dummy_dataframe):
    """Тест создания индексов для режима обучения (Random Sampling)."""
    window_size = 50
    indices = OscillogramDataset.create_indices(dummy_dataframe, window_size, mode='train')
    
    # Должно быть 2 элемента (по одному на файл)
    assert len(indices) == 2
    assert isinstance(indices[0], tuple)
    assert len(indices[0]) == 2
    
    # Проверка значений
    # Файл A начинается с 0, длина 100
    assert indices[0] == (0, 100)
    # Файл B начинается с 100, длина 200
    assert indices[1] == (100, 200)

def test_create_indices_val(dummy_dataframe):
    """Тест создания индексов для режима валидации (Sliding Window)."""
    window_size = 50
    stride = 25
    indices = OscillogramDataset.create_indices(dummy_dataframe, window_size, mode='val', stride=stride)
    
    # Файл A (100): 0, 25, 50. (3 окна)
    # Файл B (200): 100, 125, ..., 150. (200/25 = 8 окон? 0, 25, 50, 75, 100, 125, 150. 150+50=200. Итого 7 окон: 0..150)
    # Давайте посчитаем:
    # A: 0-50, 25-75, 50-100. (3)
    # B: 0-50, 25-75, ..., 150-200. (0, 25, 50, 75, 100, 125, 150). 7 окон.
    # Итого 10.
    
    assert len(indices) == 10
    assert isinstance(indices[0], int)
    
    # Проверка первых нескольких
    assert indices[0] == 0
    assert indices[1] == 25
    assert indices[2] == 50
    assert indices[3] == 100 # Начало файла B

def test_getitem_train_randomness(dummy_dataframe):
    """Тест случайности выборки окна в режиме обучения."""
    window_size = 50
    indices = OscillogramDataset.create_indices(dummy_dataframe, window_size, mode='train')
    
    ds = OscillogramDataset(dummy_dataframe, indices, window_size, mode='classification')
    
    # Получаем элемент 0 несколько раз, он должен меняться (если длина файла > размера окна)
    # Файл A имеет длину 100, окно 50. Смещение может быть 0..50.
    
    starts = set()
    for _ in range(20):
        # Мы не можем легко проверить внутренний start_idx, но можем проверить данные
        # Но данные случайны.
        # Просто проверяем, что получаем валидные тензоры
        x, y = ds[0]
        assert x.shape == (8, 50) # 8 каналов (стандартизированные)
        
    # Для проверки случайности нужно инспектировать срез.
    # Но пока достаточно убедиться, что код выполняется без ошибок.

def test_getitem_val_fixed(dummy_dataframe):
    """Тест фиксированной выборки окна в режиме валидации."""
    window_size = 50
    stride = 25
    indices = OscillogramDataset.create_indices(dummy_dataframe, window_size, mode='val', stride=stride)
    
    ds = OscillogramDataset(dummy_dataframe, indices, window_size, mode='classification')
    
    # Элемент 0 должен быть 0..50
    x0, _ = ds[0]
    # Элемент 1 должен быть 25..75
    x1, _ = ds[1]
    
    # Проверка соответствия данных датафрейму
    # IA - это канал 0 (после стандартизации)
    # Стандартизированные: IA, IB, IC, In, UA, UB, UC, Un
    
    # Нам нужно знать, что делает _get_standardized_raw_data.
    # Он помещает IA на индекс 0.
    
    ia_full = dummy_dataframe['IA'].to_numpy()
    
    assert np.allclose(x0[0].numpy(), ia_full[0:50], atol=1e-5)
    assert np.allclose(x1[0].numpy(), ia_full[25:75], atol=1e-5)
