import pytest
import torch
import numpy as np
import polars as pl
from osc_tools.ml.dataset import OscillogramDataset

class TestDownsampling:
    def test_downsampling_stride(self):
        length = 100
        # Создаём валидные трёхфазные данные, чтобы пройти проверки в `_get_standardized_raw_data`
        data = {
            'IA': np.arange(length, dtype=float),
            'IB': np.arange(length, dtype=float), # заглушка для валидности
            'IC': np.arange(length, dtype=float), # заглушка для валидности
            'IN': np.zeros(length),
            'UA': np.ones(length), # заглушка для валидности
            'UB': np.ones(length),
            'UC': np.ones(length),
            'UN': np.zeros(length),
            'target': np.zeros(length)
        }
        df = pl.DataFrame(data)
        
        ds = OscillogramDataset(
            df, [0], window_size=50,
            feature_mode='raw',
            downsampling_mode='stride',
            downsampling_stride=10
        )
        
        x, y = ds[0]
        # Форма x: (Каналы, Время)
        # Временных отсчётов: 50 / 10 = 5
        assert x.shape[1] == 5
        
        # Проверяем значения (IA — канал 0)
        # Ожидаем: 0, 10, 20, 30, 40
        expected = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0])
        assert torch.allclose(x[0], expected)

    def test_downsampling_snapshot(self):
        length = 100
        data = {
            'IA': np.arange(length, dtype=float),
            'IB': np.arange(length, dtype=float),
            'IC': np.arange(length, dtype=float),
            'IN': np.zeros(length),
            'UA': np.ones(length),
            'UB': np.ones(length),
            'UC': np.ones(length),
            'UN': np.zeros(length),
            'target': np.zeros(length)
        }
        df = pl.DataFrame(data)
        
        ds = OscillogramDataset(
            df, [0], window_size=50,
            feature_mode='raw',
            downsampling_mode='snapshot'
        )
        
        x, y = ds[0]
        # Форма x: (Каналы, Время)
        # Временных отсчётов: 2
        assert x.shape[1] == 2
        
        # Проверяем значения (IA — канал 0)
        # Ожидаем: 0 (начало) и 49 (конец окна)
        expected = torch.tensor([0.0, 49.0])
        assert torch.allclose(x[0], expected)

if __name__ == "__main__":
    pytest.main([__file__])
