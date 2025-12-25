import pytest
import pandas as pd
import numpy as np
import torch
from osc_tools.ml.dataset import OscillogramDataset

class TestOscillogramDataset:
    
    @pytest.fixture
    def sample_data(self):
        # Создаем тестовый DataFrame: 100 точек, 3 канала + таргет
        length = 100
        df = pd.DataFrame({
            'feat1': np.arange(length, dtype=float),
            'feat2': np.random.randn(length),
            'target': np.zeros(length)
        })
        # Таргет: 1 в середине
        df.loc[40:60, 'target'] = 1
        return df

    @pytest.fixture
    def sample_indices(self, sample_data):
        # Индексы начал окон. Окно 10.
        # Допустимые индексы: 0 .. 90
        return pd.DataFrame(index=np.arange(0, 90, 5))

    def test_init(self, sample_data, sample_indices):
        ds = OscillogramDataset(
            dataframe=sample_data,
            indices=sample_indices,
            window_size=10,
            mode='classification',
            feature_columns=['feat1', 'feat2'],
            target_columns='target'
        )
        assert len(ds) == len(sample_indices)

    def test_classification_mode(self, sample_data, sample_indices):
        window_size = 10
        ds = OscillogramDataset(
            dataframe=sample_data,
            indices=sample_indices,
            window_size=window_size,
            mode='classification',
            feature_columns=['feat1'],
            target_columns='target',
            target_position=window_size-1 # Последняя точка
        )
        
        x, y = ds[0] # Индекс 0 -> окно 0..9
        assert x.shape == (window_size, 1)
        assert y.shape == () or y.shape == (1,)
        
        # Проверка значений
        # x должен быть 0..9
        expected_x = np.arange(10, dtype=np.float32).reshape(-1, 1)
        assert np.allclose(x.numpy(), expected_x)
        
        # y должен быть target[9] = 0
        assert y.item() == 0.0

    def test_segmentation_mode(self, sample_data, sample_indices):
        window_size = 10
        ds = OscillogramDataset(
            dataframe=sample_data,
            indices=sample_indices,
            window_size=window_size,
            mode='segmentation',
            feature_columns=['feat1'],
            target_columns='target'
        )
        
        x, y = ds[0]
        assert x.shape == (window_size, 1)
        assert y.shape == (window_size,) # Маска той же длины
        
    def test_reconstruction_mode(self, sample_data, sample_indices):
        window_size = 10
        ds = OscillogramDataset(
            dataframe=sample_data,
            indices=sample_indices,
            window_size=window_size,
            mode='reconstruction',
            feature_columns=['feat1', 'feat2']
        )
        
        x, y = ds[0]
        assert x.shape == (window_size, 2)
        assert y.shape == (window_size, 2)
        assert torch.allclose(x, y)

    def test_invalid_mode(self, sample_data, sample_indices):
        with pytest.raises(ValueError):
            OscillogramDataset(sample_data, sample_indices, 10, mode='invalid')
