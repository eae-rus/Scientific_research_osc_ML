import pytest
import polars as pl
import numpy as np
import torch
from osc_tools.ml.dataset import OscillogramDataset

class TestOscillogramDataset:
    
    @pytest.fixture
    def sample_data(self):
        # Создаем тестовый DataFrame: 100 точек, 3 канала + таргет
        length = 100
        df = pl.DataFrame({
            'feat1': np.arange(length, dtype=float),
            'feat2': np.random.randn(length),
            'target': np.zeros(length)
        })
        # Таргет: 1 в середине
        df = df.with_columns(
            target = pl.when(pl.arange(0, length).is_between(40, 60)).then(1.0).otherwise(0.0)
        )
        return df

    @pytest.fixture
    def sample_indices(self, sample_data):
        # Индексы начал окон. Окно 10.
        # Допустимые индексы: 0 .. 90
        return pl.DataFrame({'index': np.arange(0, 90, 5)})

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
        # Shape should be (Channels, Time) -> (1, 10)
        assert x.shape == (1, window_size)
        assert y.shape == () or y.shape == (1,)
        
        # Проверка значений
        # x должен быть 0..9. Transposed: [[0, 1, ... 9]]
        expected_x = np.arange(10, dtype=np.float32).reshape(1, -1)
        assert np.allclose(x.numpy(), expected_x)
        
        # y должен быть target[9] = 0
        assert y.item() == 0.0

    def test_classification_any_in_window(self, sample_data, sample_indices):
        window_size = 10
        ds = OscillogramDataset(
            dataframe=sample_data,
            indices=sample_indices,
            window_size=window_size,
            mode='classification',
            feature_columns=['feat1'],
            target_columns='target',
            target_window_mode='any_in_window'
        )

        # Окно 0..9: метка отсутствует
        _, y0 = ds[0]
        assert y0.item() == 0.0

        # Окно 35..44 пересекается с [40, 60] => метка 1
        _, y1 = ds[7]
        assert y1.item() == 1.0

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
        # Shape: (Channels, Time) -> (1, 10)
        assert x.shape == (1, window_size)
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
        # Shape: (Channels, Time) -> (2, 10)
        assert x.shape == (2, window_size)
        # Target for reconstruction is same as input
        assert y.shape == (2, window_size)
        assert torch.allclose(x, y)

    def test_invalid_mode(self, sample_data, sample_indices):
        with pytest.raises(ValueError):
            OscillogramDataset(sample_data, sample_indices, 10, mode='invalid')

class TestOscillogramDatasetFeatures:
    
    @pytest.fixture
    def physics_data(self):
        length = 200
        t = np.linspace(0, 0.1, length)
        f = 50
        # Генерируем трёхфазные сигналы токов и напряжений
        data = {
            'IA': np.sin(2*np.pi*f*t),
            'IB': np.sin(2*np.pi*f*t - 2*np.pi/3),
            'IC': np.sin(2*np.pi*f*t + 2*np.pi/3),
            'IN': 0.1 * np.sin(2*np.pi*f*t),  # Нулевой ток
            'UA': 100 * np.sin(2*np.pi*f*t),
            'UB': 100 * np.sin(2*np.pi*f*t - 2*np.pi/3),
            'UC': 100 * np.sin(2*np.pi*f*t + 2*np.pi/3),
            'UN': 0.01 * np.ones(length),  # Малое нулевое напряжение для валидации
            'target': np.zeros(length)
        }
        return pl.DataFrame(data)

    @pytest.fixture
    def indices(self):
        return [0, 50]

    def test_power_feature_mode(self, physics_data, indices):
        ds = OscillogramDataset(
            dataframe=physics_data,
            indices=indices,
            window_size=32,
            mode='classification',
            feature_mode='power',
            sampling_rate=1600
        )
        
        x, y = ds[0]
        # Режим "power": 4 фазы * 2 (P, Q) = 8 каналов
        # Shape: (Channels, Time) -> (8, 32)
        assert x.shape == (8, 32)

    def test_combined_feature_mode(self, physics_data, indices):
        ds = OscillogramDataset(
            dataframe=physics_data,
            indices=indices,
            window_size=32,
            mode='classification',
            feature_mode=['raw', 'power'],
            feature_columns=['IA', 'UA'], # For raw part
            sampling_rate=1600
        )
        
        x, y = ds[0]
        # Raw: 2 канала (IA, UA)
        # Power: 8 каналов (4 фазы * 2)
        # Всего: 10 каналов
        # Shape: (Channels, Time) -> (10, 32)
        assert x.shape == (10, 32)

    def test_symmetric_feature_mode(self, physics_data, indices):
        ds = OscillogramDataset(
            dataframe=physics_data,
            indices=indices,
            window_size=32,
            mode='classification',
            feature_mode='symmetric',
            sampling_rate=1600
        )
        
        x, y = ds[0]
        # Режим "symmetric":
        # Для токов (I): I1, I2, I0 (действительная и мнимая части) -> 6 каналов
        # Для напряжений (U): U1, U2, U0 (действительная и мнимая части) -> 6 каналов
        # Всего: 12 каналов
        # Shape: (Channels, Time) -> (12, 32)
        assert x.shape == (12, 32)

    def test_phase_polar_h1_angle_feature_mode(self, physics_data, indices):
        num_harmonics = 3
        ds = OscillogramDataset(
            dataframe=physics_data,
            indices=indices,
            window_size=64,
            mode='classification',
            feature_mode='phase_polar_h1_angle',
            sampling_rate=1600,
            num_harmonics=num_harmonics,
        )

        x, _ = ds[0]

        # 8 каналов, для каждого: модули всех гармоник + угол только h1
        expected_channels = 8 * (num_harmonics + 1)
        assert x.shape == (expected_channels, 64)
