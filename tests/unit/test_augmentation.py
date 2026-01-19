import pytest
import torch
import numpy as np
import polars as pl
from osc_tools.ml.augmentation import TimeSeriesAugmenter
from osc_tools.ml.dataset import OscillogramDataset

class TestTimeSeriesAugmenter:
    
    @pytest.fixture
    def sample_data(self):
        # (Батч, Время, Каналы)
        # 8 каналов: IA, IB, IC, In, UA, UB, UC, Un
        batch = 2
        time = 100
        channels = 8
        return torch.ones(batch, time, channels)

    def test_inversion(self, sample_data):
        config = {"p_inversion": 1.0}
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        assert torch.allclose(output, -sample_data)

    def test_scaling(self, sample_data):
        config = {
            "p_scaling": 1.0,
            "scaling_range_current": (2.0, 2.0), # Fixed scaling
            "scaling_range_voltage": (0.5, 0.5),
            "p_inversion": 0.0,
            "p_noise": 0.0,
            "p_offset": 0.0,
            "p_phase_shuffling": 0.0,
            "p_drop_channel": 0.0
        }
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        
        # Токи (0,1,2,3) должны быть удвоены
        assert torch.allclose(output[:, :, :4], sample_data[:, :, :4] * 2.0)
        # Напряжения (4,5,6,7) должны быть уменьшены вдвое
        assert torch.allclose(output[:, :, 4:], sample_data[:, :, 4:] * 0.5)

    def test_noise(self, sample_data):
        config = {
            "p_noise": 1.0,
            "noise_std_current": 0.1,
            "noise_std_voltage": 0.1
        }
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        
        assert not torch.allclose(output, sample_data)
        # Проверяем, что среднее примерно то же (шум со средним 0)
        assert torch.abs(output.mean() - sample_data.mean()) < 0.1

    def test_offset(self, sample_data):
        config = {
            "p_offset": 1.0,
            "offset_range": (1.0, 1.0),
            "p_inversion": 0.0,
            "p_noise": 0.0,
            "p_scaling": 0.0,
            "p_phase_shuffling": 0.0,
            "p_drop_channel": 0.0
        }
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        assert torch.allclose(output, sample_data + 1.0)

    def test_phase_shuffling(self):
        # Создаём отличимые фазы
        # IA=1, IB=2, IC=3
        # UA=10, UB=20, UC=30
        data = torch.zeros(1, 10, 8)
        data[:, :, 0] = 1
        data[:, :, 1] = 2
        data[:, :, 2] = 3
        data[:, :, 4] = 10
        data[:, :, 5] = 20
        data[:, :, 6] = 30
        
        config = {
            "p_phase_shuffling": 1.0,
            "p_inversion": 0.0,
            "p_noise": 0.0,
            "p_scaling": 0.0,
            "p_offset": 0.0,
            "p_drop_channel": 0.0
        }
        augmenter = TimeSeriesAugmenter(config)
        
        # Запускаем один раз — случайность внутри может дать сдвиг 1 или 2.
        # Поскольку мы не контролируем генератор внутри, проверяем, что результат
        # соответствует либо сдвигу 1, либо сдвигу 2.
        output = augmenter(data)
        
        # Сдвиг 1: A->B, B->C, C->A
        # IA=3, IB=1, IC=2
        shift1_currents = torch.tensor([3., 1., 2.])
        shift1_voltages = torch.tensor([30., 10., 20.])
        
        # Сдвиг 2: A->C, B->A, C->B
        # IA=2, IB=3, IC=1
        shift2_currents = torch.tensor([2., 3., 1.])
        shift2_voltages = torch.tensor([20., 30., 10.])
        
        currents = output[0, 0, :3]
        voltages = output[0, 0, 4:7]
        
        is_shift1 = torch.allclose(currents, shift1_currents) and torch.allclose(voltages, shift1_voltages)
        is_shift2 = torch.allclose(currents, shift2_currents) and torch.allclose(voltages, shift2_voltages)
        
        assert is_shift1 or is_shift2

    def test_drop_channel(self, sample_data):
        config = {"p_drop_channel": 1.0}
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        
        # Проверяем, что хотя бы один канал обнулён
        # Суммируем по времени для каждого канала
        channel_sums = output.sum(dim=1) # (Batch, Channels)
        # Проверяем, есть ли канал с суммой 0
        has_zero_channel = (channel_sums == 0).any()
        assert has_zero_channel

class TestDatasetAugmentation:
    
    @pytest.fixture
    def sample_df(self):
        # Создаём фикстурную DataFrame с 8 каналами
        length = 100
        t = np.arange(length) / 1600.0
        f = 50.0
        
        # Сбалансированная трёхфазная система
        ia = np.sin(2 * np.pi * f * t)
        ib = np.sin(2 * np.pi * f * t - 2*np.pi/3)
        ic = np.sin(2 * np.pi * f * t + 2*np.pi/3)
        
        data = {
            'IA': ia,
            'IB': ib,
            'IC': ic,
            'IN': np.zeros(length), # Нулевая последовательность = 0
            'UA': ia, # Повторно используем токи как напряжения для теста
            'UB': ib,
            'UC': ic,
            'UN': np.zeros(length),
            'target': np.zeros(length)
        }
        return pl.DataFrame(data)

    def test_dataset_augmentation_integration(self, sample_df):
        indices = [0]
        window_size = 50
        
        # Конфигурация, инвертирующая сигнал
        aug_config = {
            "p_inversion": 1.0,
            "p_noise": 0.0,
            "p_scaling": 0.0,
            "p_offset": 0.0,
            "p_phase_shuffling": 0.0,
            "p_drop_channel": 0.0
        }
        
        ds = OscillogramDataset(
            sample_df, 
            indices, 
            window_size, 
            feature_mode='raw',
            augmentation_config=aug_config,
            target_columns='target'
        )
        
        x, y = ds[0]
        # x форма: (Каналы, Время)
        # Оригинальные данные — синус. После инверсии должно быть -sin.
        
        # Воссоздаём ожидаемый сигнал (первые 50 точек)
        t = np.arange(window_size) / 1600.0
        f = 50.0
        expected_ia = -np.sin(2 * np.pi * f * t)
        
        # Проверяем IA (канал 0)
        assert torch.allclose(x[0, :], torch.tensor(expected_ia, dtype=torch.float32), atol=1e-5)
        
    def test_dataset_augmentation_symmetric(self, sample_df):
        indices = [0]
        window_size = 50
        
        # Конфигурация, масштабирующая сигнал в 2 раза
        aug_config = {
            "p_scaling": 1.0, 
            "scaling_range_current": (2.0, 2.0),
            "scaling_range_voltage": (2.0, 2.0)
        }
        
        ds = OscillogramDataset(
            sample_df, 
            indices, 
            window_size, 
            feature_mode='symmetric',
            augmentation_config=aug_config,
            target_columns='target'
        )
        
        x, y = ds[0]
        # Формат выхода: [I1_re, I1_im, I2_re, I2_im, I0_re, I0_im, U1..., U2..., U0...]
        
        # Проверяем положительную последовательность тока I1 (индексы 0,1)
        # Амплитуда до масштабирования была 1.0, после масштабирования — 2.0
        i1_re = x[0, -1]
        i1_im = x[1, -1]
        i1_mag = torch.sqrt(i1_re**2 + i1_im**2)
        
        # sliding_window_fft использует окно Ханна, которое ослабляет амплитуду примерно в 2 раза.
        # Для входной амплитуды 2.0 ожидаемую величину примерно 1.0.
        # Проверим это, посчитав ожидаемое значение явно.
        from osc_tools.features.pdr_calculator import sliding_window_fft
        
        t = np.arange(window_size) / 1600.0
        f = 50.0
        # Ожидаемый сигнал масштабирован в 2.0
        expected_signal = 2.0 * np.sin(2 * np.pi * f * t)
        
        # Вычислим FFT ожидаемого сигнала.
        # sliding_window_fft возвращает валидные значения, начиная с индекса окна.
        # В нашем случае длина входа (50) > окно FFT (32), поэтому есть валидные результаты.
        fft_window = int(1600 / 50) # 32
        expected_phasor = sliding_window_fft(expected_signal, fft_window, 1)
        # Валидные значения начинаются с индекса 32.
        # Нам нужен результат в конце массива (индекс 49).
        expected_mag = np.abs(expected_phasor[-1, 0])
        
        print(f"DEBUG: Expected magnitude = {expected_mag}")

        assert torch.isclose(i1_mag, torch.tensor(expected_mag, dtype=torch.float32), atol=0.1)
        
        # Проверяем нулевую последовательность тока I0 (индексы 4,5)
        # Должна быть 0
        i0_re = x[4, -1]
        i0_im = x[5, -1]
        i0_mag = torch.sqrt(i0_re**2 + i0_im**2)
        assert torch.isclose(i0_mag, torch.tensor(0.0), atol=0.1)
