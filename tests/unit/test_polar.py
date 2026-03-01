import pytest
import torch
import numpy as np
import polars as pl
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.features.polar import calculate_polar_features

class TestPolarFeatures:
    
    def test_calculate_polar_features_absolute(self):
        # 1 + j0 -> Mag=1, Угол=0
        # 0 + j1 -> Mag=1, Угол=pi/2
        phasors = np.array([[1+0j, 0+1j]]) # (1 time step, 2 channels)
        
        feats = calculate_polar_features(phasors, reference_phasor=None)
        # Expected: [1, 0, 1, pi/2]
        expected = np.array([[1.0, 0.0, 1.0, np.pi/2]])
        
        assert np.allclose(feats, expected)

    def test_calculate_polar_features_relative(self):
        # Канал 1: угол pi/2
        # Канал 2: угол pi
        # Опорный: угол pi/2
        # Результат 1: 0
        # Результат 2: pi/2
        
        phasors = np.array([[0+1j, -1+0j]])
        ref = np.array([0+1j]) # Angle pi/2
        
        feats = calculate_polar_features(phasors, reference_phasor=ref)
        
        # Модули равны 1 для обоих
        # Угол 1: pi/2 - pi/2 = 0
        # Угол 2: pi - pi/2 = pi/2
        expected = np.array([[1.0, 0.0, 1.0, np.pi/2]])
        
        assert np.allclose(feats, expected)

    def test_dataset_polar_mode(self):
        # Генерируем синтетические данные: UA = 100*cos(wt), IA = 5*cos(wt - pi/6) (отставание 30°)
        # Используем реалистичные амплитуды, чтобы пройти порог выбора опорного фазора (1.0)
        length = 100
        t = np.arange(length) / 1600.0
        f = 50.0
        
        ua = 100.0 * np.cos(2 * np.pi * f * t) # фаза 0
        ub = 100.0 * np.cos(2 * np.pi * f * t - 2*np.pi/3)
        uc = 100.0 * np.cos(2 * np.pi * f * t + 2*np.pi/3)
        
        ia = 5.0 * np.cos(2 * np.pi * f * t - np.pi/6) # фаза -30°
        ib = 5.0 * np.cos(2 * np.pi * f * t - np.pi/6 - 2*np.pi/3)
        ic = 5.0 * np.cos(2 * np.pi * f * t - np.pi/6 + 2*np.pi/3)
        
        data = {
            'IA': ia,
            'IB': ib,
            'IC': ic,
            'IN': np.zeros(length),
            'UA': ua,
            'UB': ub,
            'UC': uc,
            'UN': np.zeros(length),
            'target': np.zeros(length)
        }
        df = pl.DataFrame(data)
        
        ds = OscillogramDataset(
            df, 
            [0], 
            window_size=50, 
            feature_mode='polar',
            target_columns='target'
        )
        
        x, y = ds[0]
        # Форма x: (Каналы*2, Время) -> (16, 50)
        # Каналы: IA, IB, IC, In, UA, UB, UC, Un
        # Индексы:
        # IA: 0 (Модуль), 1 (Угол)
        # UA: 8 (Модуль), 9 (Угол)
        
        # Опорным должен быть UA (если его амплитуда превышает порог).
        # Важно: FFT делает окна с оконной функцией Hanning, что меняет амплитуду,
        # поэтому учитываем распределение валидных индексов.
        
        # Угол UA должен быть 0 (относительно самого себя).
        # Угол IA должен быть -pi/6 (относительно UA).
        
    # Проверим валидный индекс, где вычислен FFT
    # Размер окна выборки 50, размер окна FFT 32.
    # В `pdr_calculator.sliding_window_fft` результат привязан к концу окна,
    # поэтому индексы 0..31 содержат NaN, валидные — 32..49.
        
        valid_idx = 40
        ua_angle = x[9, valid_idx]
        ia_angle = x[1, valid_idx]


