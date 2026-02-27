import pytest
import torch
import numpy as np
import polars as pl
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.features.polar import calculate_polar_features_multiharmonic


class TestPolarFeatures:

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


class TestPolarFeaturesMultiharmonic:
    """Тесты для calculate_polar_features_multiharmonic — погармоничная коррекция угла."""
    
    def test_single_harmonic_absolute_and_relative(self):
        """Для 1 гармоники: абсолютные и относительные углы корректны."""
        time_steps = 10
        n_channels = 2
        
        # Канал 0: 1+0j (mag=1, angle=0), Канал 1: 0+1j (mag=1, angle=pi/2)
        phasors_3d = np.array([[[1+0j], [0+1j]]] * time_steps)  # (T, 2, 1)
        
        # Без опорного — абсолютные углы
        result_abs = calculate_polar_features_multiharmonic(phasors_3d, ref_signal=None)
        # [ch0_h1_mag, ch0_h1_angle, ch1_h1_mag, ch1_h1_angle]
        assert np.allclose(result_abs[:, 0], 1.0)   # mag ch0
        assert np.allclose(result_abs[:, 1], 0.0)   # angle ch0 = 0
        assert np.allclose(result_abs[:, 2], 1.0)   # mag ch1
        assert np.allclose(result_abs[:, 3], np.pi/2)  # angle ch1 = pi/2
        
        # С опорным angle=pi/2 -> относительный угол ch0: 0 - pi/2 = -pi/2, ch1: pi/2 - pi/2 = 0
        ref_signal = np.array([[0+1j]] * time_steps)  # (T, 1)
        result_rel = calculate_polar_features_multiharmonic(phasors_3d, ref_signal)
        assert np.allclose(result_rel[:, 1], -np.pi/2)  # ch0 relative angle
        assert np.allclose(result_rel[:, 3], 0.0)       # ch1 relative angle
    
    def test_multiharmonic_angles_are_stable(self):
        """
        КЛЮЧЕВОЙ ТЕСТ: для чистого синусоидального сигнала с 3 гармониками
        относительные углы ВСЕХ гармоник должны быть стабильными (без осцилляций).
        
        Имитируем DFT-фазоры, вращающиеся с частотой k*ω₀.
        """
        time_steps = 64
        n_channels = 4
        n_harmonics = 3
        omega_0 = 2 * np.pi / 32  # Основная частота (период 32 отсчёта)
        
        # Генерируем "DFT-фазоры" для каждого канала и гармоники
        # Фазор k-й гармоники вращается со скоростью k*ω₀
        t = np.arange(time_steps)
        
        complex_features = np.zeros((time_steps, n_channels, n_harmonics), dtype=np.complex128)
        ref_signal = np.zeros((time_steps, n_harmonics), dtype=np.complex128)
        
        # Фиксированные фазовые сдвиги каналов (то что должно остаться после коррекции)
        channel_phases = [0.0, -np.pi/6, np.pi/3, -np.pi/2]
        ref_phase = 0.2  # Фаза опорного сигнала
        
        for h in range(n_harmonics):
            k = h + 1  # Номер гармоники (1-based)
            # Опорный сигнал: амплитуда 1.0, вращение k*ω₀
            ref_signal[:, h] = 1.0 * np.exp(1j * (k * omega_0 * t + ref_phase * k))
            
            for ch in range(n_channels):
                # Канал: амплитуда зависит от k, вращение k*ω₀, своя начальная фаза
                amp = 1.0 / k
                complex_features[:, ch, h] = amp * np.exp(
                    1j * (k * omega_0 * t + channel_phases[ch])
                )
        
        # Вычисляем полярные координаты
        result = calculate_polar_features_multiharmonic(complex_features, ref_signal)
        
        # Проверяем, что углы стабильны во времени
        for ch in range(n_channels):
            for h in range(n_harmonics):
                angle_idx = (ch * n_harmonics + h) * 2 + 1
                angles = result[:, angle_idx]
                
                # Угол должен быть постоянным (std ≈ 0)
                angle_std = np.std(angles)
                assert angle_std < 1e-10, (
                    f"Канал {ch}, гармоника {h+1}: std угла = {angle_std:.2e} "
                    f"(ожидается ≈ 0). Углы осциллируют!"
                )
    
    def test_synthetic_ref_fallback(self):
        """
        Тест фоллбека на синтетический опорный угол (k * angle_h1),
        когда k-я гармоника опорного сигнала слишком мала.
        """
        time_steps = 64
        n_channels = 2
        n_harmonics = 2
        omega_0 = 2 * np.pi / 32
        t = np.arange(time_steps)
        
        complex_features = np.zeros((time_steps, n_channels, n_harmonics), dtype=np.complex128)
        ref_signal = np.zeros((time_steps, n_harmonics), dtype=np.complex128)
        
        # Опорный: 1-я гармоника имеет нормальную амплитуду
        ref_signal[:, 0] = 1.0 * np.exp(1j * omega_0 * t)
        # Опорный: 2-я гармоника НУЛЕВАЯ (ниже порога ref_mag_threshold)
        ref_signal[:, 1] = 1e-10 * np.exp(1j * 2 * omega_0 * t)
        
        # Каналы имеют нормальные 2-е гармоники
        for ch in range(n_channels):
            complex_features[:, ch, 0] = 1.0 * np.exp(1j * (omega_0 * t + ch * 0.5))
            complex_features[:, ch, 1] = 0.5 * np.exp(1j * (2 * omega_0 * t + ch * 0.3))
        
        result = calculate_polar_features_multiharmonic(
            complex_features, ref_signal, ref_mag_threshold=1e-6
        )
        
        # Углы 2-й гармоники должны быть стабильны (фоллбек на 2*angle_h1)
        for ch in range(n_channels):
            angle_idx = (ch * n_harmonics + 1) * 2 + 1  # h=1 -> 2-я гармоника
            angles = result[:, angle_idx]
            angle_std = np.std(angles)
            assert angle_std < 1e-10, (
                f"Канал {ch}, 2-я гармоника (фоллбек): std угла = {angle_std:.2e}. "
                f"Синтетический опорный угол не устранил осцилляцию."
            )

    def test_output_shape(self):
        """Проверка формы выходного массива."""
        time_steps = 20
        n_channels = 8
        n_harmonics = 3
        
        phasors = np.random.randn(time_steps, n_channels, n_harmonics) + \
                  1j * np.random.randn(time_steps, n_channels, n_harmonics)
        ref = np.random.randn(time_steps, n_harmonics) + \
              1j * np.random.randn(time_steps, n_harmonics)
        
        result = calculate_polar_features_multiharmonic(phasors, ref)
        
        expected_cols = n_channels * n_harmonics * 2  # mag + angle
        assert result.shape == (time_steps, expected_cols)
        assert result.dtype == np.float32

    def test_no_reference(self):
        """Без опорного сигнала — абсолютные углы."""
        time_steps = 10
        n_channels = 4
        n_harmonics = 2
        
        phasors = np.array([[[1+0j, 0+1j]] * n_channels] * time_steps)
        
        result = calculate_polar_features_multiharmonic(phasors, ref_signal=None)
        
        # Mag h1 = 1, Angle h1 = 0; Mag h2 = 1, Angle h2 = pi/2
        for ch in range(n_channels):
            mag_h1_idx = (ch * n_harmonics + 0) * 2
            angle_h1_idx = mag_h1_idx + 1
            mag_h2_idx = (ch * n_harmonics + 1) * 2
            angle_h2_idx = mag_h2_idx + 1
            
            assert np.allclose(result[:, mag_h1_idx], 1.0)
            assert np.allclose(result[:, angle_h1_idx], 0.0)
            assert np.allclose(result[:, mag_h2_idx], 1.0)
            assert np.allclose(result[:, angle_h2_idx], np.pi/2)
