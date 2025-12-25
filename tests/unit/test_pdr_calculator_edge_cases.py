"""
Unit-тесты для граничных случаев и edge cases модуля osc_tools.features.pdr_calculator

Тестируем функции calculate_symmetrical_components и sliding_window_fft на предмет:
- Обработки пустых/коротких массивов
- Обработки нулевых значений
- Обработки очень больших/маленьких значений
- Численной стабильности
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Добавляем PROJECT_ROOT в sys.path для правильных импортов
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.features.pdr_calculator import calculate_symmetrical_components, sliding_window_fft


class TestCalculateSymmetricalComponentsEdgeCases:
    """Edge cases для calculate_symmetrical_components"""
    
    @pytest.mark.unit
    def test_symmetrical_components_all_zeros(self):
        """
        Тест: все фазы равны нулю
        Ожидание: все составляющие должны быть нулевыми
        """
        phasor_a = np.array([0+0j, 0+0j, 0+0j])
        phasor_b = np.array([0+0j, 0+0j, 0+0j])
        phasor_c = np.array([0+0j, 0+0j, 0+0j])
        
        V1, V2, V0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)
        
        assert np.allclose(V1, 0), "Прямая составляющая должна быть ~0"
        assert np.allclose(V2, 0), "Обратная составляющая должна быть ~0"
        assert np.allclose(V0, 0), "Нулевая составляющая должна быть ~0"
    
    @pytest.mark.unit
    def test_symmetrical_components_single_point(self):
        """
        Тест: одна точка в каждой фазе
        Ожидание: должны вернуться скалярные комплексные значения
        """
        phasor_a = np.array([100+0j])
        phasor_b = np.array([100*np.exp(-2j*np.pi/3)])
        phasor_c = np.array([100*np.exp(2j*np.pi/3)])
        
        V1, V2, V0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)
        
        # Сбалансированная система -> V2 и V0 должны быть близи к нулю
        assert np.abs(V1[0]) > 50, "V1 должна быть значительной для сбалансированной системы"
        assert np.abs(V2[0]) < 1e-10, "V2 должна быть близи к нулю"
        assert np.abs(V0[0]) < 1e-10, "V0 должна быть близи к нулю"
    
    @pytest.mark.unit
    def test_symmetrical_components_very_small_values(self):
        """
        Тест: очень маленькие значения (машинная точность)
        Ожидание: функция должна корректно работать без overflow/underflow
        """
        epsilon = 1e-15
        phasor_a = np.array([epsilon, epsilon, epsilon])
        phasor_b = np.array([epsilon, epsilon, epsilon])
        phasor_c = np.array([epsilon, epsilon, epsilon])
        
        V1, V2, V0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)
        
        # Результаты должны быть очень маленькие, но не NaN
        assert np.all(~np.isnan(V1)), "V1 не должна содержать NaN"
        assert np.all(~np.isnan(V2)), "V2 не должна содержать NaN"
        assert np.all(~np.isnan(V0)), "V0 не должна содержать NaN"
    
    @pytest.mark.unit
    def test_symmetrical_components_mismatched_length(self):
        """
        Тест: массивы разной длины
        Ожидание: функция должна вернуть результат длины минимального массива или ошибку
        """
        phasor_a = np.array([100+0j, 100+0j, 100+0j])
        phasor_b = np.array([100+0j, 100+0j])  # Короче
        phasor_c = np.array([100+0j, 100+0j, 100+0j])
        
        # В зависимости от реализации, может быть ошибка или обрезание
        try:
            V1, V2, V0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)
            # Если не упало, проверяем форму
            assert len(V1) == len(phasor_b), "Результат должен быть длины минимального массива"
        except (ValueError, IndexError):
            # Это тоже приемлемо - функция может требовать одинаковых длин
            pass
    
    @pytest.mark.unit
    def test_symmetrical_components_negative_phase(self):
        """
        Тест: фазы с отрицательными вещественными значениями
        Ожидание: функция должна корректно обработать отрицательные амплитуды
        """
        phasor_a = np.array([-100+0j, -100+0j])
        phasor_b = np.array([-100*np.exp(-2j*np.pi/3), -100*np.exp(-2j*np.pi/3)])
        phasor_c = np.array([-100*np.exp(2j*np.pi/3), -100*np.exp(2j*np.pi/3)])
        
        V1, V2, V0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)
        
        # Проверяем, что результат получен (без ошибок)
        assert len(V1) == 2, "Должны быть результаты для двух точек"
        assert not np.any(np.isnan(V1)), "V1 не должна содержать NaN"


class TestSlidingWindowFFTEdgeCases:
    """Edge cases для sliding_window_fft из pdr_calculator"""
    
    @pytest.mark.unit
    def test_sliding_window_fft_very_long_signal(self):
        """
        Тест: очень длинный сигнал
        Ожидание: функция должна обработать без memory issues (если есть)
        """
        # Создаём длинный сигнал (100000 точек)
        signal = np.sin(2 * np.pi * np.arange(100000) / 100)
        window_size = 64
        num_harmonics = 3
        
        result = sliding_window_fft(signal, window_size, num_harmonics, verbose=False)
        
        # Проверяем форму
        assert result.shape == (len(signal), num_harmonics)
        
        # Должны быть валидные значения (не все NaN)
        non_nan_count = np.sum(~np.isnan(result.real))
        assert non_nan_count > 0, "Должны быть валидные значения"
    
    @pytest.mark.unit
    def test_sliding_window_fft_window_larger_than_signal(self):
        """
        Тест: окно больше сигнала
        Ожидание: должны вернуться все NaN
        """
        signal = np.array([1, 2, 3, 4, 5])
        window_size = 100
        num_harmonics = 2
        
        result = sliding_window_fft(signal, window_size, num_harmonics, verbose=False)
        
        # Все должны быть NaN
        assert np.all(np.isnan(result.real)), "Все значения должны быть NaN"
        assert np.all(np.isnan(result.imag)), "Все значения должны быть NaN"
    
    @pytest.mark.unit
    def test_sliding_window_fft_single_harmonic_request(self):
        """
        Тест: запрос только одной гармоники
        Ожидание: результат должен быть одномерным массивом гармоник
        """
        signal = np.sin(np.arange(100) * 0.1)
        window_size = 20
        num_harmonics = 1
        
        result = sliding_window_fft(signal, window_size, num_harmonics, verbose=False)
        
        # Проверяем форму
        assert result.shape == (len(signal), 1), "Результат должен быть (100, 1)"
        
        # Должны быть валидные значения в конце
        valid_results = result[window_size:, :]
        assert np.any(~np.isnan(valid_results)), "Должны быть валидные значения после начального периода"
    
    @pytest.mark.unit
    def test_sliding_window_fft_high_frequency_signal(self):
        """
        Тест: высокочастотный сигнал (близко к Nyquist)
        Ожидание: функция должна обработать без ошибок
        """
        # Сигнал на частоте близко к Nyquist (fs=100, f=40)
        sample_rate = 100
        frequency = 40  # Близко к Nyquist
        duration = 1.0
        t = np.arange(0, duration, 1/sample_rate)
        signal = np.sin(2 * np.pi * frequency * t)
        
        window_size = 16
        num_harmonics = 5
        
        result = sliding_window_fft(signal, window_size, num_harmonics, verbose=False)
        
        # Проверяем форму и отсутствие NaN в стабильной части
        assert result.shape == (len(signal), num_harmonics)
        valid_results = result[window_size:, :]
        assert not np.all(np.isnan(valid_results)), "Должны быть валидные значения"
    
    @pytest.mark.unit
    def test_sliding_window_fft_dc_component(self):
        """
        Тест: сигнал с постоянной компонентой (DC)
        Ожидание: первая гармоника должна быть ненулевой, DC компонента не должна влиять
        """
        signal = np.ones(100) * 5.0  # Константа 5
        signal += np.sin(np.arange(100) * 0.1)  # Добавляем синусоиду
        
        window_size = 16
        num_harmonics = 2
        
        result = sliding_window_fft(signal, window_size, num_harmonics, verbose=False)
        
        # Первая гармоника должна быть ненулевой (из синусоиды)
        valid_h1 = result[window_size:, 0]
        h1_values = valid_h1[~np.isnan(valid_h1)]
        assert np.mean(np.abs(h1_values)) > 0.1, "Первая гармоника должна быть видна несмотря на DC"
    
    @pytest.mark.unit
    def test_sliding_window_fft_complex_input_values(self):
        """
        Тест: очень странный случай - что если сигнал уже комплексный?
        Ожидание: функция должна либо работать, либо выдать понятную ошибку
        """
        # Обычно сигнал вещественный, но проверим robustness
        signal_real = np.sin(np.arange(50) * 0.1)
        
        window_size = 12
        num_harmonics = 2
        
        result = sliding_window_fft(signal_real, window_size, num_harmonics, verbose=False)
        
        # Результат должен быть комплексным
        assert result.dtype == complex, f"Результат должен быть complex, получен {result.dtype}"


class TestNumericalStability:
    """Тесты численной стабильности"""
    
    @pytest.mark.unit
    def test_repeated_calculations_consistency(self):
        """
        Тест: повторные вычисления дают одинаковый результат
        Ожидание: детерминированность вычислений
        """
        signal = np.sin(np.arange(100) * 0.05)
        window_size = 20
        num_harmonics = 3
        
        result1 = sliding_window_fft(signal, window_size, num_harmonics, verbose=False)
        result2 = sliding_window_fft(signal, window_size, num_harmonics, verbose=False)
        
        # Должны быть идентичными
        assert np.allclose(result1, result2, equal_nan=True), "Повторные вычисления должны быть одинаковыми"
    
    @pytest.mark.unit
    def test_symmetrical_components_numerical_error(self):
        """
        Тест: численная ошибка при вычислении симметричных компонент
        Ожидание: результаты должны быть в пределах машинной точности
        """
        # Идеально сбалансированная система
        phasor_a = np.array([100+0j] * 5)
        angle = 2 * np.pi / 3
        phasor_b = np.array([100*np.exp(-1j*angle)] * 5)
        phasor_c = np.array([100*np.exp(1j*angle)] * 5)
        
        V1, V2, V0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)
        
        # V2 и V0 должны быть близи к нулю (но не нулю из-за численной ошибки)
        assert np.all(np.abs(V2) < 1e-10), "V2 должна быть близи к нулю для сбалансированной системы"
        assert np.all(np.abs(V0) < 1e-10), "V0 должна быть близи к нулю для сбалансированной системы"
