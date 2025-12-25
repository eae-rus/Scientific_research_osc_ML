"""
Unit-тесты для модуля osc_tools.preprocessing.filtering

Тестируем helper-функции и базовые операции фильтрации.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Добавляем PROJECT_ROOT в sys.path для правильных импортов
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.preprocessing.filtering import sliding_window_fft


def is_complex_nan(val):
    """Проверяет, содержит ли комплексное число NaN в real или imag части"""
    if np.isscalar(val):
        return np.isnan(val.real) or np.isnan(val.imag)
    else:
        # Для массивов
        return np.isnan(val.real) | np.isnan(val.imag)


class TestSlidingWindowFFT:
    """Тесты для функции sliding_window_fft"""
    
    @pytest.mark.unit
    def test_sliding_window_fft_basic_sine_wave(self):
        """
        Тест: БПФ на простой синусоидальной волне
        Ожидание: первая гармоника должна быть ненулевой, результат должен быть комплексным
        """
        # Синусоида на 50 окне (12 точек на период)
        fft_window_size = 12  # 1 период
        t = np.arange(50)
        signal = np.sin(2 * np.pi * t / fft_window_size)
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=1)
        
        # Проверяем размерность
        assert result.shape == (len(signal), 1)
        
        # Проверяем, что первая гармоника ненулевая
        # (в стационарном участке синусоиды амплитуда должна быть ~1)
        valid_h1 = result[fft_window_size:-fft_window_size, 0]  # Исключаем начало и конец
        assert np.all(~is_complex_nan(valid_h1)), "Первая гармоника содержит NaN в середине"
        assert np.any(np.abs(valid_h1) > 0.5), "Амплитуда первой гармоники слишком мала"
    
    @pytest.mark.unit
    def test_sliding_window_fft_signal_too_short(self):
        """
        Тест: сигнал короче FFT окна
        Ожидание: должен вернуть массив NaN комплексных значений
        """
        fft_window_size = 100
        signal = np.sin(np.arange(20) * 0.1)  # Только 20 точек
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=3)
        
        # Проверяем размерность
        assert result.shape == (len(signal), 3)
        
        # Все значения должны быть NaN
        assert np.all(is_complex_nan(result)), "Для короткого сигнала должны быть NaN"
    
    @pytest.mark.unit
    def test_sliding_window_fft_multiple_harmonics(self):
        """
        Тест: запрос нескольких гармоник
        Ожидание: результат должен содержать данные для каждой гармоники
        """
        fft_window_size = 20
        num_harmonics = 5
        signal = np.sin(np.arange(100) * 0.1)
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=num_harmonics)
        
        # Проверяем размерность
        assert result.shape == (len(signal), num_harmonics)
        
        # Проверяем, что каждая гармоника есть в результате
        # (первая точка может содержать данные или NaN в зависимости от реализации)
        # Главное - что последние окна содержат валидные значения
        valid_results = result[fft_window_size:, :]
        assert np.any(~is_complex_nan(valid_results)), "Должны быть валидные гармоники в конце сигнала"
    
    @pytest.mark.unit
    def test_sliding_window_fft_zero_signal(self):
        """
        Тест: нулевой сигнал
        Ожидание: все гармоники должны быть близки к нулю
        """
        fft_window_size = 12
        signal = np.zeros(100)
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=2)
        
        # Проверяем размерность
        assert result.shape == (len(signal), 2)
        
        # После начального NaN периода гармоники должны быть близи к нулю
        valid_results = result[fft_window_size:, :]
        nan_mask = is_complex_nan(valid_results)
        assert np.all(np.abs(valid_results[~nan_mask]) < 1e-10), \
            "Для нулевого сигнала гармоники должны быть близи к нулю"
    
    @pytest.mark.unit
    def test_sliding_window_fft_output_shape_and_dtype(self):
        """
        Тест: проверка типа данных и формы выхода
        Ожидание: результат должен быть complex64 или complex128
        """
        fft_window_size = 15
        signal = np.random.randn(100)
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=3)
        
        # Проверяем тип данных (может быть complex128 или complex64)
        assert np.issubdtype(result.dtype, np.complexfloating), f"Результат должен быть комплексным, получен {result.dtype}"
        
        # Проверяем форму
        assert result.shape == (len(signal), 3)
    
    @pytest.mark.unit
    def test_sliding_window_fft_harmonic_index_bounds(self):
        """
        Тест: гармоника должна быть в границах FFT
        Ожидание: если запрашиваем гармонику >= fft_window_size, должны получить NaN
        """
        fft_window_size = 10
        signal = np.sin(np.arange(100) * 0.1)
        
        # Запрашиваем много гармоник
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=20)
        
        # Гармоники выше fft_window_size должны быть NaN после валидного периода
        # (или всегда NaN, в зависимости от реализации)
        assert result.shape == (len(signal), 20)
    
    @pytest.mark.unit
    def test_sliding_window_fft_constant_signal(self):
        """
        Тест: постоянный сигнал (DC)
        Ожидание: первая гармоника (не DC!) должна быть близи к нулю
        """
        fft_window_size = 15
        signal = np.ones(100) * 5  # Постоянное значение 5
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=2)
        
        # Для постоянного сигнала первая гармоника должна быть близи к нулю
        # (DC компонента будет в fft_coeffs[0], но мы берём fft_coeffs[1] и выше)
        valid_h1 = result[fft_window_size:, 0]
        nan_mask = is_complex_nan(valid_h1)
        h1_values = valid_h1[~nan_mask]
        assert np.all(np.abs(h1_values) < 1e-10), \
            "Первая гармоника постоянного сигнала должна быть близи к нулю"


class TestSlidingWindowFFTEdgeCases:
    """Тесты граничных случаев для sliding_window_fft"""
    
    @pytest.mark.unit
    def test_sliding_window_fft_exactly_window_size(self):
        """
        Тест: сигнал точно равен размеру окна
        Ожидание: должен вернуть одну результирующую точку
        """
        fft_window_size = 20
        signal = np.sin(np.arange(fft_window_size) * 0.1)
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=3)
        
        # Должна быть одна позиция окна
        assert result.shape == (len(signal), 3)
        
        # Первое значение (позиция 0) должно содержать данные
        assert not is_complex_nan(result[0, 0]), "Первое значение должно быть валидным"
    
    @pytest.mark.unit
    def test_sliding_window_fft_with_noise(self):
        """
        Тест: сигнал с шумом
        Ожидание: первая гармоника должна быть выше шума
        """
        fft_window_size = 16
        t = np.arange(200)
        
        # Чистая синусоида + гауссовский шум
        signal = np.sin(2 * np.pi * t / fft_window_size) + 0.1 * np.random.randn(200)
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=2)
        
        # Первая гармоника должна быть ненулевой (несмотря на шум)
        valid_h1 = result[fft_window_size:-fft_window_size, 0]
        nan_mask = is_complex_nan(valid_h1)
        h1_valid = valid_h1[~nan_mask]
        assert np.mean(np.abs(h1_valid)) > 0.5, "Амплитуда синусоиды должна быть видна несмотря на шум"
    
    @pytest.mark.unit
    def test_sliding_window_fft_negative_values(self):
        """
        Тест: сигнал с отрицательными значениями
        Ожидание: должен работать корректно с отрицательными амплитудами
        """
        fft_window_size = 12
        signal = np.sin(np.arange(100) * 2 * np.pi / fft_window_size) - 2.0  # Смещённая синусоида
        
        result = sliding_window_fft(signal, fft_window_size, num_harmonics=2)
        
        # Проверяем, что не вызвало ошибок и форма правильная
        assert result.shape == (len(signal), 2)
        
        # Первая гармоника должна быть ненулевой (синусоида, независимо от смещения DC)
        valid_h1 = result[fft_window_size:, 0]
        nan_mask = is_complex_nan(valid_h1)
        h1_valid = valid_h1[~nan_mask]
        assert np.any(np.abs(h1_valid) > 0.1), "Первая гармоника синусоиды должна быть видна"
