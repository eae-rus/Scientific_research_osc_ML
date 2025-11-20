"""
Factory-функции для создания тестовых данных.

Этот модуль содержит функции для синтезирования различных тестовых сигналов,
фазоров и других данных, необходимых для тестирования без привязки к реальным файлам.
"""

import numpy as np
from typing import Dict, Tuple


def create_sinusoidal_signal(
    frequency: float = 50.0,
    sampling_rate: float = 1600.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Создаёт синусоидальный сигнал.
    
    Args:
        frequency: Частота сигнала (Гц)
        sampling_rate: Частота дискретизации (Гц)
        duration: Длительность сигнала (сек)
        amplitude: Амплитуда
        phase: Начальная фаза (рад)
    
    Returns:
        np.ndarray: Синусоидальный сигнал
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal


def create_three_phase_balanced_signal(
    frequency: float = 50.0,
    sampling_rate: float = 1600.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Создаёт сбалансированную трёхфазную систему сигналов.
    
    Args:
        frequency: Основная частота (Гц)
        sampling_rate: Частота дискретизации (Гц)
        duration: Длительность (сек)
        amplitude: Амплитуда каждой фазы
    
    Returns:
        Dict с ключами 'a', 'b', 'c' (три фазы со сдвигом 120°)
    """
    phase_shift = 2 * np.pi / 3
    
    return {
        "a": create_sinusoidal_signal(frequency, sampling_rate, duration, amplitude, 0),
        "b": create_sinusoidal_signal(frequency, sampling_rate, duration, amplitude, -phase_shift),
        "c": create_sinusoidal_signal(frequency, sampling_rate, duration, amplitude, phase_shift),
    }


def create_harmonics_signal(
    fundamental: float = 50.0,
    sampling_rate: float = 1600.0,
    duration: float = 1.0,
    harmonic_orders: list = None,
    amplitudes: list = None,
) -> np.ndarray:
    """
    Создаёт сигнал с указанными гармониками.
    
    Args:
        fundamental: Основная частота (Гц)
        sampling_rate: Частота дискретизации (Гц)
        duration: Длительность (сек)
        harmonic_orders: Список порядков гармоник (по умолчанию [1, 3, 5])
        amplitudes: Амплитуды гармоник (по умолчанию одинаковые)
    
    Returns:
        np.ndarray: Сложный сигнал с гармониками
    """
    if harmonic_orders is None:
        harmonic_orders = [1, 3, 5]
    
    if amplitudes is None:
        amplitudes = [1.0] + [1.0 / order for order in harmonic_orders[1:]]
    
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.zeros_like(t)
    
    for order, amplitude in zip(harmonic_orders, amplitudes):
        signal += amplitude * np.sin(2 * np.pi * fundamental * order * t)
    
    return signal


def create_phasor_balanced_system(
    magnitude: float = 1.0,
    phase_a_angle: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создаёт фазоры сбалансированной трёхфазной системы.
    
    Args:
        magnitude: Амплитуда фазоров
        phase_a_angle: Угол фазы A (рад)
    
    Returns:
        Tuple[phasor_a, phasor_b, phasor_c]: Три комплексных фазора
    """
    phase_shift = 2 * np.pi / 3
    
    phasor_a = magnitude * np.exp(1j * phase_a_angle)
    phasor_b = magnitude * np.exp(1j * (phase_a_angle - phase_shift))
    phasor_c = magnitude * np.exp(1j * (phase_a_angle + phase_shift))
    
    return np.array([phasor_a]), np.array([phasor_b]), np.array([phasor_c])


def create_phasor_zero_sequence(
    phasor: complex,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создаёт фазоры только с нулевой последовательностью
    (все три фазы одинаковые).
    
    Args:
        phasor: Один комплексный фазор (будет одинаков для всех трёх фаз)
    
    Returns:
        Tuple[phasor_a, phasor_b, phasor_c]: Три одинаковых фазора
    """
    phasor_array = np.array([phasor])
    return phasor_array, phasor_array, phasor_array


def create_noise(
    shape: Tuple[int, ...],
    amplitude: float = 0.1,
    seed: int = None,
) -> np.ndarray:
    """
    Создаёт гауссовский шум.
    
    Args:
        shape: Форма массива
        amplitude: Амплитуда шума
        seed: Seed для воспроизводимости
    
    Returns:
        np.ndarray: Массив шума
    """
    if seed is not None:
        np.random.seed(seed)
    
    return amplitude * np.random.randn(*shape)
