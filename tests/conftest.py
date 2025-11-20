"""
Глобальные pytest fixtures и конфигурация для всех тестов.

Этот файл автоматически загружается pytest и делает доступными
fixtures для всех тестов в проекте.
"""

import pytest
import numpy as np
import sys
import os

# Добавляем корневую директорию в путь для импортов
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def test_data_dir():
    """Возвращает путь к папке с тестовыми данными."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def balanced_three_phase():
    """
    Fixture: сбалансированная трёхфазная система (phasor).
    
    Фаза A: амплитуда 1, угол 0°
    Фаза B: амплитуда 1, угол -120°
    Фаза C: амплитуда 1, угол +120°
    """
    return {
        "phasor_a": np.array([1.0 + 0.0j]),
        "phasor_b": np.array([np.exp(1j * -2 * np.pi / 3)]),
        "phasor_c": np.array([np.exp(1j * 2 * np.pi / 3)]),
    }


@pytest.fixture
def zero_sequence_three_phase():
    """
    Fixture: система с только нулевой последовательностью.
    Все три фазы идентичны.
    """
    phasor = np.array([1.0 + 0.5j, 2.0 - 1.0j])
    return {
        "phasor_a": phasor,
        "phasor_b": phasor,
        "phasor_c": phasor,
    }


@pytest.fixture
def negative_sequence_three_phase():
    """
    Fixture: система с только обратной последовательностью.
    
    Фаза A: амплитуда 0.5, угол 0°
    Фаза B: амплитуда 0.5, угол +120° (обратный порядок!)
    Фаза C: амплитуда 0.5, угол -120°
    """
    return {
        "phasor_a": np.array([0.5 + 0.0j]),
        "phasor_b": np.array([0.5 * np.exp(1j * 2 * np.pi / 3)]),
        "phasor_c": np.array([0.5 * np.exp(1j * -2 * np.pi / 3)]),
    }


@pytest.fixture
def multi_window_signal():
    """
    Fixture: многоточечный сигнал для тестирования FFT функций.
    
    Генерирует синусоиду 50 Гц при частоте дискретизации 1600 Гц.
    """
    fs = 1600  # Частота дискретизации
    f_fundamental = 50  # Основная частота
    duration = 1.0  # Длительность в секундах
    
    t = np.arange(0, duration, 1 / fs)
    signal = np.sin(2 * np.pi * f_fundamental * t)
    
    return {
        "signal": signal,
        "fs": fs,
        "frequency": f_fundamental,
        "t": t,
    }


def pytest_configure(config):
    """Конфигурация pytest. Добавляем пользовательские маркеры."""
    config.addinivalue_line(
        "markers",
        "unit: быстрые unit-тесты без файловых операций"
    )
    config.addinivalue_line(
        "markers",
        "integration: интеграционные тесты с файловыми операциями"
    )
    config.addinivalue_line(
        "markers",
        "slow: медленные тесты"
    )
