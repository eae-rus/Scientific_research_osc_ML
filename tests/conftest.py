"""
Глобальные pytest fixtures и конфигурация для всех тестов.

Этот файл автоматически загружается pytest и делает доступными
fixtures для всех тестов в проекте.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

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


@pytest.fixture
def sample_normalized_dataframe():
    """
    Fixture: DataFrame с нормализованными 3-фазными сигналами.
    Используется для тестирования preprocessing, features, analysis модулей.
    
    Содержит:
    - 1600 samples (1 сек при fs=1600 Гц)
    - 6 напряжений (1Ua, 1Ub, 1Uc, 2Ua, 2Ub, 2Uc)
    - 6 токов (1Ia, 1Ib, 1Ic, 2Ia, 2Ib, 2Ic)
    - Синусоидальные сигналы 50 Гц с нормализацией ±1.0
    """
    fs = 1600
    f = 50
    duration = 1.0
    samples = int(fs * duration)
    
    t = np.arange(samples) / fs
    
    # Генерируем базовые 3-фазные сигналы
    phase_a = np.sin(2 * np.pi * f * t)
    phase_b = np.sin(2 * np.pi * f * t - 2 * np.pi / 3)
    phase_c = np.sin(2 * np.pi * f * t + 2 * np.pi / 3)
    
    # Создаём DataFrame с двумя трансформаторами (Bus 1, Bus 2)
    data = {
        # Напряжения шины 1
        '1Ua': phase_a * 230,
        '1Ub': phase_b * 230,
        '1Uc': phase_c * 230,
        # Токи шины 1
        '1Ia': phase_a * 10,
        '1Ib': phase_b * 10,
        '1Ic': phase_c * 10,
        # Напряжения шины 2
        '2Ua': phase_a * 110,
        '2Ub': phase_b * 110,
        '2Uc': phase_c * 110,
        # Токи шины 2
        '2Ia': phase_a * 20,
        '2Ib': phase_b * 20,
        '2Ic': phase_c * 20,
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_signal_with_event():
    """
    Fixture: DataFrame с сигналом содержащим событие (например замыкание).
    
    Первая половина (0-0.05s): нормальный режим
    Вторая половина (0.05-0.1s): событие с повышенными токами
    """
    fs = 1600
    f = 50
    samples_normal = 800  # 0.5 сек
    samples_event = 800   # 0.5 сек
    total_samples = samples_normal + samples_event
    
    t = np.arange(total_samples) / fs
    
    # Нормальный режим
    phase_a_normal = np.sin(2 * np.pi * f * t[:samples_normal])
    phase_b_normal = np.sin(2 * np.pi * f * t[:samples_normal] - 2*np.pi/3)
    phase_c_normal = np.sin(2 * np.pi * f * t[:samples_normal] + 2*np.pi/3)
    
    # Режим события (повышенные токи)
    phase_a_event = np.sin(2 * np.pi * f * t[samples_normal:])
    phase_b_event = np.sin(2 * np.pi * f * t[samples_normal:] - 2*np.pi/3)
    phase_c_event = np.sin(2 * np.pi * f * t[samples_normal:] + 2*np.pi/3)
    
    data = {
        # Напряжения (остаются примерно нормальными)
        '1Ua': np.concatenate([phase_a_normal * 230, phase_a_event * 200]),
        '1Ub': np.concatenate([phase_b_normal * 230, phase_b_event * 200]),
        '1Uc': np.concatenate([phase_c_normal * 230, phase_c_event * 200]),
        # Токи (резко возрастают при событии)
        '1Ia': np.concatenate([phase_a_normal * 10, phase_a_event * 50]),
        '1Ib': np.concatenate([phase_b_normal * 10, phase_b_event * 50]),
        '1Ic': np.concatenate([phase_c_normal * 10, phase_c_event * 50]),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def fixture_comtrade_dir(tmp_path):
    """
    Fixture: создаёт временную директорию с mock COMTRADE файлами.
    
    Возвращает Path к директории с test.cfg и test.dat файлами.
    Используется для тестирования модулей с файловыми зависимостями.
    """
    fixture_dir = tmp_path / "comtrade_fixture"
    fixture_dir.mkdir(exist_ok=True)
    
    # Создаём минимальный CFG файл COMTRADE
    cfg_content = """Station Name,Substation Name
,
1,6A,0,0
1Ua,V,,1,0,0,0,0,0,0,0,0
1Ub,V,,1,0,0,0,0,0,0,0,0
1Uc,V,,1,0,0,0,0,0,0,0,0
1Ia,A,,1,0,0,0,0,0,0,0,0
1Ib,A,,1,0,0,0,0,0,0,0,0
1Ic,A,,1,0,0,0,0,0,0,0,0
60
1600
0
2025/12/23,00:00:00.000
2025/12/23,00:00:00.100
"""
    cfg_path = fixture_dir / "test.cfg"
    cfg_path.write_text(cfg_content, encoding='utf-8')
    
    # Создаём минимальный DAT файл (бинарный формат COMTRADE)
    # Структура: 16-битные целые числа для каждого сигнала
    dat_content = np.zeros((160, 6), dtype=np.int16)  # 160 samples, 6 channels
    # Добавляем синтетические данные для трёхфазной системы (напряжение + ток)
    fs = 1600
    f = 50
    t = np.arange(0, 0.1, 1/fs)
    
    # Напряжения (U_a, U_b, U_c)
    dat_content[:len(t), 0] = (1000 * np.sin(2*np.pi*f*t)).astype(np.int16)
    dat_content[:len(t), 1] = (1000 * np.sin(2*np.pi*f*t - 2*np.pi/3)).astype(np.int16)
    dat_content[:len(t), 2] = (1000 * np.sin(2*np.pi*f*t + 2*np.pi/3)).astype(np.int16)
    
    # Токи (I_a, I_b, I_c)
    dat_content[:len(t), 3] = (100 * np.sin(2*np.pi*f*t)).astype(np.int16)
    dat_content[:len(t), 4] = (100 * np.sin(2*np.pi*f*t - 2*np.pi/3)).astype(np.int16)
    dat_content[:len(t), 5] = (100 * np.sin(2*np.pi*f*t + 2*np.pi/3)).astype(np.int16)
    
    dat_path = fixture_dir / "test.dat"
    dat_content.tofile(str(dat_path))
    
    return fixture_dir


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
