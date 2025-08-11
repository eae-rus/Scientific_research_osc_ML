import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

from osc_tools.preprocessing.filtering import EmptyOscFilter

@pytest.fixture
def empty_filter_instance() -> EmptyOscFilter:
    """
    Создает экземпляр EmptyOscFilter с базовой конфигурацией для тестов.
    Зависимости, связанные с файловым вводом-выводом, не инициализируются,
    так как мы тестируем только метод is_oscillogram_not_empty.
    """
    config = {
        'channels_to_analyze_patterns': ['i |', 'u |'],
        'current_channel_id_patterns': ['i |'],
        'voltage_channel_id_patterns': ['u |'],
        'use_norm_osc': False,
        'raw_signal_analysis': {
            'initial_window_check_periods': 1,
            'h1_vs_hx_ratio_threshold_U': 10,
            'h1_vs_hx_ratio_threshold_I': 10, # Порог для "чистого" сигнала
            'thresholds_raw_current_relative': {'delta': 0.1, 'std_dev': 0.05},
            'thresholds_raw_voltage_relative': {'delta': 0.1, 'std_dev': 0.05},
        },
        'verbose': False
    }
    # Мы не передаем реальные пути, так как не будем вызывать run_filter
    filter_instance = EmptyOscFilter(
        comtrade_root_path="",
        output_csv_path="",
        config=config
    )
    # Устанавливаем вручную, так как это делается в run_filter
    filter_instance.fft_window_size = 32
    return filter_instance


def create_test_signal(
    fft_window_size=32, num_periods=10,
    amplitude=1.0, frequency=1.0,
    event_amplitude_change=0.0, event_start_period=5
) -> pd.DataFrame:
    """Генерирует тестовый сигнал."""
    total_points = fft_window_size * num_periods
    t = np.linspace(0, num_periods * 2 * np.pi, total_points)
    signal = amplitude * np.sin(frequency * t)

    # Добавляем событие (изменение амплитуды)
    event_start_index = fft_window_size * event_start_period
    signal[event_start_index:] *= (1 + event_amplitude_change)

    return pd.DataFrame({'U | Test Channel': signal})


def test_is_oscillogram_not_empty_detects_active_signal(empty_filter_instance):
    """
    Проверяет, что сигнал с явным событием (скачком амплитуды)
    распознается как "непустой".
    """
    # Сигнал со скачком амплитуды на 50%
    active_df = create_test_signal(
        fft_window_size=32,
        amplitude=100,
        event_amplitude_change=0.5
    )

    result = empty_filter_instance.is_oscillogram_not_empty(active_df, file_name_for_norm="test_active")

    assert result is True

def test_is_oscillogram_not_empty_ignores_inactive_signal(empty_filter_instance):
    """
    Проверяет, что идеальная синусоида без изменений
    распознается как "пустая" (неактивная).
    """
    # Идеальная синусоида без изменений
    inactive_df = create_test_signal(
        fft_window_size=32,
        amplitude=100,
        event_amplitude_change=0.0
    )

    result = empty_filter_instance.is_oscillogram_not_empty(inactive_df, file_name_for_norm="test_inactive")

    assert result is False

def test_is_oscillogram_not_empty_ignores_noisy_signal(empty_filter_instance):
    """
    Проверяет, что очень шумный сигнал, который не проходит
    проверку на "чистоту", распознается как "пустой".
    """
    # Создаем в основном случайный шум
    np.random.seed(42)
    total_points = 32 * 10
    noise = np.random.randn(total_points) * 50
    # Добавляем небольшой синус, чтобы было что-то похожее на сигнал
    t = np.linspace(0, 10 * 2 * np.pi, total_points)
    signal = 10 * np.sin(t) + noise

    noisy_df = pd.DataFrame({'U | Test Channel': signal})

    result = empty_filter_instance.is_oscillogram_not_empty(noisy_df, file_name_for_norm="test_noisy")

    assert result is False
