import sys
from pathlib import Path
import numpy as np

# Добавляем корень проекта в путь импорта для тестов
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from osc_tools.features.pdr_calculator import sliding_window_fft


def test_sliding_window_fft_shape():
    """Проверка: sliding_window_fft возвращает массив нужной формы."""
    signal = np.random.randn(100)
    window_size = 16
    num_harmonics = 3

    result = sliding_window_fft(signal, window_size, num_harmonics)
    assert result.shape[1] == num_harmonics
    assert result.shape[0] == len(signal)


def test_sliding_window_fft_values_not_all_nan():
    """Простой sanity-check: при достаточной длине хотя бы часть результата не NaN."""
    signal = np.random.randn(64)
    window_size = 16
    num_harmonics = 2

    result = sliding_window_fft(signal, window_size, num_harmonics)
    assert not np.all(np.isnan(result))
