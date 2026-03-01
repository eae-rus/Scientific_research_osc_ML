import sys
from pathlib import Path
import polars as pl
import numpy as np
import torch

# Добавляем корень проекта в путь импорта для тестов
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.dataset import OscillogramDataset


def test_dataset_feature_modes_shapes():
    """Проверяем, что Dataset возвращает тензоры корректной формы для разных режимов признаков."""
    length = 1000
    t = np.linspace(0, 1, length)
    f = 50

    ia = np.sin(2*np.pi*f*t)
    ib = np.sin(2*np.pi*f*t - 2*np.pi/3)
    ic = np.sin(2*np.pi*f*t + 2*np.pi/3)

    ua = 220 * np.sin(2*np.pi*f*t + 0.1)
    ub = 220 * np.sin(2*np.pi*f*t - 2*np.pi/3 + 0.1)
    uc = 220 * np.sin(2*np.pi*f*t + 2*np.pi/3 + 0.1)

    df = pl.DataFrame({
        'IA': ia, 'IB': ib, 'IC': ic,
        'UA': ua, 'UB': ub, 'UC': uc,
        'target': np.zeros(length)
    })

    indices = [0]
    window_size = 640

    modes = ['raw', 'symmetric', 'complex_channels', 'power', 'instantaneous_power', 'alpha_beta']

    for mode in modes:
        ds = OscillogramDataset(
            dataframe=df,
            indices=indices,
            window_size=window_size,
            mode='classification',
            feature_mode=mode,
            target_columns='target'
        )
        x, y = ds[0]
        assert x is not None
        assert isinstance(x, torch.Tensor)
        assert x.shape[1] == window_size
        # Проверка на NaN
        assert not torch.isnan(x).any()
