import numpy as np
import polars as pl
import torch

from osc_tools.ml.precomputed_dataset import PrecomputedDataset


def _build_dummy_df(length: int = 8) -> pl.DataFrame:
    data = {
        'IA': np.zeros(length, dtype=np.float32),
        'IB': np.zeros(length, dtype=np.float32),
        'IC': np.zeros(length, dtype=np.float32),
        'IN': np.zeros(length, dtype=np.float32),
        'UA': np.zeros(length, dtype=np.float32),
        'UB': np.zeros(length, dtype=np.float32),
        'UC': np.zeros(length, dtype=np.float32),
        'UN': np.zeros(length, dtype=np.float32),
        'Target_A': np.zeros(length, dtype=np.float32),
        'Target_B': np.zeros(length, dtype=np.float32),
        'file_name': ['file_1'] * length
    }

    data['Target_A'][2] = 1.0
    return pl.DataFrame(data)


def test_precomputed_any_in_window_mode():
    df = _build_dummy_df()
    indices = [0]
    window_size = 4

    ds_any = PrecomputedDataset(
        dataframe=df,
        indices=indices,
        window_size=window_size,
        feature_mode='raw',
        target_columns=['Target_A', 'Target_B'],
        target_window_mode='any_in_window'
    )

    _, y_any = ds_any[0]
    assert torch.allclose(y_any, torch.tensor([1.0, 0.0]))

    ds_point = PrecomputedDataset(
        dataframe=df,
        indices=indices,
        window_size=window_size,
        feature_mode='raw',
        target_columns=['Target_A', 'Target_B'],
        target_window_mode='point'
    )

    _, y_point = ds_point[0]
    assert torch.allclose(y_point, torch.tensor([0.0, 0.0]))
