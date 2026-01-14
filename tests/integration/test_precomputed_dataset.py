"""Тестовый скрипт для проверки PrecomputedDataset."""
import sys
sys.path.insert(0, '.')

from osc_tools.ml.precomputed_dataset import create_precomputed_dataset
import torch

# Создаём dataset
ds = create_precomputed_dataset(
    'data/ml_datasets',
    window_size=320,
    feature_mode='phase_polar',
    sampling_strategy='snapshot'
)
print(f'Dataset length: {len(ds)}')

# Проверяем один элемент
x, y = ds[0]
print(f'X shape: {x.shape}')
print(f'Y shape: {y.shape}')
print(f'Y values: {y}')

# Проверяем symmetric режим
ds_sym = create_precomputed_dataset(
    'data/ml_datasets',
    window_size=320,
    feature_mode='symmetric',
    sampling_strategy='snapshot'
)
x_sym, y_sym = ds_sym[0]
print(f'Symmetric X shape: {x_sym.shape}')

# Проверяем raw режим
ds_raw = create_precomputed_dataset(
    'data/ml_datasets',
    window_size=320,
    feature_mode='raw',
    sampling_strategy='snapshot'
)
x_raw, y_raw = ds_raw[0]
print(f'Raw X shape: {x_raw.shape}')

print('SUCCESS: Все тесты пройдены!')
