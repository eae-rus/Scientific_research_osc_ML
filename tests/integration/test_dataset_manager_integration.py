"""
Smoke test для проверки работы обновлённой системы датасетов.
Запускает SimpleMLP на 1 эпоху для быстрой проверки.
"""
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import torch
from osc_tools.data_management import DatasetManager
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.labels import get_target_columns


def test_smoke():
    """Быстрый smoke test системы датасетов."""
    print("=" * 60)
    print("SMOKE TEST: Проверка системы датасетов")
    print("=" * 60)
    
    DATA_DIR = ROOT_DIR / 'data' / 'ml_datasets'
    WINDOW_SIZE = 320
    BATCH_SIZE = 16
    
    # 1. Инициализация DatasetManager
    print("\n1. Инициализация DatasetManager...")
    dm = DatasetManager(str(DATA_DIR))
    dm.ensure_train_test_split()
    dm.create_precomputed_test_csv()
    print("   OK: DatasetManager инициализирован")
    
    # 2. Загрузка данных
    print("\n2. Загрузка данных...")
    train_df = dm.load_train_df()
    train_df = train_df.with_row_index("row_nr")
    
    test_df = dm.load_test_df(precomputed=True)
    test_df = test_df.with_row_index("row_nr")
    
    print(f"   Train: {len(train_df)} строк")
    print(f"   Test:  {len(test_df)} строк")
    
    # 3. Создание индексов
    print("\n3. Создание индексов...")
    train_indices = OscillogramDataset.create_indices(
        train_df, window_size=WINDOW_SIZE, mode='train', samples_per_file=1
    )
    val_indices = PrecomputedDataset.create_indices(
        test_df, window_size=WINDOW_SIZE, mode='val'
    )
    print(f"   Train indices: {len(train_indices)}")
    print(f"   Val indices: {len(val_indices)}")
    
    # 4. Тест режима phase_polar + snapshot
    print("\n4. Тест: phase_polar + snapshot")
    
    target_cols = get_target_columns('base')
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'
    
    # Train Dataset (обычный)
    train_ds = OscillogramDataset(
        dataframe=train_df,
        indices=train_indices[:100],  # Только 100 для быстроты
        window_size=WINDOW_SIZE,
        mode='classification',
        feature_mode='phase_polar',
        target_columns=target_cols,
        physical_normalization=True,
        norm_coef_path=str(NORM_COEF_PATH),
        sampling_strategy='snapshot'
    )
    
    # Val Dataset (предрассчитанный)
    val_ds = PrecomputedDataset(
        dataframe=test_df,
        indices=val_indices[:100],  # Только 100 для быстроты
        window_size=WINDOW_SIZE,
        feature_mode='phase_polar',
        target_columns=target_cols,
        sampling_strategy='snapshot'
    )
    
    # Проверка размерностей
    train_x, train_y = train_ds[0]
    val_x, val_y = val_ds[0]
    
    print(f"   Train X shape: {train_x.shape}")
    print(f"   Val X shape:   {val_x.shape}")
    print(f"   Train Y shape: {train_y.shape}")
    print(f"   Val Y shape:   {val_y.shape}")
    
    # Проверка что размеры совпадают
    assert train_x.shape == val_x.shape, f"Shape mismatch! Train: {train_x.shape}, Val: {val_x.shape}"
    print("   OK: Размерности совпадают!")
    
    # 5. Тест DataLoader
    print("\n5. Тест DataLoader...")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Пробуем загрузить батч
    for batch_x, batch_y in train_loader:
        print(f"   Train batch X: {batch_x.shape}")
        print(f"   Train batch Y: {batch_y.shape}")
        break
    
    for batch_x, batch_y in val_loader:
        print(f"   Val batch X:   {batch_x.shape}")
        print(f"   Val batch Y:   {batch_y.shape}")
        break
    
    print("   OK: DataLoader работает!")
    
    # 6. Тест различных режимов
    print("\n6. Тест различных feature_mode...")
    
    for fm in ['raw', 'symmetric', 'phase_complex']:
        val_ds_test = PrecomputedDataset(
            dataframe=test_df,
            indices=val_indices[:10],
            window_size=WINDOW_SIZE,
            feature_mode=fm,
            target_columns=target_cols,
            sampling_strategy='snapshot'
        )
        x, y = val_ds_test[0]
        print(f"   {fm}: X shape = {x.shape}")
    
    print("\n" + "=" * 60)
    print("SMOKE TEST: УСПЕШНО!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_smoke()
