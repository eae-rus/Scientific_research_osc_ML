"""
Быстрый тест обучения модели (1 эпоха) для проверки интеграции.
"""
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import torch
import torch.nn as nn
from osc_tools.data_management import DatasetManager
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.labels import get_target_columns


class SimpleMLP(nn.Module):
    """Простая MLP для теста."""
    def __init__(self, input_size, hidden_size=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    """Обучение одной эпохи."""
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Валидация
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            # Accuracy (multi-label)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == batch_y).all(dim=1).sum().item()
            total += batch_y.size(0)
    
    val_loss /= len(val_loader)
    accuracy = correct / total if total > 0 else 0
    
    return train_loss, val_loss, accuracy


def main():
    print("=" * 60)
    print("ТЕСТ ОБУЧЕНИЯ: SimpleMLP, 1 эпоха")
    print("=" * 60)
    
    DATA_DIR = ROOT_DIR / 'data' / 'ml_datasets'
    WINDOW_SIZE = 320
    BATCH_SIZE = 32
    FEATURE_MODE = 'phase_polar'
    SAMPLING = 'snapshot'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    dm = DatasetManager(str(DATA_DIR))
    dm.ensure_train_test_split()
    
    train_df = dm.load_train_df()
    train_df = train_df.with_row_index("row_nr")
    
    test_df = dm.load_test_df(precomputed=True)
    test_df = test_df.with_row_index("row_nr")
    
    # 2. Создание датасетов
    print("\n2. Создание датасетов...")
    target_cols = get_target_columns('base')
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'
    
    train_indices = OscillogramDataset.create_indices(
        train_df, window_size=WINDOW_SIZE, mode='train', samples_per_file=2
    )[:200]  # Ограничиваем для скорости
    
    val_indices = PrecomputedDataset.create_indices(
        test_df, window_size=WINDOW_SIZE, mode='val'
    )[:100]  # Ограничиваем для скорости
    
    # Train (обычный, с расчётом на лету)
    train_ds = OscillogramDataset(
        dataframe=train_df,
        indices=train_indices,
        window_size=WINDOW_SIZE,
        mode='classification',
        feature_mode=FEATURE_MODE,
        target_columns=target_cols,
        physical_normalization=True,
        norm_coef_path=str(NORM_COEF_PATH),
        sampling_strategy=SAMPLING
    )
    
    # Val (предрассчитанный, без FFT)
    val_ds = PrecomputedDataset(
        dataframe=test_df,
        indices=val_indices,
        window_size=WINDOW_SIZE,
        feature_mode=FEATURE_MODE,
        target_columns=target_cols,
        sampling_strategy=SAMPLING
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Создание модели
    print("\n3. Создание модели...")
    sample_x, _ = train_ds[0]
    input_size = sample_x.numel()
    print(f"   Input size: {input_size}")
    
    model = SimpleMLP(input_size=input_size, hidden_size=64, num_classes=len(target_cols))
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Обучение 1 эпохи
    print("\n4. Обучение (1 эпоха)...")
    train_loss, val_loss, accuracy = train_one_epoch(
        model, train_loader, val_loader, criterion, optimizer, device
    )
    
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss:   {val_loss:.4f}")
    print(f"   Val Accuracy: {accuracy:.2%}")
    
    print("\n" + "=" * 60)
    print("ТЕСТ ОБУЧЕНИЯ: УСПЕШНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
