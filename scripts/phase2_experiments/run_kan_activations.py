import polars as pl
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import sys
import time

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.models import ConvKAN

def main():
    # 1. Setup Paths
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'kan_activations'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка данных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000, null_values=["NA", "nan", "null", ""])
    
    # 2. Обработка целевых меток
    ml_cols = [c for c in df.columns if c.startswith('ML_')]
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int8).alias(c) 
        for c in ml_cols
    ])
    
    ml_data = df.select(ml_cols).to_numpy()
    combos = []
    for row in ml_data:
        active = [ml_cols[i] for i, val in enumerate(row) if val == 1]
        if not active:
            combos.append("Normal")
        else:
            combos.append(",".join(sorted(active)))
            
    df = df.with_columns(pl.Series("target_class", combos))
    
    le = LabelEncoder()
    y_enc = le.fit_transform(combos)
    df = df.with_columns(pl.Series("target_enc", y_enc))
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes}")
    
    # 3. Сплит данных
    df = df.with_row_index("index")
    file_stats = df.group_by("file_name").agg([
        pl.col("index").min().alias("start_idx"),
        pl.len().alias("length")
    ])
    
    WINDOW_SIZE = 640
    valid_files = file_stats.filter(pl.col("length") >= WINDOW_SIZE)
    indices = valid_files["start_idx"].to_list()
    
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # === Experiment: Activation Functions ===
    print("\n=== Эксперимент: Сравнение функций активации (Basis Functions) ===")
    
    activations = {
        "SiLU": nn.SiLU,
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "GELU": nn.GELU
    }
    
    results = []
    
    for name, act_cls in activations.items():
        print(f"\n--- Testing Activation: {name} ---")
        
        # Параметры ConvKAN
        kan_params = {
            "in_channels": 12, # Симметричные признаки (3 токовых + 3 фазных) * 2 (действительная/мнимая или аналогично).
                               # Если потребуется — уточним по реализации feature_mode='symmetric' в OscillogramDataset.
                               # В плане мы ранее отмечали, что symmetric давал лучшие результаты.
                               # Ожидается: Ia, Ib, Ic, Ua, Ub, Uc (6) + I1,I2,I0,U1,U2,U0 (6) = 12 каналов.
                               # Если размер не совпадёт — обновим после проверки датасета.
            "num_classes": num_classes,
            "base_filters": 8,
            "grid_size": 5,
            "base_activation": act_cls
        }
        
        # Замечание по передаче класса активации:
        # Мы передаём сам класс `act_cls` внутри словаря параметров модели.
        # `ExperimentRunner` создаёт модель по имени и параметрам (`ModelConfig` → `ModelClass(**params)`),
        # поэтому передача объекта класса здесь корректна при выполнении в одном процессе.
        
        data_config = DataConfig(path=str(METADATA_FILE), window_size=WINDOW_SIZE, batch_size=32)
        train_config = TrainingConfig(epochs=5, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        config = ExperimentConfig(
            model=ModelConfig(name="ConvKAN", params=kan_params),
            data=data_config,
            training=train_config
        )
        
        # Датасет с симметричными признаками
        # Примечание: необходимо убедиться, что `OscillogramDataset` действительно выдаёт 12 каналов для 'symmetric'.
        # При несоответствии обновим `kan_params['in_channels']` после извлечения примера.
        train_ds = OscillogramDataset(
            dataframe=df, indices=train_indices, window_size=WINDOW_SIZE,
            mode='classification', feature_mode='symmetric',
            target_columns="target_enc"
        )
        
        val_ds = OscillogramDataset(
            dataframe=df, indices=val_indices, window_size=WINDOW_SIZE,
            mode='classification', feature_mode='symmetric',
            target_columns="target_enc"
        )
        
        # Проверяем фактическое число входных каналов в выборке
        sample, _ = train_ds[0]
        in_channels = sample.shape[0]
        print(f"Input channels: {in_channels}")
        kan_params["in_channels"] = in_channels # Обновляем параметр на реальное значение
        
        runner = ExperimentRunner(config)
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config.batch_size, shuffle=False)
        
        start_time = time.time()
        history = runner.train(train_loader, val_loader)
        end_time = time.time()
        
        best_val_loss = min(history['val_loss'])
        best_val_acc = max(history['val_acc'])
        
        results.append({
            "Activation": name,
            "Best Val Loss": best_val_loss,
            "Best Val Acc": best_val_acc,
            "Time (s)": end_time - start_time
        })
        
        # Save model
        torch.save(runner.model.state_dict(), EXPERIMENTS_DIR / f"ConvKAN_{name}.pt")

    # Print Summary
    print("\n=== Summary Results ===")
    print(f"{'Activation':<15} | {'Val Loss':<10} | {'Val Acc':<10} | {'Time (s)':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['Activation']:<15} | {r['Best Val Loss']:.4f}     | {r['Best Val Acc']:.4f}     | {r['Time (s)']:.2f}")

if __name__ == "__main__":
    main()
