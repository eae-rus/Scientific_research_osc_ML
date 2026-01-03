import polars as pl
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset

def main():
    # 1. Setup Paths
    DATA_DIR = ROOT_DIR / 'raw_data' / 'Output'
    METADATA_FILE = ROOT_DIR / 'ML_model' / 'MLOps dataset' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'kan'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка данных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000, null_values=["NA", "nan", "null", ""])
    
    # 2. Обработка целевых меток
    print("Формирование целевых меток...")
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
    print(f"Число классов: {num_classes}")
    
    # 3. Выбор признаков
    feature_cols = [c for c in df.columns if c.startswith('I') or c.startswith('U')]
    print(f"Найдено {len(feature_cols)} признаковых колонок")
    
    # 4. Создание индексов
    print("Генерация индексов...")
    df = df.with_row_count("row_nr")
    
    file_stats = df.group_by("file_name").agg([
        pl.col("row_nr").min().alias("start_idx"),
        pl.len().alias("length")
    ])
    
    WINDOW_SIZE = 640
    
    valid_files = file_stats.filter(pl.col("length") >= WINDOW_SIZE)
    print(f"Валидных файлов (>= {WINDOW_SIZE}): {len(valid_files)} из {len(file_stats)}")
    
    indices = valid_files["start_idx"].to_list()
    
    # 5. Разделение Train/Val
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # 6. Запуск экспериментов
    
    data_config = DataConfig(
        path=str(METADATA_FILE),
        window_size=WINDOW_SIZE,
        batch_size=32,
        features=feature_cols,
        target="target_enc",
        mode="classification"
    )
    
    train_config = TrainingConfig(
        epochs=10,
        learning_rate=0.001,
        device='cuda',
        save_dir=str(EXPERIMENTS_DIR),
        experiment_name="kan_experiments"
    )
    
    models_to_run = [
        ("SimpleKAN", {
            "input_size": len(feature_cols) * WINDOW_SIZE, 
            "output_size": num_classes,
            "grid_size": 5,
            "hidden_sizes": [64, 32]
        }),
        ("ConvKAN", {
            "in_channels": len(feature_cols), 
            "num_classes": num_classes,
            "grid_size": 5,
            "base_filters": 8
        })
    ]
    
    for model_name, model_params in models_to_run:
        print(f"\nЗапуск эксперимента: {model_name}")
        
        config = ExperimentConfig(
            model=ModelConfig(name=model_name, params=model_params),
            data=data_config,
            training=train_config
        )
        
        runner = ExperimentRunner(config)
        
        train_ds = OscillogramDataset(
            dataframe=df,
            indices=train_indices,
            window_size=WINDOW_SIZE,
            mode='classification',
            feature_mode='raw',
            feature_columns=feature_cols,
            target_columns="target_enc"
        )
        
        val_ds = OscillogramDataset(
            dataframe=df,
            indices=val_indices,
            window_size=WINDOW_SIZE,
            mode='classification',
            feature_mode='raw',
            feature_columns=feature_cols,
            target_columns="target_enc"
        )
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config.batch_size, shuffle=False)
        
        runner.train(train_loader, val_loader)
        
        torch.save(runner.model.state_dict(), EXPERIMENTS_DIR / f"{model_name}.pt")
        print(f"Сохранено: {model_name}")

if __name__ == "__main__":
    main()
