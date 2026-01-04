import polars as pl
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
import sys

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset

def main():
    # 1. Setup Paths
    METADATA_FILE = ROOT_DIR / 'ML_model' / 'MLOps dataset' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'kan_grid_search'
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
    
    # 3. Признаки
    feature_cols = [c for c in df.columns if c.startswith('I') or c.startswith('U')]
    
    # 4. Индексы
    df = df.with_row_count("row_nr")
    file_stats = df.group_by("file_name").agg([
        pl.col("row_nr").min().alias("start_idx"),
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
    
    # 5. Grid Search
    grid_sizes = [3, 5, 10, 15]
    
    for gs in grid_sizes:
        exp_name = f"ConvKAN_grid{gs}"
        print(f"\n=== Testing Grid Size: {gs} ===")
        
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
            experiment_name=exp_name
        )
        
        # ConvKAN params
        # Используем symmetric features (12 каналов), так как они показали лучший результат
        model_params = {
            "in_channels": 12, 
            "num_classes": num_classes, 
            "grid_size": gs, 
            "base_filters": 8
        }
        
        config = ExperimentConfig(
            model=ModelConfig(name="ConvKAN", params=model_params),
            data=data_config,
            training=train_config
        )
        
        runner = ExperimentRunner(config)
        
        train_ds = OscillogramDataset(
            dataframe=df,
            indices=train_indices,
            window_size=WINDOW_SIZE,
            mode='classification',
            feature_mode='symmetric',
            feature_columns=feature_cols,
            target_columns="target_enc"
        )
        
        val_ds = OscillogramDataset(
            dataframe=df,
            indices=val_indices,
            window_size=WINDOW_SIZE,
            mode='classification',
            feature_mode='symmetric',
            feature_columns=feature_cols,
            target_columns="target_enc"
        )
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
        
        runner.train(train_loader, val_loader)
        
        torch.save(runner.model.state_dict(), EXPERIMENTS_DIR / f"{exp_name}.pt")

if __name__ == "__main__":
    main()
