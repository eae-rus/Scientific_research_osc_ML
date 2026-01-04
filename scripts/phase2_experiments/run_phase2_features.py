import polars as pl
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset

def main():
    # 1. Setup Paths
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'features'
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
    
    # 3. Выбор признаков (базовые колонки)
    # Пытаемся найти основные 6 каналов, чтобы эксперимент был честным
    candidates = ['IA', 'IB', 'IC', 'UA', 'UB', 'UC']
    if all(c in df.columns for c in candidates):
        feature_cols = candidates
    else:
        # Fallback: берем все I/U
        feature_cols = [c for c in df.columns if c.startswith('I') or c.startswith('U')]
    
    print(f"Используемые колонки ({len(feature_cols)}): {feature_cols}")
    
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
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    # 5. Эксперименты
    feature_modes = ['raw', 'symmetric', 'alpha_beta', 'instantaneous_power']
    
    # Определяем количество каналов для каждого режима
    # raw: количество колонок в feature_cols (обычно 6: IA, IB, IC, UA, UB, UC)
    # symmetric: 12 (Re/Im для I1, I2, I0 и U1, U2, U0)
    # alpha_beta: 6 (alpha, beta, zero для I и U)
    # instantaneous_power: 3 (p для каждой фазы)
    
    # Примечание: feature_cols может содержать дополнительные колонки; функция _get_phase_data
    # внутри Dataset отфильтрует нужные фазы.
    
    channels_map = {
        'raw': len(feature_cols), 
        'symmetric': 12,
        'alpha_beta': 6,
        'instantaneous_power': 3
    }
    
    for fm in feature_modes:
        print(f"\n=== Эксперименты для режима признаков: {fm} ===")
        
        in_channels = channels_map.get(fm, 6)
        input_size = in_channels * WINDOW_SIZE
        
        # Models to test
        models = [
            ("SimpleMLP", {"input_size": input_size, "output_size": num_classes}),
            ("ConvKAN", {"in_channels": in_channels, "num_classes": num_classes, "grid_size": 5, "base_filters": 8})
        ]
        
        for model_name, model_params in models:
            exp_name = f"{model_name}_{fm}"
            print(f"  Обучение {exp_name}...")
            
            data_config = DataConfig(
                path=str(METADATA_FILE),
                window_size=WINDOW_SIZE,
                batch_size=32,
                features=feature_cols,
                target="target_enc",
                mode="classification"
            )
            
            train_config = TrainingConfig(
                epochs=5, # Короткий прогон для тестирования
                learning_rate=0.001,
                device='cuda',
                save_dir=str(EXPERIMENTS_DIR),
                experiment_name=exp_name
            )
            
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
                feature_mode=fm,
                feature_columns=feature_cols,
                target_columns="target_enc"
            )
            
            val_ds = OscillogramDataset(
                dataframe=df,
                indices=val_indices,
                window_size=WINDOW_SIZE,
                mode='classification',
                feature_mode=fm,
                feature_columns=feature_cols,
                target_columns="target_enc"
            )
            
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
            
            runner.train(train_loader, val_loader)
            
            torch.save(runner.model.state_dict(), EXPERIMENTS_DIR / f"{exp_name}.pt")

if __name__ == "__main__":
    main()
