import polars as pl
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
import sys

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.models import SimpleMLP, ConvKAN
from osc_tools.ml.labels import clean_labels, add_base_labels, get_target_columns

def main():
    # 1. Setup Paths
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'advanced_phase2'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка данных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000, null_values=["NA", "nan", "null", ""])
    
    # 2. Обработка целевых меток (Multi-Label)
    df = clean_labels(df)
    df = add_base_labels(df)
    
    target_cols = get_target_columns('base')
    num_classes = len(target_cols)
    print(f"Target Columns: {target_cols}")
    print(f"Num Classes: {num_classes}")
    
    feature_cols = [c for c in df.columns if c.startswith('I') or c.startswith('U')]
    
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

    # Путь к файлу коэффициентов нормализации по сенсорам
    norm_coef_path = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'
    print("\n=== Эксперимент 1: Сравнение при равном числе параметров ===")
    
    # 1.1 SmallMLP (симметричные признаки)
    # Вход: 12 каналов * 640 = 7680 признаков
    # Скрытый слой: 6 нейронов -> ~46k параметров
    print("\n--- Обучение SmallMLP (с равночисленными параметрами) ---")
    
    mlp_params = {
        "input_size": 12 * WINDOW_SIZE,
        "hidden_sizes": [6],
        "output_size": num_classes,
        "dropout": 0.2
    }
    
    data_config = DataConfig(path=str(METADATA_FILE), window_size=WINDOW_SIZE, batch_size=32, mode='multilabel')
    train_config = TrainingConfig(
        epochs=10, 
        learning_rate=0.001, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_pos_weight=True
    )
    
    config_mlp = ExperimentConfig(
        model=ModelConfig(name="SimpleMLP", params=mlp_params),
        data=data_config,
        training=train_config
    )
    
    runner_mlp = ExperimentRunner(config_mlp)
    
    # Dataset with Symmetric Features
    train_ds_sym = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=WINDOW_SIZE,
        mode='classification', feature_mode='symmetric',
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path)
    )
    val_ds_sym = OscillogramDataset(
        dataframe=df, indices=val_indices, window_size=WINDOW_SIZE,
        mode='classification', feature_mode='symmetric',
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path)
    )
    
    train_loader_sym = torch.utils.data.DataLoader(train_ds_sym, batch_size=32, shuffle=True)
    val_loader_sym = torch.utils.data.DataLoader(val_ds_sym, batch_size=32, shuffle=False)
    
    runner_mlp.train(train_loader_sym, val_loader_sym)
    torch.save(runner_mlp.model.state_dict(), EXPERIMENTS_DIR / "SmallMLP_Matched.pt")
    
    # 1.2 ConvKAN (Symmetric Features)
    # Base=8, Grid=10 -> ~45-50k params for 12 channels
    print("\n--- Обучение ConvKAN (с равночисленными параметрами) ---")
    
    kan_params = {
        "in_channels": 12,
        "num_classes": num_classes,
        "grid_size": 10,
        "base_filters": 8
    }
    
    config_kan = ExperimentConfig(
        model=ModelConfig(name="ConvKAN", params=kan_params),
        data=data_config,
        training=train_config
    )
    
    runner_kan = ExperimentRunner(config_kan)
    runner_kan.train(train_loader_sym, val_loader_sym)
    torch.save(runner_kan.model.state_dict(), EXPERIMENTS_DIR / "ConvKAN_Matched.pt")
    
    # === Experiment 2: Complex Features ===
    print("\n=== Эксперимент 2: Комплексные признаки (ConvKAN) ===")
    
    # 2.1 ConvKAN (комплексные каналы)
    # Вход: 12 каналов (6 вещественных + 6 мнимых)
    # Архитектура та же, что и выше
    print("\n--- Обучение ConvKAN (комплексные признаки) ---")
    
    train_ds_complex = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=WINDOW_SIZE,
        mode='classification', feature_mode='complex_channels',
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path)
    )
    val_ds_complex = OscillogramDataset(
        dataframe=df, indices=val_indices, window_size=WINDOW_SIZE,
        mode='classification', feature_mode='complex_channels',
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path)
    )
    
    train_loader_complex = torch.utils.data.DataLoader(train_ds_complex, batch_size=32, shuffle=True)
    val_loader_complex = torch.utils.data.DataLoader(val_ds_complex, batch_size=32, shuffle=False)
    
    # Re-initialize runner to reset weights
    runner_kan_complex = ExperimentRunner(config_kan)
    runner_kan_complex.train(train_loader_complex, val_loader_complex)
    torch.save(runner_kan_complex.model.state_dict(), EXPERIMENTS_DIR / "ConvKAN_Complex.pt")

if __name__ == "__main__":
    main()
