import polars as pl
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
import sys
import json

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import clean_labels, add_base_labels, get_target_columns

def run_experiment(experiment_name, model_name, model_params, data_config, train_config, df, train_indices, val_indices, feature_cols, target_cols, norm_coef_path):
    print(f"\n=== Запуск эксперимента: {experiment_name} ===")
    print(f"Модель: {model_name}")
    print(f"Параметры: {model_params}")
    
    config = ExperimentConfig(
        model=ModelConfig(name=model_name, params=model_params),
        data=data_config,
        training=train_config
    )
    
    # Обновляем имя эксперимента в конфиге
    config.training.experiment_name = experiment_name
    
    runner = ExperimentRunner(config)
    
    # Dataset
    # Важно: feature_mode='symmetric' возвращает 12 каналов: 6 токов, 6 напряжений.
    # Для PhysicsKAN нужно, чтобы они шли парами или блоками.
    # OscillogramDataset возвращает [I..., U...]
    
    train_ds = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=data_config.window_size,
        mode='classification', feature_mode='symmetric',
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path)
    )
    val_ds = OscillogramDataset(
        dataframe=df, indices=val_indices, window_size=data_config.window_size,
        mode='classification', feature_mode='symmetric',
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path)
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config.batch_size, shuffle=False, num_workers=0)
    
    history = runner.train(train_loader, val_loader)
    
    # Сохраняем историю
    history_path = runner.save_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
        
    return history

def main():
    # 1. Setup Paths
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'advanced_phase2'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'

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

    # Общие настройки
    BATCH_SIZE = 32
    EPOCHS = 20 # Увеличили количество эпох
    LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Список экспериментов
    experiments = []
    
    # 1 Базовый MLP (Symmetric Features)
    # Input: 12 channels flattened
    experiments.append({
        "name": "Baseline_MLP_Sym",
        "model": "SimpleMLP",
        "params": {
            "input_size": 12 * WINDOW_SIZE,
            "hidden_sizes": [128, 64], # Чуть больше чем раньше
            "output_size": num_classes,
            "dropout": 0.2
        }
    })
    
    # 2. ConvKAN (Standard)
    # Input: 12 channels
    experiments.append({
        "name": "ConvKAN_Standard",
        "model": "ConvKAN",
        "params": {
            "in_channels": 12,
            "num_classes": num_classes,
            "base_filters": 8,
            "kernel_size": 5,
            "grid_size": 5
        }
    })
    
    # 3. PhysicsKAN (With Arithmetic Layers)
    # Input: 12 channels (6 currents, 6 voltages)
    # PhysicsKAN expects even channels. 12 is even.
    experiments.append({
        "name": "PhysicsKAN_Arithmetic",
        "model": "PhysicsKAN",
        "params": {
            "in_channels": 12,
            "num_classes": num_classes,
            "base_filters": 8,
            "kernel_size": 5,
            "grid_size": 5
        }
    })
    
    # Запуск цикла экспериментов
    for exp in experiments:
        data_config = DataConfig(
            path=str(METADATA_FILE), 
            window_size=WINDOW_SIZE, 
            batch_size=BATCH_SIZE, 
            mode='multilabel'
        )
        
        train_config = TrainingConfig(
            epochs=EPOCHS, 
            learning_rate=LR, 
            device=DEVICE,
            use_pos_weight=True, # Включаем балансировку классов
            save_dir=str(EXPERIMENTS_DIR),
            experiment_name=exp["name"]
        )
        
        try:
            run_experiment(
                experiment_name=exp["name"],
                model_name=exp["model"],
                model_params=exp["params"],
                data_config=data_config,
                train_config=train_config,
                df=df,
                train_indices=train_indices,
                val_indices=val_indices,
                feature_cols=feature_cols,
                target_cols=target_cols,
                norm_coef_path=NORM_COEF_PATH
            )
        except Exception as e:
            print(f"Ошибка в эксперименте {exp['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
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
