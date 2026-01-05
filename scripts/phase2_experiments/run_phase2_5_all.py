import polars as pl
import numpy as np
from pathlib import Path
import torch
import sys
import json
import argparse

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import clean_labels, add_base_labels, get_target_columns, get_ml_columns

# Определение уровней сложности моделей
MODEL_COMPLEXITY = {
    'light': {
        'SimpleMLP': {'hidden_sizes': [64, 32], 'dropout': 0.2},
        'SimpleCNN': {'channels': [16, 32], 'dropout': 0.2, 'pool_every': 1},
        'ConvKAN':   {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'SimpleKAN': {'hidden_sizes': [64, 32], 'grid_size': 3, 'dropout': 0.1},
        'PhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'ResNet1D':  {'layers': [1, 1, 1], 'base_filters': 16}
    },
    'medium': {
        'SimpleMLP': {'hidden_sizes': [256, 128, 64], 'dropout': 0.3},
        'SimpleCNN': {'channels': [32, 64, 128], 'dropout': 0.3, 'pool_every': 1},
        'ConvKAN':   {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'SimpleKAN': {'hidden_sizes': [128, 64, 32], 'grid_size': 5, 'dropout': 0.2},
        'PhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'ResNet1D':  {'layers': [2, 2, 2, 2], 'base_filters': 32}
    },
    'heavy': {
        'SimpleMLP': {'hidden_sizes': [512, 256, 128, 64], 'dropout': 0.4},
        'SimpleCNN': {'channels': [64, 128, 256, 512], 'dropout': 0.4, 'pool_every': 1},
        'ConvKAN':   {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'SimpleKAN': {'hidden_sizes': [256, 128, 64, 32], 'grid_size': 5, 'dropout': 0.3},
        'PhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'ResNet1D':  {'layers': [3, 4, 6, 3], 'base_filters': 64}
    }
}

def get_model_params(model_name, complexity, input_size=None, num_classes=None, in_channels=None):
    params = MODEL_COMPLEXITY[complexity].get(model_name, {}).copy()
    
    if model_name == 'SimpleMLP':
        params['input_size'] = input_size
        params['output_size'] = num_classes
    elif model_name == 'SimpleKAN':
        params['input_size'] = input_size
        params['output_size'] = num_classes
    elif model_name in ['SimpleCNN', 'ConvKAN', 'PhysicsKAN']:
        params['in_channels'] = in_channels
        params['num_classes'] = num_classes
    elif model_name == 'ResNet1D':
        params['in_channels'] = in_channels
        params['num_classes'] = num_classes
    
    return params

def run_experiment(experiment_id, model_name, complexity, df, train_indices, val_indices, feature_cols, target_cols, 
                   norm_coef_path, feature_mode='raw', sampling_strategy='none', stride=16, 
                   use_pos_weight=True, augment=False):
    
    # Расчет входных параметров
    window_size = 320 # Базовый для Фазы 2.5
    
    # Определяем in_channels в зависимости от feature_mode
    # 'raw' -> все feature_cols
    # 'symmetric' -> 12 (6 I + 6 U)
    # 'polar' -> 12
    if feature_mode == 'raw':
        in_channels = len(feature_cols) if feature_cols else 8 # fallback
    elif feature_mode in ['symmetric', 'polar']:
        in_channels = 12
    else:
        in_channels = 6 # fallback for power etc.

    # Определяем input_size для MLP
    if sampling_strategy == 'none':
        pts = window_size
    elif sampling_strategy == 'stride':
        pts = window_size // stride
    elif sampling_strategy == 'snapshot':
        pts = 2
    else:
        pts = window_size
        
    input_size = in_channels * pts
    num_classes = len(target_cols)
    
    model_params = get_model_params(model_name, complexity, input_size, num_classes, in_channels)
    
    experiment_name = f"Exp_{experiment_id}_{model_name}_{complexity}_{feature_mode}_{sampling_strategy}"
    
    train_config = TrainingConfig(
        epochs=30,
        learning_rate=0.001,
        weight_decay=1e-4,
        use_pos_weight=use_pos_weight,
        experiment_name=experiment_name,
        save_dir=str(ROOT_DIR / 'experiments' / 'phase2_5')
    )
    
    data_config = DataConfig(
        path="", # Not used by runner directly if we provide loaders
        window_size=window_size,
        batch_size=64,
        mode='multilabel',
        features=[feature_mode]
    )
    
    config = ExperimentConfig(
        model=ModelConfig(name=model_name, params=model_params),
        data=data_config,
        training=train_config
    )
    
    runner = ExperimentRunner(config)
    
    # Аугментация
    aug_config = None
    if augment:
        aug_config = {
            'methods': ['gaussian_noise', 'amplitude_scaling', 'phase_shuffling'],
            'probs': [0.3, 0.3, 0.2]
        }

    # Datasets
    train_ds = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=window_size,
        mode='classification', feature_mode=feature_mode,
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        downsampling_mode=sampling_strategy, downsampling_stride=stride,
        augmentation_config=aug_config
    )
    
    val_ds = OscillogramDataset(
        dataframe=df, indices=val_indices, window_size=window_size,
        mode='classification', feature_mode=feature_mode,
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        downsampling_mode=sampling_strategy, downsampling_stride=stride
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config.batch_size, shuffle=False)
    
    print(f"\n>>> Starting {experiment_name}")
    history = runner.train(train_loader, val_loader)
    return history

def main():
    parser = argparse.ArgumentParser(description="Запуск экспериментов Фазы 2.5")
    parser.add_argument("--exp", type=str, default="2.5.1.0", help="ID эксперимента (например 2.5.1.0)")
    parser.add_argument("--model", type=str, choices=['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D', 'all'], default='all')
    parser.add_argument("--complexity", type=str, choices=['light', 'medium', 'heavy', 'all'], default='light')
    args = parser.parse_args()

    # 1. Setup Paths
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'

    print(f"Загрузка данных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000)
    df = clean_labels(df)
    df = add_base_labels(df)
    
    # Сплит данных (на уровне файлов для корректности)
    unique_files = df["file_name"].unique().to_list()
    np.random.seed(42)
    np.random.shuffle(unique_files)
    
    split_idx = int(len(unique_files) * 0.8)
    train_files = unique_files[:split_idx]
    val_files = unique_files[split_idx:]
    
    # Создаем индексы начал окон
    # Для train используем кортежи (start_idx, length) для Random Sliding Window
    # Для val используем фиксированные индексы
    
    window_size = 320
    
    # Вспомогательная функция для получения индексов по списку файлов
    def get_indices_for_files(df, file_list, mode='val'):
        file_df = df.filter(pl.col("file_name").is_in(file_list))
        return OscillogramDataset.create_indices(file_df, window_size, mode=mode)

    print("Подготовка индексов...")
    train_indices = get_indices_for_files(df, train_files, mode='train')
    val_indices = get_indices_for_files(df, val_files, mode='val')
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    target_cols = get_target_columns('base')
    feature_cols = None # Использовать Smart Selector
    
    models = ['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D'] if args.model == 'all' else [args.model]
    complexities = ['light', 'medium', 'heavy'] if args.complexity == 'all' else [args.complexity]
    
    # Настройка параметров в зависимости от эксперимента
    exp_params = {
        "2.5.1.0": {"feature_mode": "raw", "sampling": "none", "use_pw": False, "aug": False},
        "2.5.1.1": {"feature_mode": "raw", "sampling": "none", "use_pw": True, "aug": False},
        "2.5.1.2": {"feature_mode": "raw", "sampling": "none", "use_pw": True, "aug": True},
        "2.5.2.1": {"feature_mode": "symmetric", "sampling": "stride", "use_pw": True, "aug": True},
        "2.5.2.2": {"feature_mode": "symmetric", "sampling": "snapshot", "use_pw": True, "aug": True},
    }
    
    if args.exp not in exp_params:
        print(f"Unknown experiment ID: {args.exp}")
        return

    p = exp_params[args.exp]
    
    for model_name in models:
        for comp in complexities:
            # Маппинг сложности под стратегию (как просил пользователь)
            # Лёгкая - для ALL (dense/none)
            # Средняя - для Strided
            # Тяжёлая - для Snapshot
            # Но если пользователь явно указал --complexity, используем её.
            
            actual_comp = comp
            if args.complexity == 'light' and p['sampling'] == 'stride':
                actual_comp = 'medium'
            if args.complexity == 'light' and p['sampling'] == 'snapshot':
                actual_comp = 'heavy'
                
            run_experiment(
                args.exp, model_name, actual_comp, df, train_indices, val_indices, 
                feature_cols, target_cols, NORM_COEF_PATH,
                feature_mode=p['feature_mode'], 
                sampling_strategy=p['sampling'],
                use_pos_weight=p['use_pw'],
                augment=p['aug']
            )

if __name__ == "__main__":
    main()
