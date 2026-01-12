import polars as pl
import numpy as np
from pathlib import Path
import torch
import sys
import json
import argparse
from typing import List, Dict, Any

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import clean_labels, add_base_labels, get_target_columns

def run_single_experiment(
    exp_name: str, 
    model_name: str, 
    model_params: Dict[str, Any],
    feature_mode: str,
    sampling_strategy: str,
    downsampling_stride: int,
    df: pl.DataFrame,
    train_indices: List[int],
    val_indices: List[int],
    target_cols: List[str],
    data_config_base: DataConfig,
    train_config_base: TrainingConfig,
    norm_coef_path: Path
):
    print(f"\n>>> Запуск эксперимента: {exp_name}")
    print(f"Модель: {model_name}")
    print(f"Входные данные: {feature_mode} + {sampling_strategy}")
    
    # 1. Подготовка Dataset
    # feature_mode='phase_polar' -> 8 каналов * 2 (Ампл, Фаза) = 16 каналов
    # feature_mode='symmetric_polar' -> 6 каналов * 2 (Ампл, Фаза) = 12 каналов
    
    # Для обучения используем аугментацию (augment=True)
    train_ds = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=data_config_base.window_size,
        mode='classification', feature_mode=feature_mode,
        sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
        target_columns=target_cols, target_level='base_labels', # Используем базовые метки для тестов 2.6
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        augment=True 
    )
    
    # Для валидации отключаем аугментацию
    val_ds = OscillogramDataset(
        dataframe=df, indices=val_indices, window_size=data_config_base.window_size,
        mode='classification', feature_mode=feature_mode,
        sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
        target_columns=target_cols, target_level='base_labels',
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config_base.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config_base.batch_size, shuffle=False, num_workers=0)
    
    # 2. Настройка Config и Runner
    
    # Определяем in_channels из первого семпла
    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]
    print(f"Обнаружено входных каналов: {in_channels}, Длина входа: {sample_x.shape[1]}")
    
    # Обновляем параметры модели
    # in_channels обязателен для всех моделей
    model_params['in_channels'] = in_channels
    
    if 'input_size' in model_params:
        # Для MLP/SimpleKAN нужен расплющенный (flattened) размер
        model_params['input_size'] = in_channels * sample_x.shape[1]

    # Специальная обработка для PhysicsKAN в режиме snapshot (использует внутренний MLP)
    if model_name == 'PhysicsKAN' and sampling_strategy == 'snapshot':
        model_params['use_mlp'] = True
        model_params['input_size'] = in_channels * sample_x.shape[1] 

    config = ExperimentConfig(
        model=ModelConfig(name=model_name, params=model_params),
        data=data_config_base,
        training=train_config_base
    )
    config.training.experiment_name = exp_name
    
    runner = ExperimentRunner(config)
    
    # Запуск обучения
    history = runner.train(train_loader, val_loader)
    
    # 3. Сохранение результатов
    history_path = runner.save_dir / f"{exp_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
        
    # Сохранение весов
    torch.save(runner.model.state_dict(), runner.save_dir / f"{exp_name}.pt")
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Запуск экспериментов Фазы 2.6")
    parser.add_argument('--exp', type=str, default='all', help='ID эксперимента или группы ("all", "2.6.1", "2.6.2" и т.д.)')
    args = parser.parse_args()

    # 1. Настройка путей
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'phase2_6'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'

    print(f"Загрузка данных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
    
    # Обработка данных
    df = clean_labels(df)
    df = add_base_labels(df)
    
    # Целевые колонки (Классы повреждений)
    target_cols = get_target_columns('base')
    num_classes = len(target_cols)
    print(f"Целевые колонки: {target_cols}")

    # 2. Разделение данных
    WINDOW_SIZE = 320 # 200мс
    BATCH_SIZE = 64
    EPOCHS = 30
    
    df = df.with_row_index("index")
    # Отбираем только файлы достаточной длины
    valid_files = df.group_by("file_name").agg([
        pl.col("index").min().alias("start_idx"),
        pl.len().alias("length")
    ]).filter(pl.col("length") >= WINDOW_SIZE)
    
    indices = valid_files["start_idx"].to_list()
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Файлов для обучения: {len(train_indices)}, для валидации: {len(val_indices)}")
    
    data_config = DataConfig(
        path=str(METADATA_FILE), 
        window_size=WINDOW_SIZE, 
        batch_size=BATCH_SIZE, 
        mode='multilabel'
    )
    
    train_config = TrainingConfig(
        epochs=EPOCHS, 
        learning_rate=0.001, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_pos_weight=True,
        save_dir=str(EXPERIMENTS_DIR)
    )

    # --- Конфигурации Моделей (Средняя сложность - Medium) ---
    cnn_medium_params = {
        "num_classes": num_classes,
        "channels": [32, 64, 128, 256],
        "kernel_size": 3,
        "stride": 1
    }
    
    resnet_medium_params = {
        "num_classes": num_classes,
        "layers": [2, 2, 2, 2],
        "base_filters": 32
    }
    
    convkan_medium_params = {
        "num_classes": num_classes,
        "channels": [16, 32, 48],
        "grid_size": 5,
        "kernel_size": 3
    }
    
    physkan_medium_params = {
        "num_classes": num_classes,
        "channels": [16, 32, 48],
        "grid_size": 5,
        "kernel_size": 3
    }
    
    # Ствол для иерархических моделей: 2 независимых слоя -> 2 групповых слоя
    stem_config_medium = {'independent_layers': 2, 'grouped_layers': 2}
    
    h_cnn_params = cnn_medium_params.copy()
    h_cnn_params['stem_config'] = stem_config_medium
    
    h_mlp_params = cnn_medium_params.copy()
    if 'kernel_size' in h_mlp_params:
        del h_mlp_params['kernel_size']
    h_mlp_params['stem_config'] = stem_config_medium

    # ==========================================
    # СПИСОК ЭКСПЕРИМЕНТОВ
    # ==========================================
    
    experiments_list = [
        # === Группа 2.6.1: Базовые модели (Phase Polar + Stride) ===
        {
            "group": "2.6.1",
            "name": "Exp2.6.1_CNN_Medium_PhaseStride",
            "model": "SimpleCNN",
            "params": cnn_medium_params.copy(),
            "feature_mode": "phase_polar",
            "sampling_strategy": "stride",
            "stride": 16
        },
        {
            "group": "2.6.1",
            "name": "Exp2.6.1_ResNet_Medium_PhaseStride",
            "model": "ResNet1D",
            "params": resnet_medium_params.copy(),
            "feature_mode": "phase_polar",
            "sampling_strategy": "stride",
            "stride": 16
        },
        {
            "group": "2.6.1",
            "name": "Exp2.6.1_ConvKAN_Medium_PhaseStride",
            "model": "ConvKAN",
            "params": convkan_medium_params.copy(),
            "feature_mode": "phase_polar",
            "sampling_strategy": "stride",
            "stride": 16
        },
        {
            "group": "2.6.1",
            "name": "Exp2.6.1_PhysicsKAN_Medium_PhaseStride",
            "model": "PhysicsKAN",
            "params": physkan_medium_params.copy(),
            "feature_mode": "phase_polar",
            "sampling_strategy": "stride",
            "stride": 16
        },
        
        # Альтернатива: Snapshot (для скорости / проверки)
        {
            "group": "2.6.1",
            "name": "Exp2.6.1_PhysicsKAN_Medium_SymSnapshot",
            "model": "PhysicsKAN",
            "params": physkan_medium_params.copy(),
            "feature_mode": "symmetric_polar",
            "sampling_strategy": "snapshot",
            "stride": 32 
        },

        # === Группа 2.6.2: Иерархические модели ===
        {
            "group": "2.6.2",
            "name": "Exp2.6.2_HierarchicalCNN_Medium",
            "model": "HierarchicalCNN",
            "params": h_cnn_params,
            "feature_mode": "phase_polar",
            "sampling_strategy": "stride",
            "stride": 16
        },
        {
            "group": "2.6.2",
            "name": "Exp2.6.2_HierarchicalMLP_Medium",
            "model": "HierarchicalMLP",
            "params": h_mlp_params,
            "feature_mode": "phase_polar",
            "sampling_strategy": "stride",
            "stride": 16
        }
    ]

    print(f"Запрошен запуск: {args.exp}")
    
    for exp in experiments_list:
        # Фильтрация эксперимента по ID или имени
        should_run = (args.exp == 'all') or (args.exp == exp['group']) or (args.exp == exp['name'])
        
        if should_run:
            try:
                run_single_experiment(
                    exp_name=exp["name"],
                    model_name=exp["model"],
                    model_params=exp["params"],
                    feature_mode=exp["feature_mode"],
                    sampling_strategy=exp["sampling_strategy"],
                    downsampling_stride=exp.get("stride", 1),
                    df=df,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    target_cols=target_cols,
                    data_config_base=data_config,
                    train_config_base=train_config,
                    norm_coef_path=NORM_COEF_PATH
                )
            except KeyboardInterrupt:
                print("Остановка по требованию пользователя...")
                break
            except Exception as e:
                print(f"!!! Ошибка в {exp['name']}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
