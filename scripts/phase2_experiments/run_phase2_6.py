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
    norm_coef_path: Path,
    augment: bool = True  # Добавлен параметр управления аугментацией
):
    print(f"\n>>> Запуск эксперимента: {exp_name}")
    print(f"Модель: {model_name}")
    print(f"Входные данные: {feature_mode} + {sampling_strategy}")
    
    # 1. Подготовка Dataset
    # feature_mode='phase_polar' -> 8 каналов * 2 (Ампл, Фаза) = 16 каналов
    # feature_mode='symmetric_polar' -> 6 каналов * 2 (Ампл, Фаза) = 12 каналов
    
    # Аугментация используется только если augment=True
    train_ds = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=data_config_base.window_size,
        mode='classification', feature_mode=feature_mode,
        sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
        target_columns=target_cols, target_level='base_labels', # Используем базовые метки для тестов 2.6
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        augment=augment 
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

def main(exp: str = None, model_type: str = 'all', epochs: int = 30, samples_per_file: int = 12):
    """
    Главная точка входа для запуска экспериментов.
    Поддерживает как ручной запуск (через параметры), так и через CLI.
    """
    if exp is None:
        # Режим CLI
        parser = argparse.ArgumentParser(description="Запуск экспериментов Фазы 2.6")
        parser.add_argument('--exp', type=str, default='all', help='ID эксперимента или группы ("all", "2.6.1", "2.6.2" и т.д.)')
        parser.add_argument('--epochs', type=int, default=30, help='Количество эпох')
        parser.add_argument('--model', type=str, default='all', help='Фильтр по модели')
        parser.add_argument('--samples', type=int, default=12, help='Количество случайных окон из файла за эпоху')
        args = parser.parse_args()
        target_exp = args.exp
        target_epochs = args.epochs
        target_model = args.model
        target_spf = args.samples
    else:
        # Ручной режим
        target_exp = exp
        target_epochs = epochs
        target_model = model_type
        target_spf = samples_per_file

    print(f"Конфигурация: Exp={target_exp}, Model={target_model}, Epochs={target_epochs}, Samples/File={target_spf}")

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
    df = df.with_row_index("row_nr") # Важно для create_indices
    
    # Целевые колонки (Классы повреждений)
    target_cols = get_target_columns('base')
    num_classes = len(target_cols)
    print(f"Целевые колонки: {target_cols}")

    # 2. Разделение данных (Train/Val Split по файлам)
    WINDOW_SIZE = 320 # 200мс
    BATCH_SIZE = 64
    EPOCHS = target_epochs
    
    unique_files = df["file_name"].unique().to_list()
    np.random.seed(42)
    np.random.shuffle(unique_files)
    
    split_idx = int(len(unique_files) * 0.8)
    train_files = unique_files[:split_idx]
    val_files = unique_files[split_idx:]
    
    print(f"Файлов для обучения: {len(train_files)}, для валидации: {len(val_files)}")

    # Генерация индексов с использованием create_indices
    # Для train используем режим 'train' с samples_per_file
    train_indices = OscillogramDataset.create_indices(
        df.filter(pl.col("file_name").is_in(train_files)), 
        window_size=WINDOW_SIZE, 
        mode='train',
        samples_per_file=target_spf
    )
    
    # Для val используем режим 'val' (фиксированные окна)
    val_indices = OscillogramDataset.create_indices(
        df.filter(pl.col("file_name").is_in(val_files)), 
        window_size=WINDOW_SIZE, 
        mode='val'
    )
    
    print(f"Всего примеров на эпоху: Train={len(train_indices)}, Val={len(val_indices)}")
    
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

    # Определение уровней сложности моделей для Фазы 2.6
MODEL_COMPLEXITY = {
    'light': {
        'SimpleMLP': {'hidden_sizes': [64, 32], 'dropout': 0.2},
        'SimpleCNN': {'channels': [16, 32], 'dropout': 0.2},
        'ConvKAN':   {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'SimpleKAN': {'hidden_sizes': [64, 32], 'grid_size': 3, 'dropout': 0.1},
        'PhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'ResNet1D':  {'layers': [1, 1, 1, 1], 'base_filters': 16},
        # Иерархические модели (2.6)
        'HierarchicalCNN': {'channels': [16, 32], 'dropout': 0.2, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        'HierarchicalKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        'HierarchicalMLP': {'channels': [16, 32], 'dropout': 0.2, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}}
    },
    'medium': {
        'SimpleMLP': {'hidden_sizes': [256, 128, 64], 'dropout': 0.3},
        'SimpleCNN': {'channels': [32, 64, 128, 256], 'dropout': 0.3},
        'ConvKAN':   {'channels': [16, 32, 48], 'dropout': 0.2, 'grid_size': 5},
        'SimpleKAN': {'hidden_sizes': [128, 64, 32], 'grid_size': 5, 'dropout': 0.2},
        'PhysicsKAN': {'channels': [16, 32, 48], 'dropout': 0.2, 'grid_size': 5},
        'ResNet1D':  {'layers': [2, 2, 2, 2], 'base_filters': 32},
        # Иерархические модели (2.6)
        'HierarchicalCNN': {'channels': [32, 64, 128], 'dropout': 0.3, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        'HierarchicalKAN': {'channels': [16, 32, 48], 'dropout': 0.2, 'grid_size': 5, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        'HierarchicalMLP': {'channels': [32, 64, 128], 'dropout': 0.3, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}}
    },
    'heavy': {
        'SimpleMLP': {'hidden_sizes': [512, 256, 128, 64], 'dropout': 0.4},
        'SimpleCNN': {'channels': [64, 128, 256, 512], 'dropout': 0.4},
        'ConvKAN':   {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'SimpleKAN': {'hidden_sizes': [256, 128, 64, 32], 'grid_size': 5, 'dropout': 0.3},
        'PhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'ResNet1D':  {'layers': [3, 4, 6, 3], 'base_filters': 64},
        # Иерархические модели (2.6)
        'HierarchicalCNN': {'channels': [64, 128, 256], 'dropout': 0.4, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        'HierarchicalKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        'HierarchicalMLP': {'channels': [64, 128, 256], 'dropout': 0.4, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}}
    }
}

def get_model_params(model_name, complexity, num_classes):
    """Возвращает параметры модели на основе сложности."""
    params = MODEL_COMPLEXITY[complexity].get(model_name, {}).copy()
    params['num_classes'] = num_classes
    return params

def run_single_experiment(
    exp_name: str, 
    model_name: str, 
    complexity: str,
    feature_mode: str,
    sampling_strategy: str,
    downsampling_stride: int,
    df: pl.DataFrame,
    train_indices: List[int],
    val_indices: List[int],
    target_cols: List[str],
    data_config_base: DataConfig,
    train_config_base: TrainingConfig,
    norm_coef_path: Path,
    augment: bool = True
):
    print(f"\n>>> Запуск эксперимента: {exp_name}")
    print(f"Модель: {model_name} ({complexity})")
    
    # 1. Получение параметров модели
    num_classes = len(target_cols)
    model_params = get_model_params(model_name, complexity, num_classes)
    
    # 2. Подготовка Dataset
    train_ds = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=data_config_base.window_size,
        mode='classification', feature_mode=feature_mode,
        sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
        target_columns=target_cols, target_level='base_labels',
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        augment=augment 
    )
    
    val_ds = OscillogramDataset(
        dataframe=df, indices=val_indices, window_size=data_config_base.window_size,
        mode='classification', feature_mode=feature_mode,
        sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
        target_columns=target_cols, target_level='base_labels',
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config_base.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config_base.batch_size, shuffle=False)
    
    # 3. Определение in_channels из первого семпла
    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]
    seq_len = sample_x.shape[1]
    
    model_params['in_channels'] = in_channels
    if 'input_size' in model_params or model_name in ['SimpleMLP', 'SimpleKAN', 'HierarchicalMLP']:
        model_params['input_size'] = in_channels * seq_len

    if model_name == 'PhysicsKAN' and sampling_strategy == 'snapshot':
        model_params['use_mlp'] = True
        model_params['input_size'] = in_channels * seq_len

    config = ExperimentConfig(
        model=ModelConfig(name=model_name, params=model_params),
        data=data_config_base,
        training=train_config_base
    )
    config.training.experiment_name = exp_name
    
    runner = ExperimentRunner(config)
    history = runner.train(train_loader, val_loader)
    
    # Сохранение истории
    history_path = runner.save_dir / f"{exp_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
        
    return history

def main(exp: str = None, model: str = None, complexity: str = None, samples_per_file: int = 12, epochs: int = 30, checkpoint_frequency: int = 1):
    """Главная точка входа для запуска экспериментов Phase 2.6."""
    
    # Режим параметров
    target_exp = exp
    target_model = model or 'all'
    target_complexity = complexity or 'medium'
    target_spf = samples_per_file
    target_epochs = epochs

    # Настройка путей
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'phase2_6'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'

    df = pl.read_csv(METADATA_FILE, infer_schema_length=50000, null_values=["NA", "nan", "null", ""])
    df = clean_labels(df)
    df = add_base_labels(df)
    df = df.with_row_index("row_nr")
    
    target_cols = get_target_columns('base')
    WINDOW_SIZE = 320
    
    # Split
    unique_files = df["file_name"].unique().to_list()
    np.random.seed(42)
    np.random.shuffle(unique_files)
    split_idx = int(len(unique_files) * 0.8)
    train_files = unique_files[:split_idx]
    val_files = unique_files[split_idx:]
    
    train_indices = OscillogramDataset.create_indices(
        df.filter(pl.col("file_name").is_in(train_files)), 
        window_size=WINDOW_SIZE, mode='train', samples_per_file=target_spf
    )
    val_indices = OscillogramDataset.create_indices(
        df.filter(pl.col("file_name").is_in(val_files)), 
        window_size=WINDOW_SIZE, mode='val'
    )
    
    data_config = DataConfig(path=str(METADATA_FILE), window_size=WINDOW_SIZE, batch_size=64, mode='multilabel')
    train_config = TrainingConfig(
        epochs=target_epochs, 
        learning_rate=0.001, 
        use_pos_weight=True,
        save_dir=str(EXPERIMENTS_DIR)
    )

    # Таблица экспериментов (логика данных)
    exp_params = {
        "2.6.1_stride":   {"feature_mode": "phase_polar", "sampling": "stride",   "stride": 16, "aug": True},
        "2.6.1_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "stride": 32, "aug": True},
        "2.6.2_stride":   {"feature_mode": "phase_polar", "sampling": "stride",   "stride": 16, "aug": True},
    }

    if target_exp not in exp_params:
        print(f"!!! Ошибка: Эксперимент {target_exp} не определен в exp_params")
        return

    p = exp_params[target_exp]
    
    # Определяем список моделей для запуска
    if target_model == 'all':
        if target_exp.startswith("2.6.2"):
            models_to_run = ['HierarchicalCNN', 'HierarchicalKAN', 'HierarchicalMLP']
        else:
            models_to_run = ['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D']
    else:
        models_to_run = [target_model]

    # Определяем список сложностей
    complexities_to_run = ['light', 'medium', 'heavy'] if target_complexity == 'all' else [target_complexity]

    for m_name in models_to_run:
        for comp in complexities_to_run:
            full_exp_name = f"Exp{target_exp}_{m_name}_{comp}_{p['sampling']}"
            
            # Проверка существования
            checkpoint_path = EXPERIMENTS_DIR / full_exp_name / "final_model.pt"
            if checkpoint_path.exists():
                print(f">>> Пропуск {full_exp_name} (уже обучено)")
                continue

            try:
                run_single_experiment(
                    exp_name=full_exp_name,
                    model_name=m_name,
                    complexity=comp,
                    feature_mode=p["feature_mode"],
                    sampling_strategy=p["sampling"],
                    downsampling_stride=p.get("stride", 1),
                    df=df,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    target_cols=target_cols,
                    data_config_base=data_config,
                    train_config_base=train_config,
                    norm_coef_path=NORM_COEF_PATH,
                    augment=p.get("aug", True)
                )
            except Exception as e:
                print(f"!!! Ошибка в {full_exp_name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    # === ВЕРСИЯ 1: РУЧНОЙ ЗАПУСК (Phase 2.5 Style) ===
    
    # Набор экспериментов (группы данных)
    EXPS = ["2.6.1_stride", "2.6.1_snapshot", "2.6.2_stride"]
    
    # Тип модели ('all' - выберет автоматически подходящие для группы)
    MODEL_TYPE = "all"
    
    # Уровень сложности ('all' - light, medium, heavy)
    SELECTED_COMPLEXITY = "all"
    
    # Плотность выборки (~10 000 примеров за эпоху при 12)
    SAMPLES_PER_FILE = 12
    
    # Эпохи
    EPOCHS = 30
    
    # Частота чекпоинтов (не используется напрямую в run_single_experiment сейчас, но можно добавить)
    CHECKPOINT_FREQUENCY = EPOCHS + 1

    # Пропускать ли уже обученные модели (наличие final_model.pt)
    SKIP_EXISTING = True

    for exp_id in EXPS:
        try:
            main(
                exp=exp_id, 
                model=MODEL_TYPE, 
                complexity=SELECTED_COMPLEXITY, 
                samples_per_file=SAMPLES_PER_FILE, 
                epochs=EPOCHS, 
                checkpoint_frequency=CHECKPOINT_FREQUENCY,
                skip_existing=SKIP_EXISTING
            )
        except Exception as e:
            print(f"!!! Критическая ошибка в группе {exp_id}: {e}")

