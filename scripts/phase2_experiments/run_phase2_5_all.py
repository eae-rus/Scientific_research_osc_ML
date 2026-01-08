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
        'ResNet1D':  {'layers': [1, 1, 1, 1], 'base_filters': 16}
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

def run_experiment(experiment_id, model_name, complexity, df, train_indices, val_indices, feature_cols, target_level, 
                   norm_coef_path, feature_mode='raw', sampling_strategy='none', stride=16, 
                   use_pos_weight=True, augment=False, num_harmonics=1, epochs=30, checkpoint_frequency=1):
    
    # Расчет входных параметров
    window_size = 320 # Базовый для Фазы 2.5
    
    # Определяем in_channels в зависимости от feature_mode
    # 'raw' -> все feature_cols (обычно 8)
    # 'symmetric', 'symmetric_polar' -> 12 (6 comps * 2 values) * num_harmonics
    # 'phase_polar', 'phase_complex' -> 16 (8 channels * 2 values) * num_harmonics
    # 'power' -> 8 (4 pairs * 2 val)
    # 'alpha_beta' -> 6 (2 sets * 3 val)
    
    if feature_mode == 'raw':
        in_channels = len(feature_cols) if feature_cols else 8 # fallback
    elif feature_mode in ['symmetric', 'symmetric_polar']:
        in_channels = 12 * num_harmonics
    elif feature_mode in ['phase_polar', 'phase_complex', 'complex_channels']:
        in_channels = 16 * num_harmonics
    elif feature_mode == 'power':
        in_channels = 8
    elif feature_mode == 'alpha_beta':
        in_channels = 6
    else:
        in_channels = 8 # fallback

    # Определяем target columns
    target_cols = get_target_columns(target_level)

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
    
    experiment_name = f"Exp_{experiment_id}_{model_name}_{complexity}_{feature_mode}_{sampling_strategy}_{target_level}"
    
    train_config = TrainingConfig(
        epochs=epochs,
        learning_rate=0.001,
        weight_decay=1e-4,
        use_pos_weight=use_pos_weight,
        experiment_name=experiment_name,
        save_dir=str(ROOT_DIR / 'experiments' / 'phase2_5'),
        checkpoint_frequency=checkpoint_frequency
    )
    
    data_config = DataConfig(
        path="", # Не используется напрямую runner, если мы предоставляем загрузчики
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
        # Конфигурация аугментации (Обновлено для Фазы 2.5: меньше шума, меньше масштаб)
        aug_config = {
            'p_inversion': 0.5,
            'p_noise': 0,
            'noise_std_current': 0.005,
            'noise_std_voltage': 0.05,
            'p_scaling': 0.2,
            'scaling_range_current': (0.9, 1.1),
            'scaling_range_voltage': (0.98, 1.02),
            'p_offset': 0.0,
            'p_phase_shuffling': 0.33,
            'p_drop_channel': 0.0
        }

    # Datasets
    train_ds = OscillogramDataset(
        dataframe=df, indices=train_indices, window_size=window_size,
        mode='classification', feature_mode=feature_mode,
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        downsampling_mode=sampling_strategy, downsampling_stride=stride,
        augmentation_config=aug_config, num_harmonics=num_harmonics
    )
    
    val_ds = OscillogramDataset(
        dataframe=df, indices=val_indices, window_size=window_size,
        mode='classification', feature_mode=feature_mode,
        feature_columns=feature_cols, target_columns=target_cols,
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        downsampling_mode=sampling_strategy, downsampling_stride=stride, num_harmonics=num_harmonics
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=data_config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=data_config.batch_size, shuffle=False)
    
    print(f"\n>>> Запуск {experiment_name}")
    history = runner.train(train_loader, val_loader)
    return history

def main(exp: str = None, model: str = None, complexity: str = None, samples_per_file: int = 1, epochs: int = 30, checkpoint_frequency: int = 1, skip_existing: bool = True):
    """
    Главная точка входа для запуска экспериментов.
    
    Args:
        exp: ID эксперимента (напр. '2.5.1.0'). Если None, берется из sys.argv.
        model: Имя модели или 'all'.
        complexity: Уровень сложности или 'all'.
        samples_per_file: Количество случайных окон из каждого файла за одну эпоху.
        epochs: Количество эпох обучения.
        skip_existing: Если True, пропускать уже обученные модели (наличие final_model.pt).
    """
    if exp is not None:
        # Ручной режим (параметры переданы напрямую в функцию)
        class Args:
            pass
        args = Args()
        args.exp = exp
        args.model = model or 'all'
        args.complexity = complexity or 'light'
        args.samples_per_file = samples_per_file
        args.epochs = epochs
        args.checkpoint_frequency = checkpoint_frequency
        args.skip_existing = skip_existing
    else:
        # Стандартный запуск через командную строку
        parser = argparse.ArgumentParser(description="Запуск экспериментов Фазы 2.5")
        parser.add_argument("--exp", type=str, default="2.5.1.0", help="ID эксперимента (например 2.5.1.0)")
        parser.add_argument("--model", type=str, choices=['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D', 'all'], default='all')
        parser.add_argument("--complexity", type=str, choices=['light', 'medium', 'heavy', 'all'], default='light')
        parser.add_argument("--samples", type=int, default=1, help="Количество случайных окон из каждого файла за эпоху")
        parser.add_argument("--epochs", type=int, default=30, help="Количество эпох обучения")
        parser.add_argument("--checkpoint_freq", type=int, default=1, help="Частота сохранения чекпоинтов (каждые N эпох)")
        parser.add_argument("--no-skip", action="store_false", dest="skip_existing", help="Не пропускать уже обученные модели")
        parser.set_defaults(skip_existing=True)
        args = parser.parse_args()
        args.samples_per_file = args.samples
        args.checkpoint_frequency = args.checkpoint_freq

    # 1. Setup Paths
    METADATA_FILE = ROOT_DIR / 'data' / 'ml_datasets' / 'labeled_2025_12_03.csv'
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'

    print(f"Загрузка данных из {METADATA_FILE}")
    df = pl.read_csv(METADATA_FILE, infer_schema_length=10000)
    df = clean_labels(df)
    df = add_base_labels(df)
    df = df.with_row_index("row_nr")
    
    # Сплит данных (на уровне файлов для корректности)
    unique_files = df["file_name"].unique().to_list()
    np.random.seed(42)
    np.random.shuffle(unique_files)
    
    split_idx = int(len(unique_files) * 0.8)
    train_files = unique_files[:split_idx]
    val_files = unique_files[split_idx:]
    
    # Создаем индексы начал окон
    # Используем встроенный метод Dataset для корректной обработки границ окон и случайного сэмплирования
    window_size = 320
    
    train_indices = OscillogramDataset.create_indices(
        df.filter(pl.col("file_name").is_in(train_files)), 
        window_size=window_size, 
        mode='train',
        samples_per_file=args.samples_per_file
    )
    val_indices = OscillogramDataset.create_indices(
        df.filter(pl.col("file_name").is_in(val_files)), 
        window_size=window_size, 
        mode='val'
    )
    
    default_stride = 16 # Базовый страйд по умолчанию
    
    # target_cols = get_target_columns('base') # Moved to loop
    feature_cols = None # Использовать Smart Selector
    
    models = ['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D'] if args.model == 'all' else [args.model]
    complexities = ['light', 'medium', 'heavy'] if args.complexity == 'all' else [args.complexity]
    
    # Настройка параметров в зависимости от эксперимента
    exp_params = {
        "2.5.1.0": {"feature_mode": "raw", "sampling": "none", "use_pw": False, "aug": False, "target_level": "base"},
        "2.5.1.1": {"feature_mode": "raw", "sampling": "none", "use_pw": True, "aug": False, "target_level": "base"},
        "2.5.1.2": {"feature_mode": "raw", "sampling": "none", "use_pw": True, "aug": True, "target_level": "base"},
        
        # Исследование стратегий прореживания на сырых данных (Raw Data Downsampling)
        "2.5.2.0": {"feature_mode": "raw", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.2.1": {"feature_mode": "raw", "sampling": "snapshot", "stride": 32, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.2.2": {"feature_mode": "raw", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.2.3": {"feature_mode": "raw", "sampling": "snapshot", "stride": 32, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.2.4": {"feature_mode": "raw", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.2.5": {"feature_mode": "raw", "sampling": "snapshot", "stride": 32, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        
        # Исследование признаков (Feature Type Optimization)
        "2.5.3.0": {"feature_mode": "symmetric_polar", "sampling": "snapshot", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.1_rect":  {"feature_mode": "symmetric", "sampling": "snapshot", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.1_polar": {"feature_mode": "symmetric_polar", "sampling": "snapshot", "use_pw": True, "aug": True, "target_level": "base"},
        
        # Сравнение Фазных признаков (8 каналов)
        "2.5.3.1_phase_rect":  {"feature_mode": "phase_complex", "sampling": "snapshot", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.1_phase_polar": {"feature_mode": "phase_polar", "sampling": "snapshot", "use_pw": True, "aug": True, "target_level": "base"},

        "2.5.3.2_power": {"feature_mode": "power", "sampling": "snapshot", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.2_ab":    {"feature_mode": "alpha_beta", "sampling": "snapshot", "use_pw": True, "aug": True, "target_level": "base"},
        
        # Развитие раздела 3: Исследование признаков на временных рядах (Strided, Heavy)
        # Эти эксперименты используют Stride 16 и Heavy сложность для глубокого анализа
        "2.5.3.0_strided": {"feature_mode": "symmetric_polar", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.1_rect_strided":  {"feature_mode": "symmetric", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.1_polar_strided": {"feature_mode": "symmetric_polar", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.1_phase_rect_strided":  {"feature_mode": "phase_complex", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.1_phase_polar_strided": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.2_power_strided": {"feature_mode": "power", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.3.2_ab_strided":    {"feature_mode": "alpha_beta", "sampling": "stride", "stride": 16, "complexity": "heavy", "use_pw": True, "aug": True, "target_level": "base"},

        # Раздел 4: Исследование признаков при средней сложности (Medium)
        "2.5.4.0_strided": {"feature_mode": "symmetric_polar", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_rect_strided":  {"feature_mode": "symmetric", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_polar_strided": {"feature_mode": "symmetric_polar", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_phase_rect_strided":  {"feature_mode": "phase_complex", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_phase_polar_strided": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.2_power_strided": {"feature_mode": "power", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.2_ab_strided":    {"feature_mode": "alpha_beta", "sampling": "stride", "stride": 16, "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},

        "2.5.4.0_snapshot": {"feature_mode": "symmetric_polar", "sampling": "snapshot", "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_rect_snapshot":  {"feature_mode": "symmetric", "sampling": "snapshot", "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_polar_snapshot": {"feature_mode": "symmetric_polar", "sampling": "snapshot", "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_phase_rect_snapshot":  {"feature_mode": "phase_complex", "sampling": "snapshot", "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.1_phase_polar_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.2_power_snapshot": {"feature_mode": "power", "sampling": "snapshot", "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.4.2_ab_snapshot":    {"feature_mode": "alpha_beta", "sampling": "snapshot", "complexity": "medium", "use_pw": True, "aug": True, "target_level": "base"},

        # Раздел 5: Исследование признаков при низкой сложности (Light)
        "2.5.5.0_strided": {"feature_mode": "symmetric_polar", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_rect_strided":  {"feature_mode": "symmetric", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_polar_strided": {"feature_mode": "symmetric_polar", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_phase_rect_strided":  {"feature_mode": "phase_complex", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_phase_polar_strided": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.2_power_strided": {"feature_mode": "power", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.2_ab_strided":    {"feature_mode": "alpha_beta", "sampling": "stride", "stride": 16, "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},

        "2.5.5.0_snapshot": {"feature_mode": "symmetric_polar", "sampling": "snapshot", "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_rect_snapshot":  {"feature_mode": "symmetric", "sampling": "snapshot", "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_polar_snapshot": {"feature_mode": "symmetric_polar", "sampling": "snapshot", "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_phase_rect_snapshot":  {"feature_mode": "phase_complex", "sampling": "snapshot", "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.1_phase_polar_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.2_power_snapshot": {"feature_mode": "power", "sampling": "snapshot", "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.5.2_ab_snapshot":    {"feature_mode": "alpha_beta", "sampling": "snapshot", "complexity": "light", "use_pw": True, "aug": True, "target_level": "base"},

        "2.5.6.1":       {"feature_mode": "symmetric_polar", "sampling": "stride", "use_pw": True, "aug": True, "target_level": "base"},
        "2.5.6.2":       {"feature_mode": "symmetric_polar", "sampling": "stride", "use_pw": True, "aug": True, "target_level": "full"},
        "2.5.7.1":       {"feature_mode": "symmetric_polar", "sampling": "stride", "use_pw": True, "aug": True, "target_level": "full"},
    }
    
    if args.exp not in exp_params:
        print(f"Неизвестный ID эксперимента: {args.exp}")
        return

    p = exp_params[args.exp]
    
    for model_name in models:
        for comp in complexities:
            # Маппинг сложности под стратегию
            # Лёгкая - для ALL (dense/none)
            # Средняя - для Strided
            # Тяжёлая - для Snapshot
            # Но если пользователь явно указал --complexity, используем её.
            
            actual_comp = comp
            current_stride = p.get('stride', default_stride)
            current_harmonics = 1 # по умолчанию
            
            # Если в параметрах эксперимента жестко задана сложность, используем её (для 2.5.2.x и 2.5.3.x strided)
            if 'complexity' in p and args.complexity == 'all':
                actual_comp = p['complexity']
                # Для Heavy сложности в этих разделах по умолчанию используем 9 гармоник
                if actual_comp == 'heavy':
                    current_harmonics = 9
                elif actual_comp == 'medium':
                    current_harmonics = 3
                
                if comp != 'light': # Чтобы не запускать 3 раза одно и то же
                    continue
            
            # Логика автоматического выбора параметров для обычных экспериментов
            elif args.complexity == 'all': # Только авто-выбор
                if p['sampling'] == 'stride':
                    # Medium: Stride 32, Harmonics 3 (как в ТЗ)
                    actual_comp = 'medium'
                    current_stride = 32
                    current_harmonics = 3
                elif p['sampling'] == 'snapshot':
                    # Heavy: Snapshot (нет страйда), Harmonics 9 (как в ТЗ)
                    actual_comp = 'heavy'
                    current_harmonics = 9
                else: 
                    # Light: Stride 2 (для Dense/None?), Harmonics 1
                    actual_comp = 'light'
                    current_stride = 2 # Если sampling='none', stride игнорируется
                    current_harmonics = 1
            
                if comp != 'light': 
                    continue
            else:
                # Если сложность задана явно пользователем, подстраиваем гармоники
                if comp == 'light':
                    current_harmonics = 1
                    # stride берем из p или default
                elif comp == 'medium':
                    current_harmonics = 3
                    if p['sampling'] == 'stride' and 'stride' not in p:
                        current_stride = 32
                elif comp == 'heavy':
                    current_harmonics = 9
                    if p['sampling'] == 'stride' and 'stride' not in p:
                        current_stride = 32

            # Если ID эксперимента 2.5.2.x, убеждаемся что stride берется правильный
            if args.exp.startswith("2.5.2."):
                current_stride = p.get('stride', current_stride)
                actual_comp = p.get('complexity', actual_comp) if args.complexity == 'all' else comp

            # Проверка, обучена ли уже модель
            experiment_name = f"Exp_{args.exp}_{model_name}_{actual_comp}_{p['feature_mode']}_{p['sampling']}_{p['target_level']}"
            checkpoint_path = ROOT_DIR / 'experiments' / 'phase2_5' / experiment_name / 'final_model.pt'
            
            if args.skip_existing and checkpoint_path.exists():
                print(f">>> Пропуск {experiment_name} (уже обучено)")
                continue

            run_experiment(
                args.exp, model_name, actual_comp, df, train_indices, val_indices, 
                feature_cols, p['target_level'], NORM_COEF_PATH,
                feature_mode=p['feature_mode'], 
                sampling_strategy=p['sampling'],
                stride=current_stride,
                use_pos_weight=p['use_pw'],
                augment=p['aug'],
                num_harmonics=current_harmonics,
                epochs=args.epochs,
                checkpoint_frequency=args.checkpoint_frequency
            )

if __name__ == "__main__":
    # === ВЕРСИЯ 1: РУЧНОЙ ЗАПУСК ЧЕРЕЗ КОНСТАНТЫ (Рекомендуется для IDE/Notebooks) ===
    # Отредактируйте параметры ниже и раскомментируйте вызов main(), чтобы запустить одну модель.

    # 1. EXP_ID: Строковый ключ эксперимента (из словаря 'exp_params').
    # ПОЧЕМУ: Определяет физический смысл данных (признаки, гармоники, нормировку).
    # ЗАЧЕМ: Например, '2.5.3.1_phase_polar' активирует 8-канальные полярные признаки.
    EXPS = [
        # Medium Complexity (2.5.4.x)
        "2.5.4.0_strided", "2.5.4.1_rect_strided", "2.5.4.1_polar_strided", 
        "2.5.4.1_phase_rect_strided", "2.5.4.1_phase_polar_strided", 
        "2.5.4.2_power_strided", "2.5.4.2_ab_strided",
        "2.5.4.0_snapshot", "2.5.4.1_rect_snapshot", "2.5.4.1_polar_snapshot", 
        "2.5.4.1_phase_rect_snapshot", "2.5.4.1_phase_polar_snapshot", 
        "2.5.4.2_power_snapshot", "2.5.4.2_ab_snapshot",
        
        # Light Complexity (2.5.5.x)
        "2.5.5.0_strided", "2.5.5.1_rect_strided", "2.5.5.1_polar_strided", 
        "2.5.5.1_phase_rect_strided", "2.5.5.1_phase_polar_strided", 
        "2.5.5.2_power_strided", "2.5.5.2_ab_strided",
        "2.5.5.0_snapshot", "2.5.5.1_rect_snapshot", "2.5.5.1_polar_snapshot", 
        "2.5.5.1_phase_rect_snapshot", "2.5.5.1_phase_polar_snapshot", 
        "2.5.5.2_power_snapshot", "2.5.5.2_ab_snapshot",
    ]
    # EXPS = ["2.5.4.0_strided"]  # Пример запуска одного эксперимента

    # 2. MODEL_TYPE: Название архитектуры нейросети.
    # ПОЧЕМУ: Выбирает, какой именно класс модели будет инстанцирован и обучен.
    # ДОСТУПНО: 'SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D'.
    # ЗАЧЕМ: 'all' запускает цикл по всем доступным моделям для сравнения.
    MODEL_TYPE = "all"
    # MODEL_TYPE = "SimpleCNN"  # Пример запуска одной модели

    # 3. SELECTED_COMPLEXITY: Уровень сложности модели.
    # ПОЧЕМУ: Мальтипликатор для количества каналов/нейронов.
    # ЗАЧЕМ: 'light' - быстро, 'medium' - сбалансировано, 'heavy' - максимально мощно.
    # ПРИМЕЧАНИЕ: Для некоторых экспериментов также переключает частоту выборки.
    SELECTED_COMPLEXITY = "all"  # Использовать сложности из exp_params для каждого эксперимента

    # 4. SAMPLES_PER_FILE: Количество случайных окон из одного файла за одну эпоху.
    # ПОЧЕМУ: Увеличивает объем обучающей выборки без добавления новых файлов.
    # ЗАЧЕМ: Если у нас 800 файлов, значение 12 даст ~10 000 обучающих примеров за эпоху.
    SAMPLES_PER_FILE = 12

    # 5. EPOCHS: Количество полных проходов по обучающей выборке.
    # ЗАЧЕМ: Регулирует длительность обучения. Часто достаточно 20-30 или 50 для стабильности.
    EPOCHS = 30

    # 6. CHECKPOINT_FREQUENCY: Частота сохранения чекпоинтов.
    # ПОЧЕМУ: Позволяет восстанавливать обучение с любой эпохи.
    # ЗАЧЕМ: 1 - каждую эпоху (для отладки), 5 - каждые 5 эпох (для долгого обучения).
    CHECKPOINT_FREQUENCY = EPOCHS+1  # Сохранять только в конце

    # Раскомментируйте строку ниже для запуска с этими параметрами:
    for exp_id in EXPS:
        if MODEL_TYPE == 'all':
            for m in ['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D']:
                main(exp=exp_id, model=m, complexity=SELECTED_COMPLEXITY, samples_per_file=SAMPLES_PER_FILE, epochs=EPOCHS, checkpoint_frequency=CHECKPOINT_FREQUENCY)
        else:
            main(exp=exp_id, model=MODEL_TYPE, complexity=SELECTED_COMPLEXITY, samples_per_file=SAMPLES_PER_FILE, epochs=EPOCHS, checkpoint_frequency=CHECKPOINT_FREQUENCY)

    # === ВЕРСИЯ 2: ЗАПУСК ПО УМОЛЧАНИЮ (CLI / Аргументы командной строки) ===
    # Если вызов main() выше закомментирован, скрипт будет ждать аргументы из терминала.
    # Пример: python run_phase2_5_all.py --exp 2.5.1.0 --model SimpleMLP --complexity light --samples 12 --epochs 30
    
    # main()

