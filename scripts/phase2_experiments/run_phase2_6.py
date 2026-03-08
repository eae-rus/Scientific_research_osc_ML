import polars as pl
import numpy as np
from pathlib import Path
import torch
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.labels import clean_labels, add_base_labels, get_target_columns, prepare_labels_for_experiment
from osc_tools.data_management import DatasetManager
from osc_tools.ml.class_balancing import (
    GlobalClassBalancer, OscillogramClassBalancer, 
    BalancingConfig, get_balancing_strategy
)

# Определение уровней сложности моделей для Фазы 2.6
MODEL_COMPLEXITY = {
    'light': {
        'SimpleMLP': {'hidden_sizes': [64, 32], 'dropout': 0.2},
        'SimpleCNN': {'channels': [16, 32], 'dropout': 0.2, 'pool_every': 1},
        'ConvKAN':   {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'SimpleKAN': {'hidden_sizes': [64, 32], 'grid_size': 3, 'dropout': 0.1},
        'PhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'PhysicsKANConditional': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'cPhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'ResNet1D':  {'layers': [1, 1, 1, 1], 'base_filters': 16},
        # Иерархические модели (2.6.1, 2.6.2)
        'HierarchicalMLP': {'channels': [64, 32], 'dropout': 0.2, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        'HierarchicalCNN': {'channels': [16, 32], 'dropout': 0.2, 'pool_every': 1, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        'HierarchicalConvKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        'HierarchicalSimpleKAN': {'channels': [64, 32], 'grid_size': 3, 'dropout': 0.1, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        'HierarchicalPhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        'HierarchicalResNet': {'layers': [1, 1, 1, 1], 'base_filters': 16, 'stem_config': {'independent_layers': 1, 'grouped_layers': 1}},
        # Гибридные модели (2.6.3) — параметры уменьшены вдвое для каждой ветки
        'HybridMLP': {'hidden_sizes': [32, 16], 'dropout': 0.2},
        'HybridCNN': {'channels': [8, 16], 'dropout': 0.2},
        'HybridConvKAN': {'channels': [4, 8], 'dropout': 0.1, 'grid_size': 3},
        'HybridSimpleKAN': {'hidden_sizes': [32, 16], 'grid_size': 3, 'dropout': 0.1},
        'HybridPhysicsKAN': {'channels': [4, 8], 'dropout': 0.1, 'grid_size': 3},
        'HybridResNet': {'layers': [1, 1, 1, 1], 'base_filters': 8}
    },
    'medium': {
        'SimpleMLP': {'hidden_sizes': [256, 128, 64], 'dropout': 0.3},
        'SimpleCNN': {'channels': [32, 64, 128], 'dropout': 0.3, 'pool_every': 1},
        'ConvKAN':   {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'SimpleKAN': {'hidden_sizes': [128, 64, 32], 'grid_size': 5, 'dropout': 0.2},
        'PhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'PhysicsKANConditional': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'cPhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'ResNet1D':  {'layers': [2, 2, 2, 2], 'base_filters': 32},
        # Иерархические модели (2.6.1, 2.6.2)
        'HierarchicalMLP': {'channels': [256, 128, 64], 'dropout': 0.3, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        'HierarchicalCNN': {'channels': [32, 64, 128], 'dropout': 0.3, 'pool_every': 1, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        'HierarchicalConvKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        'HierarchicalSimpleKAN': {'channels': [128, 64, 32], 'grid_size': 5, 'dropout': 0.2, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        'HierarchicalPhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        'HierarchicalResNet': {'layers': [2, 2, 2, 2], 'base_filters': 32, 'stem_config': {'independent_layers': 2, 'grouped_layers': 2}},
        # Гибридные модели (2.6.3)
        'HybridMLP': {'hidden_sizes': [128, 64, 32], 'dropout': 0.3},
        'HybridCNN': {'channels': [16, 32, 64], 'dropout': 0.3},
        'HybridConvKAN': {'channels': [8, 16, 32], 'dropout': 0.2, 'grid_size': 5},
        'HybridSimpleKAN': {'hidden_sizes': [64, 32, 16], 'grid_size': 5, 'dropout': 0.2},
        'HybridPhysicsKAN': {'channels': [8, 16, 32], 'dropout': 0.2, 'grid_size': 5},
        'HybridResNet': {'layers': [2, 2, 2, 2], 'base_filters': 16}
    },
    'heavy': {
        'SimpleMLP': {'hidden_sizes': [512, 256, 128, 64], 'dropout': 0.4},
        'SimpleCNN': {'channels': [64, 128, 256, 512], 'dropout': 0.4, 'pool_every': 1},
        'ConvKAN':   {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'SimpleKAN': {'hidden_sizes': [256, 128, 64, 32], 'grid_size': 5, 'dropout': 0.3},
        'PhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'PhysicsKANConditional': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'cPhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'ResNet1D':  {'layers': [3, 4, 6, 3], 'base_filters': 64},
        # Иерархические модели (2.6.1, 2.6.2)
        'HierarchicalMLP': {'channels': [512, 256, 128, 64], 'dropout': 0.4, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        'HierarchicalCNN': {'channels': [64, 128, 256, 512], 'dropout': 0.4, 'pool_every': 1, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        'HierarchicalConvKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        'HierarchicalSimpleKAN': {'channels': [256, 128, 64, 32], 'grid_size': 5, 'dropout': 0.3, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        'HierarchicalPhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        'HierarchicalResNet': {'layers': [3, 4, 6, 3], 'base_filters': 64, 'stem_config': {'independent_layers': 3, 'grouped_layers': 3}},
        # Гибридные модели (2.6.3)
        'HybridMLP': {'hidden_sizes': [256, 128, 64, 32], 'dropout': 0.4},
        'HybridCNN': {'channels': [32, 64, 128, 256], 'dropout': 0.4},
        'HybridConvKAN': {'channels': [16, 32, 64], 'dropout': 0.3, 'grid_size': 8},
        'HybridSimpleKAN': {'hidden_sizes': [128, 64, 32, 16], 'grid_size': 5, 'dropout': 0.3},
        'HybridPhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.3, 'grid_size': 8},
        'HybridResNet': {'layers': [3, 4, 6, 3], 'base_filters': 32}
    }
}

def get_model_params(model_name, complexity, num_classes):
    """Возвращает параметры модели на основе сложности."""
    params = MODEL_COMPLEXITY[complexity].get(model_name, {}).copy()
    
    # Унификация имен параметров для разных архитектур
    if model_name in ['SimpleMLP', 'SimpleKAN']:
        params['output_size'] = num_classes
    else:
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
    augment: bool = True,
    val_df: pl.DataFrame = None,
    use_precomputed_val: bool = True,
    balancing_mode: str = 'none',
    balancer: Any = None,
    num_harmonics: int = 9,
    target_level: str = 'base',
    target_window_mode: str = 'point',
    model_param_overrides: Optional[Dict[str, Any]] = None
):
    print(f"\n>>> Запуск эксперимента: {exp_name}")
    print(f"Модель: {model_name} ({complexity})")
    print(f"Балансировка: {balancing_mode}")
    print(f"Уровень меток: {target_level} ({len(target_cols)} классов)")

    # === Специальная логика для гибридных моделей ===
    effective_feature_mode = feature_mode
    features_mode_for_hybrid = feature_mode
    effective_use_precomputed_val = use_precomputed_val

    if model_name.startswith('Hybrid'):
        # Гибриды ожидают raw + features
        if isinstance(feature_mode, list):
            if 'raw' not in feature_mode:
                effective_feature_mode = ['raw'] + feature_mode
            features_mode_for_hybrid = next((m for m in effective_feature_mode if m != 'raw'), 'raw')
        else:
            effective_feature_mode = ['raw', feature_mode]
            features_mode_for_hybrid = feature_mode

    if model_name == 'cPhysicsKAN':
        modes = effective_feature_mode if isinstance(effective_feature_mode, list) else [effective_feature_mode]
        if modes != ['phase_polar']:
            raise ValueError(
                f"cPhysicsKAN поддерживает только feature_mode='phase_polar', получено: {effective_feature_mode}"
            )

    def get_features_tail_len(mode: str) -> int:
        """Определяет длину хвоста для features-ветки гибридов."""
        spectral_modes = {
            'symmetric', 'symmetric_polar', 'phase_polar', 'phase_complex',
            'power', 'polar'
        }
        if mode in spectral_modes:
            return 1
        return 32

    # Настройка веса Loss функции
    use_pos_weight = (balancing_mode == 'weights')
    train_config_base.use_pos_weight = use_pos_weight
    
    # 1. Получение параметров модели
    num_classes = len(target_cols)
    model_params = get_model_params(model_name, complexity, num_classes)
    
    # Балансировка индексов (если выбрана стратегия активной выборки)
    actual_train_indices = train_indices
    actual_batch_size = data_config_base.batch_size
    
    if balancing_mode in ['global', 'oscillogram'] and balancer is not None:
        print(f"  [Активная балансировка: {balancing_mode}]")
        rng = np.random.default_rng(42)  # Фиксированный seed
        actual_train_indices = balancer.create_epoch_indices(rng)
        
        if balancing_mode == 'global':
            actual_batch_size, steps = balancer.get_batch_config()
            print(f"    Batch size: {actual_batch_size}, Steps: {steps}, Total: {len(actual_train_indices)}")

    # Маппинг target_level для OscillogramDataset (ожидает 'base_labels' вместо 'base')
    ds_target_level = 'base_labels' if target_level == 'base' else target_level
    
    # 2. Подготовка Dataset
    train_ds = OscillogramDataset(
        dataframe=df, indices=actual_train_indices, window_size=data_config_base.window_size,
        mode='classification', feature_mode=effective_feature_mode,
        sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
        target_columns=target_cols, target_level=ds_target_level,
        target_window_mode=target_window_mode,
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        augment=augment,
        num_harmonics=num_harmonics
    )
    
    # Валидационный датасет - используем предрассчитанные данные если возможно
    supported_precomputed_modes = {
        'raw', 'phase_polar', 'symmetric', 'symmetric_polar',
        'phase_complex', 'power', 'alpha_beta', 'phase_polar_h1_angle'
    }
    modes_for_precomputed = effective_feature_mode if isinstance(effective_feature_mode, list) else [effective_feature_mode]
    can_use_precomputed = (
        effective_use_precomputed_val and
        val_df is not None and
        all(m in supported_precomputed_modes for m in modes_for_precomputed)
    )
    
    if can_use_precomputed:
        try:
            print(f"  [Использую предрассчитанные признаки для валидации]")
            val_ds = PrecomputedDataset(
                dataframe=val_df, indices=val_indices, window_size=data_config_base.window_size,
                feature_mode=effective_feature_mode,
                target_columns=target_cols, target_level=target_level,
                sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
                num_harmonics=num_harmonics,
                target_window_mode=target_window_mode
            )
        except ValueError as e:
            print(f"  [!] Предрасчёт недоступен: {e}")
            print(f"  [Расчёт признаков на лету для валидации]")
            val_ds = OscillogramDataset(
                dataframe=df, indices=val_indices, window_size=data_config_base.window_size,
                mode='classification', feature_mode=effective_feature_mode,
                sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
                target_columns=target_cols, target_level=ds_target_level,
                target_window_mode=target_window_mode,
                physical_normalization=True, norm_coef_path=str(norm_coef_path),
                augment=False,
                num_harmonics=num_harmonics
            )
    else:
        print(f"  [Расчёт признаков на лету для валидации]")
        val_ds = OscillogramDataset(
            dataframe=df, indices=val_indices, window_size=data_config_base.window_size,
            mode='classification', feature_mode=effective_feature_mode,
            sampling_strategy=sampling_strategy, downsampling_stride=downsampling_stride,
            target_columns=target_cols, target_level=ds_target_level,
            target_window_mode=target_window_mode,
            physical_normalization=True, norm_coef_path=str(norm_coef_path),
            augment=False,
            num_harmonics=num_harmonics
        )
    
    # Динамическое уменьшение batch_size для тяжёлых режимов (экономия GPU памяти)
    mode_for_harmonics = features_mode_for_hybrid if model_name.startswith('Hybrid') else feature_mode
    is_harmonic_mode = mode_for_harmonics in ['phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex']
    base_batch_size = data_config_base.batch_size
    val_batch_size = 8192

    if is_harmonic_mode and num_harmonics >= 3:
        val_batch_size = 2048

    if is_harmonic_mode and complexity == 'heavy' and model_name in ['PhysicsKAN', 'PhysicsKANConditional', 'cPhysicsKAN', 'ConvKAN', 'ResNet1D', 'HierarchicalPhysicsKAN', 'HierarchicalConvKAN', 'HierarchicalResNet']:
        val_batch_size = 1024

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=base_batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    
    # 3. Определение in_channels из первого семпла
    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]
    seq_len = sample_x.shape[1]
    
    model_params['in_channels'] = in_channels
    
    # Модели, которые ожидают input_size (обычно плоские MLP/KAN)
    if model_name in ['SimpleMLP', 'SimpleKAN', 'HierarchicalSimpleKAN', 'PhysicsKAN', 'HierarchicalPhysicsKAN', 
                      'HybridMLP', 'HybridSimpleKAN']:
        model_params['input_size'] = in_channels * seq_len
        
    # SimpleMLP и SimpleKAN работают только с flatten вектором и не принимают параметр in_channels
    if model_name in ['SimpleMLP', 'SimpleKAN']:
        model_params.pop('in_channels', None)

    if model_name in ['PhysicsKAN', 'HierarchicalPhysicsKAN', 'cPhysicsKAN'] and sampling_strategy == 'snapshot':
        model_params['use_mlp'] = True
        model_params['input_size'] = in_channels * seq_len

    if model_param_overrides:
        model_params.update(model_param_overrides)
    
    # Гибридные модели нуждаются в параметрах разделения каналов
    if model_name.startswith('Hybrid'):
        # Разделение: первые 8 каналов — raw (I,U), остальные — features (phase_polar и т.д.)
        model_params['raw_channels'] = 8
        model_params['features_channels'] = in_channels - 8
        model_params['features_seq_len'] = min(get_features_tail_len(features_mode_for_hybrid), seq_len)

        # seq_len нужен только для MLP/KAN веток
        if model_name in ['HybridMLP', 'HybridSimpleKAN']:
            model_params['seq_len'] = seq_len

    # Сохраняем режим формирования меток в конфиг
    data_config_base.target_window_mode = target_window_mode

    config = ExperimentConfig(
        model=ModelConfig(name=model_name, params=model_params),
        data=data_config_base,
        training=train_config_base
    )
    config.training.experiment_name = exp_name
    
    runner = ExperimentRunner(config)
    history = runner.train(train_loader, val_loader)
    
    # Сохранение истории.
    # На Windows длинные имена файлов могут превышать MAX_PATH,
    # поэтому основное имя делаем коротким и стабильным.
    runner.save_dir.mkdir(parents=True, exist_ok=True)
    history_path = runner.save_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    # Legacy-имя сохраняем по возможности (обратная совместимость).
    legacy_history_path = runner.save_dir / f"{exp_name}_history.json"
    if legacy_history_path != history_path:
        try:
            with open(legacy_history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4, ensure_ascii=False)
        except OSError:
            pass
        
    return history

def main(exp: str = None, model: str = None, complexity: str = None, samples_per_file: int = 12, epochs: int = 30, checkpoint_frequency: int = 1, skip_existing: bool = True):
    """Главная точка входа для запуска экспериментов Phase 2.6."""
    
    # Режим параметров
    target_exp = exp
    target_model = model or 'all'
    target_complexity = complexity or 'medium'
    target_spf = samples_per_file
    target_epochs = epochs

    # Настройка путей
    DATA_DIR = ROOT_DIR / 'data' / 'ml_datasets'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments' / 'phase2_6'
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'
    METADATA_FILE = DATA_DIR / 'train.csv'
    WINDOW_SIZE = 320

    # Используем DatasetManager для гарантированного разделения данных
    print("Инициализация DatasetManager...")
    dm = DatasetManager(str(DATA_DIR), norm_coef_path=str(NORM_COEF_PATH))
    dm.ensure_train_test_split()  # Создаёт train.csv/test.csv если их нет

    # Предрасчёт с максимальным числом гармоник (для всех режимов)
    PRECOMPUTED_NUM_HARMONICS = 9

    def needs_precomputed_regen(num_harmonics: int) -> bool:
        precomputed_path = DATA_DIR / dm.PRECOMPUTED_TEST_CSV
        if not precomputed_path.exists():
            return True
        try:
            header_cols = set(pl.read_csv(precomputed_path, n_rows=1, infer_schema_length=0).columns)
        except Exception:
            return True
        required_modes = [
            'raw', 'phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex',
            'power', 'alpha_beta'
        ]
        required_cols: set[str] = set()
        for mode in required_modes:
            required_cols.update(dm.get_precomputed_feature_columns(mode, num_harmonics=num_harmonics))
        return not required_cols.issubset(header_cols)

    force_precompute = needs_precomputed_regen(PRECOMPUTED_NUM_HARMONICS)
    if force_precompute:
        print("[DatasetManager] Предрасчёт требует обновления — пересоздание файла test_precomputed.csv")
    dm.create_precomputed_test_csv(force=force_precompute, num_harmonics=PRECOMPUTED_NUM_HARMONICS)
    
    # target_cols определяется позже в зависимости от target_level эксперимента
    
    # === Предподготовка балансировщиков ===
    balancers_cache: Dict[str, Any] = {}
    
    def get_or_create_balancer(strategy: str, level: str = 'base') -> Optional[Any]:
        """Создаёт или возвращает кэшированный балансировщик."""
        if strategy == 'none' or strategy == 'weights':
            return None
        
        cache_key = f"{strategy}_{level}"
        if cache_key in balancers_cache:
            return balancers_cache[cache_key]
        
        print(f"[Предподготовка балансировщика: {strategy} level: {level}]")
        # Для балансировщика используем метки соответствующего уровня
        target_cols_for_balancing = get_target_columns(level, df)
        config = BalancingConfig(
            min_batch_size=64,
            samples_per_oscillogram=target_spf,
            total_samples_per_epoch=10000,
            cache_dir=str(DATA_DIR / 'balancing_cache')
        )
        balancer = get_balancing_strategy(strategy, df, target_cols_for_balancing, WINDOW_SIZE, config)
        if balancer:
            balancer.analyze()
        balancers_cache[cache_key] = balancer
        return balancer

    data_config = DataConfig(
        path=str(METADATA_FILE), 
        window_size=WINDOW_SIZE, 
        batch_size=64, 
        mode='multilabel',
        norm_coef_path=str(NORM_COEF_PATH)
    )
    train_config = TrainingConfig(
        epochs=target_epochs, 
        learning_rate=0.001, 
        use_pos_weight=True, # Will be updated in run loop based on strategy
        checkpoint_frequency=checkpoint_frequency,
        save_dir=str(EXPERIMENTS_DIR)
    )

    # Таблица экспериментов (логика данных)
    exp_params = {
        # === Эксперимент 2.6.1: Калибровка базовых моделей ===
        "2.6.1_stride":   {"feature_mode": "phase_polar", "sampling": "stride",   "stride": 16, "aug": True, "balancing": "weights", "target_level": "base"},
        "2.6.1_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "stride": 32, "aug": True, "balancing": "weights", "target_level": "base"},
        "2.6.1_global_stride": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "aug": True, "balancing": "global", "target_level": "base"},

        # === Эксперимент 2.6.2: Иерархические модели ===
        "2.6.2_stride":   {"feature_mode": "phase_polar", "sampling": "stride",   "stride": 16, "aug": True, "balancing": "weights", "target_level": "base"},
        "2.6.2_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "stride": 32, "aug": True, "balancing": "weights", "target_level": "base"},
        
        # === Эксперимент 2.6.3: Гибридные модели (Raw + Phase Polar) ===
        "2.6.3_stride":   {"feature_mode": "phase_polar", "sampling": "stride",   "stride": 16, "aug": True, "balancing": "weights", "target_level": "base"},
        "2.6.3_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "stride": 32, "aug": True, "balancing": "weights", "target_level": "base"},
        
        # === Эксперимент 2.6.4: Гранулярность меток (Target Granularity) ===
        # Вариант А: base_labels (4 обобщённых класса) — уже покрыто 2.6.1
        # Вариант Б: full (все ML_* колонки, независимые)
        "2.6.4_full_stride":   {"feature_mode": "phase_polar", "sampling": "stride",   "stride": 16, "aug": True, "balancing": "weights", "target_level": "full"},
        "2.6.4_full_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "stride": 32, "aug": True, "balancing": "weights", "target_level": "full"},
        "2.6.4_full_global_stride": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "aug": True, "balancing": "global", "target_level": "full"},
        # Вариант В: full_by_levels (все ML_* с иерархическим распространением)
        "2.6.4_hier_stride":   {"feature_mode": "phase_polar", "sampling": "stride",   "stride": 16, "aug": True, "balancing": "weights", "target_level": "full_by_levels"},
        "2.6.4_hier_snapshot": {"feature_mode": "phase_polar", "sampling": "snapshot", "stride": 32, "aug": True, "balancing": "weights", "target_level": "full_by_levels"},
        "2.6.4_hier_global_stride": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "aug": True, "balancing": "global", "target_level": "full_by_levels"},

        # === Эксперимент 2.6.7: Финальный тест (200 эпох) ===
        "2.6.7_baseline_200": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "aug": True, "balancing": "weights", "target_level": "base", "epochs": 200, "models_override": ["PhysicsKAN"], "complexities_override": ["heavy"],},
        "2.6.7_conditional_200": { "feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "aug": True, "balancing": "weights", "target_level": "base_sequential", "epochs": 200, "models_override": ["PhysicsKANConditional"], "complexities_override": ["heavy"]},

        # === Эксперимент 2.6.8: Метка по всему окну (сдвиг вправо) ===
        "2.6.8_stride":   {"feature_mode": "phase_polar", "sampling": "stride",  "stride": 16, "aug": True, "balancing": "weights", "target_level": "base", "target_window": "any_in_window"},

        # === Эксперимент 2.6.9: Комплексная PhysicsKAN (cPhysicsKAN) ===
        "2.6.9_stride": {"feature_mode": "phase_polar", "sampling": "stride", "stride": 16, "aug": True, "balancing": "weights", "target_level": "base", "models_override": ["cPhysicsKAN"]}
    }

    if target_exp not in exp_params:
        print(f"!!! Ошибка: Эксперимент {target_exp} не определен в exp_params")
        return

    p = exp_params[target_exp]
    
    # Определяем список моделей для запуска
    if target_model == 'all':
        if p.get("models_override"):
            models_to_run = p["models_override"]
        elif target_exp.startswith("2.6.3"):
            # Гибридные модели для эксперимента 2.6.3
            models_to_run = [
                'HybridMLP', 'HybridCNN', 'HybridResNet',
                'HybridSimpleKAN', 'HybridConvKAN', 'HybridPhysicsKAN'
            ]
        elif target_exp.startswith("2.6.2"):
            # Иерархические модели для эксперимента 2.6.2
            models_to_run = [
                'HierarchicalMLP', 'HierarchicalCNN', 'HierarchicalConvKAN', 
                'HierarchicalResNet', 'HierarchicalSimpleKAN', 'HierarchicalPhysicsKAN'
            ]
        elif target_exp.startswith("2.6.4"):
            # Базовые модели для эксперимента 2.6.4 (гранулярность меток)
            models_to_run = ['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D']
        else:
            # По умолчанию: базовые модели (2.6.1)
            models_to_run = ['SimpleMLP', 'SimpleCNN', 'ConvKAN', 'SimpleKAN', 'PhysicsKAN', 'ResNet1D']
    else:
        models_to_run = [target_model]

    # Определяем список сложностей
    if p.get("complexities_override"):
        complexities_to_run = p["complexities_override"]
    else:
        complexities_to_run = ['light', 'medium', 'heavy'] if target_complexity == 'all' else [target_complexity]
    
    # Определяем target_level из параметров эксперимента
    exp_target_level = p.get('target_level', 'base')
    exp_target_window = p.get('target_window', 'point')

    # === ПРЕДВАРИТЕЛЬНЫЙ ПРОХОД: Определяем, нужна ли загрузка данных ===
    need_training = False
    for m_name in models_to_run:
        for comp in complexities_to_run:
            exp_id_clean = target_exp.split('_')[0]
            feature_mode = p["feature_mode"]
            sampling_strategy = p["sampling"]
            
            full_exp_name = f"Exp_{exp_id_clean}_{m_name}_{comp}_{feature_mode}_{sampling_strategy}_{exp_target_level}"

            if exp_target_window == 'any_in_window':
                full_exp_name += "_win_any"
            
            # Добавляем суффиксы балансировки и аугментации в имя
            if p.get('balancing', 'none') != 'none':
                full_exp_name += f"_{p.get('balancing')}"
            if p.get("aug", True):
                full_exp_name += "_aug"
            
            # Проверка существования
            checkpoint_path = Path(train_config.save_dir) / full_exp_name / "final_model.pt"
            if not (skip_existing and checkpoint_path.exists()):
                need_training = True
                break
        if need_training:
            break
    
    # === ЗАГРУЗКА ДАННЫХ (один раз в начале, если нужно) ===
    if need_training:
        print(f"Загрузка тренировочных данных...")
        df = dm.load_train_df()
        df = df.with_row_index("row_nr")
        
        # Подготовка меток в зависимости от target_level
        if exp_target_level in ('full', 'full_by_levels', 'base_sequential'):
            print(f"  [Подготовка меток для уровня: {exp_target_level}]")
            df = prepare_labels_for_experiment(df, exp_target_level)
        
        # Создаём индексы для тренировки
        train_indices = OscillogramDataset.create_indices(
            df, 
            window_size=WINDOW_SIZE, mode='train', samples_per_file=target_spf
        )
        
        print(f"Загрузка предрассчитанных тестовых данных...")
        test_df = dm.load_test_df(precomputed=True)
        test_df = test_df.with_row_index("row_nr")
        
        # Подготовка меток для валидации
        if exp_target_level in ('full', 'full_by_levels', 'base_sequential'):
            test_df = prepare_labels_for_experiment(test_df, exp_target_level)
        
        # Создаём индексы для валидации с шагом 4 (полная валидация как в aggregate_reports)
        VAL_STRIDE = 4
        val_indices = PrecomputedDataset.create_indices(
            test_df,
            window_size=WINDOW_SIZE,
            mode='val',
            stride=VAL_STRIDE  # Шаг 4 для полного покрытия
        )
        
        # Определяем целевые колонки в зависимости от target_level
        target_cols = get_target_columns(exp_target_level, df)
        print(f"  [Целевые колонки ({exp_target_level}): {len(target_cols)} классов]")
    else:
        print("Все модели уже обучены, пропуск загрузки данных")
        return

    def get_num_harmonics_by_complexity(level: str) -> int:
        if level == 'heavy':
            return 9
        if level == 'medium':
            return 3
        return 1

    for m_name in models_to_run:
        for comp in complexities_to_run:
            exp_id_clean = target_exp.split('_')[0]
            feature_mode = p["feature_mode"]
            sampling_strategy = p["sampling"]
            
            full_exp_name = f"Exp_{exp_id_clean}_{m_name}_{comp}_{feature_mode}_{sampling_strategy}_{exp_target_level}"

            if exp_target_window == 'any_in_window':
                full_exp_name += "_win_any"
            
            # Добавляем суффиксы балансировки и аугментации в имя
            if p.get('balancing', 'none') != 'none':
                full_exp_name += f"_{p.get('balancing')}"
            if p.get("aug", True):
                full_exp_name += "_aug"
            
            # Проверка существования
            checkpoint_path = Path(train_config.save_dir) / full_exp_name / "final_model.pt"
            if skip_existing and checkpoint_path.exists():
                print(f">>> Пропуск {full_exp_name} (уже обучено)")
                continue

            try:
                exp_epochs = p.get("epochs", target_epochs)
                exp_data_mode = p.get("data_mode", data_config.mode)

                data_config_exp = DataConfig(
                    path=data_config.path,
                    window_size=data_config.window_size,
                    batch_size=data_config.batch_size,
                    mode=exp_data_mode,
                    norm_coef_path=data_config.norm_coef_path
                )
                train_config_exp = TrainingConfig(
                    epochs=exp_epochs,
                    learning_rate=train_config.learning_rate,
                    use_pos_weight=train_config.use_pos_weight,
                    checkpoint_frequency=checkpoint_frequency,
                    save_dir=train_config.save_dir,
                    ml23_loss_weight=train_config.ml23_loss_weight
                )
                current_harmonics = get_num_harmonics_by_complexity(comp)
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
                    data_config_base=data_config_exp,
                    train_config_base=train_config_exp,
                    norm_coef_path=NORM_COEF_PATH,
                    augment=p.get("aug", True),
                    val_df=test_df,
                    use_precomputed_val=True,
                    balancing_mode=p.get('balancing', 'weights'),
                    balancer=get_or_create_balancer(p.get('balancing', 'weights'), exp_target_level),
                    num_harmonics=current_harmonics,
                    target_level=exp_target_level,
                    target_window_mode=exp_target_window
                )
            except Exception as e:
                print(f"!!! Ошибка в {full_exp_name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    # === ВЕРСИЯ 1: РУЧНОЙ ЗАПУСК ===
    
    # Набор экспериментов (группы данных)
    # 2.6.1 - базовые модели (калибровка)
    # 2.6.2 - иерархические модели
    # 2.6.3 - гибридные модели
    # 2.6.4 - гранулярность меток (full, full_by_levels)
    # 2.6.7 - финальный тест (200 эпох, conditional heads)
    # 2.6.9 - комплексная PhysicsKAN (cPhysicsKAN)
    EXPS = [
        # === Эксперимент 2.6.1: Калибровка базовых моделей ===
        #"2.6.1_stride", "2.6.1_snapshot", "2.6.1_global_stride",
        
        # === Эксперимент 2.6.2: Иерархические модели ===
        # "2.6.2_stride", "2.6.2_snapshot",
        
        # === Эксперимент 2.6.3: Гибридные модели ===
        # "2.6.3_stride", "2.6.3_snapshot",
        
        # === Эксперимент 2.6.4: Гранулярность меток ===
        # Вариант Б: full (все ML_* колонки независимо)
        # "2.6.4_full_stride", "2.6.4_full_snapshot", "2.6.4_full_global_stride",
        # Вариант В: full_by_levels (все ML_* с иерархическим распространением)
        # "2.6.4_hier_stride", "2.6.4_hier_snapshot", "2.6.4_hier_global_stride",

        # === Эксперимент 2.6.7: Финальный тест (200 эпох) ===
        # "2.6.7_conditional_200",
        # "2.6.7_baseline_200", 

        # === Эксперимент 2.6.8: Метка по всему окну (сдвиг вправо) ===
        # "2.6.8_stride",
        
        # === Эксперимент 2.6.9: Комплексная PhysicsKAN (cPhysicsKAN) ===
        "2.6.9_stride"
    ]
    
    # Тип модели ('all' - выберет автоматически подходящие для группы)
    MODEL_TYPE = "all"
    
    # Уровень сложности ('all' - light, medium, heavy)
    SELECTED_COMPLEXITY = "all"
    
    # Плотность выборки (~10 000 примеров за эпоху при 12)
    SAMPLES_PER_FILE = 12
    
    # Эпохи
    EPOCHS = 60
    
    # Пропускать ли уже обученные модели (наличие final_model.pt)
    SKIP_EXISTING = True

    # Частота чекпоинтов (регулирует точки сохранения весов)
    CHECKPOINT_FREQUENCY = EPOCHS + 1 # По умолчанию сохраняем только в конце

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

