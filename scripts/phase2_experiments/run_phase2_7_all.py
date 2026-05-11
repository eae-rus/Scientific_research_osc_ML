"""
===============================================================================
Фаза 2.7: Высококачественные эксперименты для научных статей.

Улучшения по сравнению с Фазой 2.5:
1. 5-кратная перекрёстная проверка (Stratified K-Fold по осциллограммам)
2. Множественные seeds для оценки стабильности обучения
3. Полный перекрёстный перебор: типы данных × стратегии выборки × модели
4. Разделение на контексты статей (Статья 1: модели, Статья 2: входные данные)

Структура циклов:
  ц.1 — Fold (1..5): перекрёстная проверка на разных разбиениях данных
  ц.2 — Seed (1..N): проверка стабильности обучения (разные инициализации)
  ц.3 — Experiment (тип данных × стратегия выборки): перебор условий
  ц.4 — Model: выбор архитектуры (зависит от контекста статьи)
  ц.5 — Complexity: уровень сложности модели

Запуск:
  # Ручной режим (IDE/Notebook) — см. блок if __name__ == "__main__" внизу
  # CLI режим:
  python run_phase2_7_all.py --fold 1 --seed_idx 0 --exp raw_stride --model SimpleMLP --complexity medium
  python run_phase2_7_all.py --fold all --seed_idx all --exp all --model all --complexity all --epochs 1
  python run_phase2_7_all.py --article paper1 --fold 1 --seed_idx 0 --epochs 30

===============================================================================
"""
import polars as pl
import numpy as np
from pathlib import Path
import torch
import sys
import json
import argparse
from typing import Optional, List, Tuple, Dict, Any
from itertools import product

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from osc_tools.ml.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from osc_tools.ml.runner import ExperimentRunner
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.labels import clean_labels, add_base_labels, get_target_columns, get_ml_columns
from osc_tools.data_management import DatasetManager

# ==============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТОВ
# ==============================================================================

# --- Seeds для проверки стабильности обучения ---
# Генерируются с большим шагом от базового для максимального различия
BASE_SEED = 42
SEED_OFFSETS = [0, 7919, 15731, 23497, 31607]  # Простые числа для разнообразия


def get_seed(seed_idx: int) -> int:
    """Возвращает seed по индексу (0-based)."""
    return BASE_SEED + SEED_OFFSETS[seed_idx]


# --- Определение моделей ---
# Формат: (имя_модели, контексты_статей)
# 'paper1' = Статья 1 (сравнение моделей), 'paper2' = Статья 2 (сравнение данных)
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'SimpleMLP':  {'articles': ['paper1', 'paper2'], 'supports_all_features': True},
    'SimpleCNN':  {'articles': ['paper1', 'paper2'], 'supports_all_features': True},
    'ResNet1D':   {'articles': ['paper1', 'paper2'], 'supports_all_features': True},
    'SimpleKAN':  {'articles': ['paper1', 'paper2'], 'supports_all_features': True},
    'ConvKAN':    {'articles': ['paper1', 'paper2'], 'supports_all_features': True},
    'PhysicsKAN': {'articles': ['paper1'], 'supports_all_features': False, 'allowed_features': ['phase_polar']},
    'cPhysicsKAN': {'articles': ['paper1'], 'supports_all_features': False, 'allowed_features': ['phase_polar']},
    'rPhysicsKAN': {'articles': ['paper1'], 'supports_all_features': False, 'allowed_features': ['phase_polar']},
    # TODO: Будущие модели (заглушки — реализация на следующем этапе):
    # 'PatchTST':  {'articles': ['paper1', 'paper2'], 'supports_all_features': True, 'fixed_complexity': 'medium'},
    # 'TimesNet':  {'articles': ['paper1', 'paper2'], 'supports_all_features': True, 'fixed_complexity': 'medium'},
}

# --- Уровни сложности моделей ---
MODEL_COMPLEXITY = {
    'light': {
        'SimpleMLP': {'hidden_sizes': [64, 32], 'dropout': 0.2},
        'SimpleCNN': {'channels': [16, 32], 'dropout': 0.2, 'pool_every': 1},
        'ConvKAN':   {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'SimpleKAN': {'hidden_sizes': [64, 32], 'grid_size': 3, 'dropout': 0.1},
        'PhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'cPhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'rPhysicsKAN': {'channels': [8, 16], 'dropout': 0.1, 'grid_size': 3},
        'ResNet1D':  {'layers': [1, 1, 1, 1], 'base_filters': 16},
    },
    'medium': {
        'SimpleMLP': {'hidden_sizes': [256, 128, 64], 'dropout': 0.3},
        'SimpleCNN': {'channels': [32, 64, 128], 'dropout': 0.3, 'pool_every': 1},
        'ConvKAN':   {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'SimpleKAN': {'hidden_sizes': [128, 64, 32], 'grid_size': 5, 'dropout': 0.2},
        'PhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'cPhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'rPhysicsKAN': {'channels': [16, 32, 64], 'dropout': 0.2, 'grid_size': 5},
        'ResNet1D':  {'layers': [2, 2, 2, 2], 'base_filters': 32},
    },
    'heavy': {
        'SimpleMLP': {'hidden_sizes': [512, 256, 128, 64], 'dropout': 0.4},
        'SimpleCNN': {'channels': [64, 128, 256, 512], 'dropout': 0.4, 'pool_every': 1},
        'ConvKAN':   {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'SimpleKAN': {'hidden_sizes': [256, 128, 64, 32], 'grid_size': 5, 'dropout': 0.3},
        'PhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'cPhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'rPhysicsKAN': {'channels': [32, 64, 128], 'dropout': 0.3, 'grid_size': 8},
        'ResNet1D':  {'layers': [3, 4, 6, 3], 'base_filters': 64},
    }
}

# --- Типы входных данных (feature modes) ---
FEATURE_MODES = [
    'raw',              # Сырые сигналы (8 каналов)
    'phase_polar',      # Фазные полярные координаты (амплитуда + фаза для каждого канала)
    'phase_complex',    # Фазные комплексные (Re + Im)
    'symmetric',        # Симметричные составляющие (Re + Im)
    'symmetric_polar',  # Симметричные составляющие (амплитуда + фаза)
    'power',            # Мощность (P, Q)
    'alpha_beta',       # Преобразование Кларк (alpha, beta)
]

# --- Стратегии выборки (sampling) ---
SAMPLING_STRATEGIES = [
    'none',      # Dense: все 320 точек (baseline)
    'stride',    # Прореживание с шагом (stride=16 → 18 точек)
    'snapshot',  # Мгновенный снимок (1-2 точки, для raw - 64 точки: 32 начала + 32 конца, для остальных - 2 точки: первая валидная + последняя)
]

# --- Определение экспериментов как перекрёстного произведения ---
# Каждый эксперимент = (feature_mode, sampling_strategy)
# Не все комбинации валидны — фильтрация ниже


def get_experiment_list(article: str = 'both') -> List[Dict[str, str]]:
    """
    Возвращает список экспериментов (feature_mode × sampling_strategy).
    
    Args:
        article: 'paper1' — фиксированный feature_mode (phase_polar), все sampling
                 'paper2' — все feature_modes, все sampling strategies
                 'both'   — полный перебор всех валидных комбинаций
    """
    experiments = []
    
    if article == 'paper1':
        # Статья 1: Сравнение моделей. Фиксируем phase_polar, варьируем sampling.
        for sampling in SAMPLING_STRATEGIES:
            experiments.append({
                'feature_mode': 'phase_polar',
                'sampling': sampling,
            })
    elif article == 'paper2':
        # Статья 2: Сравнение входных данных. Варьируем feature_mode и sampling.
        for fm in FEATURE_MODES:
            for sampling in SAMPLING_STRATEGIES:
                experiments.append({
                    'feature_mode': fm,
                    'sampling': sampling,
                })
    else:  # 'both' — полный перебор
        for fm in FEATURE_MODES:
            for sampling in SAMPLING_STRATEGIES:
                experiments.append({
                    'feature_mode': fm,
                    'sampling': sampling,
                })
    
    return experiments


def get_models_for_article(article: str) -> List[str]:
    """Возвращает список моделей, подходящих для указанной статьи."""
    models = []
    for name, info in MODEL_REGISTRY.items():
        if article == 'both' or article in info['articles']:
            models.append(name)
    return models


def is_model_compatible(model_name: str, feature_mode: str, sampling: str) -> bool:
    """
    Проверяет, совместима ли модель с данной комбинацией feature_mode и sampling.
    
    Ограничения:
    - PhysicsKAN/cPhysicsKAN/rPhysicsKAN работают только с полярными признаками
    - Свёрточные модели (CNN, ConvKAN, ResNet1D) не могут работать со snapshot спектральных
      признаков (2 точки — слишком мало для свёртки)
    """
    info = MODEL_REGISTRY.get(model_name, {})
    
    # Проверяем допустимые feature_modes для модели
    if not info.get('supports_all_features', True):
        allowed = info.get('allowed_features', [])
        if feature_mode not in allowed:
            return False
    
    # Свёрточные модели не работают на snapshot спектральных данных (2 точки)
    if sampling == 'snapshot' and feature_mode != 'raw':
        conv_models = ['SimpleCNN', 'ConvKAN', 'ResNet1D', 'PhysicsKAN', 'cPhysicsKAN', 'rPhysicsKAN']
        if model_name in conv_models:
            return False
    
    return True


def get_complexities_for_model(model_name: str) -> List[str]:
    """Возвращает допустимые уровни сложности для модели."""
    info = MODEL_REGISTRY.get(model_name, {})
    if 'fixed_complexity' in info:
        return [info['fixed_complexity']]
    return ['light', 'medium', 'heavy']


# ==============================================================================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ==============================================================================

WINDOW_SIZE = 320
DEFAULT_STRIDE = 16
TARGET_LEVEL = 'base'
NUM_HARMONICS_BY_COMPLEXITY = {'light': 1, 'medium': 3, 'heavy': 9}
PRECOMPUTED_NUM_HARMONICS = 9  # Максимальное число гармоник для предрасчёта

# Аугментация — всегда включена (по результатам Фазы 2.5)
AUGMENTATION_ENABLED = True

# Взвешивание классов — используем pos_weight (BCE с весами частоты событий)
USE_POS_WEIGHT = True


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def compute_in_channels(feature_mode: str, num_harmonics: int) -> int:
    """Определяет количество входных каналов в зависимости от feature_mode."""
    if feature_mode == 'raw':
        return 8
    elif feature_mode in ['symmetric', 'symmetric_polar']:
        return 12 * num_harmonics
    elif feature_mode in ['phase_polar', 'phase_complex']:
        return 16 * num_harmonics
    elif feature_mode == 'power':
        return 8
    elif feature_mode == 'alpha_beta':
        return 6
    return 8  # fallback


def compute_sequence_length(feature_mode: str, sampling: str, stride: int) -> int:
    """Определяет длину последовательности (кол-во временных точек) для модели."""
    if sampling == 'none':
        return WINDOW_SIZE
    elif sampling == 'stride':
        if feature_mode != 'raw':
            return (WINDOW_SIZE - 32) // stride  # Пропускаем FFT warmup
        else:
            return WINDOW_SIZE // stride
    elif sampling == 'snapshot':
        if feature_mode == 'raw':
            return 64  # 32 начала + 32 конца
        else:
            return 2   # Первая валидная + последняя точка
    return WINDOW_SIZE


def compute_conv_params(feature_mode: str, sampling: str) -> Tuple[int, int]:
    """Определяет kernel_size и stride для свёрточных моделей."""
    if feature_mode == 'raw':
        if sampling == 'none':
            return 8, 1
        elif sampling == 'stride':
            return 8, 8
        elif sampling == 'snapshot':
            return 8, 1
    else:
        if sampling == 'none':
            return 8, 1
        elif sampling in ['stride', 'snapshot']:
            return 3, 1
    return 3, 1


def get_model_params(
    model_name: str, 
    complexity: str, 
    feature_mode: str,
    sampling: str,
    stride: int
) -> Dict[str, Any]:
    """Формирует полный набор параметров модели."""
    params = MODEL_COMPLEXITY[complexity].get(model_name, {}).copy()
    
    num_harmonics = NUM_HARMONICS_BY_COMPLEXITY[complexity]
    in_channels = compute_in_channels(feature_mode, num_harmonics)
    pts = compute_sequence_length(feature_mode, sampling, stride)
    target_cols = get_target_columns(TARGET_LEVEL)
    num_classes = len(target_cols)
    input_size = in_channels * pts
    
    kernel_size, conv_stride = compute_conv_params(feature_mode, sampling)
    
    if model_name in ['SimpleMLP', 'SimpleKAN']:
        params['input_size'] = input_size
        params['output_size'] = num_classes
    elif model_name in ['SimpleCNN', 'ConvKAN', 'PhysicsKAN', 'cPhysicsKAN', 'rPhysicsKAN']:
        params['in_channels'] = in_channels
        params['num_classes'] = num_classes
        params['kernel_size'] = kernel_size
        params['stride'] = conv_stride
    elif model_name == 'ResNet1D':
        params['in_channels'] = in_channels
        params['num_classes'] = num_classes
    
    # Для PhysicsKAN: если точек мало, включаем MLP режим
    if model_name == 'PhysicsKAN' and pts < 8:
        params['use_mlp'] = True
        params['input_size'] = input_size
    
    return params


def build_experiment_name(
    fold: int, 
    seed_idx: int, 
    feature_mode: str, 
    sampling: str,
    model_name: str, 
    complexity: str
) -> str:
    """Формирует имя эксперимента для сохранения результатов."""
    return f"Exp_2.7_f{fold}_s{seed_idx}_{model_name}_{complexity}_{feature_mode}_{sampling}"


# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА ОДНОГО ЭКСПЕРИМЕНТА
# ==============================================================================

def run_single_experiment(
    fold: int,
    seed_idx: int,
    feature_mode: str,
    sampling: str,
    model_name: str,
    complexity: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    norm_coef_path: Path,
    epochs: int = 30,
    checkpoint_frequency: int = 31,
    samples_per_file: int = 12,
    skip_existing: bool = True,
    val_precomputed_df: Optional[pl.DataFrame] = None,
) -> Optional[Dict]:
    """
    Запуск одного эксперимента обучения.
    
    Args:
        fold: Номер CV-фолда (1-based)
        seed_idx: Индекс seed-а (0-based)
        feature_mode: Тип входных признаков
        sampling: Стратегия выборки
        model_name: Название модели
        complexity: Уровень сложности
        train_df: DataFrame обучающей выборки
        test_df: DataFrame тестовой выборки
        norm_coef_path: Путь к файлу нормализации
        epochs: Количество эпох
        checkpoint_frequency: Частота сохранения чекпоинтов
        samples_per_file: Кол-во окон на файл за эпоху
        skip_existing: Пропускать уже обученные модели
        val_precomputed_df: Предрассчитанный тестовый DataFrame (опционально)
    
    Returns:
        History dict или None если пропущено
    """
    experiment_name = build_experiment_name(fold, seed_idx, feature_mode, sampling, model_name, complexity)
    save_dir = ROOT_DIR / 'experiments' / 'phase2_7'
    
    # Проверка: уже обучено?
    checkpoint_path = save_dir / experiment_name / 'final_model.pt'
    if skip_existing and checkpoint_path.exists():
        print(f">>> Пропуск {experiment_name} (уже обучено)")
        return None
    
    # Параметры
    seed = get_seed(seed_idx)
    num_harmonics = NUM_HARMONICS_BY_COMPLEXITY[complexity]
    stride = DEFAULT_STRIDE
    
    target_cols = get_target_columns(TARGET_LEVEL)
    num_classes = len(target_cols)
    in_channels = compute_in_channels(feature_mode, num_harmonics)
    pts = compute_sequence_length(feature_mode, sampling, stride)
    input_size = in_channels * pts
    
    model_params = get_model_params(model_name, complexity, feature_mode, sampling, stride)
    
    # Динамический batch_size
    is_harmonic = feature_mode in ['phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex']
    base_batch_size = 64
    val_batch_size = 8192
    if is_harmonic and num_harmonics >= 3:
        val_batch_size = 4096
    if is_harmonic and complexity == 'heavy' and model_name in ['PhysicsKAN', 'cPhysicsKAN', 'rPhysicsKAN', 'ConvKAN', 'ResNet1D']:
        val_batch_size = 2048
    
    # Конфигурация обучения
    train_config = TrainingConfig(
        epochs=epochs,
        learning_rate=0.001,
        weight_decay=1e-4,
        use_pos_weight=USE_POS_WEIGHT,
        experiment_name=experiment_name,
        save_dir=str(save_dir),
        checkpoint_frequency=checkpoint_frequency,
        seed=seed
    )
    
    data_config = DataConfig(
        path="",
        window_size=WINDOW_SIZE,
        batch_size=base_batch_size,
        mode='multilabel',
        features=[feature_mode],
        norm_coef_path=str(norm_coef_path)
    )
    
    config = ExperimentConfig(
        model=ModelConfig(name=model_name, params=model_params),
        data=data_config,
        training=train_config
    )
    
    runner = ExperimentRunner(config)
    
    # --- Подготовка данных ---
    # Индексы обучения
    train_df_indexed = train_df.with_row_index("row_nr") if "row_nr" not in train_df.columns else train_df
    train_indices = OscillogramDataset.create_indices(
        train_df_indexed, window_size=WINDOW_SIZE, mode='train', samples_per_file=samples_per_file
    )
    
    # Тренировочный датасет
    train_ds = OscillogramDataset(
        dataframe=train_df_indexed, indices=train_indices, window_size=WINDOW_SIZE,
        mode='classification', feature_mode=feature_mode,
        target_columns=target_cols, target_level=TARGET_LEVEL,
        physical_normalization=True, norm_coef_path=str(norm_coef_path),
        downsampling_mode=sampling, downsampling_stride=stride,
        augment=AUGMENTATION_ENABLED,
        num_harmonics=num_harmonics
    )
    
    # Валидационный (тестовый) датасет
    test_df_indexed = test_df.with_row_index("row_nr") if "row_nr" not in test_df.columns else test_df
    
    VAL_STRIDE = 4
    val_indices = PrecomputedDataset.create_indices(
        test_df_indexed, window_size=WINDOW_SIZE, mode='val', stride=VAL_STRIDE
    )
    
    # Пробуем предрассчитанный датасет
    supported_precomputed = ['raw', 'phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex', 'power', 'alpha_beta']
    can_precompute = val_precomputed_df is not None and feature_mode in supported_precomputed
    
    if can_precompute:
        try:
            val_ds = PrecomputedDataset(
                dataframe=val_precomputed_df, indices=val_indices, window_size=WINDOW_SIZE,
                feature_mode=feature_mode,
                target_columns=target_cols, target_level=TARGET_LEVEL,
                sampling_strategy=sampling, downsampling_stride=stride,
                num_harmonics=num_harmonics
            )
        except ValueError:
            val_ds = OscillogramDataset(
                dataframe=test_df_indexed, indices=val_indices, window_size=WINDOW_SIZE,
                mode='classification', feature_mode=feature_mode,
                target_columns=target_cols, target_level=TARGET_LEVEL,
                physical_normalization=True, norm_coef_path=str(norm_coef_path),
                downsampling_mode=sampling, downsampling_stride=stride,
                num_harmonics=num_harmonics
            )
    else:
        val_ds = OscillogramDataset(
            dataframe=test_df_indexed, indices=val_indices, window_size=WINDOW_SIZE,
            mode='classification', feature_mode=feature_mode,
            target_columns=target_cols, target_level=TARGET_LEVEL,
            physical_normalization=True, norm_coef_path=str(norm_coef_path),
            downsampling_mode=sampling, downsampling_stride=stride,
            num_harmonics=num_harmonics
        )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=base_batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    
    print(f"\n>>> Запуск {experiment_name} (seed={seed}, epochs={epochs})")
    print(f"    feature_mode={feature_mode}, sampling={sampling}, "
          f"in_channels={in_channels}, pts={pts}, num_harmonics={num_harmonics}")
    
    history = runner.train(train_loader, val_loader)
    return history


# ==============================================================================
# ГЛАВНАЯ ТОЧКА ВХОДА
# ==============================================================================

def main(
    folds: List[int] = None,
    seed_indices: List[int] = None,
    experiments: List[Dict[str, str]] = None,
    models: List[str] = None,
    complexities: List[str] = None,
    article: str = 'both',
    epochs: int = 30,
    samples_per_file: int = 12,
    checkpoint_frequency: int = 31,
    skip_existing: bool = True,
):
    """
    Главная точка входа для запуска экспериментов Фазы 2.7.
    
    Args:
        folds: Список фолдов для запуска (1-based). None = все 5.
        seed_indices: Список индексов seeds (0-based). None = [0].
        experiments: Список словарей {feature_mode, sampling}. None = все из article.
        models: Список имён моделей. None = все из article.
        complexities: Список уровней сложности. None = все.
        article: 'paper1', 'paper2', 'both' — контекст статьи.
        epochs: Количество эпох.
        samples_per_file: Окон на файл за эпоху.
        checkpoint_frequency: Частота сохранения чекпоинтов.
        skip_existing: Пропускать уже обученные.
    """
    # Значения по умолчанию
    if folds is None:
        folds = [1, 2, 3, 4, 5]
    if seed_indices is None:
        seed_indices = [0]
    if experiments is None:
        experiments = get_experiment_list(article)
    if models is None:
        models = get_models_for_article(article)
    if complexities is None:
        complexities = ['light', 'medium', 'heavy']
    
    # --- Инициализация DatasetManager и CV-разбиений ---
    DATA_DIR = ROOT_DIR / 'data' / 'ml_datasets'
    NORM_COEF_PATH = ROOT_DIR / 'raw_data' / 'norm_coef_all_v1.4.csv'
    
    dm = DatasetManager(str(DATA_DIR), norm_coef_path=str(NORM_COEF_PATH))
    
    # Генерация CV-разбиений (если ещё нет)
    dm.generate_cv_splits(n_splits=5, random_state=42)
    cv_splits = dm.load_cv_splits()
    
    print(f"\n{'='*70}")
    print(f"  ФАЗА 2.7: Высококачественные эксперименты")
    print(f"  Статья: {article}")
    print(f"  Фолды: {folds}, Seeds: {seed_indices}")
    print(f"  Экспериментов: {len(experiments)}, Моделей: {len(models)}, "
          f"Сложностей: {len(complexities)}")
    print(f"  Epochs: {epochs}, Samples/file: {samples_per_file}")
    print(f"{'='*70}\n")
    
    # --- Подсчёт общего количества экспериментов ---
    total_planned = 0
    total_skipped = 0
    total_run = 0
    total_errors = 0
    
    # === ЦИКЛ 1: По фолдам ===
    for fold in folds:
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold}/{cv_splits['n_splits']}")
        print(f"{'─'*60}")
        
        # Загрузка данных для текущего фолда
        train_df, test_df = dm.load_fold_data(fold, cv_splits)
        print(f"  Train: {len(train_df):,} строк, Test: {len(test_df):,} строк")
        
        # FIXME: Для полной поддержки предрасчёта нужно создавать precomputed для каждого fold-а.
        # Пока используем on-the-fly расчёт (val_precomputed_df=None).
        val_precomputed_df = None
        
        # === ЦИКЛ 2: По seeds ===
        for seed_idx in seed_indices:
            seed = get_seed(seed_idx)
            print(f"\n  Seed idx={seed_idx} (seed={seed})")
            
            # === ЦИКЛ 3: По экспериментам (feature_mode × sampling) ===
            for exp in experiments:
                feature_mode = exp['feature_mode']
                sampling = exp['sampling']
                
                # === ЦИКЛ 4: По моделям ===
                for model_name in models:
                    # Проверка совместимости модели с экспериментом
                    if not is_model_compatible(model_name, feature_mode, sampling):
                        continue
                    
                    # === ЦИКЛ 5: По сложностям ===
                    available_complexities = get_complexities_for_model(model_name)
                    for comp in complexities:
                        if comp not in available_complexities:
                            continue
                        
                        total_planned += 1
                        
                        try:
                            result = run_single_experiment(
                                fold=fold,
                                seed_idx=seed_idx,
                                feature_mode=feature_mode,
                                sampling=sampling,
                                model_name=model_name,
                                complexity=comp,
                                train_df=train_df,
                                test_df=test_df,
                                norm_coef_path=NORM_COEF_PATH,
                                epochs=epochs,
                                checkpoint_frequency=checkpoint_frequency,
                                samples_per_file=samples_per_file,
                                skip_existing=skip_existing,
                                val_precomputed_df=val_precomputed_df,
                            )
                            if result is None:
                                total_skipped += 1
                            else:
                                total_run += 1
                        except Exception as e:
                            total_errors += 1
                            print(f"!!! ОШИБКА: {build_experiment_name(fold, seed_idx, feature_mode, sampling, model_name, comp)}")
                            print(f"    {type(e).__name__}: {e}")
    
    # Итоговая статистика
    print(f"\n{'='*70}")
    print(f"  ИТОГО:")
    print(f"  Запланировано: {total_planned}")
    print(f"  Выполнено: {total_run}")
    print(f"  Пропущено (уже обучены): {total_skipped}")
    print(f"  Ошибки: {total_errors}")
    print(f"{'='*70}")


# ==============================================================================
# CLI ПАРСЕР
# ==============================================================================

def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Фаза 2.7: Высококачественные эксперименты для научных статей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Полный прогон для Статьи 1 (сравнение моделей)
  python run_phase2_7_all.py --article paper1 --fold all --seed_idx all --epochs 30

  # Быстрый тест (1 эпоха, 1 fold, 1 seed)
  python run_phase2_7_all.py --fold 1 --seed_idx 0 --epochs 1 --no-skip

  # Конкретный эксперимент
  python run_phase2_7_all.py --fold 1 --seed_idx 0 --exp raw_stride --model SimpleMLP --complexity medium

  # Все эксперименты для конкретной модели
  python run_phase2_7_all.py --fold all --seed_idx 0 --model SimpleCNN --complexity all --epochs 30

  # Параллельный запуск (на сервере) — каждый fold отдельно:
  python run_phase2_7_all.py --fold 1 --seed_idx all --article paper2 --epochs 30
  python run_phase2_7_all.py --fold 2 --seed_idx all --article paper2 --epochs 30
  ...
        """
    )
    
    parser.add_argument("--article", type=str, default='both',
                        choices=['paper1', 'paper2', 'both'],
                        help="Контекст статьи: paper1 (модели), paper2 (данные), both (всё)")
    
    parser.add_argument("--fold", type=str, default='all',
                        help="Номер CV-фолда: 1-5, несколько через запятую, или 'all'")
    
    parser.add_argument("--seed_idx", type=str, default='0',
                        help="Индекс seed: 0-4, несколько через запятую, или 'all'")
    
    parser.add_argument("--exp", type=str, default='all',
                        help="Эксперимент: 'feature_mode_sampling' (напр. 'raw_stride') или 'all'")
    
    parser.add_argument("--model", type=str, default='all',
                        help="Модель: имя модели или 'all'")
    
    parser.add_argument("--complexity", type=str, default='all',
                        choices=['light', 'medium', 'heavy', 'all'],
                        help="Уровень сложности модели")
    
    parser.add_argument("--epochs", type=int, default=30,
                        help="Количество эпох обучения")
    
    parser.add_argument("--samples", type=int, default=12,
                        help="Количество случайных окон из файла за эпоху")
    
    parser.add_argument("--checkpoint_freq", type=int, default=31,
                        help="Частота сохранения чекпоинтов (N>epochs → только финальная)")
    
    parser.add_argument("--no-skip", action="store_false", dest="skip_existing",
                        help="Не пропускать уже обученные модели")
    parser.set_defaults(skip_existing=True)
    
    return parser.parse_args()


def cli_main():
    """Точка входа для CLI режима."""
    args = parse_args()
    
    # Парсинг folds
    if args.fold == 'all':
        folds = [1, 2, 3, 4, 5]
    else:
        folds = [int(x) for x in args.fold.split(',')]
    
    # Парсинг seeds
    if args.seed_idx == 'all':
        seed_indices = list(range(len(SEED_OFFSETS)))
    else:
        seed_indices = [int(x) for x in args.seed_idx.split(',')]
    
    # Парсинг экспериментов
    if args.exp == 'all':
        experiments = None  # Будет определено по article
    else:
        # Формат: "feature_mode_sampling", напр. "raw_stride" или "phase_polar_none"
        # Ищем последнее вхождение sampling strategy в строке
        found = False
        for s in SAMPLING_STRATEGIES:
            if args.exp.endswith(f'_{s}'):
                fm = args.exp[: -(len(s) + 1)]
                experiments = [{'feature_mode': fm, 'sampling': s}]
                found = True
                break
        if not found:
            print(f"Ошибка: не удалось распарсить эксперимент '{args.exp}'")
            print(f"Формат: feature_mode_sampling (напр. raw_stride, phase_polar_none)")
            print(f"Доступные sampling strategies: {SAMPLING_STRATEGIES}")
            return
    
    # Парсинг моделей
    if args.model == 'all':
        models = None  # Будет определено по article
    else:
        models = [args.model]
    
    # Парсинг сложностей
    if args.complexity == 'all':
        complexities = None
    else:
        complexities = [args.complexity]
    
    main(
        folds=folds,
        seed_indices=seed_indices,
        experiments=experiments,
        models=models,
        complexities=complexities,
        article=args.article,
        epochs=args.epochs,
        samples_per_file=args.samples,
        checkpoint_frequency=args.checkpoint_freq,
        skip_existing=args.skip_existing,
    )


# ==============================================================================
# РУЧНОЙ ЗАПУСК (IDE / Notebooks)
# ==============================================================================

if __name__ == "__main__":
    # Автоматический выбор режима:
    # Если переданы аргументы командной строки (кроме имени скрипта) → CLI режим
    # Иначе → ручной режим с константами
    
    if len(sys.argv) > 1:
        # === CLI РЕЖИМ ===
        cli_main()
    else:
        # === РУЧНОЙ ЗАПУСК ЧЕРЕЗ КОНСТАНТЫ (Рекомендуется для IDE/Notebooks) ===
        
        # Параметры запуска — настройте под текущую задачу:
        
        # 1. FOLDS: Какие фолды запускать (1-5)
        FOLDS = [1]  # Для теста — один фолд. Для полного запуска: [1, 2, 3, 4, 5]
        
        # 2. SEED_INDICES: Какие seeds использовать (0-4)
        SEED_INDICES = [0]  # Для теста — один seed. Для полного: [0, 1, 2, 3, 4]
        
        # 3. ARTICLE: Контекст статьи
        # 'paper1' — сравнение моделей (phase_polar, все sampling, ВСЕ модели)
        # 'paper2' — сравнение данных (все feature_modes, все sampling, базовые модели)
        # 'both'   — полный перебор
        ARTICLE = 'both'
        
        # 4. MODELS: Какие модели запускать (None = все из ARTICLE)
        MODELS = None  # или ['SimpleMLP', 'SimpleCNN'] для конкретных
        
        # 5. COMPLEXITIES: Уровни сложности (None = все)
        COMPLEXITIES = None  # или ['light', 'medium']
        
        # 6. EXPERIMENTS: Конкретные эксперименты (None = все из ARTICLE)
        # Формат: [{'feature_mode': 'raw', 'sampling': 'stride'}, ...]
        EXPERIMENTS = None
        
        # 7. EPOCHS: Количество эпох
        EPOCHS = 1  # <<< ДЛЯ ТЕСТОВОГО ПРОГОНА. Для полного: 30
        
        # 8. SAMPLES_PER_FILE: Окон на файл
        SAMPLES_PER_FILE = 12
        
        # 9. CHECKPOINT_FREQUENCY: Частота сохранения
        CHECKPOINT_FREQUENCY = 32  # > EPOCHS → только финальная модель
        
        # --- Запуск ---
        main(
            folds=FOLDS,
            seed_indices=SEED_INDICES,
            experiments=EXPERIMENTS,
            models=MODELS,
            complexities=COMPLEXITIES,
            article=ARTICLE,
            epochs=EPOCHS,
            samples_per_file=SAMPLES_PER_FILE,
            checkpoint_frequency=CHECKPOINT_FREQUENCY,
            skip_existing=False,  # Для тестового прогона не пропускаем
        )
