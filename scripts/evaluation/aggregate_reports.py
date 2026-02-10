import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional, Tuple
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from tqdm import tqdm
import re
import logging

# Добавляем корень проекта в путь импорта
ROOT_DIR_PROJECT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR_PROJECT))

# Глобальный логгер (инициализируется в aggregate_reports)
_eval_logger: Optional[logging.Logger] = None

from osc_tools.data_management.dataset_manager import DatasetManager
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment

# Импорт расширенного визуализатора
try:
    from scripts.visualization.advanced_plots import AdvancedVisualizer
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False
    print("[Warning] advanced_plots.py не найден - расширенная визуализация недоступна")


def load_existing_summary(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Загружает существующий summary_report.csv если он есть.
    
    Returns:
        DataFrame с предыдущими результатами или None
    """
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            print(f"[Cache] Загружен существующий отчёт: {csv_path} ({len(df)} записей)")
            return df
        except Exception as e:
            print(f"[Cache] Ошибка загрузки {csv_path}: {e}")
    return None


def needs_full_eval_recalc(existing_row: Optional[pd.Series], require_hierarchical: bool = False) -> bool:
    """
    Проверяет, нужен ли пересчёт full_eval для данной модели.
    
    Возвращает True если:
    - Нет существующих данных (existing_row is None)
    - Full Best Acc/F1 или Full Final Acc/F1 равны 0
    - Full Eval Time (s) или Full Eval Samples равны 0
    - (Опционально) отсутствуют метрики Hierarchical Accuracy
    """
    if existing_row is None:
        return True
    
    # Проверяем ключевые метрики на 0
    full_eval_cols = [
        'Full Best Acc', 'Full Best F1', 'Full Final Acc', 'Full Final F1',
        'Full Eval Time (s)', 'Full Eval Samples'
    ]
    
    for col in full_eval_cols:
        if col in existing_row.index:
            val = existing_row[col]
            # Проверяем на 0, NaN или пустое значение
            if pd.isna(val) or val == 0 or val == 0.0:
                return True
    
    if require_hierarchical:
        hier_cols = [
            'Hier Base F1 (Best)', 'Hier Base Acc (Best)', 'Hier Base Exact (Best)',
            'Hier Base F1 (Final)', 'Hier Base Acc (Final)', 'Hier Base Exact (Final)',
            'Num Classes',
            'Full Per Class F1 Count',
            'Full Per Class Support Count'
        ]
        for col in hier_cols:
            if col in existing_row.index:
                val = existing_row[col]
                if pd.isna(val) or val == 0 or val == 0.0:
                    return True
            else:
                return True

    return False


def needs_benchmark_recalc(existing_row: Optional[pd.Series]) -> bool:
    """
    Проверяет, нужен ли пересчёт benchmark для данной модели.
    
    Возвращает True если:
    - Нет существующих данных
    - CPU Inf (ms) равен 0 или NaN
    """
    if existing_row is None:
        return True
    
    if 'CPU Inf (ms)' in existing_row.index:
        val = existing_row['CPU Inf (ms)']
        if pd.isna(val) or val == 0 or val == 0.0:
            return True
    
    return False


def get_existing_row(existing_df: Optional[pd.DataFrame], exp_name: str) -> Optional[pd.Series]:
    """
    Ищет строку в существующем DataFrame по имени эксперимента.
    """
    if existing_df is None:
        return None
    
    matches = existing_df[existing_df['Path'] == exp_name]
    if len(matches) > 0:
        return matches.iloc[0]
    
    # Fallback: поиск по Experiment
    matches = existing_df[existing_df['Experiment'] == exp_name]
    if len(matches) > 0:
        return matches.iloc[0]
    
    return None


def load_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    """Загрузка метрик из файла jsonl."""
    metrics = []
    try:
        with open(metrics_path, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
    except Exception as e:
        print(f"Ошибка чтения {metrics_path}: {e}", file=sys.stderr)
    return metrics

def load_config(config_path: Path) -> Dict[str, Any]:
    """Загрузка конфигурации из файла json."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {config_path}: {e}", file=sys.stderr)
        return {}

def plot_learning_curves(metrics: List[Dict[str, Any]], save_path: Path):
    """Генерация графиков обучения."""
    df = pd.DataFrame(metrics)
    if df.empty:
        return
        
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Learning Curves (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy/F1
    plt.subplot(1, 2, 2)
    if 'val_acc' in df.columns:
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    if 'val_f1' in df.columns:
        plt.plot(df['epoch'], df['val_f1'], label='Val F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison(all_histories: Dict[str, List[Dict[str, Any]]], save_dir: Path):
    """
    Генерирует сравнительные графики для всех экспериментов.
    """
    if not all_histories:
        return
        
    metrics_to_plot = ['train_loss', 'val_loss', 'val_acc', 'val_f1']
    titles = ['Comparison: Train Loss', 'Comparison: Validation Loss', 'Comparison: Validation Accuracy', 'Comparison: Validation F1 (Macro)']
    
    plt.figure(figsize=(24, 5))
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        plt.subplot(1, 4, i + 1)
        
        for exp_name, history in all_histories.items():
            df = pd.DataFrame(history)
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], label=exp_name)
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(metric.split('_')[-1].capitalize())
        if len(all_histories) < 10: # Чтобы легенда не перекрывала всё, если много моделей
            plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_metrics.png")
    plt.close()

def benchmark_model_cpu(exp_dir: Path, config: Dict[str, Any], iterations: int = 100) -> float:
    """Замер скорости инференса на CPU для конкретной модели на пустых (dummy) данных."""
    try:
        model_name = config.get('model', {}).get('name')
        params = config.get('model', {}).get('params', {}).copy() # Копия, чтобы не портить оригинал
        
        # Импорт из общего пакета моделей
        from osc_tools.ml import models as ml_models
        
        # Определение класса модели
        if not hasattr(ml_models, model_name):
            try:
                if model_name in ['SimpleMLP', 'SimpleCNN']:
                    from osc_tools.ml.models.baseline import SimpleMLP, SimpleCNN
                    models_map = {'SimpleMLP': SimpleMLP, 'SimpleCNN': SimpleCNN}
                elif model_name == 'ResNet1D':
                    from osc_tools.ml.models.resnet import ResNet1D
                    models_map = {'ResNet1D': ResNet1D}
                elif model_name in ['SimpleKAN', 'ConvKAN', 'PhysicsKAN']:
                    from osc_tools.ml.models.kan import SimpleKAN, ConvKAN, PhysicsKAN
                    models_map = {'SimpleKAN': SimpleKAN, 'ConvKAN': ConvKAN, 'PhysicsKAN': PhysicsKAN}
                elif model_name.startswith('Hierarchical'):
                    from osc_tools.ml.models.advanced import (
                        HierarchicalCNN, HierarchicalConvKAN, HierarchicalMLP,
                        HierarchicalResNet, HierarchicalSimpleKAN, HierarchicalPhysicsKAN
                    )
                    models_map = {
                        'HierarchicalCNN': HierarchicalCNN,
                        'HierarchicalConvKAN': HierarchicalConvKAN,
                        'HierarchicalMLP': HierarchicalMLP,
                        'HierarchicalResNet': HierarchicalResNet,
                        'HierarchicalSimpleKAN': HierarchicalSimpleKAN,
                        'HierarchicalPhysicsKAN': HierarchicalPhysicsKAN
                    }
                elif model_name.startswith('Hybrid'):
                    from osc_tools.ml.models.hybrid import (
                        HybridMLP, HybridCNN, HybridConvKAN,
                        HybridSimpleKAN, HybridPhysicsKAN, HybridResNet
                    )
                    models_map = {
                        'HybridMLP': HybridMLP,
                        'HybridCNN': HybridCNN,
                        'HybridConvKAN': HybridConvKAN,
                        'HybridSimpleKAN': HybridSimpleKAN,
                        'HybridPhysicsKAN': HybridPhysicsKAN,
                        'HybridResNet': HybridResNet
                    }
                else:
                    return 0.0
                model_cls = models_map.get(model_name)
            except:
                return 0.0
        else:
            model_cls = getattr(ml_models, model_name)
            
        if model_cls is None:
            return 0.0
            
        # Очистка параметров от лишних полей
        valid_params = {}
        import inspect
        try:
            sig = inspect.signature(model_cls)
            for p_name, p_val in params.items():
                if p_name in sig.parameters:
                    valid_params[p_name] = p_val
        except:
            valid_params = params

        model = model_cls(**valid_params).cpu()
        model.eval()
        
        # Загрузка весов (если есть)
        model_path = exp_dir / "best_model.pt"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                pass # Если веса не подошли, бенчмарк все равно можно прогнать на случайных

        # Подстановка данных (dummy)
        use_mlp = params.get('use_mlp', False)
        if model_name in ['SimpleMLP', 'SimpleKAN'] or use_mlp:
            input_size = params.get('input_size', 2560)
            # Пробуем 2D (B, N)
            dummy_input = torch.randn(1, input_size)
            try:
                model(dummy_input)
            except:
                # Если PhysicsKAN в MLP режиме, он может хотеть 3D (B, C, T) даже для MLP
                in_channels = params.get('in_channels', 8)
                pts = input_size // in_channels
                dummy_input = torch.randn(1, in_channels, pts)
        else:
            in_channels = params.get('in_channels', 8)
            # Пытаемся определить размер временного окна (320, 64, 20, 2)
            # Добавим проверку от большего к меньшему
            pts_options = [320, 64, 20, 10, 2]
            dummy_input = None
            for pts in pts_options:
                try:
                    test_input = torch.randn(1, in_channels, pts)
                    with torch.no_grad():
                        model(test_input)
                    dummy_input = test_input
                    break
                except Exception:
                    continue
            
            if dummy_input is None:
                # print(f"Не удалось подобрать вход для {model_name}")
                return 0.0

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(dummy_input)
        
        avg_time_ms = ((time.perf_counter() - start) * 1000) / iterations
        return avg_time_ms
        
    except Exception as e:
        print(f"Benchmark ошибка для {exp_dir.name} ({model_name}): {e}")
        return 0.0


# =============================================================================
# ПОЛНАЯ ОЦЕНКА НА ТЕСТОВОМ ДАТАСЕТЕ
# =============================================================================

# Глобальный кэш для данных тестового датасета (загружается один раз)
_test_data_cache: Dict[str, Any] = {}


def _get_test_data_cached(
    data_dir: str, 
    window_size: int = 320,
    eval_stride: int = 1,  # Stride=1 для полного перебора ВСЕХ точек
    norm_coef_path: Optional[str] = None,
    target_level: str = 'base'
) -> Tuple[Any, List[int], List[str]]:
    """
    Возвращает кэшированные данные тестового датасета (PrecomputedDataset).
    
    После исправления FFT и нормализации, PrecomputedDataset и OscillogramDataset
    выдают идентичные данные, поэтому используем PrecomputedDataset для скорости.
    
    Стратегия: stride=1 для полного покрытия + большие батчи на GPU для скорости.
    PrecomputedDataset хранит предрассчитанные признаки, поэтому извлечение быстрое.
    
    Args:
        data_dir: Путь к директории с данными
        window_size: Размер окна
        eval_stride: Stride между окнами (1 = все точки, default)
        norm_coef_path: Путь к файлу нормализации
        target_level: Уровень детализации меток ('base', 'full', 'full_by_levels')
    
    Returns:
        (test_df, indices, target_cols)
    """
    import polars as pl
    
    # Включаем target_level в ключ кэша
    cache_key = f"{data_dir}_{window_size}_{eval_stride}_{norm_coef_path}_{target_level}"
    
    if cache_key not in _test_data_cache:
        dm = DatasetManager(data_dir, norm_coef_path=norm_coef_path)
        
        print(f"[Full Eval] Загрузка тестового датасета (target_level={target_level})...")
        # Убеждаемся что предрассчитанный файл существует
        dm.create_precomputed_test_csv()
        
        # Загружаем
        test_df = dm.load_test_df(precomputed=True)
        test_df = test_df.with_row_index("row_nr")
        
        # Подготавливаем метки в зависимости от target_level
        test_df = prepare_labels_for_experiment(test_df, target_level)
        
        # Получаем целевые колонки
        target_cols = get_target_columns(target_level, test_df)
        
        # Создаём индексы для полной оценки
        indices = PrecomputedDataset.create_indices(
            test_df,
            window_size=window_size,
            mode='val',
            stride=eval_stride
        )
        print(f"[Full Eval] Загружено {len(indices):,} точек (stride={eval_stride}, {len(target_cols)} классов)")
        
        _test_data_cache[cache_key] = (test_df, indices, target_cols)
    
    return _test_data_cache[cache_key]


def _create_model_from_config(config: Dict[str, Any]) -> Optional[nn.Module]:
    """
    Создаёт экземпляр модели по конфигурации.
    
    Returns:
        Модель или None если не удалось создать
    """
    from osc_tools.ml import models as ml_models
    import inspect
    
    model_name = config.get('model', {}).get('name')
    params = config.get('model', {}).get('params', {}).copy()
    
    if not model_name:
        return None
    
    # Определение класса модели
    model_cls = None
    
    # Сначала пробуем из __init__
    if hasattr(ml_models, model_name):
        model_cls = getattr(ml_models, model_name)
    else:
        # Fallback на прямой импорт
        try:
            if model_name in ['SimpleMLP', 'SimpleCNN']:
                from osc_tools.ml.models.baseline import SimpleMLP, SimpleCNN
                models_map = {'SimpleMLP': SimpleMLP, 'SimpleCNN': SimpleCNN}
                model_cls = models_map.get(model_name)
            elif model_name == 'ResNet1D':
                from osc_tools.ml.models.resnet import ResNet1D
                model_cls = ResNet1D
            elif model_name in ['SimpleKAN', 'ConvKAN', 'PhysicsKAN', 'PhysicsKANConditional']:
                from osc_tools.ml.models.kan import SimpleKAN, ConvKAN, PhysicsKAN, PhysicsKANConditional
                models_map = {
                    'SimpleKAN': SimpleKAN,
                    'ConvKAN': ConvKAN,
                    'PhysicsKAN': PhysicsKAN,
                    'PhysicsKANConditional': PhysicsKANConditional
                }
                model_cls = models_map.get(model_name)
            elif model_name.startswith('Hierarchical'):
                from osc_tools.ml.models.advanced import (
                    HierarchicalCNN, HierarchicalConvKAN, HierarchicalMLP,
                    HierarchicalResNet, HierarchicalSimpleKAN, HierarchicalPhysicsKAN
                )
                models_map = {
                    'HierarchicalCNN': HierarchicalCNN,
                    'HierarchicalConvKAN': HierarchicalConvKAN,
                    'HierarchicalMLP': HierarchicalMLP,
                    'HierarchicalResNet': HierarchicalResNet,
                    'HierarchicalSimpleKAN': HierarchicalSimpleKAN,
                    'HierarchicalPhysicsKAN': HierarchicalPhysicsKAN
                }
                model_cls = models_map.get(model_name)
            elif model_name.startswith('Hybrid'):
                from osc_tools.ml.models.hybrid import (
                    HybridMLP, HybridCNN, HybridConvKAN,
                    HybridSimpleKAN, HybridPhysicsKAN, HybridResNet
                )
                models_map = {
                    'HybridMLP': HybridMLP,
                    'HybridCNN': HybridCNN,
                    'HybridConvKAN': HybridConvKAN,
                    'HybridSimpleKAN': HybridSimpleKAN,
                    'HybridPhysicsKAN': HybridPhysicsKAN,
                    'HybridResNet': HybridResNet
                }
                model_cls = models_map.get(model_name)
        except Exception as e:
            print(f"  [!] Не удалось импортировать модель {model_name}: {e}")
            return None
    
    if model_cls is None:
        return None
    
    # Очистка параметров от лишних полей
    valid_params = {}
    try:
        sig = inspect.signature(model_cls)
        for p_name, p_val in params.items():
            if p_name in sig.parameters:
                valid_params[p_name] = p_val
    except:
        valid_params = params
    
    try:
        model = model_cls(**valid_params)
        return model
    except Exception as e:
        print(f"  [!] Ошибка создания модели {model_name}: {e}")
        return None


def _get_eval_batch_size(
    config: Dict[str, Any],
    model_name: str = None,
    feature_mode: str = None,
    num_harmonics: int = 1
) -> int:
    """
    Определяет оптимальный batch_size для оценки модели,
    учитывая архитектуру и особенности данных.
    
    Основано на логике из run_phase2_6.py (строки 184-195):
    - KAN модели с harmonic режимами требуют меньше памяти
    - Heavy режимы требуют еще меньше
    
    Args:
        config: Конфиг эксперимента
        model_name: Имя модели
        feature_mode: Режим признаков (raw, phase_polar и т.д.)
        num_harmonics: Количество гармоник
    
    Returns:
        Оптимальный batch_size для оценки
    """
    if model_name is None:
        model_name = config.get('model', {}).get('name', '')
    
    if feature_mode is None:
        data_features = config.get('data', {}).get('features', [])
        if isinstance(data_features, list) and data_features:
            feature_mode = data_features[0]
        elif isinstance(data_features, str):
            feature_mode = data_features
    
    # Дефолтный батч
    batch_size = 8192
    val_batch_size = 8192
    
    # Проверяем, это ли harmonic режим (спектральные данные требуют больше памяти)
    is_harmonic_mode = feature_mode in ['phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex']
    
    # Уменьшаем для harmonic режимов
    if is_harmonic_mode and num_harmonics >= 3:
        batch_size = 32
        val_batch_size = 2048
    
    # Дополнительно уменьшаем для heavy режимов + KAN модели
    if is_harmonic_mode and model_name in ['PhysicsKAN', 'ConvKAN', 'HierarchicalPhysicsKAN', 'HierarchicalConvKAN']:
        batch_size = 16
        val_batch_size = 1024
    
    return val_batch_size


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Нормализует ключи state_dict для совместимости старых чекпоинтов.
    
    Причина: в части экспериментов структура имен слоёв была изменена
    (например, physics.processing_net.net.* -> physics.processing_net.features.*).
    """
    if not state_dict:
        return state_dict
    
    remap_prefixes = {
        'physics.processing_net.net.': 'physics.processing_net.features.'
    }
    
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in remap_prefixes.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        new_state_dict[new_key] = value
    
    return new_state_dict


def _load_state_dict_safe(
    model: nn.Module,
    checkpoint: Dict[str, Any],
    exp_name: str,
    tag: str
) -> None:
    """
    Загружает state_dict с совместимостью по ключам и строгим логированием.
    """
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)
    
    load_result = model.load_state_dict(state_dict, strict=False)
    if _eval_logger:
        if load_result.missing_keys:
            _eval_logger.error(
                f"{exp_name} - {tag}: Missing keys: {load_result.missing_keys}"
            )
        if load_result.unexpected_keys:
            _eval_logger.error(
                f"{exp_name} - {tag}: Unexpected keys: {load_result.unexpected_keys}"
            )


def _evaluate_with_batch_fallback(
    model: nn.Module,
    test_ds: torch.utils.data.Dataset,
    base_batch_size: int,
    device: torch.device,
    mode: str,
    exp_name: str,
    tag: str
) -> Tuple[Dict[str, float], int, torch.device]:
    """
    Оценка с понижением batch_size при OOM и fallback на CPU.
    """
    current_batch_size = base_batch_size
    current_device = device
    
    while True:
        test_loader = DataLoader(
            test_ds,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(current_device.type == 'cuda')
        )
        
        try:
            if current_device.type == 'cuda':
                torch.cuda.empty_cache()
            metrics = evaluate_model_full(
                model,
                test_loader,
                current_device,
                mode,
                raise_on_oom=True
            )
            return metrics, current_batch_size, current_device
        except Exception as e:
            err_text = str(e)
            is_oom = 'CUDA out of memory' in err_text
            if is_oom and current_device.type == 'cuda':
                if _eval_logger:
                    _eval_logger.error(
                        f"{exp_name} - {tag}: OOM при batch_size={current_batch_size}. Уменьшаем и повторяем."
                    )
                # Уменьшаем batch_size до минимума, затем fallback на CPU
                if current_batch_size > 16:
                    current_batch_size = max(16, current_batch_size // 2)
                    continue
                
                if _eval_logger:
                    _eval_logger.error(
                        f"{exp_name} - {tag}: OOM даже при batch_size=16. Переходим на CPU."
                    )
                current_device = torch.device('cpu')
                model = model.to(current_device)
                continue
            
            if _eval_logger:
                _eval_logger.error(f"{exp_name} - {tag}: Ошибка оценки: {e}")
            return {}, current_batch_size, current_device


def _evaluate_hierarchical_with_batch_fallback(
    model: nn.Module,
    test_ds: torch.utils.data.Dataset,
    base_batch_size: int,
    device: torch.device,
    exp_name: str,
    tag: str,
    target_columns: List[str]
) -> Tuple[Dict[str, Any], int, torch.device]:
    """
    Оценка Hierarchical Accuracy с понижением batch_size при OOM и fallback на CPU.
    """
    from osc_tools.ml.evaluation import evaluate_model_multilevel

    current_batch_size = base_batch_size
    current_device = device

    while True:
        test_loader = DataLoader(
            test_ds,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(current_device.type == 'cuda')
        )

        try:
            if current_device.type == 'cuda':
                torch.cuda.empty_cache()
            results = evaluate_model_multilevel(
                model,
                test_loader,
                current_device,
                target_columns
            )
            return results, current_batch_size, current_device
        except Exception as e:
            err_text = str(e)
            is_oom = 'CUDA out of memory' in err_text
            if is_oom and current_device.type == 'cuda':
                if _eval_logger:
                    _eval_logger.error(
                        f"{exp_name} - {tag}: OOM при Hier Eval batch_size={current_batch_size}. Уменьшаем и повторяем."
                    )
                if current_batch_size > 16:
                    current_batch_size = max(16, current_batch_size // 2)
                    continue

                if _eval_logger:
                    _eval_logger.error(
                        f"{exp_name} - {tag}: OOM при Hier Eval даже при batch_size=16. Переходим на CPU."
                    )
                current_device = torch.device('cpu')
                model = model.to(current_device)
                continue

            if _eval_logger:
                _eval_logger.error(f"{exp_name} - {tag}: Ошибка Hier Eval: {e}")
            return {}, current_batch_size, current_device


def evaluate_model_full(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mode: str = 'multilabel',
    raise_on_oom: bool = False
) -> Dict[str, float]:
    """
    Полная оценка модели на тестовом датасете.
    
    Args:
        model: Модель для оценки
        test_loader: DataLoader с тестовыми данными
        device: Устройство (cuda/cpu)
        mode: 'multilabel' или 'classification'
    
    Returns:
        Словарь с метриками: acc, f1, balanced_acc, per_class_f1
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_targets_ml23 = []
    all_preds_ml23 = []
    
    try:
        with torch.no_grad():
            for x, y in test_loader:  # Без tqdm здесь - итерация быстрая
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                if mode == 'classification':
                    y = y.long()
                    if y.dim() > 1 and y.shape[1] == 1:
                        y = y.squeeze(1)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                elif mode == 'multilabel':
                    y = y.float()
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                elif mode == 'multitask_conditional':
                    y = y.float()
                    probs = torch.sigmoid(outputs[:, :4])
                    pred_binary = (probs > 0.5).int().cpu().numpy()

                    # Ограничение: если ML_3=1, то ML_2=0
                    pred_binary[:, 2] = np.where(pred_binary[:, 3] == 1, 0, pred_binary[:, 2])

                    all_preds.extend(pred_binary)
                    all_targets.extend(y[:, :4].cpu().numpy())
                    all_preds_ml23.extend([])
                    all_targets_ml23.extend([])
                else:
                    raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        # Очищаем GPU при ошибке
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if raise_on_oom and 'CUDA out of memory' in str(e):
            raise
        if _eval_logger:
            _eval_logger.error(f"Ошибка при оценке модели: {e}")
        return {}
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Расчёт метрик
    per_class_support = []

    if mode == 'multitask_conditional':
        y_true = all_targets
        y_pred = all_preds

        f1_normal = f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0)
        f1_ml1 = f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0)
        f1_ml2 = f1_score(y_true[:, 2], y_pred[:, 2], zero_division=0)
        f1_ml3 = f1_score(y_true[:, 3], y_pred[:, 3], zero_division=0)

        acc_normal = accuracy_score(y_true[:, 0], y_pred[:, 0])
        acc_ml1 = accuracy_score(y_true[:, 1], y_pred[:, 1])
        acc_ml2 = accuracy_score(y_true[:, 2], y_pred[:, 2])
        acc_ml3 = accuracy_score(y_true[:, 3], y_pred[:, 3])

        acc = float(np.mean([acc_normal, acc_ml1, acc_ml2, acc_ml3]))
        f1 = float(np.mean([f1_normal, f1_ml1, f1_ml2, f1_ml3]))
        balanced_acc = 0.0
        per_class_f1 = [float(f1_normal), float(f1_ml1), float(f1_ml2), float(f1_ml3)]
    else:
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        if mode == 'classification':
            balanced_acc = balanced_accuracy_score(all_targets, all_preds)
            class_counts = np.bincount(all_targets.astype(int), minlength=len(set(all_targets.tolist())))
            per_class_support = class_counts.tolist()
        else:
            balanced_acc = 0.0
            if all_targets.ndim == 2 and all_targets.size > 0:
                per_class_support = all_targets.sum(axis=0).astype(int).tolist()

        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    
    return {
        'acc': acc,
        'f1': f1,
        'balanced_acc': balanced_acc,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support
    }


def evaluate_full_test_dataset(
    exp_dir: Path,
    config: Dict[str, Any],
    data_dir: str,
    batch_size: int = 8192,  # Параметр сохранен для обратной совместимости (игнорируется, батч определяется динамически)
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Полная оценка моделей (best и final) на всём тестовом датасете.
    
    Стратегия оптимизации:
    - stride=1: перебираем ВСЕ возможные окна (полное покрытие)
    - Динамический batch_size: автоматически уменьшается для тяжелых моделей (KAN + harmonic)
    - PrecomputedDataset: данные уже предрассчитаны, извлечение быстрое
    
    Args:
        exp_dir: Путь к директории эксперимента
        config: Конфигурация эксперимента
        data_dir: Путь к директории с датасетами
        batch_size: (Игнорируется) Размер батча теперь определяется динамически на основе архитектуры модели
        use_gpu: Использовать GPU
    
    Returns:
        Словарь с метриками для best и final моделей
    """
    results = {
        'full_best_acc': 0.0,
        'full_best_f1': 0.0,
        'full_best_per_class_f1': [],
        'full_final_acc': 0.0,
        'full_final_f1': 0.0,
        'full_final_per_class_f1': [],
        'full_eval_time_s': 0.0,
        'full_eval_samples': 0,
        'num_classes': 0,
        'hier_base_f1_best': 0.0,
        'hier_base_acc_best': 0.0,
        'hier_base_exact_best': 0.0,
        'hier_base_f1_final': 0.0,
        'hier_base_acc_final': 0.0,
        'hier_base_exact_final': 0.0,
        'full_best_per_class_f1_map': {},
        'full_final_per_class_f1_map': {},
        'full_best_per_class_support_map': {},
        'full_final_per_class_support_map': {},
        'full_per_class_f1_count': 0,
        'full_per_class_support_count': 0
    }

    config_path = exp_dir / "config.json"
    
    try:
        # Определяем устройство
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Параметры из конфига
        window_size = config.get('data', {}).get('window_size', 320)
        mode = config.get('data', {}).get('mode', 'multilabel')
        
        # Определяем sampling_strategy из имени эксперимента
        # Это критично для правильного формата данных
        exp_name = exp_dir.name.lower()
        
        # ВАЖНО: Для экспериментов 2.6.4 слово 'full' означает target_level, а не sampling
        # Поэтому сначала проверяем stride/snapshot, а 'none' только если нет других маркеров
        if 'stride' in exp_name:
            sampling_strategy = 'stride'
            downsampling_stride = 16
        elif 'snapshot' in exp_name:
            sampling_strategy = 'snapshot'
            downsampling_stride = 32
        elif 'none_sampl' in exp_name or ('none' in exp_name and 'full' not in exp_name):
            # Только если явно указано 'none' в контексте sampling (не target_level)
            sampling_strategy = 'none'
            downsampling_stride = 1
        else:
            # Пробуем определить по input_size модели
            input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
            in_channels_cfg = config.get('model', {}).get('params', {}).get('in_channels', None)
            
            if in_channels_cfg:
                seq_len = input_size // in_channels_cfg
            elif input_size > 0:
                # Для MLP: input_size = in_channels * seq_len
                # Phase_polar: 16 каналов, stride=16 → ~18 точек → input_size ~288
                # Symmetric: 12 каналов, stride=16 → ~18 точек → input_size ~216
                # Snapshot (phase_polar): 16 каналов * 2 точки = 32
                # Snapshot (symmetric): 12 каналов * 2 точки = 24
                if input_size <= 32:
                    # Snapshot mode
                    seq_len = 2
                elif input_size <= 64:
                    # Может быть snapshot с raw (8*2=16) или другой
                    seq_len = 2 if input_size in [16, 24, 32] else input_size // 16
                else:
                    # Stride mode - предполагаем phase_polar (16 каналов)
                    seq_len = input_size // 16
            else:
                seq_len = 18  # Default для stride
            
            # Если seq_len совпадает с window_size, это 'none'
            if seq_len >= window_size:
                sampling_strategy = 'none'
                downsampling_stride = 1
            # Snapshot дает 2 точки, stride дает ~18-20
            elif seq_len <= 4:
                sampling_strategy = 'snapshot'
                downsampling_stride = 32
            else:
                sampling_strategy = 'stride'
                downsampling_stride = 16
        
        model_name = config.get('model', {}).get('name', '')

        # Feature mode - сначала пробуем взять из конфига данных
        # 16 = phase_polar (8 сигналов × 2: amp + phase)
        # 12 = symmetric (6 составляющих × 2: amp + phase)
        # 8 = raw (8 сигналов без FFT)
        in_channels = config.get('model', {}).get('params', {}).get('in_channels', None)
        feature_mode = None

        data_features = config.get('data', {}).get('features')
        if isinstance(data_features, list) and data_features:
            feature_mode = data_features[0]
        elif isinstance(data_features, str) and data_features:
            feature_mode = data_features
        
        if feature_mode is None and in_channels is None:
            # Сначала пытаемся найти в названии
            if 'raw' in exp_name:
                in_channels = 8
            elif 'symmetric' in exp_name:
                in_channels = 12
            elif 'phase_polar' in exp_name or 'polar' in exp_name:
                in_channels = 16
            
            # Если не нашли в названии, вычисляем из input_size
            if in_channels is None:
                input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
                if input_size > 0:
                    if sampling_strategy == 'snapshot':
                        # input_size = in_channels * 2
                        in_channels = input_size // 2
                    elif sampling_strategy == 'none':
                        # input_size = in_channels * window_size
                        in_channels = input_size // window_size
                    else:
                        # input_size = in_channels * seq_len (~18)
                        # Пробуем сначала 16 (phase_polar), потом 12 (symmetric)
                        if input_size % 16 == 0:
                            in_channels = 16
                        elif input_size % 12 == 0:
                            in_channels = 12
                        elif input_size % 8 == 0:
                            in_channels = 8
                        else:
                            in_channels = 16  # Default
                else:
                    in_channels = 16  # Default

        # Если feature_mode так и не определён, определяем его по in_channels
        if feature_mode is None:
            if in_channels == 12:
                feature_mode = 'symmetric'
            elif in_channels == 8:
                feature_mode = 'raw'
            else:
                feature_mode = 'phase_polar'  # Default для Phase 2.x

        # Гибридные модели ожидают raw + features
        if model_name.startswith('Hybrid'):
            if isinstance(feature_mode, list):
                if 'raw' not in feature_mode:
                    feature_mode = ['raw'] + feature_mode
            else:
                feature_mode = ['raw', feature_mode]

        modes_list = feature_mode if isinstance(feature_mode, list) else [feature_mode]
        has_spectral = any(m != 'raw' for m in modes_list)

        # Определяем базовые каналы
        raw_base_ch = 8 if 'raw' in modes_list else 0
        spectral_mode = next((m for m in modes_list if m != 'raw'), None)
        spectral_base_ch = 0
        if spectral_mode in ['symmetric', 'symmetric_polar']:
            spectral_base_ch = 12
        elif spectral_mode in ['phase_polar', 'phase_complex']:
            spectral_base_ch = 16
        elif spectral_mode == 'power':
            spectral_base_ch = 8
        elif spectral_mode == 'alpha_beta':
            spectral_base_ch = 6
        elif spectral_mode is None and raw_base_ch > 0:
            spectral_base_ch = 0
        
        # Определяем число гармоник:
        # 1. Из features_channels (для гибридных): num_harmonics = features_channels / base_ch
        # 2. Из in_channels (если есть): num_harmonics = in_channels / base_ch
        # 3. Из input_size (для MLP/KAN без in_channels):
        #    - snapshot: input_size = in_channels * 2 → in_channels = input_size / 2
        #    - stride: input_size = in_channels * seq_len (~18)
        #    - none: input_size = in_channels * window_size
        # ВАЖНО: если in_channels был угадан по названию (например, 16),
        #        но input_size говорит о большем числе гармоник, мы обязаны обновить.
        num_harmonics = 1
        input_size_cfg = config.get('model', {}).get('params', {}).get('input_size', 0)
        features_channels_cfg = config.get('model', {}).get('params', {}).get('features_channels', 0)
        derived_in_channels = None
        
        # Для гибридных моделей: вычисляем num_harmonics из features_channels напрямую
        if model_name.startswith('Hybrid') and features_channels_cfg > 0 and spectral_base_ch > 0:
            if features_channels_cfg % spectral_base_ch == 0:
                num_harmonics = max(1, features_channels_cfg // spectral_base_ch)
        elif input_size_cfg > 0 and (spectral_base_ch > 0 or raw_base_ch > 0):
            # Вычисляем in_channels из input_size и sampling_strategy
            if sampling_strategy == 'snapshot':
                # Для спектральных: 2 точки, для raw: 64 точки
                pts = 2 if has_spectral else 64
                derived_in_channels = input_size_cfg // pts
            elif sampling_strategy == 'none':
                derived_in_channels = input_size_cfg // window_size
            else:  # stride
                # Для спектральных: (window_size - 32) // stride точек
                if not has_spectral:
                    pts = window_size // downsampling_stride
                else:
                    pts = (window_size - 32) // downsampling_stride
                pts = max(1, pts)
                derived_in_channels = input_size_cfg // pts
        
        # Приоритетно обновляем in_channels по input_size (если там больше гармоник)
        if derived_in_channels:
            if raw_base_ch > 0 and spectral_base_ch > 0:
                spectral_ch = max(0, derived_in_channels - raw_base_ch)
                if spectral_ch % spectral_base_ch == 0:
                    derived_harmonics = max(1, spectral_ch // spectral_base_ch)
                    if (in_channels is None) or (derived_in_channels != in_channels and derived_harmonics > 1):
                        in_channels = derived_in_channels
                    num_harmonics = max(num_harmonics, derived_harmonics)
            elif spectral_base_ch > 0 and derived_in_channels % spectral_base_ch == 0:
                derived_harmonics = max(1, derived_in_channels // spectral_base_ch)
                if (in_channels is None) or (derived_in_channels != in_channels and derived_harmonics > 1):
                    in_channels = derived_in_channels
                num_harmonics = max(num_harmonics, derived_harmonics)
        elif in_channels and spectral_base_ch > 0 and in_channels % spectral_base_ch == 0:
            num_harmonics = max(1, in_channels // spectral_base_ch)
        
        # Определяем target_level из имени эксперимента
        # Примеры: 2.6.4_full_stride → full, 2.6.4_hier_stride → full_by_levels
        if 'base_sequential' in exp_name:
            target_level = 'base_sequential'
        elif 'full_by_levels' in exp_name or ('hier_' in exp_name and '2.6.4' in exp_name):
            target_level = 'full_by_levels'
        elif '2.6.4' in exp_name and 'full' in exp_name:
            target_level = 'full'
        else:
            target_level = 'base'
        
        # Загружаем кэшированные данные (PrecomputedDataset - быстро и точно после исправлений)
        # stride=1 для полного перебора ВСЕХ точек
        norm_coef_path = config.get('data', {}).get('norm_coef_path')
        test_df, all_indices, target_cols = _get_test_data_cached(
            data_dir, window_size, eval_stride=1, norm_coef_path=norm_coef_path,
            target_level=target_level
        )

        results['num_classes'] = len(target_cols)
        
        # Определяем ds_target_level для PrecomputedDataset
        # PrecomputedDataset ожидает 'base' или расширенные уровни (не 'full'/'full_by_levels')
        ds_target_level = 'base' if target_level == 'base' else target_level
        
        # Создаём PrecomputedDataset для быстрой оценки
        test_ds = PrecomputedDataset(
            dataframe=test_df,
            indices=all_indices,
            window_size=window_size,
            feature_mode=feature_mode,
            target_columns=target_cols,
            target_level=ds_target_level,
            sampling_strategy=sampling_strategy,
            downsampling_stride=downsampling_stride,
            num_harmonics=num_harmonics
        )
        
        # Проверяем форму данных соответствует ожиданиям модели
        sample_x, _ = test_ds[0]
        actual_shape = sample_x.shape  # (C, T)
        actual_input_size = sample_x.numel()
        expected_input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
        expected_in_channels = config.get('model', {}).get('params', {}).get('in_channels', 0)
        
        # Для CNN/Conv моделей проверяем in_channels вместо input_size
        shape_mismatch = False
        if expected_input_size > 0 and actual_input_size != expected_input_size:
            shape_mismatch = True
        elif expected_in_channels > 0 and actual_shape[0] != expected_in_channels:
            shape_mismatch = True
        
        if shape_mismatch:
            error_msg = (
                f"[!] Несоответствие размера данных:\n"
                f"    Эксперимент: {exp_dir.name}\n"
                f"    Модель: {model_name}\n"
                f"    Ожидалось: in_channels={expected_in_channels}, input_size={expected_input_size}\n"
                f"    Получено: shape={actual_shape}, numel={actual_input_size}\n"
                f"    feature_mode={feature_mode} | sampling={sampling_strategy} | num_harmonics={num_harmonics}\n"
            )
            print(error_msg)
            if _eval_logger:
                _eval_logger.error(error_msg)
            
            # Возвращаем пустые результаты, т.к. модель не сможет работать с такими данными
            return results
        
        # Динамическое определение batch_size для сложных моделей
        # Основано на логике run_phase2_6.py (для экономии GPU памяти)
        eval_batch_size = _get_eval_batch_size(
            config=config,
            model_name=config.get('model', {}).get('name'),
            feature_mode=feature_mode,
            num_harmonics=num_harmonics
        )
        
        if _eval_logger:
            _eval_logger.info(f"Batch size для {exp_dir.name}: {eval_batch_size} (модель: {config.get('model', {}).get('name')}, features: {feature_mode})")
        
        results['full_eval_samples'] = len(all_indices)
        
        start_time = time.perf_counter()
        
        # Создаём модель
        model = _create_model_from_config(config)
        if model is None:
            return results
        
        model = model.to(device)

        # Нужно ли считать Hierarchical Accuracy (только для full / full_by_levels)
        use_hier_eval = (mode == 'multilabel' and target_level in ('full', 'full_by_levels'))
        
        # Оценка best_model.pt
        best_path = exp_dir / "best_model.pt"
        if best_path.exists():
            try:
                # weights_only=False т.к. чекпоинт содержит config (ExperimentConfig)
                checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                _load_state_dict_safe(model, checkpoint, exp_dir.name, 'best_model')
                
                best_metrics, used_batch, used_device = _evaluate_with_batch_fallback(
                    model=model,
                    test_ds=test_ds,
                    base_batch_size=eval_batch_size,
                    device=device,
                    mode=mode,
                    exp_name=exp_dir.name,
                    tag='best_model'
                )
                if best_metrics:  # Проверяем что оценка прошла успешно
                    results['full_best_acc'] = best_metrics['acc']
                    results['full_best_f1'] = best_metrics['f1']
                    results['full_best_per_class_f1'] = best_metrics['per_class_f1']
                    best_support = best_metrics.get('per_class_support', [])
                    if len(results['full_best_per_class_f1']) == len(target_cols):
                        results['full_best_per_class_f1_map'] = dict(zip(target_cols, results['full_best_per_class_f1']))
                        results['full_per_class_f1_count'] = len(results['full_best_per_class_f1'])
                    if best_support and len(best_support) == len(target_cols):
                        results['full_best_per_class_support_map'] = dict(zip(target_cols, best_support))
                        results['full_per_class_support_count'] = len(best_support)
                if used_device.type != device.type:
                    device = used_device

                if use_hier_eval:
                    hier_results, hier_batch, hier_device = _evaluate_hierarchical_with_batch_fallback(
                        model=model,
                        test_ds=test_ds,
                        base_batch_size=eval_batch_size,
                        device=device,
                        exp_name=exp_dir.name,
                        tag='best_model_hier',
                        target_columns=target_cols
                    )
                    if hier_results and 'base_metrics' in hier_results:
                        base_m = hier_results['base_metrics']
                        results['hier_base_f1_best'] = base_m.get('f1', 0.0)
                        results['hier_base_acc_best'] = base_m.get('accuracy', 0.0)
                        results['hier_base_exact_best'] = base_m.get('exact_match', 0.0)
                    if hier_device.type != device.type:
                        device = hier_device
            except Exception as e:
                error_msg = f"    [!] Ошибка загрузки best_model: {e}"
                print(error_msg)
                if _eval_logger:
                    _eval_logger.error(f"{exp_dir.name} - best_model: {e}")
                # Очищаем GPU память при ошибке
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Оценка final_model.pt
        final_path = exp_dir / "final_model.pt"
        if final_path.exists():
            try:
                # weights_only=False т.к. чекпоинт содержит config (ExperimentConfig)
                checkpoint = torch.load(final_path, map_location='cpu', weights_only=False)
                _load_state_dict_safe(model, checkpoint, exp_dir.name, 'final_model')
                
                final_metrics, used_batch, used_device = _evaluate_with_batch_fallback(
                    model=model,
                    test_ds=test_ds,
                    base_batch_size=eval_batch_size,
                    device=device,
                    mode=mode,
                    exp_name=exp_dir.name,
                    tag='final_model'
                )
                if final_metrics:  # Проверяем что оценка прошла успешно
                    results['full_final_acc'] = final_metrics['acc']
                    results['full_final_f1'] = final_metrics['f1']
                    results['full_final_per_class_f1'] = final_metrics['per_class_f1']
                    final_support = final_metrics.get('per_class_support', [])
                    if len(results['full_final_per_class_f1']) == len(target_cols):
                        results['full_final_per_class_f1_map'] = dict(zip(target_cols, results['full_final_per_class_f1']))
                        results['full_per_class_f1_count'] = max(
                            results.get('full_per_class_f1_count', 0),
                            len(results['full_final_per_class_f1'])
                        )
                    if final_support and len(final_support) == len(target_cols):
                        results['full_final_per_class_support_map'] = dict(zip(target_cols, final_support))
                        results['full_per_class_support_count'] = max(
                            results.get('full_per_class_support_count', 0),
                            len(final_support)
                        )
                if used_device.type != device.type:
                    device = used_device

                if use_hier_eval:
                    hier_results, hier_batch, hier_device = _evaluate_hierarchical_with_batch_fallback(
                        model=model,
                        test_ds=test_ds,
                        base_batch_size=eval_batch_size,
                        device=device,
                        exp_name=exp_dir.name,
                        tag='final_model_hier',
                        target_columns=target_cols
                    )
                    if hier_results and 'base_metrics' in hier_results:
                        base_m = hier_results['base_metrics']
                        results['hier_base_f1_final'] = base_m.get('f1', 0.0)
                        results['hier_base_acc_final'] = base_m.get('accuracy', 0.0)
                        results['hier_base_exact_final'] = base_m.get('exact_match', 0.0)
                    if hier_device.type != device.type:
                        device = hier_device
            except Exception as e:
                error_msg = f"    [!] Ошибка загрузки final_model: {e}"
                print(error_msg)
                if _eval_logger:
                    _eval_logger.error(f"{exp_dir.name} - final_model: {e}")
                # Очищаем GPU память при ошибке
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        results['full_eval_time_s'] = time.perf_counter() - start_time
        
    except Exception as e:
        error_msg = f"  [!] Ошибка full evaluation для {exp_dir.name}: {e}"
        print(error_msg)
        if _eval_logger:
            _eval_logger.error(error_msg)
        import traceback
        traceback.print_exc()
        # Очищаем GPU при критической ошибке
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def combine_training_histories(experiments_dir: Path, output_path: Path):
    """
    Объединяет файлы metrics.jsonl всех экспериментов в один текстовый файл.
    Рекурсивно ищет файлы во всех подпапках.
    """
    print(f"Combining training histories into {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Рекурсивно ищем все файлы metrics.jsonl
        metrics_files = sorted(list(experiments_dir.rglob("metrics.jsonl")))
        
        for metrics_path in metrics_files:
            exp_dir = metrics_path.parent
            
            # Записываем заголовок модели для удобства чтения
            outfile.write(f"Model: {exp_dir.name}\n")
            
            # Читаем и записываем метрики
            try:
                with open(metrics_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                outfile.write("\n") # Разделитель
            except Exception as e:
                print(f"Error reading {metrics_path}: {e}")

# =============================================================================
# УТИЛИТЫ ПАРСИНГА
# =============================================================================

def parse_experiment_info(folder_name: str) -> Dict[str, str]:
    """
    Разбирает имя папки на структурированные теги.
    Пример: Exp_2.6.1_HierarchicalCNN_medium...
    """
    info = {
        "exp_id": "Unknown",
        "model_family": "Unknown",
        "complexity": "Unknown",
        "feature_mode": "Unknown",
        "sampling": "Unknown",
        "target": "Unknown",
        "target_level": "base",  # Уровень детализации меток: base, full, full_by_levels
        "is_aug": "No",
        "balancing": "None",
        "arch_type": "Base"  # Тип архитектуры: Base, Hierarchical или Hybrid
    }
    
    parts = folder_name.split('_')
    
    # Пытаемся вытащить ID опыта
    id_match = re.search(r'(\d+\.\d+\.\d+\.?\w*)', folder_name)
    if id_match:
        info["exp_id"] = id_match.group(1)

    # Определяем архитектуру (порядок важен: сначала проверяем более специфичные)
    if 'Hybrid' in folder_name:
        info["arch_type"] = "Hybrid"
    elif 'Hierarchical' in folder_name:
        info["arch_type"] = "Hierarchical"
    
    # Определяем target_level из имени эксперимента
    # Примеры: full_stride, full_snapshot, hier_stride (full_by_levels)
    if 'base_sequential' in folder_name.lower():
        info["target_level"] = "base_sequential"
    elif 'full_by_levels' in folder_name or ('hier_' in folder_name.lower() and '2.6.4' in folder_name):
        info["target_level"] = "full_by_levels"
    elif 'full' in folder_name.lower() and '2.6.4' in folder_name:
        # Только для 2.6.4 эксперимента, чтобы не спутать с 'full' в других контекстах
        info["target_level"] = "full"
    else:
        info["target_level"] = "base"

    # Поиск ключевого семейства моделей (чистим от префиксов)
    models_map = {
        'SimpleMLP': 'MLP',
        'SimpleCNN': 'CNN',
        'ConvKAN': 'ConvKAN',
        'SimpleKAN': 'SimpleKAN',
        'PhysicsKAN': 'PhysicsKAN',
        'ResNet1D': 'ResNet',
    }
    
    # Сначала ищем полное соответствие в имени
    found_model = False
    for pattern, clean_name in models_map.items():
        if pattern in folder_name:
            info["model_family"] = clean_name
            found_model = True
            break
            
    # Если это иерархическая или гибридная модель и мы не нашли семейство явно
    if not found_model and info["arch_type"] in ("Hierarchical", "Hybrid"):
        # Вытаскиваем то, что идет после слова Hierarchical или Hybrid
        match = re.search(r'(?:Hierarchical|Hybrid)([a-zA-Z0-9]+)', folder_name)
        if match:
            info["model_family"] = match.group(1)
    
    # Добавляем префикс архитектуры к model_family для Hybrid/Hierarchical
    if info["arch_type"] == "Hybrid" and not info["model_family"].startswith("Hybrid"):
        info["model_family"] = "Hybrid" + info["model_family"]
    elif info["arch_type"] == "Hierarchical" and not info["model_family"].startswith("Hier"):
        info["model_family"] = "Hier" + info["model_family"]

    if 'light' in parts: info["complexity"] = 'Light'
    elif 'medium' in parts: info["complexity"] = 'Medium'
    elif 'heavy' in parts: info["complexity"] = 'Heavy'

    if 'raw' in parts: info["feature_mode"] = 'Raw'
    elif 'phase_polar' in folder_name: info["feature_mode"] = 'PhasePolar'
    elif 'phase_rect' in folder_name: info["feature_mode"] = 'PhaseRect'
    elif 'symmetric' in folder_name: info["feature_mode"] = 'Symmetric'
    elif 'power' in parts: info["feature_mode"] = 'Power'
    elif 'ab' in parts: info["feature_mode"] = 'AB'
    
    if 'stride' in parts: info["sampling"] = 'Stride'
    elif 'snapshot' in parts: info["sampling"] = 'Snapshot'
    elif 'none_sampl' in parts: info["sampling"] = 'NoneSampling'
    
    if 'aug' in parts: info["is_aug"] = 'Yes'
    
    if 'weights' in parts: info["balancing"] = 'Weights'
    elif 'global' in parts: info["balancing"] = 'Global'
    elif 'oscillogram' in parts: info["balancing"] = 'Oscillogram'
    elif 'none_weights' in parts: info["balancing"] = 'NoneWeights'

    return info


def extract_base_exp_id(text: str) -> str:
    """
    Извлекает базовый ID опыта из строки (например, 2.5.1.0 или 2.6.4).
    Используется для устойчивой группировки и понятных заголовков графиков.
    """
    match = re.search(r'(\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+)', text)
    if match:
        return match.group(1)
    return "Unknown"

# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================

class ReportVisualizer:
    # Словарь переводов
    TEXTS = {
        'ru': {
            'loss_title': "Опыт {}: Лосс валидации (Обрезано по 2.0)",
            'f1_title': "Опыт {}: F1-Macro (Валидация)",
            'epoch': "Эпоха",
            'loss': "Лосс",
            'f1': "F1 Score",
            'pareto_title': "Кривая Парето: Точность vs Скорость инференса (CPU)",
            'latency': "Задержка (мс) - Лог. шкала",
            'val_f1_label': "F1-Macro (Валидация)",
            'model': "Модель",
            'complexity': "Сложность",
            'groups_desc': "Генерация групповых графиков"
        },
        'en': {
            'loss_title': "Exp {}: Validation Loss (Clipped at 2.0)",
            'f1_title': "Exp {}: F1-Macro Score",
            'epoch': "Epoch",
            'loss': "Loss",
            'f1': "F1 Score",
            'pareto_title': "Pareto Frontier: Accuracy vs Inference Speed (CPU)",
            'latency': "Latency (ms) - Log Scale",
            'val_f1_label': "Validation F1-Macro",
            'model': "Model",
            'complexity': "Complexity",
            'groups_desc': "Generating group plots"
        }
    }

    def __init__(self, output_root: Path, lang: str = 'ru'):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lang = lang if lang in self.TEXTS else 'ru'
        self.t = self.TEXTS[self.lang]
        
        # Цвета для моделей, чтобы они были одинаковыми на всех графиках
        self.color_map = {
            'MLP': '#95a5a6',
            'CNN': '#3498db',
            'ResNet': '#2ecc71',
            'SimpleKAN': '#e67e22',
            'ConvKAN': '#e74c3c',
            'PhysicsKAN': '#9b59b6',
            'Unknown': '#34495e'
        }

    def plot_group_curves(self, group_label: str, group_df: pd.DataFrame, histories: Dict[str, List[Dict]], group_file_id: str):
        """Рисует сравнение всех моделей внутри одной группы (по заданным параметрам)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for name, history in histories.items():
            if not history: continue
            df_h = pd.DataFrame(history)
            info = parse_experiment_info(name)
            
            # Формируем метку: CNN (H) или CNN (Base)
            arch_suffix = "H" if info['arch_type'] == "Hierarchical" else "B"
            model_label = f"{info['model_family']} ({arch_suffix}-{info['complexity']})"
            
            color = self.color_map.get(info['model_family'], self.color_map['Unknown'])
            # Иерархические модели рисуем пунктиром для отличия
            ls = '--' if info['arch_type'] == "Hierarchical" else '-'
            
            # 1. Loss (с ограничением, чтобы не "взрывалось")
            val_loss = df_h['val_loss'].clip(upper=2.0) # Защита от бесконечности
            ax1.plot(df_h['epoch'], val_loss, label=model_label, color=color, linestyle=ls, alpha=0.8)
            
            # 2. F1 Score
            if 'val_f1' in df_h.columns:
                ax2.plot(df_h['epoch'], df_h['val_f1'], label=model_label, color=color, linestyle=ls, alpha=0.8)

        ax1.set_title(self.t['loss_title'].format(group_label))
        ax1.set_xlabel(self.t['epoch'])
        ax1.set_ylabel(self.t['loss'])
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title(self.t['f1_title'].format(group_label))
        ax2.set_xlabel(self.t['epoch'])
        ax2.set_ylabel(self.t['f1'])
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        save_path = self.output_root / "groups" / f"{group_file_id}_comparison_{self.lang}.png"
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_pareto(self, summary_df: pd.DataFrame):
        """Рисует график 'Точность vs Скорость'."""
        plt.figure(figsize=(10, 7))
        # Очистка данных от 0 или некорректных замеров времени
        # Проверяем наличие колонки
        if 'CPU Inf (ms)' not in summary_df.columns:
            return
            
        df_plot = summary_df[summary_df['CPU Inf (ms)'] > 0].copy()
        if df_plot.empty: return

        # Создаем колонку для отображения: CNN (H)
        df_plot['DisplayModel'] = df_plot.apply(
            lambda x: f"{x['Model']} ({x['arch_type'][0]})" if 'arch_type' in x else x['Model'], axis=1
        )

        sns.scatterplot(
            data=df_plot, 
            x='CPU Inf (ms)', 
            y='Val F1', 
            hue='Model', 
            style='Complexity',
            s=120, 
            alpha=0.8
        )
        
        plt.xscale('log')
        plt.title(self.t['pareto_title'])
        plt.xlabel(self.t['latency'])
        plt.ylabel(self.t['val_f1_label'])
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_root / f"summary_pareto_{self.lang}.png", dpi=200)
        plt.close()

def aggregate_reports(
    root_dir: str, 
    output_dir: str = None, 
    plot: bool = False, 
    benchmark: bool = False, 
    full_eval: bool = False, 
    data_dir: str = None, 
    lang: str = 'ru',
    advanced_plots: bool = False
):
    """
    Агрегирует отчеты обучения из всех поддиректорий.
    
    Args:
        root_dir: Корневая директория, содержащая папки экспериментов.
        output_dir: Опциональный путь к папке для сохранения всех отчетов и графиков.
        plot: Генерировать ли графики обучения для каждого эксперимента.
        benchmark: Выполнять ли глубокий бенчмарк на CPU.
        full_eval: Выполнять ли полную оценку на тестовом датасете (GPU, все точки).
        data_dir: Путь к директории с датасетами (для full_eval).
        lang: Язык графиков ('ru' или 'en').
        advanced_plots: Генерировать ли расширенные графики (Парето, heatmaps и др.).
    
    Оптимизация:
        - Загружает существующий summary_report.csv для пропуска уже посчитанных моделей
        - Пропускает full_eval/benchmark если метрики уже валидны (не 0)
        - Пропускает генерацию графиков если файлы уже существуют (кроме pareto)
    """
    root_path = Path(root_dir)
    experiments = []
    all_histories = {}
    
    # Подготовка путей вывода
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        figures_path = out_path / "figures"
        csv_path = out_path / "summary_report.csv"
        history_path = out_path / "combined_metrics_history.txt"
        error_log_path = out_path / "processing_errors.log"
        eval_log_path = out_path / "eval_log.txt"
        # Очищаем логи при новом запуске
        if error_log_path.exists():
            error_log_path.unlink()
        if eval_log_path.exists():
            eval_log_path.unlink()
    else:
        out_path = None
        figures_path = root_path / "figures"
        csv_path = None
        history_path = None
        error_log_path = None
        eval_log_path = None

    # Инициализация логгера для оценки
    global _eval_logger
    if eval_log_path:
        _eval_logger = logging.getLogger('eval_logger')
        _eval_logger.setLevel(logging.DEBUG)
        # Удаляем старые обработчики если есть
        _eval_logger.handlers.clear()
        # Добавляем обработчик для файла
        fh = logging.FileHandler(eval_log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        _eval_logger.addHandler(fh)
        print(f"[Log] Оценка будет логироваться в: {eval_log_path}")
    else:
        _eval_logger = None

    # Путь к датасетам по умолчанию
    if data_dir is None:
        data_dir = str(ROOT_DIR_PROJECT / 'data' / 'ml_datasets')

    # Загружаем существующий отчёт для пропуска уже посчитанных моделей
    existing_df = load_existing_summary(csv_path) if csv_path else None
    
    # Счётчики для статистики пропусков
    skip_stats = {
        'full_eval_skipped': 0,
        'full_eval_recalc': 0,
        'benchmark_skipped': 0,
        'benchmark_recalc': 0,
        'plots_skipped': 0,
        'plots_generated': 0
    }

    print(f"Сканирование {root_path} (Язык: {lang})...")
    
    # Сначала собираем все эксперименты
    all_exp_dirs = []
    for metrics_file in root_path.rglob("metrics.jsonl"):
        all_exp_dirs.append(metrics_file.parent)
    
    print(f"Найдено {len(all_exp_dirs)} экспериментов")

    # Найти все файлы metrics.jsonl
    for exp_idx, exp_dir in enumerate(tqdm(all_exp_dirs, desc="Сбор логов")):
        try:
            metrics_file = exp_dir / "metrics.jsonl"
            config_file = exp_dir / "config.json"
            
            # Загрузка данных
            metrics = load_metrics(metrics_file)
            config = load_config(config_file) if config_file.exists() else {}
            
            if not metrics:
                continue
                
            all_histories[exp_dir.name] = metrics

            # Умный парсинг инфо из имени папки
            info = parse_experiment_info(exp_dir.name)
            
            # Получаем существующие данные для этого эксперимента
            existing_row = get_existing_row(existing_df, exp_dir.name)

            # Определяем target_level для логики пересчёта
            current_target_level = info.get("target_level", "base")
            if existing_row is not None and 'TargetLevel' in existing_row.index:
                current_target_level = existing_row.get('TargetLevel', current_target_level)

            # Генерация графиков (пропускаем если файл уже существует)
            if plot:
                plot_path = exp_dir / "learning_curves.png"
                if plot_path.exists():
                    skip_stats['plots_skipped'] += 1
                else:
                    plot_learning_curves(metrics, plot_path)
                    skip_stats['plots_generated'] += 1

            # Найти лучшую эпоху (на основе min val_loss, с защитой от пустых данных)
            if any('val_loss' in m for m in metrics):
                best_epoch = min(metrics, key=lambda x: x.get('val_loss', 999.0))
            else:
                best_epoch = metrics[-1]

            # Извлечение информации
            # Сначала пробуем взять из нового поля model_info в конфиге
            num_params = config.get('model_info', {}).get('num_params', best_epoch.get('num_params', 0))
            
            cpu_inf = best_epoch.get('cpu_inf_time_ms', 0.0)
            
            # Benchmark: проверяем нужен ли пересчёт
            if benchmark:
                if needs_benchmark_recalc(existing_row):
                    cpu_inf = benchmark_model_cpu(exp_dir, config, iterations=1000)
                    skip_stats['benchmark_recalc'] += 1
                else:
                    # Используем существующее значение из кэша
                    cpu_inf = existing_row['CPU Inf (ms)']
                    skip_stats['benchmark_skipped'] += 1

            # Формируем данные: приоритет отдаем существующей строке (вашим ручным правкам в CSV)
            if existing_row is not None:
                exp_data = {
                    "ExpID": existing_row.get("ExpID", info["exp_id"]),
                    "Experiment": exp_dir.name,
                    "Model": existing_row.get("Model", info["model_family"]),
                    "Complexity": existing_row.get("Complexity", info["complexity"]),
                    "Features": existing_row.get("Features", info["feature_mode"]),
                    "Sampling": existing_row.get("Sampling", info["sampling"]),
                    "TargetLevel": existing_row.get("TargetLevel", info["target_level"]),
                    "Balancing": existing_row.get("Balancing", info["balancing"]),
                    "Aug": existing_row.get("Aug", info["is_aug"]),
                    "arch_type": existing_row.get("arch_type", info["arch_type"]),
                    "Val F1": best_epoch.get('val_f1', 0.0),
                    "Val Loss": best_epoch.get('val_loss', 0.0),
                    "Val Acc": best_epoch.get('val_acc', 0.0),
                    "CPU Inf (ms)": cpu_inf,
                    "Params": num_params,
                    "Epochs": len(metrics),
                    "Best Epoch": best_epoch.get('epoch'),
                    "Path": exp_dir.name
                }
            else:
                exp_data = {
                    "ExpID": info["exp_id"],
                    "Experiment": exp_dir.name,
                    "Model": info["model_family"],
                    "Complexity": info["complexity"],
                    "Features": info["feature_mode"],
                    "Sampling": info["sampling"],
                    "TargetLevel": info["target_level"],
                    "Balancing": info["balancing"],
                    "Aug": info["is_aug"],
                    "arch_type": info["arch_type"],
                    "Val F1": best_epoch.get('val_f1', 0.0),
                    "Val Loss": best_epoch.get('val_loss', 0.0),
                    "Val Acc": best_epoch.get('val_acc', 0.0),
                    "CPU Inf (ms)": cpu_inf,
                    "Params": num_params,
                    "Epochs": len(metrics),
                    "Best Epoch": best_epoch.get('epoch'),
                    "Path": exp_dir.name
                }
            
            if 'data' in config:
                exp_data['Window'] = config['data'].get('window_size')
                exp_data['Batch'] = config['data'].get('batch_size')
            
            # Полная оценка на тестовом датасете (GPU)
            if full_eval:
                require_hier = current_target_level in ('full', 'full_by_levels')
                if needs_full_eval_recalc(existing_row, require_hierarchical=require_hier):
                    # Нужен пересчёт
                    full_metrics = evaluate_full_test_dataset(exp_dir, config, data_dir)
                    exp_data['Full Best Acc'] = full_metrics['full_best_acc']
                    exp_data['Full Best F1'] = full_metrics['full_best_f1']
                    exp_data['Full Final Acc'] = full_metrics['full_final_acc']
                    exp_data['Full Final F1'] = full_metrics['full_final_f1']
                    exp_data['Full Eval Time (s)'] = full_metrics['full_eval_time_s']
                    exp_data['Full Eval Samples'] = full_metrics['full_eval_samples']
                    exp_data['Num Classes'] = full_metrics.get('num_classes', 0)
                    exp_data['Full Per Class F1 Count'] = full_metrics.get('full_per_class_f1_count', 0)
                    exp_data['Full Per Class Support Count'] = full_metrics.get('full_per_class_support_count', 0)

                    # Hierarchical Accuracy (только для full/full_by_levels)
                    if require_hier:
                        exp_data['Hier Base F1 (Best)'] = full_metrics.get('hier_base_f1_best', 0.0)
                        exp_data['Hier Base Acc (Best)'] = full_metrics.get('hier_base_acc_best', 0.0)
                        exp_data['Hier Base Exact (Best)'] = full_metrics.get('hier_base_exact_best', 0.0)
                        exp_data['Hier Base F1 (Final)'] = full_metrics.get('hier_base_f1_final', 0.0)
                        exp_data['Hier Base Acc (Final)'] = full_metrics.get('hier_base_acc_final', 0.0)
                        exp_data['Hier Base Exact (Final)'] = full_metrics.get('hier_base_exact_final', 0.0)

                        # Per-class F1/Support для всех классов (полный набор меток)
                        best_f1_map = full_metrics.get('full_best_per_class_f1_map', {})
                        final_f1_map = full_metrics.get('full_final_per_class_f1_map', {})
                        best_sup_map = full_metrics.get('full_best_per_class_support_map', {})
                        final_sup_map = full_metrics.get('full_final_per_class_support_map', {})

                        for cls_name, val in best_f1_map.items():
                            exp_data[f"Full F1 {cls_name} (Best)"] = val
                        for cls_name, val in final_f1_map.items():
                            exp_data[f"Full F1 {cls_name} (Final)"] = val
                        for cls_name, val in best_sup_map.items():
                            exp_data[f"Full Support {cls_name} (Best)"] = val
                        for cls_name, val in final_sup_map.items():
                            exp_data[f"Full Support {cls_name} (Final)"] = val
                    
                    # Per-class F1 для radar charts
                    # Берём лучшие per_class метрики (от best модели)
                    best_per_class = full_metrics.get('full_best_per_class_f1', [])
                    if best_per_class and len(best_per_class) >= 4:
                        exp_data['Class_0_F1'] = best_per_class[0]  # Normal
                        exp_data['Class_1_F1'] = best_per_class[1]  # Switching
                        exp_data['Class_2_F1'] = best_per_class[2]  # Abnormal
                        exp_data['Class_3_F1'] = best_per_class[3]  # Fault
                    
                    skip_stats['full_eval_recalc'] += 1
                else:
                    # Используем существующие данные из кэша
                    exp_data['Full Best Acc'] = existing_row.get('Full Best Acc', 0.0)
                    exp_data['Full Best F1'] = existing_row.get('Full Best F1', 0.0)
                    exp_data['Full Final Acc'] = existing_row.get('Full Final Acc', 0.0)
                    exp_data['Full Final F1'] = existing_row.get('Full Final F1', 0.0)
                    exp_data['Full Eval Time (s)'] = existing_row.get('Full Eval Time (s)', 0.0)
                    exp_data['Full Eval Samples'] = existing_row.get('Full Eval Samples', 0)
                    exp_data['Num Classes'] = existing_row.get('Num Classes', 0)
                    exp_data['Full Per Class F1 Count'] = existing_row.get('Full Per Class F1 Count', 0)
                    exp_data['Full Per Class Support Count'] = existing_row.get('Full Per Class Support Count', 0)

                    # Hierarchical Accuracy из кэша (если есть)
                    if require_hier:
                        exp_data['Hier Base F1 (Best)'] = existing_row.get('Hier Base F1 (Best)', 0.0)
                        exp_data['Hier Base Acc (Best)'] = existing_row.get('Hier Base Acc (Best)', 0.0)
                        exp_data['Hier Base Exact (Best)'] = existing_row.get('Hier Base Exact (Best)', 0.0)
                        exp_data['Hier Base F1 (Final)'] = existing_row.get('Hier Base F1 (Final)', 0.0)
                        exp_data['Hier Base Acc (Final)'] = existing_row.get('Hier Base Acc (Final)', 0.0)
                        exp_data['Hier Base Exact (Final)'] = existing_row.get('Hier Base Exact (Final)', 0.0)

                        # Переносим все per-class колонки из кэша
                        for col_name in existing_row.index:
                            if col_name.startswith('Full F1 ') or col_name.startswith('Full Support '):
                                exp_data[col_name] = existing_row.get(col_name, 0)
                    
                    # Per-class F1 из кэша (если есть)
                    for i in range(4):
                        col = f'Class_{i}_F1'
                        if col in existing_row.index:
                            exp_data[col] = existing_row.get(col, None)
                    
                    skip_stats['full_eval_skipped'] += 1
            
            experiments.append(exp_data)
            
        except Exception as e:
            error_msg = f"Ошибка при обработке опыта {exp_dir.name}: {str(e)}"
            print(f"\n[!] {error_msg}")
            
            # Логируем ошибку в файл, если задан out_dir
            if output_dir:
                try:
                    error_log = Path(output_dir) / "processing_errors.log"
                    with open(error_log, 'a', encoding='utf-8') as f:
                        f.write(f"=== {exp_dir.name} ===\n")
                        f.write(f"{error_msg}\n")
                        import traceback
                        f.write(traceback.format_exc())
                        f.write("-" * 50 + "\n\n")
                except:
                    pass
            continue
    
    if not experiments:
        print("Эксперименты не найдены.")
        return

    # Создать DataFrame и заполнить пропуски
    df = pd.DataFrame(experiments).fillna(0)
    
    # --- ВИЗУАЛИЗАЦИЯ (Report Engine 2.0) ---
    if plot:
        viz = ReportVisualizer(figures_path, lang=lang)
        
        # 1. Групповые графики по параметрам (Complexity, Features, Sampling, TargetLevel, Balancing, Aug, arch_type)
        group_cols = [
            'Complexity', 'Features', 'Sampling', 'TargetLevel',
            'Balancing', 'Aug', 'arch_type'
        ]

        grouped = df.groupby(group_cols, dropna=False)
        for idx, (group_keys, group_df) in enumerate(tqdm(grouped, desc=viz.t['groups_desc'])):
            # Список базовых ID опытов в группе
            base_ids = sorted({extract_base_exp_id(str(x)) for x in group_df['Experiment'].tolist()})
            base_ids = [x for x in base_ids if x != "Unknown"]

            # Формируем читаемую метку группы
            if len(base_ids) == 1:
                group_label = base_ids[0]
                file_prefix = f"exp_{base_ids[0]}"
            elif len(base_ids) > 1:
                group_label = ", ".join(base_ids)
                file_prefix = f"exp_multi_{idx:03d}"
            else:
                group_label = f"Group {idx:03d}"
                file_prefix = f"group_{idx:03d}"

            # Небольшой суффикс по параметрам для уникальности файлов
            params_suffix = "_".join([str(k) for k in group_keys])
            params_suffix = re.sub(r'[^a-zA-Z0-9._-]+', '_', params_suffix)[:80]
            group_file_id = f"{file_prefix}_{params_suffix}" if params_suffix else file_prefix

            # Проверяем существование графика
            group_plot_path = figures_path / "groups" / f"{group_file_id}_comparison_{lang}.png"
            if group_plot_path.exists():
                skip_stats['plots_skipped'] += 1
                continue

            # Находим все папки, относящиеся к этой группе
            exp_folders = group_df['Path'].tolist()
            group_histories = {f: all_histories[f] for f in exp_folders if f in all_histories}
            viz.plot_group_curves(group_label, group_df, group_histories, group_file_id)
            skip_stats['plots_generated'] += 1
            
        # 2. Паррето график скорость/точность (ВСЕГДА обновляем)
        viz.plot_pareto(df)

    # Сортировка
    if full_eval and 'Full Best F1' in df.columns:
        df = df.sort_values(['ExpID', 'Full Best F1'], ascending=[True, False])
    else:
        df = df.sort_values(['ExpID', 'Val F1'], ascending=[True, False])

    # Отобразить топ-результаты
    print("\nАгрегированный отчет (Топ-20):")
    cols_to_show = ['ExpID', 'Model', 'Complexity', 'Val F1', 'CPU Inf (ms)']
    if full_eval: cols_to_show.extend(['Full Best F1'])
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(df[cols_to_show].head(20).to_markdown(index=False, floatfmt=".4f"))

    # Сохранение файлов
    if out_path:
        df.to_csv(csv_path, index=False)
        print(f"\nCSV-отчет сохранен: {csv_path}")
        
        combine_training_histories(root_path, history_path)
        print(f"История обучения объединена: {history_path}")

    if plot:
        plot_comparison(all_histories, out_path if out_path else root_path)
    
    # --- РАСШИРЕННАЯ ВИЗУАЛИЗАЦИЯ ---
    if advanced_plots and ADVANCED_VIZ_AVAILABLE and out_path:
        print("\n=== Генерация расширенных графиков ===")
        advanced_figures_path = out_path / "figures_advanced"
        adv_viz = AdvancedVisualizer(advanced_figures_path, lang=lang)
        adv_viz.generate_all_plots(df, histories=all_histories)
    elif advanced_plots and not ADVANCED_VIZ_AVAILABLE:
        print("[!] Расширенная визуализация недоступна (модуль не найден)")
    
    # Выводим статистику пропусков
    print(f"\n=== Статистика оптимизации ===")
    if full_eval:
        print(f"  Full Eval: пересчитано {skip_stats['full_eval_recalc']}, пропущено {skip_stats['full_eval_skipped']}")
    if benchmark:
        print(f"  Benchmark: пересчитано {skip_stats['benchmark_recalc']}, пропущено {skip_stats['benchmark_skipped']}")
    if plot:
        print(f"  Графики: сгенерировано {skip_stats['plots_generated']}, пропущено {skip_stats['plots_skipped']}")

if __name__ == "__main__":
    # Сначала проверяем CLI, если нет аргументов - MANUAL_RUN
    parser = argparse.ArgumentParser(description="Агрегация отчетов экспериментов.")
    parser.add_argument("--root", type=str, default=None, help="Корневая директория экспериментов")
    parser.add_argument("--out-dir", type=str, default=None, help="Путь к папке для сохранения всех отчетов")
    parser.add_argument("--plot", action="store_true", help="Генерировать графики обучения")
    parser.add_argument("--benchmark", action="store_true", help="Выполнить глубокий бенчмарк на CPU")
    parser.add_argument("--full-eval", action="store_true", help="Выполнить полную оценку на тестовом датасете (GPU)")
    parser.add_argument("--advanced-plots", action="store_true", help="Генерировать расширенные графики (Парето, heatmaps)")
    parser.add_argument("--data-dir", type=str, default=None, help="Путь к директории с датасетами")
    parser.add_argument("--lang", type=str, default="ru", choices=["ru", "en"], help="Язык графиков (ru/en)")
    
    args, unknown = parser.parse_known_args()

    # === ВЕРСИЯ 1: РУЧНОЙ ЗАПУСК ЧЕРЕЗ КОНСТАНТЫ ===
    # Отредактируйте параметры ниже для быстрого запуска без аргументов командной строки.
    MANUAL_RUN = True
    
    # Если аргументы не переданы, используем MANUAL-конфиг
    if MANUAL_RUN or len(sys.argv) <= 1 or args.root is None:
        # === MANUAL CONFIG ===
        # ROOT_DIR: Папка, где лежат результаты ваших экспериментов (metrics.jsonl, config.json).
        ROOT_DIR = "experiments/phase2_5" #_новые опыты/Exp_2.5.1/Exp_2.5.1.0_ConvKAN_light_raw_none_base" # пока сюда скопировал часть 2.6
        # OUTPUT_DIR: Папка, куда сохранять ВСЕ отчёты / файлы / картинки
        OUTPUT_DIR = "reports/Exp_2_5_and_start_Exp_2_6"
        # GENERATE_PLOTS: Если True, для каждого эксперимента будут построены графики обучения.
        GENERATE_PLOTS = True
        # RUN_BENCHMARK: Если True, будет выполнен глубокий замер скорости инференса на CPU (1000 итераций).
        RUN_BENCHMARK = True
        # RUN_FULL_EVAL: Если True, будет выполнена полная оценка на всём тестовом датасете (GPU).
        # Это более точная оценка, чем Val Acc/F1 из обучения (там используется подвыборка).
        RUN_FULL_EVAL = True
        # ADVANCED_PLOTS: Если True, строятся расширенные графики (8 Парето, heatmaps, boxplots и др.)
        ADVANCED_PLOTS = True
        # LANG: Язык графиков ('ru' или 'en').
        LANG = "ru"
        
        print(f"[!] Запуск в ручном режиме (MANUAL_RUN)")
    else:
        # === CLI CONFIG ===
        ROOT_DIR = args.root
        OUTPUT_DIR = args.out_dir
        GENERATE_PLOTS = args.plot
        RUN_BENCHMARK = args.benchmark
        RUN_FULL_EVAL = args.full_eval
        ADVANCED_PLOTS = args.advanced_plots
        LANG = args.lang

    if ROOT_DIR:
        aggregate_reports(
            ROOT_DIR, 
            OUTPUT_DIR, 
            GENERATE_PLOTS, 
            RUN_BENCHMARK,
            full_eval=RUN_FULL_EVAL,
            lang=LANG,
            data_dir=args.data_dir,
            advanced_plots=ADVANCED_PLOTS)