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

# Добавляем корень проекта в путь импорта
ROOT_DIR_PROJECT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR_PROJECT))

from osc_tools.data_management.dataset_manager import DatasetManager
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.dataset import OscillogramDataset
from osc_tools.ml.labels import get_target_columns

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
    norm_coef_path: Optional[str] = None
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
    
    Returns:
        (test_df, indices, target_cols)
    """
    import polars as pl
    
    # Включаем norm_coef_path в ключ кэша
    cache_key = f"{data_dir}_{window_size}_{eval_stride}_{norm_coef_path}"
    
    if cache_key not in _test_data_cache:
        dm = DatasetManager(data_dir, norm_coef_path=norm_coef_path)
        target_cols = get_target_columns('base')
        
        print(f"[Full Eval] Загрузка тестового датасета (PrecomputedDataset)...")
        # Убеждаемся что предрассчитанный файл существует
        # Если пришёл нестандартный путь к нормализации, форсируем пересоздание если файл старый?
        # Пока просто создаем если нет.
        dm.create_precomputed_test_csv()
        
        # Загружаем
        test_df = dm.load_test_df(precomputed=True)
        test_df = test_df.with_row_index("row_nr")
        
        # Создаём индексы для полной оценки
        indices = PrecomputedDataset.create_indices(
            test_df,
            window_size=window_size,
            mode='val',
            stride=eval_stride
        )
        print(f"[Full Eval] Загружено {len(indices):,} точек (stride={eval_stride})")
        
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
            elif model_name in ['SimpleKAN', 'ConvKAN', 'PhysicsKAN']:
                from osc_tools.ml.models.kan import SimpleKAN, ConvKAN, PhysicsKAN
                models_map = {'SimpleKAN': SimpleKAN, 'ConvKAN': ConvKAN, 'PhysicsKAN': PhysicsKAN}
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


def evaluate_model_full(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mode: str = 'multilabel'
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
            else:  # multilabel
                y = y.float()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Расчёт метрик
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    if mode == 'classification':
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    else:
        balanced_acc = 0.0
    
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    
    return {
        'acc': acc,
        'f1': f1,
        'balanced_acc': balanced_acc,
        'per_class_f1': per_class_f1
    }


def evaluate_full_test_dataset(
    exp_dir: Path,
    config: Dict[str, Any],
    data_dir: str,
    batch_size: int = 8192,  # Большой батч для эффективного использования GPU
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Полная оценка моделей (best и final) на всём тестовом датасете.
    
    Стратегия оптимизации:
    - stride=1: перебираем ВСЕ возможные окна (полное покрытие)
    - batch_size=8192: GPU обрабатывает большие батчи параллельно
    - PrecomputedDataset: данные уже предрассчитаны, извлечение быстрое
    
    Args:
        exp_dir: Путь к директории эксперимента
        config: Конфигурация эксперимента
        data_dir: Путь к директории с датасетами
        batch_size: Размер батча (8192 по умолчанию для GPU)
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
        'full_eval_samples': 0
    }
    
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
        
        if 'stride' in exp_name:
            sampling_strategy = 'stride'
            downsampling_stride = 16
        elif 'snapshot' in exp_name:
            sampling_strategy = 'snapshot'
            downsampling_stride = 32
        elif 'none' in exp_name or 'full' in exp_name:
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
        
        # Feature mode - определяем по in_channels или input_size
        # 16 = phase_polar (8 сигналов × 2: amp + phase)
        # 12 = symmetric (6 составляющих × 2: amp + phase)
        # 8 = raw (8 сигналов без FFT)
        in_channels = config.get('model', {}).get('params', {}).get('in_channels', None)
        
        if in_channels is None:
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
        
        if in_channels == 12:
            feature_mode = 'symmetric'
        elif in_channels == 8:
            feature_mode = 'raw'
        else:
            feature_mode = 'phase_polar'  # Default для Phase 2.6
        
        # Загружаем кэшированные данные (PrecomputedDataset - быстро и точно после исправлений)
        # stride=1 для полного перебора ВСЕХ точек
        norm_coef_path = config.get('data', {}).get('norm_coef_path')
        test_df, all_indices, target_cols = _get_test_data_cached(
            data_dir, window_size, eval_stride=1, norm_coef_path=norm_coef_path
        )
        
        # Создаём PrecomputedDataset для быстрой оценки
        test_ds = PrecomputedDataset(
            dataframe=test_df,
            indices=all_indices,
            window_size=window_size,
            feature_mode=feature_mode,
            target_columns=target_cols,
            target_level='base',
            sampling_strategy=sampling_strategy,
            downsampling_stride=downsampling_stride
        )
        
        # Проверяем форму данных соответствует ожиданиям модели
        sample_x, _ = test_ds[0]
        actual_shape = sample_x.shape  # (C, T)
        actual_input_size = sample_x.numel()
        expected_input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
        
        if expected_input_size > 0 and actual_input_size != expected_input_size:
            print(f"    [!] Несоответствие размера: ожидалось {expected_input_size}, получено {actual_input_size}")
            print(f"        Проверьте sampling_strategy={sampling_strategy}")
            # Возвращаем пустые результаты, т.к. модель не сможет работать с такими данными
            return results
        
        # DataLoader с большим батчем для скорости
        # На Windows num_workers > 0 может вызвать проблемы с multiprocessing,
        # поэтому используем 0 по умолчанию (данные уже предрассчитаны, загрузка быстрая)
        test_loader = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # На Windows безопаснее 0 (данные предрассчитаны, загрузка быстрая)
            pin_memory=(device.type == 'cuda')
        )
        
        results['full_eval_samples'] = len(all_indices)
        
        start_time = time.perf_counter()
        
        # Создаём модель
        model = _create_model_from_config(config)
        if model is None:
            return results
        
        model = model.to(device)
        
        # Оценка best_model.pt
        best_path = exp_dir / "best_model.pt"
        if best_path.exists():
            try:
                # weights_only=False т.к. чекпоинт содержит config (ExperimentConfig)
                checkpoint = torch.load(best_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                best_metrics = evaluate_model_full(model, test_loader, device, mode)
                results['full_best_acc'] = best_metrics['acc']
                results['full_best_f1'] = best_metrics['f1']
                results['full_best_per_class_f1'] = best_metrics['per_class_f1']
            except Exception as e:
                print(f"    [!] Ошибка загрузки best_model: {e}")
        
        # Оценка final_model.pt
        final_path = exp_dir / "final_model.pt"
        if final_path.exists():
            try:
                # weights_only=False т.к. чекпоинт содержит config (ExperimentConfig)
                checkpoint = torch.load(final_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                final_metrics = evaluate_model_full(model, test_loader, device, mode)
                results['full_final_acc'] = final_metrics['acc']
                results['full_final_f1'] = final_metrics['f1']
                results['full_final_per_class_f1'] = final_metrics['per_class_f1']
            except Exception as e:
                print(f"    [!] Ошибка загрузки final_model: {e}")
        
        results['full_eval_time_s'] = time.perf_counter() - start_time
        
    except Exception as e:
        print(f"  [!] Ошибка full evaluation для {exp_dir.name}: {e}")
        import traceback
        traceback.print_exc()
    
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
        "is_aug": "No",
        "balancing": "None",
        "arch_type": "Base"  # Тип архитектуры: Base или Hierarchical
    }
    
    parts = folder_name.split('_')
    
    # Пытаемся вытащить ID опыта
    id_match = re.search(r'(\d+\.\d+\.\d+\.?\w*)', folder_name)
    if id_match:
        info["exp_id"] = id_match.group(1)

    # Определяем архитектуру
    if 'Hierarchical' in folder_name:
        info["arch_type"] = "Hierarchical"

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
            
    # Если это иерархическая модель и мы не нашли семейство явно (напр. HierarchicalCNN)
    if not found_model and info["arch_type"] == "Hierarchical":
        # Вытаскиваем то, что идет после слова Hierarchical
        match = re.search(r'Hierarchical([a-zA-Z0-9]+)', folder_name)
        if match:
            info["model_family"] = match.group(1)

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

    def plot_group_curves(self, exp_id: str, group_df: pd.DataFrame, histories: Dict[str, List[Dict]]):
        """Рисует сравнение всех моделей внутри одного ExpID."""
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

        ax1.set_title(self.t['loss_title'].format(exp_id))
        ax1.set_xlabel(self.t['epoch'])
        ax1.set_ylabel(self.t['loss'])
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title(self.t['f1_title'].format(exp_id))
        ax2.set_xlabel(self.t['epoch'])
        ax2.set_ylabel(self.t['f1'])
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        save_path = self.output_root / "groups" / f"exp_{exp_id}_comparison_{self.lang}.png"
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

def aggregate_reports(root_dir: str, output_dir: str = None, plot: bool = False, benchmark: bool = False, full_eval: bool = False, data_dir: str = None, lang: str = 'ru'):
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
        # Очищаем лог ошибок при новом запуске
        if error_log_path.exists():
            error_log_path.unlink()
    else:
        out_path = None
        figures_path = root_path / "figures"
        csv_path = None
        history_path = None
        error_log_path = None

    # Путь к датасетам по умолчанию
    if data_dir is None:
        data_dir = str(ROOT_DIR_PROJECT / 'data' / 'ml_datasets')

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

            # Генерация графиков
            if plot:
                plot_path = exp_dir / "learning_curves.png"
                plot_learning_curves(metrics, plot_path)

            # Найти лучшую эпоху (на основе min val_loss, с защитой от пустых данных)
            if any('val_loss' in m for m in metrics):
                best_epoch = min(metrics, key=lambda x: x.get('val_loss', 999.0))
            else:
                best_epoch = metrics[-1]

            # Извлечение информации
            # Сначала пробуем взять из нового поля model_info в конфиге
            num_params = config.get('model_info', {}).get('num_params', best_epoch.get('num_params', 0))
            
            cpu_inf = best_epoch.get('cpu_inf_time_ms', 0.0)
            if benchmark:
                cpu_inf = benchmark_model_cpu(exp_dir, config, iterations=1000)

            exp_data = {
                "ExpID": info["exp_id"],
                "Experiment": exp_dir.name,
                "Model": info["model_family"],
                "Complexity": info["complexity"],
                "Features": info["feature_mode"],
                "Sampling": info["sampling"],
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
                full_metrics = evaluate_full_test_dataset(exp_dir, config, data_dir)
                exp_data['Full Best Acc'] = full_metrics['full_best_acc']
                exp_data['Full Best F1'] = full_metrics['full_best_f1']
                exp_data['Full Final Acc'] = full_metrics['full_final_acc']
                exp_data['Full Final F1'] = full_metrics['full_final_f1']
                exp_data['Full Eval Time (s)'] = full_metrics['full_eval_time_s']
                exp_data['Full Eval Samples'] = full_metrics['full_eval_samples']
            
            experiments.append(exp_data)
            
        except Exception as e:
            error_msg = f"Ошибка при обработке опыта {exp_dir.name}: {str(e)}"
            print(f"\n[!] {error_msg}")
            if error_log_path:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"=== {exp_dir.name} ===\n")
                    f.write(f"Error: {str(e)}\n")
                    import traceback
                    f.write(traceback.format_exc())
                    f.write("-" * 50 + "\n\n")
            continue
    
    if not experiments:
        print("Эксперименты не найдены.")
        return

    # Создать DataFrame и заполнить пропуски
    df = pd.DataFrame(experiments).fillna(0)
    
    # --- ВИЗУАЛИЗАЦИЯ (Report Engine 2.0) ---
    if plot:
        viz = ReportVisualizer(figures_path, lang=lang)
        
        # 1. Групповые графики по ExpID
        unique_exps = df['ExpID'].unique()
        for eid in tqdm(unique_exps, desc=viz.t['groups_desc']):
            if eid == "Unknown": continue
            # Находим все папки, относящиеся к этому опыту
            exp_folders = df[df['ExpID'] == eid]['Path'].tolist()
            group_histories = {f: all_histories[f] for f in exp_folders if f in all_histories}
            viz.plot_group_curves(eid, df, group_histories)
            
        # 2. Паррето график скорость/точность
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

if __name__ == "__main__":
    # Сначала проверяем CLI, если нет аргументов - MANUAL_RUN
    parser = argparse.ArgumentParser(description="Агрегация отчетов экспериментов.")
    parser.add_argument("--root", type=str, default=None, help="Корневая директория экспериментов")
    parser.add_argument("--out-dir", type=str, default=None, help="Путь к папке для сохранения всех отчетов")
    parser.add_argument("--plot", action="store_true", help="Генерировать графики обучения")
    parser.add_argument("--benchmark", action="store_true", help="Выполнить глубокий бенчмарк на CPU")
    parser.add_argument("--full-eval", action="store_true", help="Выполнить полную оценку на тестовом датасете (GPU)")
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
        ROOT_DIR = "experiments/phase2_5" # пока сюда скопировал часть 2.6
        # OUTPUT_DIR: Папка, куда сохранять ВСЕ отчёты / файлы / картинки
        OUTPUT_DIR = "reports/Exp_2_5_and_start_Exp_2_6"
        # GENERATE_PLOTS: Если True, для каждого эксперимента будут построены графики обучения.
        GENERATE_PLOTS = True
        # RUN_BENCHMARK: Если True, будет выполнен глубокий замер скорости инференса на CPU (1000 итераций).
        RUN_BENCHMARK = True
        # RUN_FULL_EVAL: Если True, будет выполнена полная оценка на всём тестовом датасете (GPU).
        # Это более точная оценка, чем Val Acc/F1 из обучения (там используется подвыборка).
        RUN_FULL_EVAL = True
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
        LANG = args.lang

    if ROOT_DIR:
        aggregate_reports(
            ROOT_DIR, 
            OUTPUT_DIR, 
            GENERATE_PLOTS, 
            RUN_BENCHMARK,
            full_eval=RUN_FULL_EVAL,
            lang=LANG,
            data_dir=args.data_dir
        )
