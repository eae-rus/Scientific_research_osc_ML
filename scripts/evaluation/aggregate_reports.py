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
    eval_stride: int = 1  # Stride=1 для полного перебора ВСЕХ точек
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
    
    Returns:
        (test_df, indices, target_cols)
    """
    import polars as pl
    
    cache_key = f"{data_dir}_{window_size}_{eval_stride}"
    
    if cache_key not in _test_data_cache:
        dm = DatasetManager(data_dir)
        target_cols = get_target_columns('base')
        
        print(f"[Full Eval] Загрузка тестового датасета (PrecomputedDataset)...")
        # Убеждаемся что предрассчитанный файл существует
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
    batch_size: int = 2048,  # Большой батч для эффективного использования GPU
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Полная оценка моделей (best и final) на всём тестовом датасете.
    
    Стратегия оптимизации:
    - stride=1: перебираем ВСЕ возможные окна (полное покрытие)
    - batch_size=2048: GPU обрабатывает большие батчи параллельно
    - PrecomputedDataset: данные уже предрассчитаны, извлечение быстрое
    
    Args:
        exp_dir: Путь к директории эксперимента
        config: Конфигурация эксперимента
        data_dir: Путь к директории с датасетами
        batch_size: Размер батча (2048 по умолчанию для GPU)
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
            
            # Snapshot дает 2 точки, stride дает ~18-20
            if seq_len <= 4:
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
            # Вычисляем из input_size
            input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
            if input_size > 0:
                if sampling_strategy == 'snapshot':
                    # input_size = in_channels * 2
                    in_channels = input_size // 2
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
        test_df, all_indices, target_cols = _get_test_data_cached(
            data_dir, window_size, eval_stride=1
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

def aggregate_reports(root_dir: str, output_file: str = None, plot: bool = False, benchmark: bool = False, full_eval: bool = False, data_dir: str = None):
    """
    Агрегирует отчеты обучения из всех поддиректорий.
    
    Args:
        root_dir: Корневая директория, содержащая папки экспериментов.
        output_file: Опциональный путь для сохранения агрегированного отчета (CSV).
        plot: Генерировать ли графики обучения для каждого эксперимента.
        benchmark: Выполнять ли глубокий бенчмарк на CPU.
        full_eval: Выполнять ли полную оценку на тестовом датасете (GPU, все точки).
        data_dir: Путь к директории с датасетами (для full_eval).
    """
    root_path = Path(root_dir)
    experiments = []
    all_histories = {}
    
    # Путь к датасетам по умолчанию
    if data_dir is None:
        data_dir = str(ROOT_DIR_PROJECT / 'data' / 'ml_datasets')

    print(f"Сканирование {root_path} на наличие экспериментов...")
    
    # Сначала собираем все эксперименты
    all_exp_dirs = []
    for metrics_file in root_path.rglob("metrics.jsonl"):
        all_exp_dirs.append(metrics_file.parent)
    
    print(f"Найдено {len(all_exp_dirs)} экспериментов")
    
    # Если full_eval включен, используем tqdm для прогресс-бара
    pbar = None
    if full_eval:
        pbar = tqdm(total=len(all_exp_dirs), desc="Полная оценка моделей", unit="exp")

    # Найти все файлы metrics.jsonl
    for exp_idx, exp_dir in enumerate(all_exp_dirs):
        metrics_file = exp_dir / "metrics.jsonl"
        config_file = exp_dir / "config.json"
        
        # Загрузка данных
        metrics = load_metrics(metrics_file)
        config = load_config(config_file) if config_file.exists() else {}
        
        if not metrics:
            continue
            
        all_histories[exp_dir.name] = metrics

        # Генерация графиков
        if plot:
            plot_path = exp_dir / "learning_curves.png"
            plot_learning_curves(metrics, plot_path)

        # Найти лучшую эпоху (на основе min val_loss)
        if any('val_loss' in m for m in metrics):
            best_epoch = min(metrics, key=lambda x: x.get('val_loss', float('inf')))
        else:
            best_epoch = metrics[-1]

        # Извлечение информации
        # Сначала пробуем взять из нового поля model_info в конфиге
        num_params = config.get('model_info', {}).get('num_params', best_epoch.get('num_params', 0))
        
        cpu_inf = best_epoch.get('cpu_inf_time_ms', 0.0)
        if benchmark:
            print(f"\nБенчмарк CPU для {exp_dir.name}...")
            cpu_inf = benchmark_model_cpu(exp_dir, config, iterations=1000)

        exp_data = {
            "Experiment": exp_dir.name,
            "Model": config.get('model', {}).get('name', 'Unknown'),
            "Params": str(config.get('model', {}).get('params', {})),
            "Epochs": len(metrics),
            "Best Epoch": best_epoch.get('epoch'),
            "Num Params": num_params,
            "Train Loss": best_epoch.get('train_loss'),
            "Val Loss": best_epoch.get('val_loss'),
            "Val Acc": best_epoch.get('val_acc'),
            "Val F1": best_epoch.get('val_f1'),
            "CPU Inf (ms)": cpu_inf,
            "Time (s)": best_epoch.get('epoch_time', best_epoch.get('time', 0.0)),
            "Path": str(exp_dir)
        }
        
        # Добавить некоторые детали конфигурации, если доступны
        if 'data' in config:
            exp_data['Window'] = config['data'].get('window_size')
            exp_data['Batch'] = config['data'].get('batch_size')
        
        # Полная оценка на тестовом датасете (GPU)
        if full_eval:
            # Обновляем прогресс-бар
            if pbar:
                pbar.set_postfix(model=exp_dir.name[:30])
                pbar.update(1)
            full_metrics = evaluate_full_test_dataset(exp_dir, config, data_dir)
            
            exp_data['Full Best Acc'] = full_metrics['full_best_acc']
            exp_data['Full Best F1'] = full_metrics['full_best_f1']
            exp_data['Full Final Acc'] = full_metrics['full_final_acc']
            exp_data['Full Final F1'] = full_metrics['full_final_f1']
            exp_data['Full Eval Time (s)'] = full_metrics['full_eval_time_s']
            exp_data['Full Eval Samples'] = full_metrics['full_eval_samples']
        
        experiments.append(exp_data)
    
    # Закрываем прогресс-бар
    if pbar:
        pbar.close()

    if not experiments:
        print("Эксперименты не найдены.")
        return

    # Создать DataFrame
    df = pd.DataFrame(experiments)
    
    # Сортировать по Val Loss (или по Full Best F1 если есть full_eval)
    if full_eval and 'Full Best F1' in df.columns:
        df = df.sort_values('Full Best F1', ascending=False)
    elif 'Val Loss' in df.columns:
        df = df.sort_values('Val Loss')

    # Отобразить
    print("\nАгрегированный отчет:")
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # Сохранить
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nОтчет сохранен в {output_file}")
        
        # Также объединяем историю метрик для удобного анализа
        history_file = Path(output_file).parent / "combined_metrics_history.txt"
        combine_training_histories(root_path, history_file)
        print(f"Объединенная история обучения сохранена в {history_file}")

    # Сравнительные графики
    if plot:
        # Сохраняем рядом с CSV или в папку root_path
        save_dir = Path(output_file).parent if output_file else root_path
        plot_comparison(all_histories, save_dir)
        print(f"Сравнительные графики сохранены в {save_dir / 'comparison_metrics.png'}")

if __name__ == "__main__":
    # === ВЕРСИЯ 1: РУЧНОЙ ЗАПУСК ЧЕРЕЗ КОНСТАНТЫ ===
    # Отредактируйте параметры ниже для быстрого запуска без аргументов командной строки.
    MANUAL_RUN = True
    
    # ROOT_DIR: Папка, где лежат результаты ваших экспериментов (metrics.jsonl, config.json).
    ROOT_DIR = "experiments/phase2_6"
    
    # OUTPUT_CSV: Имя файла для сохранения таблицы с результатами.
    OUTPUT_CSV = "reports/phase2_6_full.csv"
    
    # GENERATE_PLOTS: Если True, для каждого эксперимента будут построены графики обучения.
    GENERATE_PLOTS = False
    
    # RUN_BENCHMARK: Если True, будет выполнен глубокий замер скорости инференса на CPU (1000 итераций).
    RUN_BENCHMARK = False  # Отключаем CPU бенчмарк, если включён full_eval (нужен GPU)
    
    # RUN_FULL_EVAL: Если True, будет выполнена полная оценка на всём тестовом датасете (GPU).
    # Это более точная оценка, чем Val Acc/F1 из обучения (там используется подвыборка).
    RUN_FULL_EVAL = True

    if MANUAL_RUN:
        # Создаем папку для отчетов, если её нет
        Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        aggregate_reports(
            ROOT_DIR, 
            OUTPUT_CSV, 
            GENERATE_PLOTS, 
            RUN_BENCHMARK,
            full_eval=RUN_FULL_EVAL
        )
    else:
        # === ВЕРСИЯ 2: ЗАПУСК ПО УМОЛЧАНИЮ (CLI / Командная строка) ===
        parser = argparse.ArgumentParser(description="Агрегация отчетов экспериментов.")
        parser.add_argument("root_dir", type=str, help="Корневая директория, содержащая эксперименты (например, experiments/)")
        parser.add_argument("--output", type=str, default=None, help="Путь к выходному файлу CSV")
        parser.add_argument("--plot", action="store_true", help="Генерировать графики обучения")
        parser.add_argument("--benchmark", action="store_true", help="Выполнить глубокий бенчмарк на CPU")
        parser.add_argument("--full-eval", action="store_true", help="Выполнить полную оценку на тестовом датасете (GPU)")
        parser.add_argument("--data-dir", type=str, default=None, help="Путь к директории с датасетами")
        
        args = parser.parse_args()
        aggregate_reports(
            args.root_dir, 
            args.output, 
            args.plot, 
            args.benchmark,
            full_eval=args.full_eval,
            data_dir=args.data_dir
        )
