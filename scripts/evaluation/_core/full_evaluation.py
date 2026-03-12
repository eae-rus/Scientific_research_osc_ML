"""
Полная оценка моделей на тестовом датасете.

Содержит evaluate_model_full, evaluate_full_test_dataset, кэш данных,
batch fallback и иерархическую оценку.
"""
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

from osc_tools.data_management.dataset_manager import DatasetManager
from osc_tools.ml.precomputed_dataset import PrecomputedDataset
from osc_tools.ml.labels import get_target_columns, prepare_labels_for_experiment

from scripts.evaluation._core.model_utils import (
    _create_model_from_config,
    _load_state_dict_safe,
    _get_eval_batch_size,
    get_eval_logger,
)
from scripts.evaluation._core.predictions import _save_predictions_csv

# Добавляем корень проекта в путь импорта
ROOT_DIR_PROJECT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR_PROJECT))

# Глобальный кэш для данных тестового датасета (загружается один раз)
_test_data_cache: Dict[str, Any] = {}


def _get_test_data_cached(
    data_dir: str, 
    window_size: int = 320,
    eval_stride: int = 1,
    norm_coef_path: Optional[str] = None,
    target_level: str = 'base',
    eval_split: str = 'test'
) -> Tuple[Any, List[int], List[str]]:
    """
    Возвращает кэшированные данные датасета для выбранного split (PrecomputedDataset).
    
    Args:
        data_dir: Путь к директории с данными
        window_size: Размер окна
        eval_stride: Stride между окнами (1 = все точки, default)
        norm_coef_path: Путь к файлу нормализации
        target_level: Уровень детализации меток ('base', 'full', 'full_by_levels')
        eval_split: Сплит для оценки ('test' или 'train')
    
    Returns:
        (eval_df, indices, target_cols)
    """
    import polars as pl
    
    split = str(eval_split or 'test').strip().lower()
    if split not in ('test', 'train'):
        raise ValueError(f"Неподдерживаемый eval_split: {eval_split}. Ожидается 'test' или 'train'.")

    cache_key = f"{data_dir}_{window_size}_{eval_stride}_{norm_coef_path}_{target_level}_{split}"
    
    if cache_key not in _test_data_cache:
        dm = DatasetManager(data_dir, norm_coef_path=norm_coef_path)
        
        print(f"[Full Eval] Загрузка {split}-датасета (target_level={target_level})...")
        if split == 'test':
            dm.create_precomputed_test_csv(source_split='test')
            eval_df = dm.load_test_df(precomputed=True)
        else:
            dm.create_precomputed_test_csv(source_split='train')
            eval_df = dm.load_train_df(precomputed=True)

        eval_df = eval_df.with_row_index("row_nr")

        eval_df = prepare_labels_for_experiment(eval_df, target_level)
        target_cols = get_target_columns(target_level, eval_df)
        
        indices = PrecomputedDataset.create_indices(
            eval_df,
            window_size=window_size,
            mode='val',
            stride=eval_stride
        )
        print(f"[Full Eval] Загружено {len(indices):,} точек для split='{split}' (stride={eval_stride}, {len(target_cols)} классов)")
        
        _test_data_cache[cache_key] = (eval_df, indices, target_cols)
    
    return _test_data_cache[cache_key]


def _evaluate_with_batch_fallback(
    model: nn.Module,
    test_ds: torch.utils.data.Dataset,
    base_batch_size: int,
    device: torch.device,
    mode: str,
    exp_name: str,
    tag: str
) -> Tuple[Dict[str, float], int, torch.device]:
    """Оценка с понижением batch_size при OOM и fallback на CPU."""
    _eval_logger = get_eval_logger()
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
    """Оценка Hierarchical Accuracy с понижением batch_size при OOM и fallback на CPU."""
    from osc_tools.ml.evaluation import evaluate_model_multilevel

    _eval_logger = get_eval_logger()
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
    _eval_logger = get_eval_logger()
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    try:
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                if mode == 'classification':
                    y = y.long()
                    if y.dim() > 1 and y.shape[1] == 1:
                        y = y.squeeze(1)
                    _, predicted = torch.max(outputs.data, 1)
                    probs = torch.softmax(outputs, dim=1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                elif mode == 'multilabel':
                    y = y.float()
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                elif mode == 'multitask_conditional':
                    y = y.float()
                    probs = torch.sigmoid(outputs[:, :4])
                    pred_binary = (probs > 0.5).int().cpu().numpy()
                    pred_binary[:, 2] = np.where(pred_binary[:, 3] == 1, 0, pred_binary[:, 2])
                    all_preds.extend(pred_binary)
                    all_targets.extend(y[:, :4].cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                else:
                    raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if raise_on_oom and 'CUDA out of memory' in str(e):
            raise
        if _eval_logger:
            _eval_logger.error(f"Ошибка при оценке модели: {e}")
        return {}
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs) if all_probs else np.array([])
    
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
    
    # === ROC-AUC ===
    roc_auc = 0.0
    per_class_roc_auc = []
    
    if all_probs.size > 0:
        try:
            if mode == 'classification':
                roc_auc = roc_auc_score(
                    all_targets, all_probs,
                    average='macro', multi_class='ovr'
                )
                num_classes = all_probs.shape[1]
                for cls_idx in range(num_classes):
                    y_true_bin = (all_targets == cls_idx).astype(int)
                    if len(np.unique(y_true_bin)) < 2:
                        per_class_roc_auc.append(0.0)
                    else:
                        cls_auc = roc_auc_score(y_true_bin, all_probs[:, cls_idx])
                        per_class_roc_auc.append(float(cls_auc))
            elif mode in ('multilabel', 'multitask_conditional'):
                num_classes = all_targets.shape[1] if all_targets.ndim == 2 else 1
                valid_aucs = []
                for cls_idx in range(num_classes):
                    y_true_cls = all_targets[:, cls_idx]
                    y_prob_cls = all_probs[:, cls_idx]
                    if len(np.unique(y_true_cls)) < 2:
                        per_class_roc_auc.append(0.0)
                    else:
                        cls_auc = roc_auc_score(y_true_cls, y_prob_cls)
                        per_class_roc_auc.append(float(cls_auc))
                        valid_aucs.append(cls_auc)
                roc_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
        except Exception as e:
            if _eval_logger:
                _eval_logger.warning(f"Не удалось вычислить ROC-AUC: {e}")
            roc_auc = 0.0
            per_class_roc_auc = []
    
    return {
        'acc': acc,
        'f1': f1,
        'balanced_acc': balanced_acc,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        'roc_auc': roc_auc,
        'per_class_roc_auc': per_class_roc_auc,
        'y_true': all_targets,
        'y_pred': all_preds
    }


def evaluate_full_test_dataset(
    exp_dir: Path,
    config: Dict[str, Any],
    data_dir: str,
    batch_size: int = 8192,
    use_gpu: bool = True,
    eval_split: str = 'test'
) -> Dict[str, Any]:
    """
    Полная оценка моделей (best и final) на всём датасете выбранного split.
    
    Стратегия оптимизации:
    - stride=1: перебираем ВСЕ возможные окна (полное покрытие)
    - Динамический batch_size: автоматически уменьшается для тяжелых моделей (KAN + harmonic)
    - PrecomputedDataset: данные уже предрассчитаны, извлечение быстрое
    """
    _eval_logger = get_eval_logger()

    save_predictions = True

    results = {
        'full_best_acc': 0.0,
        'full_best_f1': 0.0,
        'full_best_per_class_f1': [],
        'full_best_roc_auc': 0.0,
        'full_best_per_class_roc_auc': [],
        'full_final_acc': 0.0,
        'full_final_f1': 0.0,
        'full_final_per_class_f1': [],
        'full_final_roc_auc': 0.0,
        'full_final_per_class_roc_auc': [],
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

    try:
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        window_size = config.get('data', {}).get('window_size', 320)
        mode = config.get('data', {}).get('mode', 'multilabel')
        
        exp_name = exp_dir.name.lower()
        
        # ВАЖНО: Для экспериментов 2.6.4 слово 'full' означает target_level, а не sampling
        if 'stride' in exp_name:
            sampling_strategy = 'stride'
            downsampling_stride = 16
        elif 'snapshot' in exp_name:
            sampling_strategy = 'snapshot'
            downsampling_stride = 32
        elif 'none_sampl' in exp_name or ('none' in exp_name and 'full' not in exp_name):
            sampling_strategy = 'none'
            downsampling_stride = 1
        else:
            input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
            in_channels_cfg = config.get('model', {}).get('params', {}).get('in_channels', None)
            
            if in_channels_cfg:
                seq_len = input_size // in_channels_cfg
            elif input_size > 0:
                if input_size <= 32:
                    seq_len = 2
                elif input_size <= 64:
                    seq_len = 2 if input_size in [16, 24, 32] else input_size // 16
                else:
                    seq_len = input_size // 16
            else:
                seq_len = 18
            
            if seq_len >= window_size:
                sampling_strategy = 'none'
                downsampling_stride = 1
            elif seq_len <= 4:
                sampling_strategy = 'snapshot'
                downsampling_stride = 32
            else:
                sampling_strategy = 'stride'
                downsampling_stride = 16
        
        model_name = config.get('model', {}).get('name', '')

        in_channels = config.get('model', {}).get('params', {}).get('in_channels', None)
        feature_mode = None

        data_features = config.get('data', {}).get('features')
        if isinstance(data_features, list) and data_features:
            feature_mode = data_features[0]
        elif isinstance(data_features, str) and data_features:
            feature_mode = data_features
        
        if feature_mode is None and in_channels is None:
            if 'raw' in exp_name:
                in_channels = 8
            elif 'symmetric' in exp_name:
                in_channels = 12
            elif 'phase_polar' in exp_name or 'polar' in exp_name:
                in_channels = 16
            
            if in_channels is None:
                input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
                if input_size > 0:
                    if sampling_strategy == 'snapshot':
                        in_channels = input_size // 2
                    elif sampling_strategy == 'none':
                        in_channels = input_size // window_size
                    else:
                        if input_size % 16 == 0:
                            in_channels = 16
                        elif input_size % 12 == 0:
                            in_channels = 12
                        elif input_size % 8 == 0:
                            in_channels = 8
                        else:
                            in_channels = 16
                else:
                    in_channels = 16

        if feature_mode is None:
            if in_channels == 12:
                feature_mode = 'symmetric'
            elif in_channels == 8:
                feature_mode = 'raw'
            else:
                feature_mode = 'phase_polar'

        if model_name.startswith('Hybrid'):
            if isinstance(feature_mode, list):
                if 'raw' not in feature_mode:
                    feature_mode = ['raw'] + feature_mode
            else:
                feature_mode = ['raw', feature_mode]

        modes_list = feature_mode if isinstance(feature_mode, list) else [feature_mode]
        has_spectral = any(m != 'raw' for m in modes_list)

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
        
        num_harmonics = 1
        input_size_cfg = config.get('model', {}).get('params', {}).get('input_size', 0)
        features_channels_cfg = config.get('model', {}).get('params', {}).get('features_channels', 0)
        derived_in_channels = None
        
        if model_name.startswith('Hybrid') and features_channels_cfg > 0 and spectral_base_ch > 0:
            if features_channels_cfg % spectral_base_ch == 0:
                num_harmonics = max(1, features_channels_cfg // spectral_base_ch)
        elif input_size_cfg > 0 and (spectral_base_ch > 0 or raw_base_ch > 0):
            if sampling_strategy == 'snapshot':
                pts = 2 if has_spectral else 64
                derived_in_channels = input_size_cfg // pts
            elif sampling_strategy == 'none':
                derived_in_channels = input_size_cfg // window_size
            else:
                if not has_spectral:
                    pts = window_size // downsampling_stride
                else:
                    pts = (window_size - 32) // downsampling_stride
                pts = max(1, pts)
                derived_in_channels = input_size_cfg // pts
        
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
        
        cfg_target_level = str(config.get('data', {}).get('target_level', '')).strip().lower()
        if cfg_target_level:
            target_level = cfg_target_level
        elif 'base_sequential' in exp_name:
            target_level = 'base_sequential'
        elif 'ozz' in exp_name or '2.6.11' in exp_name:
            target_level = 'ozz'
        elif 'full_by_levels' in exp_name or ('hier_' in exp_name and '2.6.4' in exp_name):
            target_level = 'full_by_levels'
        elif '2.6.4' in exp_name and 'full' in exp_name:
            target_level = 'full'
        else:
            target_level = 'base'
        
        norm_coef_path = config.get('data', {}).get('norm_coef_path')
        test_df, all_indices, target_cols = _get_test_data_cached(
            data_dir, window_size, eval_stride=1, norm_coef_path=norm_coef_path,
            target_level=target_level, eval_split=eval_split
        )

        results['num_classes'] = len(target_cols)
        
        ds_target_level = 'base' if target_level == 'base' else target_level
        
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
        
        sample_x, _ = test_ds[0]
        actual_shape = sample_x.shape
        actual_input_size = sample_x.numel()
        expected_input_size = config.get('model', {}).get('params', {}).get('input_size', 0)
        expected_in_channels = config.get('model', {}).get('params', {}).get('in_channels', 0)
        
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
            return results
        
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
        
        model = _create_model_from_config(config)
        if model is None:
            return results
        
        model = model.to(device)

        use_hier_eval = (mode == 'multilabel' and target_level in ('full', 'full_by_levels'))
        
        # Оценка best_model.pt
        best_path = exp_dir / "best_model.pt"
        if best_path.exists():
            try:
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
                if best_metrics:
                    results['full_best_acc'] = best_metrics['acc']
                    results['full_best_f1'] = best_metrics['f1']
                    results['full_best_per_class_f1'] = best_metrics['per_class_f1']
                    results['full_best_roc_auc'] = best_metrics.get('roc_auc', 0.0)
                    results['full_best_per_class_roc_auc'] = best_metrics.get('per_class_roc_auc', [])
                    best_support = best_metrics.get('per_class_support', [])
                    if len(results['full_best_per_class_f1']) == len(target_cols):
                        results['full_best_per_class_f1_map'] = dict(zip(target_cols, results['full_best_per_class_f1']))
                        results['full_per_class_f1_count'] = len(results['full_best_per_class_f1'])
                    if best_support and len(best_support) == len(target_cols):
                        results['full_best_per_class_support_map'] = dict(zip(target_cols, best_support))
                        results['full_per_class_support_count'] = len(best_support)
                    if save_predictions and 'y_true' in best_metrics and 'y_pred' in best_metrics:
                        _save_predictions_csv(
                            best_metrics['y_true'],
                            best_metrics['y_pred'],
                            exp_dir / f'{str(eval_split).lower()}_predictions_best.csv',
                            target_level=target_level,
                            target_cols=target_cols
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
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Оценка final_model.pt
        final_path = exp_dir / "final_model.pt"
        if final_path.exists():
            try:
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
                if final_metrics:
                    results['full_final_acc'] = final_metrics['acc']
                    results['full_final_f1'] = final_metrics['f1']
                    results['full_final_per_class_f1'] = final_metrics['per_class_f1']
                    results['full_final_roc_auc'] = final_metrics.get('roc_auc', 0.0)
                    results['full_final_per_class_roc_auc'] = final_metrics.get('per_class_roc_auc', [])
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
                    if save_predictions and 'y_true' in final_metrics and 'y_pred' in final_metrics:
                        _save_predictions_csv(
                            final_metrics['y_true'],
                            final_metrics['y_pred'],
                            exp_dir / f'{str(eval_split).lower()}_predictions_final.csv',
                            target_level=target_level,
                            target_cols=target_cols
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
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results
