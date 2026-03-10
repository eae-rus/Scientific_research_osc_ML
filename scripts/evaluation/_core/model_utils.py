"""
Утилиты работы с моделями: создание, загрузка весов, бенчмарк.

Единая точка для фабрики моделей и загрузки чекпоинтов.
"""
import time
import inspect
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np


# Глобальный логгер (инициализируется в aggregate_reports)
_eval_logger: Optional[logging.Logger] = None


def set_eval_logger(logger: Optional[logging.Logger]) -> None:
    """Устанавливает глобальный логгер для модуля."""
    global _eval_logger
    _eval_logger = logger


def get_eval_logger() -> Optional[logging.Logger]:
    """Возвращает текущий глобальный логгер."""
    return _eval_logger


def _create_model_from_config(config: Dict[str, Any]) -> Optional[nn.Module]:
    """
    Создаёт экземпляр модели по конфигурации.
    
    Returns:
        Модель или None если не удалось создать
    """
    from osc_tools.ml import models as ml_models

    model_name = config.get('model', {}).get('name')
    params = config.get('model', {}).get('params', {}).copy()
    
    if not model_name:
        return None
    
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
            elif model_name in ['SimpleKAN', 'ConvKAN', 'PhysicsKAN', 'cPhysicsKAN', 'PhysicsKANConditional']:
                from osc_tools.ml.models.kan import SimpleKAN, ConvKAN, PhysicsKAN, cPhysicsKAN, PhysicsKANConditional
                models_map = {
                    'SimpleKAN': SimpleKAN,
                    'ConvKAN': ConvKAN,
                    'PhysicsKAN': PhysicsKAN,
                    'cPhysicsKAN': cPhysicsKAN,
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


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Нормализует ключи state_dict для совместимости старых чекпоинтов.
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


def _get_eval_batch_size(
    config: Dict[str, Any],
    model_name: str = None,
    feature_mode: str = None,
    num_harmonics: int = 9
) -> int:
    """
    Определяет оптимальный batch_size для оценки модели,
    учитывая архитектуру и особенности данных.
    """
    if model_name is None:
        model_name = config.get('model', {}).get('name', '')
    
    if feature_mode is None:
        data_features = config.get('data', {}).get('features', [])
        if isinstance(data_features, list) and data_features:
            feature_mode = data_features[0]
        elif isinstance(data_features, str):
            feature_mode = data_features
    
    batch_size = 8192
    val_batch_size = 8192
    
    is_harmonic_mode = feature_mode in ['phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex']
    
    if is_harmonic_mode and num_harmonics >= 3:
        batch_size = 32
        val_batch_size = 2048
    
    if is_harmonic_mode and model_name in ['PhysicsKAN', 'ConvKAN', 'HierarchicalPhysicsKAN', 'HierarchicalConvKAN']:
        batch_size = 16
        val_batch_size = 1024
    
    return val_batch_size


def benchmark_model_cpu(exp_dir: Path, config: Dict[str, Any], iterations: int = 100) -> float:
    """Замер скорости инференса на CPU для конкретной модели на пустых (dummy) данных."""
    try:
        model_name = config.get('model', {}).get('name')
        params = config.get('model', {}).get('params', {}).copy()
        
        model = _create_model_from_config(config)
        if model is None:
            return 0.0

        model = model.cpu()
        model.eval()
        
        # Загрузка весов (если есть)
        model_path = exp_dir / "best_model.pt"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                pass

        # Подстановка данных (dummy)
        use_mlp = params.get('use_mlp', False)
        if model_name in ['SimpleMLP', 'SimpleKAN'] or use_mlp:
            input_size = params.get('input_size', 2560)
            dummy_input = torch.randn(1, input_size)
            try:
                model(dummy_input)
            except:
                in_channels = params.get('in_channels', 8)
                pts = input_size // in_channels
                dummy_input = torch.randn(1, in_channels, pts)
        else:
            in_channels = params.get('in_channels', 8)
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
        print(f"Benchmark ошибка для {exp_dir.name} ({config.get('model', {}).get('name')}): {e}")
        return 0.0
