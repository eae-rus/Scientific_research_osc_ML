"""
Парсинг информации из имён экспериментов и извлечение ID.

Функции для разбора структурированных имён папок экспериментов.
"""
import re
from typing import Dict, List, Any


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
        "target_level": "base",
        "is_aug": "No",
        "balancing": "None",
        "arch_type": "Base"
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
    if 'base_sequential' in folder_name.lower():
        info["target_level"] = "base_sequential"
    elif 'ozz' in folder_name.lower() or '2.6.11' in folder_name:
        info["target_level"] = "ozz"
    elif 'full_by_levels' in folder_name or ('hier_' in folder_name.lower() and '2.6.4' in folder_name):
        info["target_level"] = "full_by_levels"
    elif 'full' in folder_name.lower() and '2.6.4' in folder_name:
        info["target_level"] = "full"
    else:
        info["target_level"] = "base"

    # Поиск ключевого семейства моделей
    models_map = {
        'SimpleMLP': 'MLP',
        'SimpleCNN': 'CNN',
        'cPhysicsKAN': 'cPhysicsKAN',
        'ConvKAN': 'ConvKAN',
        'SimpleKAN': 'SimpleKAN',
        'PhysicsKAN': 'PhysicsKAN',
        'PhysicsBaseline': 'PhysicsBaseline',
        'ResNet1D': 'ResNet',
    }
    
    found_model = False
    
    # Сначала проверяем cPhysicsKAN, так как PhysicsKAN является подстрокой
    if 'cPhysicsKAN' in folder_name:
        info["model_family"] = "cPhysicsKAN"
        found_model = True
    
    if not found_model:
        for pattern, clean_name in models_map.items():
            if pattern in folder_name:
                info["model_family"] = clean_name
                found_model = True
                break
            
    # Если это иерархическая или гибридная модель и мы не нашли семейство явно
    if not found_model and info["arch_type"] in ("Hierarchical", "Hybrid"):
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


def _coerce_per_class_f1_values(raw_value: Any) -> List[float]:
    """Нормализует val_f1_per_class из history/metrics в список float."""
    values: List[float] = []
    if isinstance(raw_value, dict):
        for _, val in raw_value.items():
            if isinstance(val, list):
                cur = val[-1] if val else 0.0
            else:
                cur = val
            try:
                values.append(float(cur))
            except Exception:
                values.append(0.0)
        return values

    if isinstance(raw_value, list):
        for val in raw_value:
            try:
                values.append(float(val))
            except Exception:
                values.append(0.0)
    return values
