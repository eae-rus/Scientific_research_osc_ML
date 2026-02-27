"""
Генератор реестра моделей для тестирования.

Сканирует папки experiments/phase2_5 и experiments/phase2_6,
выбирает репрезентативные модели разных архитектур и сложностей,
и создаёт JSON реестр для быстрого доступа в тестах.

Стратегия выбора:
- По одной лучшей модели для каждой архитектуры (SimpleMLP, SimpleCNN, etc.)
- Приоритет: phase2_5 базовые эксперименты > phase2_6 специализированные
- Быстрые модели (light/medium) > тяжёлые для скорости тестирования
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """
    Парсит имя эксперимента для извлечения метаданных.
    
    Форматы:
    - Phase 2.5: Exp_2.5.X.Y_ModelName_Complexity_Features_Sampling_BaseWeights_Aug
    - Phase 2.6: Exp2.6.X_ModelName_Complexity_Features_Sampling/other
    """
    result = {
        'phase': None,
        'model': None,
        'complexity': None,
        'features': None,
        'sampling': None,
    }
    
    # Phase detection
    if 'Exp_2.5' in exp_name:
        result['phase'] = '2.5'
    elif 'Exp2.6' in exp_name or 'Exp_2.6' in exp_name:
        result['phase'] = '2.6'
    else:
        return result
    
    # Model name extraction
    # Модели: SimpleMLP, SimpleCNN, SimpleKAN, ConvKAN, PhysicsKAN, ResNet1D, Hierarchical*
    model_pattern = r'(SimpleMLP|SimpleCNN|SimpleKAN|ConvKAN|PhysicsKAN|ResNet1D|HierarchicalMLP|HierarchicalCNN|HierarchicalConvKAN|HierarchicalPhysicsKAN|HierarchicalSimpleKAN|HierarchicalResNet)'
    match = re.search(model_pattern, exp_name)
    if match:
        result['model'] = match.group(1)
    
    # Complexity extraction
    complexity_pattern = r'(light|medium|heavy)'
    match = re.search(complexity_pattern, exp_name)
    if match:
        result['complexity'] = match.group(1)
    
    # Features extraction
    features_pattern = r'(raw|symmetric|polar|phase|power|alpha_beta|complex)'
    match = re.search(features_pattern, exp_name)
    if match:
        result['features'] = match.group(1)
    
    # Sampling extraction
    sampling_pattern = r'(stride|snapshot)'
    match = re.search(sampling_pattern, exp_name)
    if match:
        result['sampling'] = match.group(1)
    
    return result


def find_best_models(
    phase2_5_dir: Path,
    phase2_6_dir: Path,
    max_per_architecture: int = 3
) -> Dict[str, List[Dict]]:
    """
    Находит репрезентативные модели по каждой архитектуре.
    
    Приоритет отбора:
    1. phase2_5 перед phase2_6 (более базовые)
    2. light/medium перед heavy (для скорости)
    3. raw features перед другими (проще)
    4. stride перед snapshot (проще)
    
    Возвращает список моделей по архитектуре (несколько комбинаций входов).
    """
    
    models_by_arch: Dict[str, List[Dict]] = {}
    
    # Сканируем phase2_5 (рекурсивно — эксперименты вложены в подпапки)
    for exp_dir in sorted(phase2_5_dir.glob('**/Exp_*')):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        metadata = parse_experiment_name(exp_name)
        
        if not metadata['model']:
            continue
        
        model_arch = metadata['model']
        
        # Пропускаем hierarchical модели из 2.5
        if 'Hierarchical' in model_arch:
            continue
        
        config_path = exp_dir / 'config.json'
        if not config_path.exists():
            continue
        
        # Проверяем что модель есть
        if not (exp_dir / 'best_model.pt').exists():
            continue
        
        # Приоритет: light/medium > heavy
        complexity_priority = {
            'light': 0,
            'medium': 1,
            'heavy': 2,
        }
        priority = complexity_priority.get(metadata['complexity'], 999)
        phase_priority = 0  # phase2_5
        
        if model_arch not in models_by_arch:
            models_by_arch[model_arch] = []
        
        models_by_arch[model_arch].append({
            'path': exp_dir,
            'metadata': metadata,
            'priority': priority,
            'phase_priority': phase_priority,
        })
    
    # Сканируем phase2_6 (рекурсивно — эксперименты вложены в подпапки)
    for exp_dir in sorted(phase2_6_dir.glob('**/Exp*')):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        metadata = parse_experiment_name(exp_name)
        
        if not metadata['model']:
            continue
        
        model_arch = metadata['model']
        
        config_path = exp_dir / 'config.json'
        if not config_path.exists():
            continue
        
        # Проверяем что модель есть
        if not (exp_dir / 'best_model.pt').exists():
            continue
        
        complexity_priority = {
            'light': 0,
            'medium': 1,
            'heavy': 2,
        }
        priority = complexity_priority.get(metadata['complexity'], 999)
        phase_priority = 1  # phase2_6
        
        if model_arch not in models_by_arch:
            models_by_arch[model_arch] = []
        
        models_by_arch[model_arch].append({
            'path': exp_dir,
            'metadata': metadata,
            'priority': priority,
            'phase_priority': phase_priority,
        })
    
    # Выбираем лучшие по приоритету
    best_models: Dict[str, List[Dict]] = {}
    for arch, candidates in models_by_arch.items():
        # Сортируем: сначала phase2_5, потом по сложности
        candidates.sort(key=lambda x: (x['phase_priority'], x['priority']))
        
        selected = []
        used_combos = set()
        
        for candidate in candidates:
            metadata = candidate['metadata']
            combo_key = (metadata.get('features'), metadata.get('sampling'))
            if combo_key in used_combos:
                continue
            used_combos.add(combo_key)
            selected.append(candidate)
            if len(selected) >= max_per_architecture:
                break
        
        best_models[arch] = selected
    
    return best_models


def load_model_config(config_path: Path) -> Dict:
    """Загружает конфиг модели."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_registry(
    phase2_5_dir: Path,
    phase2_6_dir: Path,
    output_path: Path
) -> None:
    """
    Генерирует JSON реестр моделей для тестирования.
    """
    
    best_models = find_best_models(phase2_5_dir, phase2_6_dir)
    
    registry = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'models': {}
    }
    
    for arch, candidates in best_models.items():
        for candidate in candidates:
            model_dir = candidate['path']
            metadata = candidate['metadata']
            config_path = model_dir / 'config.json'
            
            try:
                config = load_model_config(config_path)
            except Exception as e:
                print(f"⚠️  Ошибка при загрузке конфига для {arch}: {e}")
                continue
            
            entry_key = f"{arch}__{model_dir.name}"
            registry['models'][entry_key] = {
                'model_name': arch,
                'experiment_dir': str(model_dir),
                'model_path': str(model_dir / 'best_model.pt'),
                'config_path': str(config_path),
                'metadata': metadata,
                'model_config': config.get('model', {}),
                'data_config': config.get('data', {}),
            }
            
            print(f"✅ {entry_key}")
    
    # Сохраняем реестр
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Реестр сохранён в {output_path}")
    print(f"📊 Всего моделей: {len(registry['models'])}")


if __name__ == '__main__':
    ROOT_DIR = Path(__file__).parent.parent
    phase2_5_dir = ROOT_DIR / 'experiments' / 'phase2_5'
    phase2_6_dir = ROOT_DIR / 'experiments' / 'phase2_6'
    output_path = ROOT_DIR / 'tests' / 'fixtures' / 'model_registry.json'
    
    if not phase2_5_dir.exists():
        print(f"❌ Директория phase2_5 не найдена: {phase2_5_dir}")
        exit(1)
    
    if not phase2_6_dir.exists():
        print(f"❌ Директория phase2_6 не найдена: {phase2_6_dir}")
        exit(1)
    
    generate_registry(phase2_5_dir, phase2_6_dir, output_path)
