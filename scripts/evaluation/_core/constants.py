"""
Константы инженерной визуализации и утилитарные словари.

Вынесено из aggregate_reports.py для уменьшения размера головного модуля.
"""
from typing import Dict, List, Optional
import numpy as np

# =============================================================================
# КОНСТАНТЫ ИНЖЕНЕРНОЙ ВИЗУАЛИЗАЦИИ
# =============================================================================

ENGINEERING_CLASS_MAP: Dict[int, str] = {
    0: 'Норма',
    1: 'Коммутация',
    2: 'Аномалия',
    3: 'Авария'
}

ENGINEERING_CLASS_MAP_OZZ: Dict[int, str] = {
    0: 'ОЗЗ (обнаружение)',
    1: 'Затухающее ОЗЗ',
    2: 'ДПОЗЗ'
}

# Столбцы multilabel CSV для OZZ (3 GT + 3 pred)
OZZ_MULTILABEL_TRUE_COLS = ['y_true_ozz', 'y_true_decay', 'y_true_dpozz']
OZZ_MULTILABEL_PRED_COLS = ['y_pred_ozz', 'y_pred_decay', 'y_pred_dpozz']
OZZ_MULTILABEL_ALL_COLS = OZZ_MULTILABEL_TRUE_COLS + OZZ_MULTILABEL_PRED_COLS


def get_engineering_class_map(target_level: str, y_values: Optional[np.ndarray] = None) -> Dict[int, str]:
    """Выбирает карту классов для инженерных графиков по target_level."""
    level = str(target_level or 'base').lower()
    if level == 'ozz':
        return ENGINEERING_CLASS_MAP_OZZ
    return ENGINEERING_CLASS_MAP


# Тонкая настройка: любые отдельные графики можно отключать через этот словарь.
PLOT_SWITCHES_DEFAULT: Dict[str, bool] = {
    'engineering_bars_per_model_absolute': True,
    'engineering_bars_per_model_relative': True,
    'engineering_bars_combined_absolute': True,
    'engineering_bars_combined_relative': True,
    'custom_cm_per_model_absolute': True,
    'custom_cm_per_model_relative': True
}

ENGINEERING_PLOTS_SUBDIR_DEFAULT = 'engineering_plots'
MAX_MODELS_FOR_COMBINED_DEFAULT = 10

# Частые шаблоны названий файлов с предсказаниями.
PREDICTION_FILE_PATTERNS: Dict[str, List[str]] = {
    'best': [
        'test_predictions_best.csv',
        'predictions_best.csv',
        'best_predictions.csv',
        'best_test_predictions.csv'
    ],
    'final': [
        'test_predictions_final.csv',
        'predictions_final.csv',
        'final_predictions.csv',
        'final_test_predictions.csv'
    ]
}
