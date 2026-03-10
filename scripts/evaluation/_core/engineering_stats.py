"""
Инженерная статистика: расчёт TP/FP/FN, выбор предсказаний по моделям.

Функции build_engineering_stats, choose_selected_predictions, collapse_selected_to_model_level.
"""
import re
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.evaluation._core.constants import (
    OZZ_MULTILABEL_TRUE_COLS,
    OZZ_MULTILABEL_PRED_COLS,
)
from scripts.evaluation._core.predictions import (
    _is_multilabel_pred_df,
    select_best_final_by_test_f1,
)


def build_engineering_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_map: Dict[int, str]
) -> pd.DataFrame:
    """
    Считает инженерные метрики по классам: TP, FP, FN, ошибки и размер GT.
    Используется для single-class (base) экспериментов.
    """
    rows: List[Dict[str, Any]] = []
    for class_id, class_name in class_map.items():
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)

        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        gt = int(np.sum(true_mask))
        err = fp + fn

        rows.append({
            'class_id': class_id,
            'class_name': class_name,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'errors': err,
            'gt': gt
        })

    return pd.DataFrame(rows)


def build_engineering_stats_multilabel(
    pred_df: pd.DataFrame,
    class_map: Dict[int, str]
) -> pd.DataFrame:
    """
    Считает инженерные метрики для OZZ multilabel: каждый таргет
    оценивается как независимый бинарный классификатор.
    """
    rows: List[Dict[str, Any]] = []
    for class_id, class_name in class_map.items():
        tc = OZZ_MULTILABEL_TRUE_COLS[class_id]
        pc = OZZ_MULTILABEL_PRED_COLS[class_id]
        gt_col = pred_df[tc].to_numpy()
        pred_col = pred_df[pc].to_numpy()

        tp = int(np.sum((gt_col > 0) & (pred_col > 0)))
        fp = int(np.sum((gt_col == 0) & (pred_col > 0)))
        fn = int(np.sum((gt_col > 0) & (pred_col == 0)))
        gt = int(np.sum(gt_col > 0))

        rows.append({
            'class_id': class_id,
            'class_name': class_name,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'errors': fp + fn,
            'gt': gt
        })

    return pd.DataFrame(rows)


def choose_selected_predictions_per_experiment(
    exp_predictions: Dict[str, Dict[str, Optional[pd.DataFrame]]]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Для каждого эксперимента выбирает Best/Final по максимальному F1-macro."""
    selection_rows: List[Dict[str, Any]] = []
    selected: Dict[str, pd.DataFrame] = {}

    for exp_name, versions in exp_predictions.items():
        picked = select_best_final_by_test_f1(versions.get('best'), versions.get('final'))
        selection_rows.append({
            'Experiment': exp_name,
            'Best Test F1': picked['best_f1'],
            'Final Test F1': picked['final_f1'],
            'Selected Weights': picked['selected_version'],
            'Selected Test F1': picked['selected_f1']
        })
        if picked['selected_df'] is not None:
            selected[exp_name] = picked['selected_df']

    return pd.DataFrame(selection_rows), selected


def collapse_selected_to_model_level(
    selected_by_experiment: Dict[str, pd.DataFrame],
    selection_df: pd.DataFrame,
    summary_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Переход от уровня эксперимента к уровню типа модели.
    Для каждой модели берётся лучший (по Selected Test F1) эксперимент.
    """
    if selection_df.empty or not selected_by_experiment:
        return {}

    merged = selection_df.merge(
        summary_df[['Experiment', 'Model']].drop_duplicates(),
        how='left',
        on='Experiment'
    )
    merged = merged.sort_values('Selected Test F1', ascending=False)

    selected_model_level: Dict[str, Dict[str, Any]] = {}
    for _, row in merged.iterrows():
        model_name = str(row.get('Model', 'Unknown'))
        exp_name = str(row['Experiment'])
        if model_name in selected_model_level:
            continue
        if exp_name in selected_by_experiment:
            target_level = 'base'
            matched = summary_df[summary_df['Experiment'] == exp_name]
            if not matched.empty and 'TargetLevel' in matched.columns:
                target_level = str(matched.iloc[0].get('TargetLevel', 'base'))
            selected_model_level[model_name] = {
                'pred_df': selected_by_experiment[exp_name],
                'target_level': target_level,
                'experiment': exp_name,
            }
    return selected_model_level
