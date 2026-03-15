"""
Загрузка, нормализация и выбор предсказаний (Best/Final).

Multilabel OZZ конвертация, выбор лучших весов по F1, вспомогательные функции.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from scripts.evaluation._core.constants import (
    OZZ_MULTILABEL_TRUE_COLS,
    OZZ_MULTILABEL_PRED_COLS,
    OZZ_MULTILABEL_ALL_COLS,
    PREDICTION_FILE_PATTERNS,
    get_prediction_file_patterns,
)
from scripts.evaluation._core.model_utils import get_eval_logger


def _is_multilabel_pred_df(df: pd.DataFrame) -> bool:
    """Проверяет, содержит ли pred_df multilabel OZZ столбцы."""
    return 'y_true_ozz' in df.columns


def _ensure_multilabel_ozz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Конвертирует OZZ single-class предсказания (-1/0/1/2) в multilabel формат.

    Маппинг (иерархия ОЗЗ):
      -1 → [0, 0, 0]  (нет ОЗЗ)
       0 → [1, 0, 0]  (устойчивое ОЗЗ)
       1 → [1, 1, 0]  (затухающее ОЗЗ, является ОЗЗ)
       2 → [1, 0, 1]  (ДПОЗЗ, является ОЗЗ)

    Если уже multilabel — возвращает как есть.
    """
    if _is_multilabel_pred_df(df):
        return df

    result = pd.DataFrame(index=df.index)
    for prefix, src_col in [('y_true', 'y_true'), ('y_pred', 'y_pred')]:
        vals = df[src_col].to_numpy(dtype=int)
        result[f'{prefix}_ozz'] = (vals >= 0).astype(int)
        result[f'{prefix}_decay'] = (vals == 1).astype(int)
        result[f'{prefix}_dpozz'] = (vals == 2).astype(int)

    return result


def _normalize_prediction_columns(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит таблицу предсказаний к стандартному формату.
    Для OZZ multilabel (6 столбцов) — возвращает как есть.
    Для OZZ single-class (-1/0/1/2) — автоматически конвертирует в multilabel.
    Для single-class base — нормализует к y_true / y_pred.
    """
    # Multilabel OZZ формат (6 столбцов)
    if 'y_true_ozz' in df_pred.columns:
        result = df_pred[OZZ_MULTILABEL_ALL_COLS].copy()
        for col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0).astype(int)
        return result

    # Single-class формат
    candidate_true = ['y_true', 'true', 'target', 'label_true', 'gt']
    candidate_pred = ['y_pred', 'pred', 'prediction', 'label_pred']

    result = df_pred.copy()
    true_col = next((c for c in candidate_true if c in result.columns), None)
    pred_col = next((c for c in candidate_pred if c in result.columns), None)

    if true_col is None or pred_col is None:
        raise ValueError("В таблице предсказаний не найдены колонки y_true/y_pred")

    if true_col != 'y_true':
        result = result.rename(columns={true_col: 'y_true'})
    if pred_col != 'y_pred':
        result = result.rename(columns={pred_col: 'y_pred'})

    result = result.dropna(subset=['y_true', 'y_pred']).copy()
    result['y_true'] = pd.to_numeric(result['y_true'], errors='coerce')
    result['y_pred'] = pd.to_numeric(result['y_pred'], errors='coerce')
    result = result.dropna(subset=['y_true', 'y_pred']).copy()
    result['y_true'] = result['y_true'].astype(int)
    result['y_pred'] = result['y_pred'].astype(int)

    # Авто-конвертация OZZ single-class → multilabel.
    if result['y_true'].min() == -1:
        return _ensure_multilabel_ozz(result)

    return result


def _load_prediction_file_safe(file_path: Path) -> Optional[pd.DataFrame]:
    """Безопасная загрузка CSV с предсказаниями."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        return _normalize_prediction_columns(df)
    except Exception as exc:
        _eval_logger = get_eval_logger()
        if _eval_logger:
            _eval_logger.warning(f"Не удалось прочитать {file_path}: {exc}")
        return None


def load_best_final_predictions(exp_dir: Path, eval_split: str = 'test') -> Dict[str, Optional[pd.DataFrame]]:
    """Ищет файлы предсказаний Best/Final в каталоге эксперимента для заданного split."""
    loaded: Dict[str, Optional[pd.DataFrame]] = {'best': None, 'final': None}
    file_patterns = get_prediction_file_patterns(eval_split)
    for version in ('best', 'final'):
        # Сначала split-специфичные шаблоны, затем исторические fallback-имена.
        for pattern in file_patterns.get(version, PREDICTION_FILE_PATTERNS[version]):
            candidate = exp_dir / pattern
            df_pred = _load_prediction_file_safe(candidate)
            if df_pred is not None:
                loaded[version] = df_pred
                break
    return loaded


def _compute_f1_for_pred_df(df: pd.DataFrame) -> float:
    """Универсальный расчёт macro F1 для pred_df (single-class или multilabel)."""
    if _is_multilabel_pred_df(df):
        f1_vals = []
        for tc, pc in zip(OZZ_MULTILABEL_TRUE_COLS, OZZ_MULTILABEL_PRED_COLS):
            f1_vals.append(f1_score(df[tc], df[pc], zero_division=0))
        return float(np.mean(f1_vals))
    return float(f1_score(df['y_true'], df['y_pred'], average='macro', zero_division=0))


def select_best_final_by_test_f1(
    df_best: Optional[pd.DataFrame],
    df_final: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """Выбирает версию весов по максимальному F1-macro на тесте."""
    f1_best = np.nan
    f1_final = np.nan

    if df_best is not None and not df_best.empty:
        f1_best = _compute_f1_for_pred_df(df_best)
    if df_final is not None and not df_final.empty:
        f1_final = _compute_f1_for_pred_df(df_final)

    if np.isnan(f1_best) and np.isnan(f1_final):
        return {
            'selected_version': 'N/A',
            'selected_f1': np.nan,
            'best_f1': np.nan,
            'final_f1': np.nan,
            'selected_df': None
        }

    if np.isnan(f1_final) or (not np.isnan(f1_best) and f1_best >= f1_final):
        return {
            'selected_version': 'Best',
            'selected_f1': float(f1_best),
            'best_f1': float(f1_best),
            'final_f1': float(f1_final) if not np.isnan(f1_final) else np.nan,
            'selected_df': df_best
        }

    return {
        'selected_version': 'Final',
        'selected_f1': float(f1_final),
        'best_f1': float(f1_best) if not np.isnan(f1_best) else np.nan,
        'final_f1': float(f1_final),
        'selected_df': df_final
    }


def add_best_final_selection_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет в сводную таблицу колонки выбора версии весов (Best/Final)."""
    df = summary_df.copy()
    if 'Full Best F1' not in df.columns or 'Full Final F1' not in df.columns:
        return df

    best_vals = pd.to_numeric(df['Full Best F1'], errors='coerce').fillna(-1.0)
    final_vals = pd.to_numeric(df['Full Final F1'], errors='coerce').fillna(-1.0)

    df['Selected Weights'] = np.where(best_vals >= final_vals, 'Best', 'Final')
    df['Selected Test F1'] = np.where(best_vals >= final_vals, best_vals, final_vals)
    return df


def _save_predictions_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    target_level: str = 'base',
    target_cols: Optional[List[str]] = None
) -> None:
    """
    Сохраняет предсказания в CSV.
    Для OZZ multilabel сохраняет 6 столбцов (3 GT + 3 pred) без потери информации.
    """
    _eval_logger = get_eval_logger()
    try:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if str(target_level).lower() == 'ozz' and y_true_arr.ndim == 2 and y_true_arr.shape[1] >= 3:
            df_out = pd.DataFrame({
                'y_true_ozz': y_true_arr[:, 0].astype(int),
                'y_true_decay': y_true_arr[:, 1].astype(int),
                'y_true_dpozz': y_true_arr[:, 2].astype(int),
                'y_pred_ozz': y_pred_arr[:, 0].astype(int),
                'y_pred_decay': y_pred_arr[:, 1].astype(int),
                'y_pred_dpozz': y_pred_arr[:, 2].astype(int),
            })
        else:
            if y_true_arr.ndim == 2:
                y_true_cls = y_true_arr.argmax(axis=1)
                y_pred_cls = y_pred_arr.argmax(axis=1)
            else:
                y_true_cls = y_true_arr.ravel()
                y_pred_cls = y_pred_arr.ravel()
            df_out = pd.DataFrame({'y_true': y_true_cls, 'y_pred': y_pred_cls})

        df_out.to_csv(save_path, index=False)
    except Exception as exc:
        if _eval_logger:
            _eval_logger.warning(f"Не удалось сохранить предсказания в {save_path}: {exc}")
