"""
Кэширование, загрузка данных и утилиты работы с отчётами.

Функции загрузки метрик, конфигов и кэша summary_report.
"""
import json
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


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


def needs_full_eval_recalc(
    existing_row: Optional[pd.Series],
    require_hierarchical: bool = False,
    eval_split: str = 'test'
) -> bool:
    """
    Проверяет, нужен ли пересчёт full_eval для данной модели.
    
    Возвращает True если:
    - Нет существующих данных (existing_row is None)
    - Full Best Acc/F1 или Full Final Acc/F1 равны 0
    - Full Eval Time (s) или Full Eval Samples равны 0
    - Full Best ROC-AUC отсутствует или равен 0 (новая метрика)
    - (Опционально) отсутствуют метрики Hierarchical Accuracy
    """
    if existing_row is None:
        return True

    # Если ранее метрики считались на другом split, нужен полный пересчёт.
    row_split = str(existing_row.get('Full Eval Split', 'test')).strip().lower()
    req_split = str(eval_split or 'test').strip().lower()
    if row_split != req_split:
        return True
    
    # Проверяем ключевые метрики на 0
    full_eval_cols = [
        'Full Best Acc', 'Full Best F1', 'Full Final Acc', 'Full Final F1',
        'Full Eval Time (s)', 'Full Eval Samples'
    ]
    
    for col in full_eval_cols:
        if col in existing_row.index:
            val = existing_row[col]
            if pd.isna(val) or val == 0 or val == 0.0:
                return True
    
    # ROC-AUC: если колонки нет или значение 0/NaN — нужен пересчёт
    if 'Full Best ROC-AUC' not in existing_row.index:
        return True
    roc_val = existing_row.get('Full Best ROC-AUC', 0)
    if pd.isna(roc_val) or roc_val == 0 or roc_val == 0.0:
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


def load_history_as_metrics(history_path: Path) -> List[Dict[str, Any]]:
    """Преобразует history.json в формат списка эпох (как metrics.jsonl)."""
    if not history_path.exists():
        return []
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        def _last_float(key: str, default: float = 0.0) -> float:
            value = history.get(key, default)
            if isinstance(value, list):
                value = value[-1] if value else default
            try:
                return float(value)
            except Exception:
                return float(default)

        return [{
            'epoch': 1,
            'train_loss': _last_float('train_loss', 0.0),
            'val_loss': _last_float('val_loss', 0.0),
            'val_f1': _last_float('val_f1_macro', 0.0),
            'val_acc': _last_float('val_precision_macro', 0.0),
            'val_f1_per_class': history.get('val_f1_per_class', {}),
            'cpu_inf_time_ms': 0.0,
            'num_params': 0,
        }]
    except Exception as e:
        print(f"Ошибка чтения {history_path}: {e}", file=sys.stderr)
        return []


def load_config(config_path: Path) -> Dict[str, Any]:
    """Загрузка конфигурации из файла json."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {config_path}: {e}", file=sys.stderr)
        return {}
