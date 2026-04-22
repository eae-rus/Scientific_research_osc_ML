"""
Фильтрация и подготовка реальных осциллограмм без ОЗЗ для 5-го класса.

Из ``data/ml_datasets/labeled_2025_12_03.csv`` отбираются осциллограммы,
у которых **все OZZ-метки == 0** (ML_2_1, ML_2_1_1, ML_2_1_2, ML_2_1_3, ML_2_2).

Эти осциллограммы формируют класс «Normal / не ОЗЗ» в объединённом
обучении вместе с 4 классами из Simulated_OZZ_v1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from osc_tools.data_management.ozz_split import OZZ_LABEL_COLS


# Все колонки, кодирующие ОЗЗ (включая ML_2_2)
ALL_OZZ_COLS = ['ML_2_1', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3', 'ML_2_2']


def get_no_ozz_files(
    df: pl.DataFrame,
    *,
    verbose: bool = True,
) -> Tuple[List[str], Dict[str, int]]:
    """Возвращает имена файлов, которые **не содержат** ни одного события ОЗЗ.

    Args:
        df: Полный DataFrame с колонками ``file_name`` и ``ML_2_*``.

    Returns:
        (no_ozz_file_names, stats)
        stats: количество файлов каждого типа ОЗЗ (для отчёта).
    """
    available = set(df.columns)
    ozz_cols_present = [c for c in ALL_OZZ_COLS if c in available]

    if not ozz_cols_present:
        raise ValueError(f"В DataFrame нет ни одной OZZ-колонки: {ALL_OZZ_COLS}")

    # Приводим к числовому типу (могут быть String в CSV)
    cast_exprs = []
    for c in ozz_cols_present:
        cast_exprs.append(
            pl.col(c).cast(pl.Float32, strict=False).fill_null(0).alias(c)
        )
    df_clean = df.with_columns(cast_exprs)

    # Макс OZZ-флагов по каждому файлу
    agg_exprs = [pl.col(c).max().alias(f'max_{c}') for c in ozz_cols_present]
    file_agg = df_clean.group_by('file_name').agg(agg_exprs)

    # Суммарный OZZ-флаг
    file_agg = file_agg.with_columns(
        sum(pl.col(f'max_{c}') for c in ozz_cols_present).alias('ozz_sum')
    )

    n_total = len(file_agg)
    n_ozz = file_agg.filter(pl.col('ozz_sum') > 0).height
    n_no_ozz = n_total - n_ozz

    stats = {
        'total_files': n_total,
        'with_ozz': n_ozz,
        'without_ozz': n_no_ozz,
    }

    no_ozz_names = file_agg.filter(pl.col('ozz_sum') == 0)['file_name'].to_list()

    if verbose:
        print(f"[get_no_ozz_files] Всего файлов: {n_total}, "
              f"с ОЗЗ: {n_ozz}, без ОЗЗ: {n_no_ozz}")

    return no_ozz_names, stats


def filter_no_ozz_dataframe(
    df: pl.DataFrame,
    no_ozz_files: Optional[List[str]] = None,
    *,
    verbose: bool = True,
) -> pl.DataFrame:
    """Возвращает DataFrame, содержащий **только** строки файлов без ОЗЗ.

    Если ``no_ozz_files=None``, сначала вычисляет список через
    :func:`get_no_ozz_files`.
    """
    if no_ozz_files is None:
        no_ozz_files, _ = get_no_ozz_files(df, verbose=verbose)

    df_filtered = df.filter(pl.col('file_name').is_in(no_ozz_files))

    if verbose:
        print(f"[filter_no_ozz_dataframe] "
              f"Строк: {df.height:,} -> {df_filtered.height:,}")

    return df_filtered
