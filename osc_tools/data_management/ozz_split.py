"""
Стратифицированное разбиение данных по осциллограммам для задачи ОЗЗ/ДПОЗЗ.

Обеспечивает:
- Разделение на уровне целых файлов (исключение утечки данных)
- Гарантированное представительство каждого целевого класса в test
- Иерархическую приоритизацию файлов (ДПОЗЗ > Затухающее > Устойчивое)

Семантика меток (multi-label, НЕ взаимоисключающие):
    Target_OZZ      — любое ОЗЗ (включая затухающее и ДПОЗЗ)
    Target_OZZ_decay  — затухающее ОЗЗ (подмножество Target_OZZ)
    Target_OZZ_dpozz  — ДПОЗЗ (подмножество Target_OZZ)
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# Колонки разметки для ОЗЗ-подтипов
OZZ_LABEL_COLS = {
    'stable': ['ML_2_1', 'ML_2_1_1'],      # Устойчивое ОЗЗ
    'decay':  ['ML_2_1_2'],                  # Затухающее ОЗЗ
    'dpozz':  ['ML_2_1_3'],                  # ДПОЗЗ
}

# Целевые колонки для модели (3 класса, multi-label)
OZZ_TARGET_COLS = ['Target_OZZ', 'Target_OZZ_decay', 'Target_OZZ_dpozz']


def classify_file_ozz(df_file: pl.DataFrame) -> str:
    """
    Определяет класс файла по иерархии приоритетов.

    Иерархия (от самого редкого к обычному):
        ДПОЗЗ > Затухающее ОЗЗ > Устойчивое ОЗЗ > Нет ОЗЗ

    Args:
        df_file: DataFrame одного файла

    Returns:
        Строка-класс: 'dpozz', 'decay', 'stable', 'no_ozz'
    """
    cols = set(df_file.columns)

    # Проверяем от самого редкого
    for col in OZZ_LABEL_COLS['dpozz']:
        if col in cols and df_file[col].max() > 0:
            return 'dpozz'

    for col in OZZ_LABEL_COLS['decay']:
        if col in cols and df_file[col].max() > 0:
            return 'decay'

    for col in OZZ_LABEL_COLS['stable']:
        if col in cols and df_file[col].max() > 0:
            return 'stable'

    return 'no_ozz'


def stratified_ozz_split(
    df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    min_test_per_class: int = 1,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Стратифицированное разбиение файлов на train/test с гарантией
    представительства каждого класса ОЗЗ в тестовой выборке.

    Args:
        df: Полный DataFrame с колонкой 'file_name' и ML_2_1* метками.
        test_size: Доля тестовой выборки (0.0–1.0).
        random_state: Seed для воспроизводимости.
        min_test_per_class: Минимальное число файлов каждого класса в test.

    Returns:
        (train_files, test_files, stats)
        stats — словарь со статистикой по классам в train и test.
    """
    rng = np.random.default_rng(random_state)

    # 1. Классифицируем каждый файл
    file_names = df['file_name'].unique().to_list()
    file_classes: Dict[str, str] = {}
    for fname in file_names:
        file_df = df.filter(pl.col('file_name') == fname)
        file_classes[fname] = classify_file_ozz(file_df)

    # 2. Группируем файлы по классам
    class_files: Dict[str, List[str]] = {
        'dpozz': [], 'decay': [], 'stable': [], 'no_ozz': []
    }
    for fname, cls in file_classes.items():
        class_files[cls].append(fname)

    # Перемешиваем внутри групп
    for cls in class_files:
        rng.shuffle(class_files[cls])

    # 3. Гарантируем минимум min_test_per_class файлов каждого класса в test
    test_files = []
    remaining: Dict[str, List[str]] = {}

    for cls in ['dpozz', 'decay', 'stable', 'no_ozz']:
        files = class_files[cls]
        n_test = min(min_test_per_class, len(files))
        test_files.extend(files[:n_test])
        remaining[cls] = files[n_test:]

    # 4. Заполняем до нужного размера пропорционально
    all_remaining = []
    for cls in remaining:
        all_remaining.extend(remaining[cls])

    total_files = len(file_names)
    target_test_count = max(int(total_files * test_size), len(test_files))
    n_extra = target_test_count - len(test_files)

    if n_extra > 0 and len(all_remaining) > 0:
        rng.shuffle(all_remaining)
        extra = all_remaining[:n_extra]
        test_files.extend(extra)
        all_remaining = all_remaining[n_extra:]

    test_set = set(test_files)
    train_files = [f for f in file_names if f not in test_set]

    # 5. Статистика
    stats: Dict[str, Dict[str, int]] = {'train': {}, 'test': {}}
    for split_name, split_files in [('train', train_files), ('test', test_files)]:
        counts = {'dpozz': 0, 'decay': 0, 'stable': 0, 'no_ozz': 0}
        for f in split_files:
            counts[file_classes[f]] += 1
        stats[split_name] = counts

    return train_files, test_files, stats


def add_ozz_target_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Добавляет целевые колонки для задачи ОЗЗ (3 колонки, multi-label).

    Target_OZZ       — любое ОЗЗ (ML_2_1 ∨ ML_2_1_1 ∨ ML_2_1_2 ∨ ML_2_1_3)
    Target_OZZ_decay — затухающее ОЗЗ (ML_2_1_2) — подмножество Target_OZZ
    Target_OZZ_dpozz — ДПОЗЗ (ML_2_1_3) — подмножество Target_OZZ

    Классы НЕ взаимоисключающие: если Target_OZZ_decay=1 или Target_OZZ_dpozz=1,
    то Target_OZZ тоже обязательно = 1.

    Args:
        df: DataFrame с ML_2_1* колонками

    Returns:
        DataFrame с добавленными Target_OZZ / Target_OZZ_decay / Target_OZZ_dpozz
    """
    available = set(df.columns)

    def _get_col(name: str):
        if name in available:
            return pl.col(name).cast(pl.Int8).fill_null(0)
        return pl.lit(0, dtype=pl.Int8)

    # ДПОЗЗ
    dpozz_expr = _get_col('ML_2_1_3')

    # Затухающее ОЗЗ
    decay_expr = _get_col('ML_2_1_2')

    # ОЗЗ (общее) = ML_2_1 ∨ ML_2_1_1 ∨ ML_2_1_2 ∨ ML_2_1_3
    ozz_expr = pl.max_horizontal([
        _get_col('ML_2_1'), _get_col('ML_2_1_1'),
        _get_col('ML_2_1_2'), _get_col('ML_2_1_3'),
    ]).fill_null(0).cast(pl.Int8)

    df = df.with_columns([
        ozz_expr.alias('Target_OZZ'),
        decay_expr.alias('Target_OZZ_decay'),
        dpozz_expr.alias('Target_OZZ_dpozz'),
    ])

    return df
