"""
Стратифицированное разбиение файлов датасета `Simulated_OZZ_v1` на train/val.

Стратификация — по типу дуги `X` (1..4), извлечённому из имени файла
`OZZ_X_R_L_P_T.csv`. Это гарантирует, что все 4 типа дуги присутствуют
как в train, так и в val (а при желании — и test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from osc_tools.ml.simulated_ozz_dataset import parse_filename, ARC_TYPES


def stratified_sim_ozz_split(
    file_names: List[str],
    val_split: float = 0.2,
    test_split: float = 0.0,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, int]]]:
    """Стратифицированный сплит по типу X из имени файла.

    Args:
        file_names: список имён файлов (без пути, напр. `OZZ_3_1_10_1_200.csv`)
        val_split: доля val (0..1)
        test_split: доля test (0..1). Если 0, возвращается пустой список.
        seed: random seed

    Returns:
        (train, val, test, stats), где stats = {split: {type_name: count}}
    """
    if not 0 <= val_split < 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if not 0 <= test_split < 1:
        raise ValueError(f"test_split must be in [0, 1), got {test_split}")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split должны быть < 1")

    # Группировка по X
    groups: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}
    for name in file_names:
        meta = parse_filename(name)
        if meta is None:
            continue
        x = meta['x']
        if x in groups:
            groups[x].append(name)

    rng = np.random.default_rng(seed)
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    for x, files in groups.items():
        if not files:
            continue
        files_sorted = sorted(files)
        rng.shuffle(files_sorted)
        n = len(files_sorted)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        # Гарантируем хотя бы 1 в val, если split > 0
        if val_split > 0 and n_val == 0 and n >= 2:
            n_val = 1

        test.extend(files_sorted[:n_test])
        val.extend(files_sorted[n_test:n_test + n_val])
        train.extend(files_sorted[n_test + n_val:])

    # Статистика
    def _count_by_type(names: List[str]) -> Dict[str, int]:
        counts = {t: 0 for t in ARC_TYPES}
        for nm in names:
            m = parse_filename(nm)
            if m is not None:
                counts[ARC_TYPES[m['x'] - 1]] += 1
        return counts

    stats = {
        'train': _count_by_type(train),
        'val': _count_by_type(val),
        'test': _count_by_type(test),
    }
    return train, val, test, stats
