"""
Загрузчик датасета `data/Simulated_OZZ_v1/*.csv` (RTDS-симуляции ОЗЗ/ДПОЗЗ).

Пайплайн:
1. Парсим имя файла `OZZ_X_R_L_P_T.csv`:
   - X ∈ {1,2,3,4} — тип дуги (1=Stable, 2=Petersen, 3=Peters_Slepian, 4=Beliakov)
   - R — переходное сопротивление, Ом
   - L — удалённость, % длины линии
   - P ∈ {1,2,3} — повреждённая фаза (A/B/C)
   - T — длительность ОЗЗ, мс
2. Читаем CSV: колонки Subsystem|...|VA/VB/VC и |TLINES|...|IA/IB/IC + флаги OZZ, Stable, ...
3. Ресэмплируем к `target_fs` (по умолчанию 1600 Гц — совместимо с пайплайном Phase 4).
4. Подставляем IN=0, UN=0 (изолированная нейтраль).
5. Формируем целевые колонки: Target_OZZ_Stable/Petersen/PetersSlepian/Beliakov —
   = 1 в точках, где `OZZ==1` И X из имени файла соответствует.

Используется в `scripts/phase4_experiments/run_phase4_finetune_sim_ozz.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

DEFAULT_TARGET_FS = 1600.0   # Гц, совместимо с пайплайном Phase 4 (SAMPLES_PER_PERIOD=32)
DEFAULT_F_NETWORK = 50.0      # Гц

# Отображение X → имя теории (индекс 0..3 в Target_OZZ_*)
ARC_TYPES: List[str] = ['Stable', 'Petersen', 'PetersSlepian', 'Beliakov']

TARGET_COLUMNS: List[str] = [f'Target_OZZ_{t}' for t in ARC_TYPES]

# Паттерн имени поддерживает два варианта:
#   OZZ_X_R_L_P_T.csv        (5 параметров: нет угла зажигания I)
#   OZZ_X_R_L_P_I_T.csv      (6 параметров: с моментом зажигания I — актуален для X=1 Stable)
_FILENAME_RE = re.compile(
    r'^OZZ_(\d+)_(\d+)_(\d+)_(\d+)(?:_(\d+))?_(\d+)\.csv$',
    re.IGNORECASE,
)

# Карта «сырое имя колонки RTDS → короткое»
_COLUMN_ALIAS = {
    'S1_VA': 'UA', 'VA': 'UA',
    'S1_VB': 'UB', 'VB': 'UB',
    'S1_VC': 'UC', 'VC': 'UC',
}


# ---------------------------------------------------------------------------
# Парсинг имени файла
# ---------------------------------------------------------------------------

def parse_filename(path: str | Path) -> Optional[Dict[str, int]]:
    """Парсит имя файла набора `Simulated_OZZ_v1` в словарь параметров.

    Поддерживает два варианта:
      * `OZZ_X_R_L_P_T.csv`        — 5 параметров (без момента зажигания).
      * `OZZ_X_R_L_P_I_T.csv`      — 6 параметров (с моментом зажигания I,
        актуален для X=1 Stable).

    Returns:
        dict с ключами x, r, l, p, i, t. Если поле I отсутствует в имени —
        ставится i=0. Возвращает None, если имя не соответствует шаблону.
    """
    name = Path(path).name
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    x, r, l, p, i_opt, t = m.groups()
    return {
        'x': int(x),
        'r': int(r),
        'l': int(l),
        'p': int(p),
        'i': int(i_opt) if i_opt is not None else 0,
        't': int(t),
    }


# ---------------------------------------------------------------------------
# Загрузка одного CSV
# ---------------------------------------------------------------------------

def _rename_rtds_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Сокращает длинные имена колонок RTDS `Subsystem #1|...|X` → `X`."""
    rename_map = {}
    for col in df.columns:
        # Берём имя после последней `|`, заменяем пробелы/скобки
        short = col.split('|')[-1].strip()
        short = short.replace(') ', '_').replace(' ', '_').replace(')', '')
        # Финальный алиас: S1_VA → UA и т.д.
        short = _COLUMN_ALIAS.get(short, short)
        if short != col:
            rename_map[col] = short
    if rename_map:
        df = df.rename(rename_map)
    return df


def _resample_linear(
    signal: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """Линейная интерполяция сигнала на новую сетку времени."""
    return np.interp(t_dst, t_src, signal).astype(np.float32)


def load_simulated_ozz_csv(
    path: str | Path,
    target_fs: float = DEFAULT_TARGET_FS,
) -> Optional[pl.DataFrame]:
    """Загружает один CSV и возвращает polars DataFrame в формате пайплайна.

    Формат выходного DataFrame (колонки):
        file_name, IA, IB, IC, IN, UA, UB, UC, UN,
        Target_OZZ_Stable, Target_OZZ_Petersen,
        Target_OZZ_PetersSlepian, Target_OZZ_Beliakov

    Args:
        path: путь к CSV
        target_fs: целевая частота дискретизации, Гц (по умолчанию 1600)

    Returns:
        DataFrame или None, если файл не подходит под формат
    """
    path = Path(path)
    meta = parse_filename(path)
    if meta is None:
        return None

    df = pl.read_csv(str(path), infer_schema_length=1000)
    df = _rename_rtds_columns(df)

    required = {'Time', 'IA', 'IB', 'IC', 'UA', 'UB', 'UC', 'OZZ'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path.name} нет колонок: {missing}")

    # --- 1. Исходная и целевая сетка времени ---
    t_src = df['Time'].to_numpy().astype(np.float64)
    if len(t_src) < 2:
        return None
    duration = t_src[-1]
    n_dst = int(np.floor(duration * target_fs)) + 1
    if n_dst < 2:
        return None
    t_dst = np.arange(n_dst, dtype=np.float64) / target_fs

    # --- 2. Ресэмплирование аналоговых сигналов (линейно) ---
    out: Dict[str, np.ndarray] = {}
    for ch in ('IA', 'IB', 'IC', 'UA', 'UB', 'UC'):
        out[ch] = _resample_linear(df[ch].to_numpy().astype(np.float64), t_src, t_dst)

    # Отсутствующие каналы в датасете → нули (изолированная нейтраль)
    out['IN'] = np.zeros(n_dst, dtype=np.float32)
    out['UN'] = np.zeros(n_dst, dtype=np.float32)

    # --- 3. Ресэмплирование дискретных флагов (nearest neighbour) ---
    # Использую np.interp и округление — флаги ступенчатые, NN эквивалентно.
    ozz_src = df['OZZ'].to_numpy().astype(np.float64)
    # Для дискретных флагов — ближайшая точка (nearest), чтобы не размывать границы
    idx_nn = np.searchsorted(t_src, t_dst, side='right') - 1
    idx_nn = np.clip(idx_nn, 0, len(t_src) - 1)
    ozz_dst = (ozz_src[idx_nn] > 0.5).astype(np.int8)

    # --- 4. Целевые колонки: 4 класса, one-hot по X файла, активно при OZZ==1 ---
    x = meta['x']
    for i, arc_name in enumerate(ARC_TYPES):
        col = f'Target_OZZ_{arc_name}'
        if (i + 1) == x:
            out[col] = ozz_dst
        else:
            out[col] = np.zeros(n_dst, dtype=np.int8)

    # --- 5. file_name ---
    out['file_name'] = np.full(n_dst, path.name, dtype=object)

    # Собираем DataFrame в ожидаемом порядке колонок
    df_out = pl.DataFrame({
        'file_name': out['file_name'],
        'IA': out['IA'], 'IB': out['IB'], 'IC': out['IC'], 'IN': out['IN'],
        'UA': out['UA'], 'UB': out['UB'], 'UC': out['UC'], 'UN': out['UN'],
        'Target_OZZ_Stable': out['Target_OZZ_Stable'],
        'Target_OZZ_Petersen': out['Target_OZZ_Petersen'],
        'Target_OZZ_PetersSlepian': out['Target_OZZ_PetersSlepian'],
        'Target_OZZ_Beliakov': out['Target_OZZ_Beliakov'],
    })
    return df_out


# ---------------------------------------------------------------------------
# Сборка полного DataFrame из директории
# ---------------------------------------------------------------------------

def build_simulated_ozz_dataframe(
    data_dir: str | Path,
    target_fs: float = DEFAULT_TARGET_FS,
    max_files: Optional[int] = None,
    file_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pl.DataFrame, Dict[str, Dict[str, int]]]:
    """Сканирует директорию, загружает все `OZZ_*.csv` и объединяет в один DataFrame.

    Args:
        data_dir: путь к директории с CSV
        target_fs: целевая частота (Гц)
        max_files: если задан — ограничение на число файлов (для отладки)
        file_filter: если задан — список конкретных имён файлов (без пути)
        verbose: печатать прогресс

    Returns:
        (df, meta_per_file), где:
            - df: склеенный polars DataFrame
            - meta_per_file: dict[file_name -> {'x','r','l','p','t'}]
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {data_dir}")

    # Собираем список файлов
    if file_filter is not None:
        files = [data_dir / name for name in file_filter]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(data_dir.glob('OZZ_*.csv'))

    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise ValueError(f"Не найдено файлов в {data_dir}")

    if verbose:
        print(f"[SimulatedOZZ] Загрузка {len(files)} файлов из {data_dir} (Fs={target_fs} Гц)...")

    frames: List[pl.DataFrame] = []
    meta_per_file: Dict[str, Dict[str, int]] = {}
    skipped = 0

    for i, f in enumerate(files):
        meta = parse_filename(f)
        if meta is None:
            skipped += 1
            continue
        try:
            df_one = load_simulated_ozz_csv(f, target_fs=target_fs)
        except Exception as e:
            if verbose:
                print(f"  [!] Пропуск {f.name}: {e}")
            skipped += 1
            continue
        if df_one is None:
            skipped += 1
            continue
        frames.append(df_one)
        meta_per_file[f.name] = meta

        if verbose and ((i + 1) % 200 == 0 or i + 1 == len(files)):
            print(f"  [{i + 1}/{len(files)}] загружено (строк всего: "
                  f"{sum(len(fr) for fr in frames):,})")

    if not frames:
        raise ValueError("Ни одного файла не удалось загрузить.")

    df = pl.concat(frames, how='vertical')
    if verbose:
        print(f"[SimulatedOZZ] Готово: файлов={len(frames)}, строк={df.height:,}, "
              f"пропущено={skipped}")
        # Сводка по типам
        type_counts = {t: 0 for t in ARC_TYPES}
        for m in meta_per_file.values():
            type_counts[ARC_TYPES[m['x'] - 1]] += 1
        print(f"  Типы (по X из имени): {type_counts}")

    return df, meta_per_file


def compute_file_boundaries_from_df(df: pl.DataFrame) -> List[Tuple[int, int]]:
    """Возвращает список (start, length) для каждого файла в df (по порядку появления).

    Необходимо для AugmentedSpectralDataset.
    """
    fnames = df['file_name'].to_numpy()
    boundaries: List[Tuple[int, int]] = []
    if len(fnames) == 0:
        return boundaries

    start = 0
    current = fnames[0]
    for i in range(1, len(fnames)):
        if fnames[i] != current:
            boundaries.append((start, i - start))
            start = i
            current = fnames[i]
    boundaries.append((start, len(fnames) - start))
    return boundaries
