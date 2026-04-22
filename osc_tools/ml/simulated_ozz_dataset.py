"""
Lazy-загрузчик датасета ``data/Simulated_OZZ_v1/*.csv`` (RTDS-симуляции ОЗЗ/ДПОЗЗ).

Ключевые принципы:
- **Без ресэмплирования:** работаем с сырыми данными (Fs ~ 38.4..40 кГц). FFT вычисляется
  напрямую по сырому сигналу — гармоники считаются корректно при любой Fs.
- **Per-file Fs:** файлы с L=10,20 имеют Fs=38462 (spp=769, stride=96),
  файлы с L=30..100 имеют Fs=40000 (spp=800, stride=100). Параметры вычисляются
  для каждого файла индивидуально; число зон всегда = (num_periods_window - 1) * stride_fraction.
- **3I0 → IN, 3U0 → UN:** каналы нулевой последовательности присутствуют в CSV.
- **Lazy loading:** 62 ГБ не влезают в RAM. Файлы подгружаются по требованию
  с LRU-кэшем (по умолчанию 500 файлов ≈ 500 МБ).
- **Stride = 1/8 периода:** для зонирования (96..100 отсчётов в зависимости от Fs).
- **Совместимость:** выходной формат (C, T) совпадает с AugmentedSpectralDataset.

Пайплайн ``__getitem__``::

    CSV → сырые 8 каналов → [аугментация] → FFT (compute_spectral_from_raw) → (X, Y)

Использование::

    from osc_tools.ml.simulated_ozz_dataset import (
        SimOZZFileIndex, SimOZZLazyDataset, parse_filename,
    )

    index = SimOZZFileIndex.from_directory('data/Simulated_OZZ_v1')
    train_paths, val_paths = ...  # из sim_ozz_split
    ds = SimOZZLazyDataset(
        file_paths=train_paths,
        file_index=index,
        mode='classify',
    )
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from osc_tools.ml.augmented_dataset import (
    DEFAULT_SUB_PERIODS,
    NUM_RAW,
    RAW_CHANNELS,
    compute_num_channels,
    compute_spectral_from_raw,
)
from osc_tools.ml.labels import get_target_columns


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

DEFAULT_F_NETWORK = 50.0  # Гц

# Отображение X → имя теории (индекс 0..3 в Target_OZZ_*)
ARC_TYPES: List[str] = ['Stable', 'Petersen', 'PetersSlepian', 'Beliakov']
TARGET_COLUMNS: List[str] = [f'Target_OZZ_{t}' for t in ARC_TYPES]

# ---------------------------------------------------------------------------
# Номинальные коэффициенты для нормализации Simulated_OZZ_v1 (сеть 10 кВ).
# Сигналы делятся на эти значения перед FFT, чтобы привести к ~1.0 в нормальном
# режиме (как физическая нормализация реальных осциллограмм через ТТ/ТН).
#
# ВАЖНО: Эти значения нужно уточнить по результатам
# Номинальные значения для нормализации SimOZZ (данные в кВ / кА).
# Определены по результатам scan_sim_ozz_statistics.py (19199 файлов).
# Логика: эмулируем ТТ/ТН реальной подстанции 10 кВ.
# ---------------------------------------------------------------------------
SIM_NOMINAL: Dict[str, float] = {
    # Фазные напряжения: U_лин = 10 кВ → данные в кВ
    'UA': 10.0,
    'UB': 10.0,
    'UC': 10.0,
    # UN (3U0): утроенное фазное = 3 × 10/√3 ≈ 17.333 кВ
    'UN': 3.0 * 10.0 / 1.7320508075688772,   # ≈ 17.321 кВ
    # Фазные токи: ТТ на 200 А → 0.2 кА
    # (median h1 ≈ 0.038 кА → после нормализации ~0.19)
    'IA': 0.2,
    'IB': 0.2,
    'IC': 0.2,
    # Ток нулевой последовательности: ТТНП на 30 А → 0.03 кА
    # (median h1 ≈ 0.009 кА → после нормализации ~0.28)
    'IN': 0.03,
}

# Паттерн имени поддерживает два варианта:
#   OZZ_X_R_L_P_T.csv        (5 числовых параметров: нет угла зажигания I)
#   OZZ_X_R_L_P_I_T.csv      (6 числовых параметров: с моментом зажигания I)
_FILENAME_RE = re.compile(
    r'^OZZ_(\d+)_(\d+)_(\d+)_(\d+)(?:_(\d+))?_(\d+)\.csv$',
    re.IGNORECASE,
)

# Карта «сырое имя колонки RTDS → короткое»
_COLUMN_ALIAS: Dict[str, str] = {
    'S1_VA': 'UA', 'VA': 'UA',
    'S1_VB': 'UB', 'VB': 'UB',
    'S1_VC': 'UC', 'VC': 'UC',
    '3I0': 'IN',
    '3U0': 'UN',
}


# ---------------------------------------------------------------------------
# Парсинг имени файла
# ---------------------------------------------------------------------------

def parse_filename(path: str | Path) -> Optional[Dict[str, int]]:
    """Парсит имя файла набора ``Simulated_OZZ_v1`` в словарь параметров.

    Поддерживает два варианта:
      * ``OZZ_X_R_L_P_T.csv``      — 5 параметров (без момента зажигания).
      * ``OZZ_X_R_L_P_I_T.csv``    — 6 параметров (с моментом зажигания I).

    Returns:
        dict с ключами ``x, r, l, p, i, t``.  Если поле *I* отсутствует —
        ставится ``i=0``.  ``None``, если имя не соответствует шаблону.
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
# Загрузка одного CSV (без ресэмплирования)
# ---------------------------------------------------------------------------

def _rename_rtds_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Сокращает имена колонок RTDS ``Subsystem #1|...|X`` → ``X``, далее алиасы."""
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        short = col.split('|')[-1].strip()
        short = short.replace(') ', '_').replace(' ', '_').replace(')', '')
        short = _COLUMN_ALIAS.get(short, short)
        if short != col:
            rename_map[col] = short
    if rename_map:
        df = df.rename(rename_map)
    return df


def load_raw_csv(
    path: str | Path,
    normalize: bool = True,
    nominal: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Загружает один CSV и возвращает сырые numpy-массивы БЕЗ ресэмплирования.

    Args:
        path: путь к CSV.
        normalize: делить ли каналы на номинальные коэффициенты.
        nominal: dict канал→делитель (по умолчанию SIM_NOMINAL).

    Returns:
        dict с ключами:
            ``raw``      — (T, 8) float32, каналы [IA, IB, IC, IN, UA, UB, UC, UN]
            ``targets``  — (T, 4) int8, по одному столбцу на тип дуги
            ``ozz``      — (T,) int8, бинарный флаг OZZ
            ``meta``     — dict из parse_filename
            ``dt``       — float, шаг по времени (сек)
            ``n_samples``— int
        или ``None``, если файл не совпадает по формату.
    """
    path = Path(path)
    meta = parse_filename(path)
    if meta is None:
        return None

    df = pl.read_csv(str(path), infer_schema_length=1000)
    df = _rename_rtds_columns(df)

    required = {'Time', 'IA', 'IB', 'IC', 'UA', 'UB', 'UC', 'OZZ', 'IN', 'UN'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path.name} нет колонок: {missing}")

    n = len(df)
    if n < 2:
        return None

    # Шаг дискретизации из первых двух отсчётов
    t_col = df['Time'].to_numpy().astype(np.float64)
    dt = float(t_col[1] - t_col[0])

    # Сырые 8 каналов (порядок совместим с RAW_CHANNELS)
    raw = np.empty((n, NUM_RAW), dtype=np.float32)
    for i, ch in enumerate(RAW_CHANNELS):
        raw[:, i] = df[ch].to_numpy().astype(np.float32)

    # Нормализация: делим каждый канал на номинальное значение
    if normalize:
        nom = nominal if nominal is not None else SIM_NOMINAL
        for i, ch in enumerate(RAW_CHANNELS):
            divisor = nom.get(ch, 1.0)
            if divisor != 1.0:
                raw[:, i] /= divisor

    # Бинарный флаг OZZ
    ozz = (df['OZZ'].to_numpy() > 0.5).astype(np.int8)

    # Целевые 4 класса: one-hot по X файла × OZZ flag
    x = meta['x']
    targets = np.zeros((n, len(ARC_TYPES)), dtype=np.int8)
    if 1 <= x <= len(ARC_TYPES):
        targets[:, x - 1] = ozz

    return {
        'raw': raw,
        'targets': targets,
        'ozz': ozz,
        'meta': meta,
        'dt': dt,
        'n_samples': n,
    }


# ---------------------------------------------------------------------------
# Индекс файлов (без загрузки данных)
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Метаданные одного CSV (без загрузки данных)."""
    path: Path
    meta: Dict[str, int]
    n_samples: int
    dt: float    # шаг дискретизации (с)
    fs: float    # Fs = 1/dt


@dataclass
class SimOZZFileIndex:
    """Каталог CSV-файлов набора Simulated_OZZ_v1.

    Сканирует директорию, считывает число строк и шаг дискретизации
    для каждого файла, **без загрузки самих данных**. Используется для
    создания ``SimOZZLazyDataset`` и для стратифицированного split.
    """
    files: List[FileInfo] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        max_files: Optional[int] = None,
        verbose: bool = True,
    ) -> 'SimOZZFileIndex':
        """Сканирует ``data_dir``, считывает метаданные каждого файла."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {data_dir}")

        csv_paths = sorted(data_dir.glob('OZZ_*.csv'))
        if max_files is not None:
            csv_paths = csv_paths[:max_files]
        if not csv_paths:
            raise ValueError(f"Не найдено файлов OZZ_*.csv в {data_dir}")

        if verbose:
            print(f"[SimOZZFileIndex] Сканирование {len(csv_paths)} файлов "
                  f"в {data_dir}...")

        infos: List[FileInfo] = []
        skipped = 0
        for i, p in enumerate(csv_paths):
            meta = parse_filename(p)
            if meta is None:
                skipped += 1
                continue
            try:
                # Быстрое чтение: только Time, 2 строки (для dt) + подсчёт строк
                head = pl.read_csv(str(p), n_rows=2, columns=['Time'],
                                   infer_schema_length=2)
                dt = float(head['Time'][1] - head['Time'][0])
                n_samples = pl.scan_csv(str(p)).select(pl.len()).collect().item()
            except Exception:
                skipped += 1
                continue

            infos.append(FileInfo(
                path=p, meta=meta, n_samples=n_samples,
                dt=dt, fs=1.0 / dt,
            ))

            if verbose and ((i + 1) % 2000 == 0):
                print(f"  [{i + 1}/{len(csv_paths)}]...")

        if verbose:
            print(f"[SimOZZFileIndex] Готово: {len(infos)} файлов, "
                  f"пропущено {skipped}")
            if infos:
                fs0 = infos[0].fs
                print(f"  Fs ~ {fs0:.1f} Гц (dt = {infos[0].dt:.2e} с), "
                      f"samples/period ~ {fs0 / DEFAULT_F_NETWORK:.1f}")

        return cls(files=infos)

    # ------------------------------------------------------------------
    def paths(self) -> List[Path]:
        """Все пути файлов."""
        return [fi.path for fi in self.files]

    def get_by_name(self, name: str) -> Optional[FileInfo]:
        """Поиск FileInfo по имени файла."""
        for fi in self.files:
            if fi.path.name == name:
                return fi
        return None

    def __len__(self) -> int:
        return len(self.files)


# ---------------------------------------------------------------------------
# LRU-кэш загруженных файлов
# ---------------------------------------------------------------------------

class _FileCache:
    """Простой LRU-кэш для (raw, targets) numpy-массивов."""

    def __init__(self, maxsize: int = 500):
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, path: Path, data: Dict[str, np.ndarray]) -> None:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = data


# ---------------------------------------------------------------------------
# PyTorch Dataset (lazy)
# ---------------------------------------------------------------------------

class SimOZZLazyDataset(Dataset):
    """Lazy-загрузчик для Simulated_OZZ_v1 с on-the-fly FFT.

    Вместо предзагрузки всех данных в RAM подгружает CSV по требованию
    и кэширует последние ``cache_size`` файлов (~1 МБ/файл → 500 файлов ≈ 500 МБ).

    Выходной формат ``__getitem__``::

        (X, Y) — X: Tensor (C, T_zones), Y: Tensor (T_zones, n_classes)

    Совместим по размерности C с ``AugmentedSpectralDataset`` при тех же
    ``num_harmonics`` и ``sub_periods``.
    """

    def __init__(
        self,
        file_paths: List[Path],
        file_index: SimOZZFileIndex,
        *,
        f_network: float = DEFAULT_F_NETWORK,
        num_periods_window: int = 10,
        stride_fraction: int = 8,
        num_harmonics: int = 9,
        sub_periods: Optional[List[int]] = None,
        include_symmetric: bool = True,
        target_columns: Optional[List[str]] = None,
        target_level: str = 'sim_ozz',
        mode: str = 'classify',
        zone_target_aggregation: str = 'max',
        augmenter: Optional[object] = None,
        cache_size: int = 500,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            file_paths: список путей к CSV (подмножество из file_index).
            file_index: полный индекс (для n_samples, dt).
            f_network: промышленная частота (Гц), по умолчанию 50.
            num_periods_window: длина окна в периодах (10 → ~7690 отсчётов).
            stride_fraction: доля периода для stride (8 → 1/8 = ~96 отсчётов).
            num_harmonics: число стандартных гармоник FFT (9).
            sub_periods: суб-периоды для низших гармоник [2, 4, 6, 10].
            include_symmetric: добавлять симметричные составляющие.
            target_columns: имена колонок меток (None → из target_level).
            target_level: уровень меток ('sim_ozz').
            mode: 'classify' (позонная разметка).
            zone_target_aggregation: агрегация внутри зоны ('max' / 'mean').
            augmenter: аугментатор на сырых данных (None = без аугментации).
            cache_size: макс. кэшированных файлов.
            verbose: печатать статистику.
        """
        super().__init__()

        self.mode = mode
        self.augmenter = augmenter
        self.num_harmonics = num_harmonics
        self.sub_periods = (sub_periods if sub_periods is not None
                            else list(DEFAULT_SUB_PERIODS))
        self.include_symmetric = include_symmetric
        self.zone_target_aggregation = zone_target_aggregation.lower()
        if self.zone_target_aggregation not in {'max', 'mean'}:
            raise ValueError(
                f"zone_target_aggregation: ожидается 'max'/'mean', "
                f"получено {zone_target_aggregation!r}",
            )

        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = get_target_columns(target_level)
        self.n_classes = len(self.target_columns)

        # --- Сохраняем параметры для per-file вычислений ---
        self.f_network = f_network
        self.stride_fraction = stride_fraction
        self.num_periods_window = num_periods_window

        # --- Индекс файлов ---
        self._file_infos: Dict[str, FileInfo] = {}
        for fp in file_paths:
            fi = file_index.get_by_name(fp.name)
            if fi is not None:
                self._file_infos[fp.name] = fi
        if not self._file_infos:
            raise ValueError("Ни один из file_paths не найден в file_index")

        # Число каналов (не зависит от Fs)
        self.num_output_channels: int = compute_num_channels(
            num_harmonics, len(self.sub_periods), include_symmetric,
        )

        # Число зон ВСЕГДА одинаково:
        #   (num_periods_window * spp - spp) / (spp / stride_fraction)
        #   = (num_periods_window - 1) * stride_fraction
        self.num_steps: int = (num_periods_window - 1) * stride_fraction

        # --- Индексация: (file_path, window_start) --- per-file Fs ---
        self._indices: List[Tuple[Path, int]] = []
        fs_set: set = set()
        for name, fi in self._file_infos.items():
            spp, stride, _, window_size, _ = self._compute_params(fi.fs)
            fs_set.add(fi.fs)
            max_start = fi.n_samples - window_size
            if max_start < 0:
                # Файл короче окна -- пропускаем  # TODO: обработка коротких файлов
                continue
            for ws in range(0, max_start + 1, stride):
                self._indices.append((fi.path, ws))

        # Кэш
        self._cache = _FileCache(maxsize=cache_size)

        if verbose:
            fs_list = sorted(fs_set)
            first_fi = next(iter(self._file_infos.values()))
            spp0, stride0, _, ws0, ctx0 = self._compute_params(first_fi.fs)
            fs_str = ', '.join(f'{f:.0f}' for f in fs_list)
            print(
                f"[SimOZZLazyDataset] mode={mode}, "
                f"files={len(self._file_infos)}, "
                f"windows={len(self._indices):,}, "
                f"Fs=[{fs_str}] Гц (per-file), "
                f"spp(example)={spp0}, "
                f"stride(example)={stride0} (1/{stride_fraction} периода), "
                f"window(example)={ws0} ({num_periods_window}T), "
                f"channels={self.num_output_channels}, "
                f"ctx_before(example)={ctx0}, "
                f"zones/window={self.num_steps}, "
                f"cache={cache_size}",
            )

    # ------------------------------------------------------------------
    def _compute_params(
        self, fs: float,
    ) -> Tuple[int, int, int, int, int]:
        """Вычисляет (spp, stride, fft_window, window_size, ctx_before) для данной Fs."""
        spp = round(fs / self.f_network)
        stride = max(1, spp // self.stride_fraction)
        fft_window = spp
        window_size = self.num_periods_window * spp
        ctx_before = max(self.sub_periods) * spp
        return spp, stride, fft_window, window_size, ctx_before

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._indices)

    # ------------------------------------------------------------------
    def _load_file(self, path: Path) -> Dict[str, np.ndarray]:
        """Загружает CSV (из кэша или с диска)."""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        result = load_raw_csv(path)
        if result is None:
            raise RuntimeError(f"Не удалось загрузить {path}")
        self._cache.put(path, result)
        return result

    # ------------------------------------------------------------------
    def _load_raw_window(
        self,
        file_data: Dict[str, np.ndarray],
        window_start: int,
        window_size: int,
        ctx_before: int,
    ) -> np.ndarray:
        """Извлекает окно (context + window) из raw-массива файла.

        Если для backward-looking контекста не хватает данных слева —
        недостающие позиции заполняются NaN (модель научится игнорировать).

        Returns:
            (ctx_before + window_size, 8) float32
        """
        raw_full = file_data['raw']   # (T_file, 8)
        n_file = raw_full.shape[0]

        ctx_start = window_start - ctx_before
        raw_end = min(window_start + window_size, n_file)

        if ctx_start >= 0:
            raw = raw_full[ctx_start:raw_end].copy()
        else:
            # Не хватает данных слева — NaN-pad
            nan_len = -ctx_start
            raw_part = raw_full[0:raw_end].copy()
            nan_pad = np.full((nan_len, NUM_RAW), np.nan, dtype=np.float32)
            raw = np.concatenate([nan_pad, raw_part], axis=0)

        needed = ctx_before + window_size
        if raw.shape[0] < needed:
            raw = np.concatenate([
                raw,
                np.full((needed - raw.shape[0], NUM_RAW), np.nan,
                        dtype=np.float32),
            ], axis=0)

        return raw

    # ------------------------------------------------------------------
    def _extract_zone_targets(
        self,
        file_data: Dict[str, np.ndarray],
        window_start: int,
        fft_window: int,
        stride: int,
    ) -> np.ndarray:
        """Позонная агрегация меток.

        Returns:
            (num_steps, n_classes) float32
        """
        targets = file_data['targets']   # (T_file, 4)
        n_file = targets.shape[0]

        Y = np.zeros((self.num_steps, self.n_classes), dtype=np.float32)

        for z in range(self.num_steps):
            z_start = window_start + fft_window + z * stride
            z_end = min(z_start + stride, n_file)
            z_start = min(z_start, n_file - 1)
            if z_start < z_end:
                zone = targets[z_start:z_end].astype(np.float32)
                if self.zone_target_aggregation == 'mean':
                    Y[z] = np.mean(zone, axis=0)
                else:
                    Y[z] = np.max(zone, axis=0)
            else:
                Y[z] = targets[z_start].astype(np.float32)

        return Y

    # ------------------------------------------------------------------
    def __getitem__(
        self, idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, window_start = self._indices[idx]

        # 0. Per-file параметры дискретизации
        fi = self._file_infos[file_path.name]
        spp, stride, fft_window, window_size, ctx_before = (
            self._compute_params(fi.fs)
        )

        # 1. Загружаем файл (из кэша или диска)
        file_data = self._load_file(file_path)

        # 2. Извлекаем окно с контекстом
        raw = self._load_raw_window(
            file_data, window_start, window_size, ctx_before,
        )

        # 3. Аугментация на сырых данных (до FFT)
        if self.augmenter is not None:
            raw = self.augmenter(raw)
            if isinstance(raw, torch.Tensor):
                raw = raw.numpy()
            raw = np.asarray(raw, dtype=np.float32)

        # 4. FFT → спектральные признаки (per-file fft_window / spp)
        spectral = compute_spectral_from_raw(
            raw,
            num_harmonics=self.num_harmonics,
            sub_periods=self.sub_periods,
            include_symmetric=self.include_symmetric,
            stride=stride,
            warmup=ctx_before + fft_window,
            fft_window=fft_window,
            samples_per_period=spp,
        )   # (T_zones, C)

        # Обрезаем до num_steps (при нецелочисленном spp//stride_fraction
        # может быть на 1 зону больше)
        spectral = spectral[:self.num_steps]

        X = torch.from_numpy(spectral.T.copy())   # (C, T_zones)

        # 5. Метки зон (per-file stride, fft_window)
        Y = self._extract_zone_targets(
            file_data, window_start, fft_window, stride,
        )
        Y = torch.from_numpy(Y)   # (T_zones, n_classes)

        return X, Y
