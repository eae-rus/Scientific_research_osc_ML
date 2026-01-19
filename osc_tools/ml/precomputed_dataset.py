"""
PrecomputedDataset - Dataset для работы с предрассчитанными признаками.

Используется для ускорения валидации/тестирования, когда FFT и другие 
вычисления уже выполнены и сохранены в CSV.

Использование:
    from osc_tools.ml.precomputed_dataset import PrecomputedDataset
    
    dataset = PrecomputedDataset(
        dataframe=df,  # Предрассчитанный DataFrame
        indices=val_indices,
        window_size=320,
        feature_mode='phase_polar',
        sampling_strategy='snapshot'
    )
"""

import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from typing import List, Union, Optional, Tuple

from osc_tools.ml.labels import get_target_columns


class PrecomputedDataset(Dataset):
    """
    Dataset для работы с предрассчитанными признаками (из test_precomputed.csv).
    
    Преимущества:
    - Не выполняет FFT расчёты при каждом __getitem__
    - Значительно быстрее на валидации/тесте
    - Поддерживает те же режимы (phase_polar, symmetric, phase_complex, power, alpha_beta)
    
    Ограничения:
    - Только для тестирования (без аугментации)
    - FFT_WARMUP = 32 точки (первые 31 точка = 0)
    """
    
    # Константы
    FFT_WARMUP = 32  # Точка с которой данные валидны (0-indexed: 31)
    
    ANALOG_CHANNELS = ['IA', 'IB', 'IC', 'IN', 'UA', 'UB', 'UC', 'UN']
    SYMMETRIC_COMPONENTS = ['I1', 'I2', 'I0', 'U1', 'U2', 'U0']
    POWER_COLUMNS = ['P_IA', 'Q_IA', 'P_IB', 'Q_IB', 'P_IC', 'Q_IC', 'P_IN', 'Q_IN']
    ALPHA_BETA_COLUMNS = ['I_alpha', 'I_beta', 'I_zero', 'U_alpha', 'U_beta', 'U_zero']

    @staticmethod
    def _harmonic_suffix(harmonic_idx: int) -> str:
        """Суффикс гармоники (h=1 без суффикса)."""
        return "" if harmonic_idx == 1 else f"_h{harmonic_idx}"

    @classmethod
    def _build_feature_columns(cls, feature_mode: str, num_harmonics: int, legacy_symmetric: bool = False) -> List[str]:
        """Формирует список колонок для выбранного режима и числа гармоник."""
        num_harmonics = max(1, int(num_harmonics))

        if feature_mode == 'raw':
            return cls.ANALOG_CHANNELS.copy()

        if feature_mode == 'phase_polar':
            cols = []
            for ch in cls.ANALOG_CHANNELS:
                for h in range(1, num_harmonics + 1):
                    suffix = cls._harmonic_suffix(h)
                    cols.extend([f'{ch}{suffix}_mag', f'{ch}{suffix}_angle'])
            return cols

        if feature_mode == 'phase_complex':
            cols = []
            for ch in cls.ANALOG_CHANNELS:
                for h in range(1, num_harmonics + 1):
                    suffix = cls._harmonic_suffix(h)
                    cols.extend([f'{ch}{suffix}_re', f'{ch}{suffix}_im'])
            return cols

        if feature_mode == 'symmetric':
            cols = []
            for comp in cls.SYMMETRIC_COMPONENTS:
                for h in range(1, num_harmonics + 1):
                    suffix = cls._harmonic_suffix(h)
                    if legacy_symmetric:
                        cols.extend([f'{comp}{suffix}_mag', f'{comp}{suffix}_angle'])
                    else:
                        cols.extend([f'{comp}{suffix}_re', f'{comp}{suffix}_im'])
            return cols

        if feature_mode == 'symmetric_polar':
            cols = []
            for comp in cls.SYMMETRIC_COMPONENTS:
                for h in range(1, num_harmonics + 1):
                    suffix = cls._harmonic_suffix(h)
                    cols.extend([f'{comp}{suffix}_mag', f'{comp}{suffix}_angle'])
            return cols

        if feature_mode == 'power':
            return cls.POWER_COLUMNS.copy()

        if feature_mode == 'alpha_beta':
            return cls.ALPHA_BETA_COLUMNS.copy()

        raise ValueError(f"Неподдерживаемый feature_mode: {feature_mode}")
    
    def __init__(
        self,
        dataframe: Union[pl.DataFrame, pl.LazyFrame],
        indices: Union[List[int], List[Tuple[int, int]]],
        window_size: int,
        feature_mode: Union[str, List[str]] = 'phase_polar',
        target_columns: Optional[List[str]] = None,
        target_level: str = 'base',
        sampling_strategy: str = 'snapshot',
        downsampling_stride: int = 16,
        target_position: Optional[int] = None,
        num_harmonics: int = 1
    ):
        """
        Args:
            dataframe: Предрассчитанный DataFrame (из test_precomputed.csv)
            indices: Список индексов начал окон (int) или кортежей (start, length)
            window_size: Размер окна
            feature_mode: Режим признаков ('raw', 'phase_polar', 'symmetric', 'symmetric_polar', 'phase_complex', 'power', 'alpha_beta')
            target_columns: Колонки меток (если None, используется target_level)
            target_level: 'base' (4 класса) или указанный список
            sampling_strategy: 'none', 'stride', 'snapshot'
            downsampling_stride: Шаг для режима 'stride'
            target_position: Позиция метки в окне (по умолчанию window_size - 1)
            num_harmonics: Количество гармоник в предрасчитанном датасете
        """
        if isinstance(dataframe, pl.LazyFrame):
            self.data = dataframe.collect()
        else:
            self.data = dataframe
        
        self.indices = indices
        self.window_size = window_size
        self.feature_mode = feature_mode if isinstance(feature_mode, list) else [feature_mode]
        self.sampling_strategy = sampling_strategy
        self.downsampling_stride = downsampling_stride
        self.num_harmonics = max(1, int(num_harmonics))
        
        # Определяем колонки меток
        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = get_target_columns(target_level)
        
        # Позиция метки
        self.target_position = target_position if target_position is not None else window_size - 1
        
        # Определяем колонки признаков с учётом гармоник
        self.feature_columns = []
        for fm in self.feature_mode:
            cols = self._build_feature_columns(fm, self.num_harmonics, legacy_symmetric=False)
            missing = set(cols) - set(self.data.columns)

            if missing and fm == 'symmetric' and self.num_harmonics == 1:
                legacy_cols = self._build_feature_columns(fm, self.num_harmonics, legacy_symmetric=True)
                legacy_missing = set(legacy_cols) - set(self.data.columns)
                if not legacy_missing:
                    print("    [PrecomputedDataset] Использую legacy-колонки symmetric (mag/angle) для h1")
                    cols = legacy_cols
                    missing = set()

            if missing:
                raise ValueError(f"Отсутствуют колонки в DataFrame: {missing}")

            self.feature_columns.extend(cols)
        
        # Определяем является ли режим спектральным (с warmup)
        self.is_spectral = any(fm != 'raw' for fm in self.feature_mode)
        
        # === ОПТИМИЗАЦИЯ: Предзагрузка всех данных в numpy массивы ===
        # Это критично для скорости — избегаем конвертации Polars→NumPy при каждом __getitem__
        self._preload_data()
    
    def _preload_data(self):
        """Предзагружает данные в numpy массивы для быстрого доступа."""
        import time
        start_t = time.perf_counter()
        
        # Признаки: (N, C) → транспонируем позже при извлечении
        self._features_np = self.data.select(self.feature_columns).to_numpy().astype(np.float32)
        self._features_np = np.nan_to_num(self._features_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Метки: (N, num_classes)
        self._targets_np = self.data.select(self.target_columns).to_numpy().astype(np.float32)
        
        elapsed = time.perf_counter() - start_t
        mem_mb = (self._features_np.nbytes + self._targets_np.nbytes) / (1024 * 1024)
        print(f"    [PrecomputedDataset] Предзагружено {len(self.indices):,} индексов, "
              f"память: {mem_mb:.1f} MB, время: {elapsed:.2f}s")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает (X, Y) для указанного индекса.
        
        X: Тензор признаков (Channels, Time) 
        Y: Тензор меток
        """
        # Получаем стартовый индекс
        index_item = self.indices[idx]
        
        if isinstance(index_item, (tuple, list)) and len(index_item) == 2:
            # Random Sampling Mode: (file_start_idx, file_length)
            file_start, file_len = index_item
            if file_len <= self.window_size:
                start_idx = file_start
            else:
                max_offset = file_len - self.window_size
                offset = np.random.randint(0, max_offset + 1)
                start_idx = file_start + offset
        else:
            start_idx = index_item
        
        # Извлекаем признаки из предзагруженного массива (Time, Channels)
        end_idx = start_idx + self.window_size
        x_data = self._features_np[start_idx:end_idx]  # (window_size, n_features)
        
        # Конвертируем в тензор (Channels, Time)
        x = torch.from_numpy(x_data.T.copy())  # .T даёт (Channels, Time), copy() для contiguous
        
        # Применяем sampling strategy
        x = self._apply_sampling(x)
        
        # Извлекаем метки
        target_idx = start_idx + self.target_position
        y = torch.from_numpy(self._targets_np[target_idx].copy())
        
        return x, y
    
    def _apply_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет стратегию прореживания к тензору.
        
        Args:
            x: Тензор (Channels, Time)
            
        Returns:
            Прореженный тензор
        """
        if self.sampling_strategy == 'none':
            return x
        
        elif self.sampling_strategy == 'stride':
            # Для спектральных признаков пропускаем warmup
            if self.is_spectral:
                if x.shape[1] > self.FFT_WARMUP:
                    x = x[:, self.FFT_WARMUP:]
            x = x[:, ::self.downsampling_stride]
            return x
        
        elif self.sampling_strategy == 'snapshot':
            # Для спектральных: точка 31 (первая валидная) и последняя
            if self.is_spectral:
                warmup_idx = self.FFT_WARMUP - 1  # 31
                last_idx = x.shape[1] - 1
                
                if last_idx > warmup_idx:
                    x = x[:, [warmup_idx, last_idx]]
                else:
                    # Fallback если окно слишком короткое
                    x = x[:, [warmup_idx, warmup_idx]]
            else:
                # Для raw: 32 точки начала + 32 конца
                window_len = 32
                if x.shape[1] >= window_len * 2:
                    head = x[:, :window_len]
                    tail = x[:, -window_len:]
                    x = torch.cat([head, tail], dim=1)
            return x
        
        return x
    
    @staticmethod
    def create_indices(
        df: pl.DataFrame,
        window_size: int,
        mode: str = 'val',
        stride: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> List[int]:
        """
        Создает индексы для Dataset.
        
        Для тестирования обычно используется mode='val' со скользящим окном.
        
        Args:
            df: DataFrame с данными
            window_size: Размер окна
            mode: 'val' или 'test' (sliding window)
            stride: Шаг окна (по умолчанию window_size // 2)
            min_length: Минимальная длина файла
            
        Returns:
            Список индексов начал окон
        """
        if 'file_name' not in df.columns:
            raise ValueError("DataFrame должен содержать колонку 'file_name'")
        
        if min_length is None:
            min_length = window_size
        if stride is None:
            stride = window_size // 2
        
        # Добавляем row_nr если нет
        if 'row_nr' not in df.columns:
            df = df.with_row_index("row_nr")
        
        # Группируем по файлам
        file_stats = df.group_by("file_name").agg([
            pl.col("row_nr").min().alias("start_idx"),
            pl.len().alias("length")
        ]).sort("start_idx")
        
        # Фильтруем короткие файлы
        valid_files = file_stats.filter(pl.col("length") >= min_length)
        
        indices = []
        
        for row in valid_files.iter_rows(named=True):
            start = row['start_idx']
            length = row['length']
            
            # Генерируем окна со скользящим шагом
            curr = 0
            while curr + window_size <= length:
                indices.append(start + curr)
                curr += stride
        
        return indices


def create_precomputed_dataset(
    data_dir: str,
    window_size: int = 320,
    feature_mode: str = 'phase_polar',
    sampling_strategy: str = 'snapshot',
    target_level: str = 'base',
    num_harmonics: int = 1
) -> PrecomputedDataset:
    """
    Утилитная функция для создания PrecomputedDataset из test_precomputed.csv.
    
    Args:
        data_dir: Путь к директории с датасетами
        window_size: Размер окна
        feature_mode: Режим признаков
        sampling_strategy: Стратегия прореживания
        target_level: Уровень меток
        num_harmonics: Количество гармоник
        
    Returns:
        Настроенный PrecomputedDataset
    """
    from pathlib import Path
    from osc_tools.data_management.dataset_manager import DatasetManager
    
    dm = DatasetManager(data_dir)
    
    # Убеждаемся что файл существует
    dm.create_precomputed_test_csv(num_harmonics=num_harmonics)
    
    # Загружаем данные
    df = dm.load_test_df(precomputed=True)
    
    # Создаём индексы
    indices = PrecomputedDataset.create_indices(
        df, window_size=window_size, mode='val'
    )
    
    # Создаём dataset
    dataset = PrecomputedDataset(
        dataframe=df,
        indices=indices,
        window_size=window_size,
        feature_mode=feature_mode,
        sampling_strategy=sampling_strategy,
        target_level=target_level,
        num_harmonics=num_harmonics
    )
    
    return dataset
