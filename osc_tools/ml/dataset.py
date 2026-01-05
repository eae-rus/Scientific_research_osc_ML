import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from typing import List, Union, Optional, Tuple
from osc_tools.features.pdr_calculator import sliding_window_fft
from osc_tools.features.phasor import (
    calculate_symmetrical_components, 
    calculate_power, 
    calculate_symmetrical_components_from_line
)
from osc_tools.ml.augmentation import TimeSeriesAugmenter

class OscillogramDataset(Dataset):
    """
    Универсальный Dataset для осциллограмм.
    Поддерживает режимы классификации, сегментации и реконструкции.
    Поддерживает различные режимы формирования признаков (feature_mode).
    Поддерживает Polars DataFrame.
    """
    def __init__(
        self, 
        dataframe: Union[pl.DataFrame, pl.LazyFrame], 
        indices: Union[pl.DataFrame, List[int], np.ndarray], 
        window_size: int, 
        mode: str = 'classification',
        feature_mode: Union[str, List[str]] = 'raw',
        sampling_rate: int = 1600,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        target_position: Optional[int] = None,
        physical_normalization: bool = False,
        norm_coef_path: Optional[str] = None,
        augmentation_config: Optional[dict] = None
    ):
        """
        Args:
            dataframe: Исходный DataFrame с данными (Polars).
            indices: DataFrame или список индексов начал окон.
            window_size: Размер окна (количество отсчетов).
            mode: Режим работы ('classification', 'segmentation', 'reconstruction').
            feature_mode: Режим признаков ('raw', 'symmetric', 'complex_channels', 'power'). 
                          Может быть списком.
            sampling_rate: Частота дискретизации (нужна для feature_mode != 'raw').
            feature_columns: Список колонок для входа (X). Если None, используется умный селектор (стандартизация).
            target_columns: Колонка(и) для выхода (Y).
            target_position: Позиция целевого значения внутри окна (для classification). 
                             По умолчанию - последнее значение (window_size - 1).
            physical_normalization: Применять ли физическую нормализацию по коэффициентам.
            norm_coef_path: Путь к CSV файлу с коэффициентами нормализации.
            augmentation_config: Конфигурация аугментации (только для mode='classification'/'segmentation' в train).
        """
        if isinstance(dataframe, pl.LazyFrame):
            self.data = dataframe.collect()
        else:
            self.data = dataframe
            
        self.indices = indices
        self.window_size = window_size
        self.mode = mode
        self.feature_mode = feature_mode if isinstance(feature_mode, list) else [feature_mode]
        self.sampling_rate = sampling_rate
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.physical_normalization = physical_normalization
        
        self.augmenter = None
        if augmentation_config:
            self.augmenter = TimeSeriesAugmenter(augmentation_config)
        
        self.norm_coef_df = None
        if self.physical_normalization:
            if norm_coef_path is None:
                raise ValueError("norm_coef_path must be provided if physical_normalization is True")
            try:
                self.norm_coef_df = pl.read_csv(norm_coef_path, infer_schema_length=100000)
            except Exception as e:
                print(f"Предупреждение: не удалось загрузить коэффициенты нормализации из {norm_coef_path}: {e}")
        
        if target_position is not None:
            self.target_position = target_position
        else:
            self.target_position = window_size - 1

        # Проверка режима
        valid_modes = ['classification', 'segmentation', 'reconstruction']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")
            
        valid_feature_modes = ['raw', 'symmetric', 'complex_channels', 'power', 'instantaneous_power', 'alpha_beta']
        for fm in self.feature_mode:
            if fm not in valid_feature_modes:
                raise ValueError(f"Unknown feature_mode: {fm}. Valid modes: {valid_feature_modes}")

    @staticmethod
    def create_indices(
        df: pl.DataFrame, 
        window_size: int, 
        mode: str = 'train', 
        stride: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> List[Union[int, Tuple[int, int]]]:
        """
        Создает индексы для Dataset.
        
        Args:
            df: DataFrame с данными. Должен содержать колонку 'file_name'.
            window_size: Размер окна.
            mode: 'train' (random sampling) или 'val'/'test' (sliding window).
            stride: Шаг для sliding window (только для val/test). По умолчанию window_size // 2.
            min_length: Минимальная длина файла для включения. По умолчанию window_size.
            
        Returns:
            Список индексов (int для val, tuple для train).
        """
        if 'file_name' not in df.columns:
            raise ValueError("DataFrame must contain 'file_name' column to generate indices.")
            
        if min_length is None:
            min_length = window_size
            
        if stride is None:
            stride = window_size // 2

        # Добавляем номера строк, если их нет
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
        
        if mode == 'train':
            # Для обучения возвращаем кортежи (start, length)
            # Dataset сам выберет случайное окно внутри
            for row in valid_files.iter_rows(named=True):
                indices.append((row['start_idx'], row['length']))
                
        else: # val, test
            # Для валидации генерируем фиксированные окна с шагом
            for row in valid_files.iter_rows(named=True):
                start = row['start_idx']
                length = row['length']
                
                # Генерируем окна: start, start+stride, ..., пока end <= start+length
                curr = 0
                while curr + window_size <= length:
                    indices.append(start + curr)
                    curr += stride
                    
                # Опционально: добавить последнее окно, если оно не покрыто?
                # Пока оставим только полные окна
                
        return indices

    def __len__(self):
        return len(self.indices)

    def _apply_physical_normalization(self, df: pl.DataFrame, file_name: str) -> pl.DataFrame:
        """
        Применяет физическую нормализацию к DataFrame на основе коэффициентов.
        """
        if self.norm_coef_df is None:
            return df
            
        # Парсинг имени файла: hash_Bus X -> hash, X
        # Пример: 00d4242f4fa66c50a89a7fc565f8ea58_Bus 1
        try:
            parts = file_name[0].split('_Bus ')
            if len(parts) != 2:
                return df
            file_hash = parts[0]
            bus_num = int(parts[1].split('_')[0]) # На случай если там еще что-то есть
        except Exception:
            return df
            
        # Поиск строки в коэффициентах
        norm_row = self.norm_coef_df.filter(pl.col("name") == file_hash)
        if norm_row.is_empty():
            return df
            
        # Проверка флага norm (YES...)
        norm_val = str(norm_row.get_column("norm")[0])
        if "YES" not in norm_val:
            return df
            
        new_cols = {}
        
        # Предварительно соберем списки колонок для ускорения
        ip_cols = ['IA', 'IB', 'IC']
        iz_cols = ['IN']
        ub_cols = [c for c in df.columns if 'BB' in c]
        uc_cols = [c for c in df.columns if 'CL' in c]
        
        # 1. Фазные токи (Ip)
        col_ip = f"{bus_num}Ip_base"
        if col_ip in norm_row.columns:
            val = norm_row.get_column(col_ip)[0]
            if val is not None:
                nominal = 20 * float(val)
                if nominal > 0:
                    for c in ip_cols:
                        if c in df.columns:
                            new_cols[c] = pl.col(c).cast(pl.Float32) / nominal

        # 2. Ток нулевой последовательности (Iz)
        col_iz = f"{bus_num}Iz_base"
        if col_iz in norm_row.columns:
            val = norm_row.get_column(col_iz)[0]
            if val is not None:
                nominal = 5 * float(val)
                if nominal > 0:
                    for c in iz_cols:
                        if c in df.columns:
                            new_cols[c] = pl.col(c).cast(pl.Float32) / nominal

        # 3. Напряжения СШ (Ub) -> * BB
        col_ub = f"{bus_num}Ub_base"
        if col_ub in norm_row.columns:
            val = norm_row.get_column(col_ub)[0]
            if val is not None:
                nominal = 3 * float(val)
                if nominal > 0:
                    for c in ub_cols:
                        new_cols[c] = pl.col(c).cast(pl.Float32) / nominal

        # 4. Напряжения КЛ (Uc) -> * CL
        col_uc = f"{bus_num}Uc_base"
        if col_uc in norm_row.columns:
            val = norm_row.get_column(col_uc)[0]
            if val is not None:
                nominal = 3 * float(val)
                if nominal > 0:
                    for c in uc_cols:
                        new_cols[c] = pl.col(c).cast(pl.Float32) / nominal

        if new_cols:
            return df.with_columns(**new_cols)
            
        return df

    def _get_best_voltage_channels(self, df: pl.DataFrame) -> Tuple[List[np.ndarray], str]:
        """
        Возвращает лучшие доступные каналы напряжения и их тип ('phase' или 'line').
        Проверяет не только наличие колонок, но и наличие валидных данных (не все нули/NaN).
        Включает нулевое напряжение Un.
        """
        cols = df.columns
        
        def is_valid_channel(name: str) -> bool:
            if name not in cols:
                return False
            # Проверка: есть ли в канале значения, отличные от 0 (с учетом шума)
            # Берем numpy array для скорости
            try:
                data = df[name].cast(pl.Float32).to_numpy()
            except Exception:
                return False
            
            # Простая проверка: если max(abs) < epsilon=1e-4, считаем канал пустым
            if np.max(np.abs(np.nan_to_num(data))) < 1e-4:
                return False
            return True

        # 1. Фазные напряжения (Шины - Bus Bar) + Un
        candidates = ['UA BB', 'UB BB', 'UC BB']
        un_candidates = ['UN BB']
        if all(is_valid_channel(c) for c in candidates) and is_valid_channel(un_candidates[0]):
            return [df[c].cast(pl.Float32).to_numpy() for c in candidates + un_candidates], 'phase'
            
        # 2. Фазные напряжения (Линия - Cable Line) + Un
        candidates = ['UA CL', 'UB CL', 'UC CL']
        un_candidates = ['UN CL']
        if all(is_valid_channel(c) for c in candidates) and is_valid_channel(un_candidates[0]):
            return [df[c].cast(pl.Float32).to_numpy() for c in candidates + un_candidates], 'phase'
            
        # 3. Фазные напряжения (Простые) + Un
        candidates = ['UA', 'UB', 'UC']
        un_candidates = ['UN']
        if all(is_valid_channel(c) for c in candidates) and is_valid_channel(un_candidates[0]):
            return [df[c].cast(pl.Float32).to_numpy() for c in candidates + un_candidates], 'phase'
            
        # 4. Линейные напряжения (Шины - Bus Bar) + Un
        candidates = ['UAB BB', 'UBC BB', 'UCA BB']
        un_candidates = ['UN BB']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].cast(pl.Float32).to_numpy() for c in candidates]
            un_data = df[un_candidates[0]].cast(pl.Float32).to_numpy() if is_valid_channel(un_candidates[0]) else np.zeros(len(df), dtype=np.float32)
            return voltages + [un_data], 'line'
        
        # 5. Линейные напряжения (Линия - Cable Line) + Un
        candidates = ['UAB CL', 'UBC CL', 'UCA CL']
        un_candidates = ['UN CL']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].cast(pl.Float32).to_numpy() for c in candidates]
            un_data = df[un_candidates[0]].cast(pl.Float32).to_numpy() if is_valid_channel(un_candidates[0]) else np.zeros(len(df), dtype=np.float32)
            return voltages + [un_data], 'line'
        
        # 6. Линейные напряжения + Un
        candidates = ['UAB', 'UBC', 'UCA']
        un_candidates = ['UN']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].cast(pl.Float32).to_numpy() for c in candidates]
            un_data = df[un_candidates[0]].cast(pl.Float32).to_numpy() if is_valid_channel(un_candidates[0]) else np.zeros(len(df), dtype=np.float32)
            return voltages + [un_data], 'line'
             
        return [], 'none'

    def _get_best_current_channels(self, df: pl.DataFrame) -> List[np.ndarray]:
        """
        Возвращает каналы токов [IA, IB, IC, In].
        Если IB отсутствует или пустой, но есть IA и IC, восстанавливает IB = -IA - IC.
        In добавляется, если присутствует.
        """
        cols = df.columns
        
        def get_data_if_valid(name: str) -> Optional[np.ndarray]:
            if name not in cols:
                return None
            try:
                data = df[name].cast(pl.Float32).to_numpy()
            except Exception:
                return None
            if np.max(np.abs(np.nan_to_num(data))) < 1e-6:
                return None
            return data

        ia = get_data_if_valid('IA')
        ib = get_data_if_valid('IB')
        ic = get_data_if_valid('IC')
        in_ = get_data_if_valid('IN')  # Нулевой ток
        
        if ia is not None and ic is not None:
            if ib is None:
                # Восстановление IB (без учёта In)
                ib = -ia - ic
            if in_ is None:
                in_ = np.zeros(len(df), dtype=np.float32)
            return [ia, ib, ic, in_]
            
        # Если нет IA или IC, то полноценной тройки нет
        return []

    def _get_standardized_raw_data(self, df: pl.DataFrame) -> np.ndarray:
        """
        Возвращает стандартизированный набор сигналов [IA, IB, IC, In, UA, UB, UC, Un].
        Если фазные напряжения отсутствуют, пытается восстановить из линейных.
        Если сигналы отсутствуют, заполняет нулями.
        """
        # Токи
        i_channels = self._get_best_current_channels(df)
        if not i_channels:
            i_channels = [np.zeros(len(df), dtype=np.float32) for _ in range(4)]  # IA, IB, IC, In
            
        # Напряжения
        u_channels, u_type = self._get_best_voltage_channels(df)
        
        if u_type == 'line':
            # Восстановление фазных из линейных
            # (предполагая U0=0, а если Un есть - то учитывается отдельно)
            uab, ubc, uca, un = u_channels
            ua = (2 * uab + ubc) / 3
            ub = (2 * ubc + uca) / 3
            uc = (2 * uca + uab) / 3
            u_channels = [ua, ub, uc, un]
        elif u_type == 'none':
            u_channels = [np.zeros(len(df), dtype=np.float32) for _ in range(4)]  # UA, UB, UC, Un
            
        # Объединение: IA, IB, IC, In, UA, UB, UC, Un
        all_channels = i_channels + u_channels
        return np.stack(all_channels, axis=1) # (Time, 8)

    def __getitem__(self, idx):
        # Получаем стартовый индекс
        if isinstance(self.indices, pl.DataFrame):
            # Если `indices` передан как DataFrame — ожидаем одну колонку со стартовыми индексами окон
            start_idx = self.indices.row(idx)[0]
        else:
            index_item = self.indices[idx]
            
            if isinstance(index_item, (tuple, list)) and len(index_item) == 2:
                # Random Sampling Mode: (file_start_idx, file_length)
                # Используется для обучения: выбираем случайное окно внутри файла
                file_start, file_len = index_item
                
                if file_len <= self.window_size:
                    start_idx = file_start
                else:
                    # Random offset
                    max_offset = file_len - self.window_size
                    offset = np.random.randint(0, max_offset + 1)
                    start_idx = file_start + offset
            else:
                # Fixed Window Mode: index_item is start_idx
                # Используется для валидации/теста: фиксированное окно
                start_idx = index_item

        # Проверка на выход за границы
        if start_idx + self.window_size > len(self.data):
            raise IndexError(f"Index {idx} (start_idx={start_idx}) + window_size {self.window_size} > len(data) {len(self.data)}")

        # Извлечение окна данных
        # Polars slicing (эффективно)
        sample_df = self.data.slice(start_idx, self.window_size)
        
        # Физическая нормализация (если включена)
        if self.physical_normalization and 'file_name' in sample_df.columns:
            try:
                file_name = sample_df['file_name'][0]
                sample_df = self._apply_physical_normalization(sample_df, file_name)
            except Exception:
                pass # Fallback to raw data if something goes wrong
        
        # 1. Получение стандартизированных сырых данных (8 каналов)
        # [IA, IB, IC, In, UA, UB, UC, Un]
        # Это база для всех остальных признаков и аугментации
        raw_data = self._get_standardized_raw_data(sample_df)
        
        # 2. Аугментация (если включена)
        if self.augmenter:
            print("DEBUG: Applying augmentation")
            raw_data = self.augmenter(raw_data)
            if isinstance(raw_data, torch.Tensor):
                raw_data = raw_data.numpy()

        collected_features = []
        
        for fm in self.feature_mode:
            if fm == 'raw':
                if self.feature_columns is not None:
                    # Если заданы конкретные колонки, берем их из DataFrame (без аугментации пока что)
                    collected_features.append(sample_df.select(self.feature_columns).to_numpy())
                else:
                    collected_features.append(raw_data)
                
            elif fm == 'symmetric':
                # Расчет симметричных составляющих из raw_data
                fft_window = int(self.sampling_rate / 50)
                
                # Токи (каналы 0, 1, 2, 3)
                phasors_i = [sliding_window_fft(raw_data[:, i], fft_window, 1)[:, 0] for i in range(3)]
                i1, i2, i0 = calculate_symmetrical_components(*phasors_i)
                
                # Если In (канал 3) не пустой, используем его для I0
                if not np.allclose(raw_data[:, 3], 0):
                    in_phasor = sliding_window_fft(raw_data[:, 3], fft_window, 1)[:, 0]
                    i0 = in_phasor / 3
                
                features_i = np.stack([i1.real, i1.imag, i2.real, i2.imag, i0.real, i0.imag], axis=1)
                
                # Напряжения (каналы 4, 5, 6, 7)
                phasors_u = [sliding_window_fft(raw_data[:, i], fft_window, 1)[:, 0] for i in range(4, 7)]
                u1, u2, u0 = calculate_symmetrical_components(*phasors_u)
                
                # Если Un (канал 7) не пустой, используем его для U0
                if not np.allclose(raw_data[:, 7], 0):
                    un_phasor = sliding_window_fft(raw_data[:, 7], fft_window, 1)[:, 0]
                    u0 = un_phasor / 3
                    
                features_u = np.stack([u1.real, u1.imag, u2.real, u2.imag, u0.real, u0.imag], axis=1)
                
                collected_features.append(np.concatenate([features_i, features_u], axis=1))

            elif fm == 'complex_channels':
                fft_window = int(self.sampling_rate / 50)
                features = []
                # Проходим по всем 8 каналам
                for i in range(8):
                    phasor = sliding_window_fft(raw_data[:, i], fft_window, 1)[:, 0]
                    features.append(np.stack([phasor.real, phasor.imag], axis=1))
                
                data = np.concatenate(features, axis=1)
                collected_features.append(np.nan_to_num(data))
                
            elif fm == 'power':
                fft_window = int(self.sampling_rate / 50)
                features = []
                
                # Пары (IA, UA), (IB, UB), (IC, UC), (In, Un)
                # Indices: (0, 4), (1, 5), (2, 6), (3, 7)
                for i_idx, u_idx in zip(range(4), range(4, 8)):
                    i_phasor = sliding_window_fft(raw_data[:, i_idx], fft_window, 1)[:, 0]
                    u_phasor = sliding_window_fft(raw_data[:, u_idx], fft_window, 1)[:, 0]
                    _, p_act, q_react = calculate_power(u_phasor, i_phasor)
                    features.append(np.stack([p_act, q_react], axis=1))
                
                data = np.concatenate(features, axis=1)
                collected_features.append(np.nan_to_num(data))

            elif fm == 'instantaneous_power':
                features = []
                # Пары (IA, UA), (IB, UB), (IC, UC), (In, Un)
                for i_idx, u_idx in zip(range(4), range(4, 8)):
                    p_inst = raw_data[:, u_idx] * raw_data[:, i_idx]
                    features.append(p_inst[:, None])
                
                data = np.concatenate(features, axis=1)
                collected_features.append(np.nan_to_num(data))

            elif fm == 'alpha_beta':
                features = []
                
                # Currents (0, 1, 2)
                a, b, c = raw_data[:, 0], raw_data[:, 1], raw_data[:, 2]
                alpha = (2/3) * (a - 0.5*b - 0.5*c)
                beta = (2/3) * (np.sqrt(3)/2 * (b - c))
                zero = (1/3) * (a + b + c)
                features.append(np.stack([alpha, beta, zero], axis=1))
                    
                # Voltages (4, 5, 6)
                a, b, c = raw_data[:, 4], raw_data[:, 5], raw_data[:, 6]
                alpha = (2/3) * (a - 0.5*b - 0.5*c)
                beta = (2/3) * (np.sqrt(3)/2 * (b - c))
                zero = (1/3) * (a + b + c)
                features.append(np.stack([alpha, beta, zero], axis=1))
                
                data = np.concatenate(features, axis=1)
                collected_features.append(np.nan_to_num(data))

        if not collected_features:
            # Fallback
            if self.feature_columns:
                x_data = sample_df.select(self.feature_columns).to_numpy()
            else:
                x_data = np.zeros((self.window_size, 1), dtype=np.float32) # Should not happen

        else:
             x_data = np.concatenate(collected_features, axis=1)

        # Ensure x_data is float32 (handle object type if any)
        if x_data.dtype == object:
            x_data = x_data.astype(np.float32)
            
        # Handle NaNs and Infs
        x_data = np.nan_to_num(x_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to tensor and transpose to (Channels, Time)
        x = torch.tensor(x_data, dtype=torch.float32).transpose(0, 1)

        # Извлечение целевой переменной (Y)
        y = None
        if self.mode == 'classification':
            if self.target_columns:
                # Polars indexing
                target_idx = start_idx + self.target_position
                # row() returns tuple, we need specific columns
                # select().row() is safer
                if isinstance(self.target_columns, list):
                    y_val = self.data.select(self.target_columns).row(target_idx)
                    y = torch.tensor(y_val, dtype=torch.float32)
                else:
                    y_val = self.data.select(self.target_columns).row(target_idx)[0]
                    y = torch.tensor(y_val, dtype=torch.long)
                    
        elif self.mode == 'segmentation':
            if self.target_columns:
                y_data = sample_df.select(self.target_columns).to_numpy()
                y = torch.tensor(y_data, dtype=torch.long)
                if isinstance(self.target_columns, str):
                    y = y.squeeze(-1)
                
        elif self.mode == 'reconstruction':
            y = x.clone()

        return x, y
