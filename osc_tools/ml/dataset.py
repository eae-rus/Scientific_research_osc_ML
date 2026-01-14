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
from osc_tools.features.polar import calculate_polar_features
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
        augmentation_config: Optional[dict] = None,
        downsampling_mode: str = 'none',
        downsampling_stride: int = 16,
        num_harmonics: int = 1,
        augment: bool = False,
        sampling_strategy: Optional[str] = None,
        target_level: str = 'full'
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
            downsampling_mode: Режим прореживания ('none', 'stride', 'snapshot').
            downsampling_stride: Шаг прореживания (для mode='stride').
            num_harmonics: Количество гармоник для спектрального анализа (feature_mode='symmetric' etc.).
            augment: Быстрое включение стандартной аугментации (если True, игнорирует augmentation_config).
            sampling_strategy: Алиас для downsampling_mode (для совместимости с конфигами экспериментов).
            target_level: Уровень детализации меток ('full', 'base_labels'). Влияет на output (Y).
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
        self.target_level = target_level
        self.physical_normalization = physical_normalization
        
        # Алиас обработка
        self.downsampling_mode = sampling_strategy if sampling_strategy is not None else downsampling_mode
        self.downsampling_stride = downsampling_stride
        self.num_harmonics = num_harmonics

        self.augmenter = None
        if augment:
             # Стандартная конфигурация аугментации (восстановлена "легкая" схема Фазы 2)
             # Инверсия, масштабирование, сдвиг фаз/времени. Шум отключен по умолчанию.
             default_aug_config = {
                 'p_inversion': 0.5,
                 'p_scaling': 0.3,
                 'scaling_range_current': (0.9, 1.1),
                 'scaling_range_voltage': (0.95, 1.05),
                 'p_phase_shuffling': 0.3, 
                 'p_noise': 0.0,
                 'p_drop_channel': 0.0,
                 'p_offset': 0.0
             }
             self.augmenter = TimeSeriesAugmenter(default_aug_config)
        elif augmentation_config:
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
            
        valid_feature_modes = [
            'raw', 'symmetric', 'complex_channels', 'power', 'instantaneous_power', 
            'alpha_beta', 'polar', 'symmetric_polar', 'phase_polar', 'phase_complex'
        ]
        for fm in self.feature_mode:
            if fm not in valid_feature_modes:
                raise ValueError(f"Unknown feature_mode: {fm}. Valid modes: {valid_feature_modes}")

    @staticmethod
    def create_indices(
        df: pl.DataFrame, 
        window_size: int, 
        mode: str = 'train', 
        stride: Optional[int] = None,
        min_length: Optional[int] = None,
        samples_per_file: int = 1
    ) -> List[Union[int, Tuple[int, int]]]:
        """
        Создает индексы для Dataset.
        
        Args:
            df: DataFrame с данными. Должен содержать колонку 'file_name'.
            window_size: Размер окна.
            mode: 'train' (random sampling) или 'val'/'test' (sliding window).
            stride: Шаг для sliding window (только для val/test). По умолчанию window_size // 2.
            min_length: Минимальная длина файла для включения. По умолчанию window_size.
            samples_per_file: Количество случайных окон из каждого файла за одну эпоху (только для train).
            
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
                # Клонируем записи о файле n раз, чтобы DataLoader прошелся по нему n раз за эпоху
                for _ in range(samples_per_file):
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
            # file_name уже строка - не нужно индексирование
            parts = file_name.split('_Bus ')
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

        # Вспомогательная функция для получения Un
        def get_un(name: str) -> np.ndarray:
            if is_valid_channel(name):
                return df[name].cast(pl.Float32).to_numpy()
            return np.zeros(len(df), dtype=np.float32)

        # 1. Фазные напряжения (Шины - Bus Bar) + Un
        candidates = ['UA BB', 'UB BB', 'UC BB']
        un_candidates = ['UN BB']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].cast(pl.Float32).to_numpy() for c in candidates]
            return voltages + [get_un(un_candidates[0])], 'phase'
            
        # 2. Фазные напряжения (Линия - Cable Line) + Un
        candidates = ['UA CL', 'UB CL', 'UC CL']
        un_candidates = ['UN CL']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].cast(pl.Float32).to_numpy() for c in candidates]
            return voltages + [get_un(un_candidates[0])], 'phase'
            
        # 3. Фазные напряжения (Простые) + Un
        candidates = ['UA', 'UB', 'UC']
        un_candidates = ['UN']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].cast(pl.Float32).to_numpy() for c in candidates]
            return voltages + [get_un(un_candidates[0])], 'phase'
            
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
            # print("DEBUG: Applying augmentation") # Закомментировано для чистоты вывода с tqdm
            raw_data = self.augmenter(raw_data)
            if isinstance(raw_data, torch.Tensor):
                raw_data = raw_data.numpy()

        collected_features = []
        fft_window = int(self.sampling_rate / 50)
        
        for fm in self.feature_mode:
            if fm == 'raw':
                if self.feature_columns is not None:
                    # Если заданы конкретные колонки, берем их из DataFrame (без аугментации пока что)
                    collected_features.append(sample_df.select(self.feature_columns).to_numpy())
                else:
                    collected_features.append(raw_data)
                
            elif fm == 'symmetric':
                # sliding_window_fft returns (Time, NumHarmonics)
                # Берем все запрошенные гармоники
                phasors_i = [sliding_window_fft(raw_data[:, i], fft_window, self.num_harmonics) for i in range(3)]
                i1, i2, i0 = calculate_symmetrical_components(*phasors_i)
                
                if not np.allclose(raw_data[:, 3], 0):
                    in_phasor = sliding_window_fft(raw_data[:, 3], fft_window, self.num_harmonics)
                    i0 = in_phasor / 3
                
                # Сглаживание гармоник в каналы if num_harmonics > 1
                # Форма i1: (Время, Гармоники)
                # Мы объединяем компоненты: I1, I2, I0
                # Полуитоговая форма перед сглаживанием: (Время, 3 компоненты, Гармоники)
                
                # Напряжения (каналы 4, 5, 6, 7)
                phasors_u = [sliding_window_fft(raw_data[:, i], fft_window, self.num_harmonics) for i in range(4, 7)]
                u1, u2, u0 = calculate_symmetrical_components(*phasors_u)
                
                if not np.allclose(raw_data[:, 7], 0):
                    un_phasor = sliding_window_fft(raw_data[:, 7], fft_window, self.num_harmonics)
                    u0 = un_phasor / 3

                # Сборка фичей
                # Для каждой компоненты (I1, I2... U0) берем Re и Im для всех гармоник
                # Итоговый порядок: 
                # [I1_h1_re, I1_h1_im, I1_h2_re, ... I2_h1..., I0..., U1..., U2..., U0...]
                
                components = [i1, i2, i0, u1, u2, u0] # List of (Time, Harmonics) arrays
                
                feature_list = []
                for comp in components:
                    # comp: (Time, Harmonics)
                    for h in range(self.num_harmonics):
                        feature_list.append(comp[:, h].real)
                        feature_list.append(comp[:, h].imag)
                        
                # Stack all features -> (Time, 12 * NumHarmonics)
                collected_features.append(np.stack(feature_list, axis=1))

            elif fm == 'symmetric_polar':
                # Расчет симметричных составляющих с переводом в полярные координаты
                # Токи (I1, I2, I0)
                phasors_i = [sliding_window_fft(raw_data[:, i], fft_window, self.num_harmonics) for i in range(3)]
                i1, i2, i0 = calculate_symmetrical_components(*phasors_i)
                
                if not np.allclose(raw_data[:, 3], 0):
                    in_phasor = sliding_window_fft(raw_data[:, 3], fft_window, self.num_harmonics)
                    i0 = in_phasor / 3
                
                # Напряжения (U1, U2, U0)
                phasors_u = [sliding_window_fft(raw_data[:, i], fft_window, self.num_harmonics) for i in range(4, 7)]
                u1, u2, u0 = calculate_symmetrical_components(*phasors_u)
                
                if not np.allclose(raw_data[:, 7], 0):
                    un_phasor = sliding_window_fft(raw_data[:, 7], fft_window, self.num_harmonics)
                    u0 = un_phasor / 3
                
                # Собираем все 6 комплексных компонент: (Time, 6, NumHarmonics)
                # stack axis=1 -> (Time, 6, Harmonics)
                complex_features = np.stack([i1, i2, i0, u1, u2, u0], axis=1)
                
                # Определяем Reference (UA или IA) - только по первой гармонике (фундаментальной)
                ua_phasor = phasors_u[0][:, 0] # (Time,)
                ia_phasor = phasors_i[0][:, 0] # (Time,)
                
                ua_mag = np.nanmean(np.abs(ua_phasor))
                if ua_mag > 1e-4:
                    ref_phasor = ua_phasor
                else:
                    ia_mag = np.nanmean(np.abs(ia_phasor))
                    if ia_mag > 1e-4:
                        ref_phasor = ia_phasor
                    else:
                        ref_phasor = None
                
                # Конвертация в полярные
                # Если num_harmonics > 1, нам нужно обработать каждую гармонику отдельно или вместе?
                # calculate_polar_features принимает (Time, Channels). 
                # У нас (Time, 6, Harmonics). Можно сделать reshape -> (Time, 6*Harmonics)
                
                time_steps, n_comps, n_harm = complex_features.shape
                complex_features_flat = complex_features.reshape(time_steps, n_comps * n_harm)
                
                # Конвертация в полярные
                polar_feats = calculate_polar_features(complex_features_flat, ref_phasor)
                collected_features.append(np.nan_to_num(polar_feats))

            elif fm == 'phase_polar':
                # Поблочный расчет для каждой фазы (IA, IB, IC, IN, UA, UB, UC, UN)
                # Собираем комплексные фазоры для всех 8 каналов
                all_phasors = []
                for i in range(8):
                    p = sliding_window_fft(raw_data[:, i], fft_window, self.num_harmonics)
                    all_phasors.append(p)
                
                # stack -> (Time, 8, Harmonics)
                complex_features = np.stack(all_phasors, axis=1)
                
                # Опорный сигнал для фазы (UA или IA)
                ua_phasor = all_phasors[4][:, 0] # UA_h1
                ia_phasor = all_phasors[0][:, 0] # IA_h1
                
                ua_mag = np.nanmean(np.abs(ua_phasor))
                ref_phasor = ua_phasor if ua_mag > 1e-4 else (ia_phasor if np.nanmean(np.abs(ia_phasor)) > 1e-4 else None)
                
                time_steps, n_signals, n_harm = complex_features.shape
                # Reshape to (Time, 8 * Harmonics) for calculate_polar_features
                complex_features_flat = complex_features.reshape(time_steps, n_signals * n_harm)
                polar_feats = calculate_polar_features(complex_features_flat, ref_phasor)
                
                collected_features.append(np.nan_to_num(polar_feats))

            elif fm == 'phase_complex':
                # Режим Re/Im (Rectangular) для всех 8 фаз (как в Фазе 2)
                feature_list = []
                for i in range(8):
                    # p: (Time, Harmonics)
                    p = sliding_window_fft(raw_data[:, i], fft_window, self.num_harmonics)
                    for h in range(self.num_harmonics):
                        feature_list.append(p[:, h].real)
                        feature_list.append(p[:, h].imag)
                
                # Stack all features -> (Time, 16 * NumHarmonics)
                collected_features.append(np.stack(feature_list, axis=1))

            elif fm == 'complex_channels':
                # Оставляем для обратной совместимости (1 гармоника, Re/Im, 8 каналов)
                # (Defining it inside for safety, though fft_window is already defined)
                pass
                
            elif fm == 'power':
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
                
                # Токи (0, 1, 2)
                a, b, c = raw_data[:, 0], raw_data[:, 1], raw_data[:, 2]
                alpha = (2/3) * (a - 0.5*b - 0.5*c)
                beta = (2/3) * (np.sqrt(3)/2 * (b - c))
                zero = (1/3) * (a + b + c)
                features.append(np.stack([alpha, beta, zero], axis=1))
                    
                # Напряжение (4, 5, 6)
                a, b, c = raw_data[:, 4], raw_data[:, 5], raw_data[:, 6]
                alpha = (2/3) * (a - 0.5*b - 0.5*c)
                beta = (2/3) * (np.sqrt(3)/2 * (b - c))
                zero = (1/3) * (a + b + c)
                features.append(np.stack([alpha, beta, zero], axis=1))
                
                data = np.concatenate(features, axis=1)
                collected_features.append(np.nan_to_num(data))

            elif fm == 'polar':
                # 1. Вычисляем фазоры для всех 8 каналов
                # [IA, IB, IC, In, UA, UB, UC, Un]
                phasors = []
                for i in range(8):
                    p = sliding_window_fft(raw_data[:, i], fft_window, 1)[:, 0]
                    phasors.append(p)
                
                phasors = np.stack(phasors, axis=1) # (Time, 8)
                
                # 2. Определяем опорный фазор (Reference Phasor)
                # Приоритет: UA (idx 4) -> UAB (если бы были линейные, но тут у нас уже фазные восстановленные) -> IA (idx 0)
                # В raw_data у нас всегда [IA, IB, IC, In, UA, UB, UC, Un]
                # Если UA валидный (не нули), берем его. Иначе IA.
                
                # Проверка на "валидность" UA (амплитуда > порога)
                # Берем среднюю амплитуду по окну (игнорируя NaN)
                ua_mag = np.nanmean(np.abs(phasors[:, 4]))
                
                if ua_mag > 1e-4: # т.к. напряжение в о.е.
                    ref_phasor = phasors[:, 4]
                else:
                    # Если напряжения нет, пробуем ток фазы А
                    ia_mag = np.nanmean(np.abs(phasors[:, 0]))
                    if ia_mag > 1e-4: # ток в о.е.
                        ref_phasor = phasors[:, 0]
                    else:
                        # Если и тока нет, то фаза 0 (абсолютная)
                        ref_phasor = None
                
                # 3. Расчет Magnitude/Angle
                polar_feats = calculate_polar_features(phasors, ref_phasor)
                collected_features.append(np.nan_to_num(polar_feats))

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
        
        # Прореживание (downsampling)
        is_spectral = any(m in ['symmetric', 'symmetric_polar', 'phase_polar', 'phase_complex', 'power', 'alpha_beta', 'polar'] for m in self.feature_mode)
        is_raw = 'raw' in self.feature_mode and len(self.feature_mode) == 1
        
        if self.downsampling_mode == 'stride':
             # Для спектральных/Фурье фичей пропускаем "разгон" FFT (первые 32 точки)
             # Т.к. там нули/NaN
             if is_spectral:
                 fft_warmup = 32
                 if x.shape[1] > fft_warmup:
                      x = x[:, fft_warmup:]
                      
             x = x[:, ::self.downsampling_stride]
             
        elif self.downsampling_mode == 'snapshot':
             if is_raw:
                 # Для сырых данных берем окна (начало + конец) вместо точек
                 # 32 точки = 1 период пром. частоты (20 мс)
                 window_len = 32
                 if x.shape[1] >= window_len * 2:
                     head = x[:, :window_len]
                     tail = x[:, -window_len:]
                     x = torch.cat([head, tail], dim=1) # (Channels, 64)
                 elif x.shape[1] >= 2:
                     # Fallback для коротких окон: просто начало и конец пополам
                     mid = x.shape[1] // 2
                     head = x[:, :mid]
                     tail = x[:, mid:] # чтобы сумма длин была равна исходной? Нет, лучше фиксированный размер.
                     # Если меньше 64 точек, просто возвращаем все?
                     # Или ресайзим? Пока оставим как есть - вернем все x
                     pass
             else:
                 # Для спектральных данных
                 # Берем первую валидную точку (индекс 31, т.к. окно 32) и последнюю
                 fft_warmup = 31
                 if x.shape[1] > fft_warmup:
                     last_idx = x.shape[1] - 1
                     if last_idx > fft_warmup:
                         x = x[:, [fft_warmup, last_idx]]
                     else:
                         # Если длина всего 32, то 31-й индекс это и есть последняя
                         x = x[:, [fft_warmup, fft_warmup]]
                 else:
                     # Fallback (если вдруг окно меньше 32)
                     if x.shape[1] >= 2:
                         x = x[:, [0, -1]]
                     else:
                         x = x.repeat(1, 2)[:, :2]

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
