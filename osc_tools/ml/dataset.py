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
        target_position: Optional[int] = None
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

    def __len__(self):
        return len(self.indices)

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
            data = df[name].to_numpy()
            # Простая проверка: если max(abs) < epsilon, считаем канал пустым
            if np.max(np.abs(np.nan_to_num(data))) < 1e-4:
                return False
            return True

        # 1. Фазные напряжения (Шины - Bus Bar) + Un
        candidates = ['UA BB', 'UB BB', 'UC BB']
        un_candidates = ['UN BB']
        if all(is_valid_channel(c) for c in candidates) and is_valid_channel(un_candidates[0]):
            return [df[c].to_numpy() for c in candidates + un_candidates], 'phase'
            
        # 2. Фазные напряжения (Линия - Cable Line) + Un
        candidates = ['UA CL', 'UB CL', 'UC CL']
        un_candidates = ['UN CL']
        if all(is_valid_channel(c) for c in candidates) and is_valid_channel(un_candidates[0]):
            return [df[c].to_numpy() for c in candidates + un_candidates], 'phase'
            
        # 3. Фазные напряжения (Простые) + Un
        candidates = ['UA', 'UB', 'UC']
        un_candidates = ['UN']
        if all(is_valid_channel(c) for c in candidates) and is_valid_channel(un_candidates[0]):
            return [df[c].to_numpy() for c in candidates + un_candidates], 'phase'
            
        # 4. Линейные напряжения (Шины - Bus Bar) + Un
        candidates = ['UAB BB', 'UBC BB', 'UCA BB']
        un_candidates = ['UN BB']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].to_numpy() for c in candidates]
            un_data = df[un_candidates[0]].to_numpy() if is_valid_channel(un_candidates[0]) else np.zeros(len(df), dtype=np.float32)
            return voltages + [un_data], 'line'
        
        # 5. Линейные напряжения (Линия - Cable Line) + Un
        candidates = ['UAB CL', 'UBC CL', 'UCA CL']
        un_candidates = ['UN CL']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].to_numpy() for c in candidates]
            un_data = df[un_candidates[0]].to_numpy() if is_valid_channel(un_candidates[0]) else np.zeros(len(df), dtype=np.float32)
            return voltages + [un_data], 'line'
        
        # 6. Линейные напряжения + Un
        candidates = ['UAB', 'UBC', 'UCA']
        un_candidates = ['UN']
        if all(is_valid_channel(c) for c in candidates):
            voltages = [df[c].to_numpy() for c in candidates]
            un_data = df[un_candidates[0]].to_numpy() if is_valid_channel(un_candidates[0]) else np.zeros(len(df), dtype=np.float32)
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
            data = df[name].to_numpy()
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
            start_idx = self.indices[idx]

        # Проверка на выход за границы
        if start_idx + self.window_size > len(self.data):
            raise IndexError(f"Index {idx} (start_idx={start_idx}) + window_size {self.window_size} > len(data) {len(self.data)}")

        # Извлечение окна данных
        # Polars slicing (эффективно)
        sample_df = self.data.slice(start_idx, self.window_size)
        
        collected_features = []
        
        for fm in self.feature_mode:
            if fm == 'raw':
                if self.feature_columns is not None:
                    collected_features.append(sample_df.select(self.feature_columns).to_numpy())
                else:
                    # Умный выбор / Стандартизация
                    collected_features.append(self._get_standardized_raw_data(sample_df))
                
            elif fm == 'symmetric':
                # Расчет симметричных составляющих
                fft_window = int(self.sampling_rate / 50)
                features = []
                
                i_phases = self._get_best_current_channels(sample_df)
                if len(i_phases) >= 3:  # IA, IB, IC минимум
                    phasors = [sliding_window_fft(p, fft_window, 1)[:, 0] for p in i_phases[:3]]  # IA, IB, IC
                    i1, i2, i0 = calculate_symmetrical_components(*phasors)
                    if len(i_phases) >= 4 and i_phases[3] is not None:  # In присутствует
                        in_phasor = sliding_window_fft(i_phases[3], fft_window, 1)[:, 0]
                        i0 = in_phasor / 3  # I0 = In / 3
                    features.append(np.stack([i1.real, i1.imag, i2.real, i2.imag, i0.real, i0.imag], axis=1))
                else:
                    # Нули, если нет токов
                    features.append(np.zeros((self.window_size, 6), dtype=np.float32))
                    
                u_phases, u_type = self._get_best_voltage_channels(sample_df)
                
                if u_type == 'phase' and len(u_phases) >= 3:
                    phasors = [sliding_window_fft(p, fft_window, 1)[:, 0] for p in u_phases[:3]]  # UA, UB, UC
                    u1, u2, u0 = calculate_symmetrical_components(*phasors)
                    if len(u_phases) >= 4 and u_phases[3] is not None:  # Un присутствует
                        un_phasor = sliding_window_fft(u_phases[3], fft_window, 1)[:, 0]
                        u0 = un_phasor / 3  # U0 = Un / 3
                    features.append(np.stack([u1.real, u1.imag, u2.real, u2.imag, u0.real, u0.imag], axis=1))
                elif u_type == 'line' and len(u_phases) >= 3:
                    phasors = [sliding_window_fft(p, fft_window, 1)[:, 0] for p in u_phases[:3]]  # UAB, UBC, UCA
                    u1, u2 = calculate_symmetrical_components_from_line(*phasors)
                    u0 = np.zeros_like(u1)
                    if len(u_phases) >= 4 and u_phases[3] is not None:  # Un присутствует
                        un_phasor = sliding_window_fft(u_phases[3], fft_window, 1)[:, 0]
                        u0 = un_phasor / 3
                    features.append(np.stack([u1.real, u1.imag, u2.real, u2.imag, u0.real, u0.imag], axis=1))
                else:
                    # Нули, если нет напряжений
                    features.append(np.zeros((self.window_size, 6), dtype=np.float32))
                
                if features:
                    data = np.concatenate(features, axis=1)
                    collected_features.append(np.nan_to_num(data))

            elif fm == 'complex_channels':
                fft_window = int(self.sampling_rate / 50)
                features = []
                
                # Токи
                i_phases = self._get_best_current_channels(sample_df)
                if i_phases:
                    for p in i_phases:
                        phasor = sliding_window_fft(p, fft_window, 1)[:, 0]
                        features.append(np.stack([phasor.real, phasor.imag], axis=1))
                
                # Напряжения
                u_phases, u_type = self._get_best_voltage_channels(sample_df)
                if u_type == 'line':
                     # Восстановление фазных напряжений из линейных
                     uab, ubc, uca, un = u_phases[:4] if len(u_phases) >= 4 else u_phases + [None]
                     ua = (2 * uab + ubc) / 3 if uab is not None and ubc is not None else None
                     ub = (2 * ubc + uca) / 3 if ubc is not None and uca is not None else None
                     uc = (2 * uca + uab) / 3 if uca is not None and uab is not None else None
                     u_phases = [ua, ub, uc, un]
                
                if u_phases:
                    for p in u_phases:
                        # Расчет Фурье-фазора для каждого канала
                        if p is not None:
                            phasor = sliding_window_fft(p, fft_window, 1)[:, 0]
                            features.append(np.stack([phasor.real, phasor.imag], axis=1))
                        else:
                            features.append(np.zeros((self.window_size, 2), dtype=np.float32))
                
                if features:
                    data = np.concatenate(features, axis=1)
                    collected_features.append(np.nan_to_num(data))
                else:
                    if self.feature_columns:
                        collected_features.append(sample_df.select(self.feature_columns).to_numpy())
                    else:
                        # 8 каналов * 2 (re/im)
                        collected_features.append(np.zeros((self.window_size, 16), dtype=np.float32))
            elif fm == 'power':
                fft_window = int(self.sampling_rate / 50)
                features = []
                
                i_phases = self._get_best_current_channels(sample_df)
                u_phases, u_type = self._get_best_voltage_channels(sample_df)
                
                if u_type == 'line':
                     uab, ubc, uca, un = u_phases[:4] if len(u_phases) >= 4 else u_phases + [None]
                     ua = (2 * uab + ubc) / 3 if uab is not None and ubc is not None else None
                     ub = (2 * ubc + uca) / 3 if ubc is not None and uca is not None else None
                     uc = (2 * uca + uab) / 3 if uca is not None and uab is not None else None
                     u_phases = [ua, ub, uc, un]
                
                if i_phases and u_phases:
                    for i_p, u_p in zip(i_phases, u_phases):
                        if i_p is not None and u_p is not None:
                            i_phasor = sliding_window_fft(i_p, fft_window, 1)[:, 0]
                            u_phasor = sliding_window_fft(u_p, fft_window, 1)[:, 0]
                            _, p_act, q_react = calculate_power(u_phasor, i_phasor)
                            features.append(np.stack([p_act, q_react], axis=1))
                        else:
                            features.append(np.zeros((self.window_size, 2), dtype=np.float32))
                
                if features:
                    data = np.concatenate(features, axis=1)
                    collected_features.append(np.nan_to_num(data))
                else:
                    # Fallback
                    if self.feature_columns:
                        collected_features.append(sample_df.select(self.feature_columns).to_numpy())
                    else:
                        # 4 фазы * 2 (P, Q)
                        collected_features.append(np.zeros((self.window_size, 8), dtype=np.float32))

            elif fm == 'instantaneous_power':
                features = []
                i_phases = self._get_best_current_channels(sample_df)
                u_phases, u_type = self._get_best_voltage_channels(sample_df)
                
                if u_type == 'line':
                     uab, ubc, uca, un = u_phases[:4] if len(u_phases) >= 4 else u_phases + [None]
                     ua = (2 * uab + ubc) / 3 if uab is not None and ubc is not None else None
                     ub = (2 * ubc + uca) / 3 if ubc is not None and uca is not None else None
                     uc = (2 * uca + uab) / 3 if uca is not None and uab is not None else None
                     u_phases = [ua, ub, uc, un]
                
                if i_phases and u_phases:
                    for i_p, u_p in zip(i_phases, u_phases):
                        if i_p is not None and u_p is not None:
                            # p(t) = u(t) * i(t)
                            p_inst = u_p * i_p
                            features.append(p_inst[:, None]) # (T, 1)
                        else:
                            features.append(np.zeros((self.window_size, 1), dtype=np.float32))
                
                if features:
                    data = np.concatenate(features, axis=1)
                    collected_features.append(np.nan_to_num(data))
                else:
                    if self.feature_columns:
                        collected_features.append(sample_df.select(self.feature_columns).to_numpy())
                    else:
                        collected_features.append(np.zeros((self.window_size, 4), dtype=np.float32))

            elif fm == 'alpha_beta':
                features = []
                
                # Currents
                i_phases = self._get_best_current_channels(sample_df)
                if len(i_phases) == 3:
                    a, b, c = i_phases
                    alpha = (2/3) * (a - 0.5*b - 0.5*c)
                    beta = (2/3) * (np.sqrt(3)/2 * (b - c))
                    zero = (1/3) * (a + b + c)
                    features.append(np.stack([alpha, beta, zero], axis=1))
                else:
                    features.append(np.zeros((self.window_size, 3), dtype=np.float32))
                    
                # Voltages
                u_phases, u_type = self._get_best_voltage_channels(sample_df)
                if u_type == 'line':
                     uab, ubc, uca = u_phases
                     ua = (2 * uab + ubc) / 3
                     ub = (2 * ubc + uca) / 3
                     uc = (2 * uca + uab) / 3
                     u_phases = [ua, ub, uc]
                
                if len(u_phases) == 3:
                    a, b, c = u_phases
                    alpha = (2/3) * (a - 0.5*b - 0.5*c)
                    beta = (2/3) * (np.sqrt(3)/2 * (b - c))
                    zero = (1/3) * (a + b + c)
                    features.append(np.stack([alpha, beta, zero], axis=1))
                else:
                    features.append(np.zeros((self.window_size, 3), dtype=np.float32))
                
                if features:
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
