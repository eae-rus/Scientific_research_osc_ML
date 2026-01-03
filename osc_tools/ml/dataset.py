import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from typing import List, Union, Optional
from osc_tools.features.pdr_calculator import sliding_window_fft, calculate_symmetrical_components, calculate_power

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
            feature_columns: Список колонок для входа (X). Если None, берутся все.
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
        self.feature_columns = feature_columns if feature_columns is not None else dataframe.columns
        self.target_columns = target_columns
        
        if target_position is not None:
            self.target_position = target_position
        else:
            self.target_position = window_size - 1

        # Проверка режима
        valid_modes = ['classification', 'segmentation', 'reconstruction']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")
            
        valid_feature_modes = ['raw', 'symmetric', 'complex_channels', 'power']
        for fm in self.feature_mode:
            if fm not in valid_feature_modes:
                raise ValueError(f"Unknown feature_mode: {fm}. Valid modes: {valid_feature_modes}")

    def __len__(self):
        return len(self.indices)

    def _get_phase_data(self, df: pl.DataFrame, prefix: str) -> List[np.ndarray]:
        """Helper to extract phase data (A, B, C) based on column names."""
        cols = df.columns
        
        if prefix == 'I':
            candidates = ['IA', 'IB', 'IC']
        elif prefix == 'U':
            candidates_bb = ['UA BB', 'UB BB', 'UC BB']
            candidates_cl = ['UA CL', 'UB CL', 'UC CL']
            candidates_simple = ['UA', 'UB', 'UC']
            
            if all(c in cols for c in candidates_bb):
                candidates = candidates_bb
            elif all(c in cols for c in candidates_cl):
                candidates = candidates_cl
            elif all(c in cols for c in candidates_simple):
                candidates = candidates_simple
            else:
                candidates = ['UA CL', 'UB CL', 'UC CL'] # Default fallback, though likely to fail if not present
        else:
            return []

        found = []
        for c in candidates:
            if c in cols:
                found.append(df[c].to_numpy())
            else:
                return []
        
        if len(found) == 3:
            return found
        return []

    def __getitem__(self, idx):
        # Получаем стартовый индекс
        if isinstance(self.indices, pl.DataFrame):
             # Assuming indices is a DataFrame with one column or specific structure
             # If it's just a list of indices wrapped in DF
             start_idx = self.indices.row(idx)[0]
        else:
            start_idx = self.indices[idx]

        # Проверка на выход за границы
        if start_idx + self.window_size > len(self.data):
             return None, None

        # Извлечение окна данных
        # Polars slicing (эффективно)
        sample_df = self.data.slice(start_idx, self.window_size)
        
        collected_features = []
        
        for fm in self.feature_mode:
            if fm == 'raw':
                collected_features.append(sample_df.select(self.feature_columns).to_numpy())
                
            elif fm == 'symmetric':
                # Расчет симметричных составляющих
                fft_window = int(self.sampling_rate / 50)
                features = []
                
                i_phases = self._get_phase_data(sample_df, 'I')
                if i_phases:
                    phasors = [sliding_window_fft(p, fft_window, 1)[:, 0] for p in i_phases]
                    i1, i2, i0 = calculate_symmetrical_components(*phasors)
                    features.append(np.stack([i1.real, i1.imag, i2.real, i2.imag, i0.real, i0.imag], axis=1))
                    
                u_phases = self._get_phase_data(sample_df, 'U')
                if u_phases:
                    phasors = [sliding_window_fft(p, fft_window, 1)[:, 0] for p in u_phases]
                    u1, u2, u0 = calculate_symmetrical_components(*phasors)
                    features.append(np.stack([u1.real, u1.imag, u2.real, u2.imag, u0.real, u0.imag], axis=1))
                
                if features:
                    data = np.concatenate(features, axis=1)
                    collected_features.append(np.nan_to_num(data))
                else:
                    # Fallback if phases not found
                    collected_features.append(sample_df.select(self.feature_columns).to_numpy())

            elif fm == 'complex_channels':
                fft_window = int(self.sampling_rate / 50)
                features = []
                
                for prefix in ['I', 'U']:
                    phases = self._get_phase_data(sample_df, prefix)
                    if phases:
                        for p in phases:
                            phasor = sliding_window_fft(p, fft_window, 1)[:, 0]
                            features.append(np.stack([phasor.real, phasor.imag], axis=1))
                
                if features:
                    data = np.concatenate(features, axis=1)
                    collected_features.append(np.nan_to_num(data))
                else:
                    collected_features.append(sample_df.select(self.feature_columns).to_numpy())
            
            elif fm == 'power':
                fft_window = int(self.sampling_rate / 50)
                features = []
                
                i_phases = self._get_phase_data(sample_df, 'I')
                u_phases = self._get_phase_data(sample_df, 'U')
                
                if i_phases and u_phases:
                    for i_p, u_p in zip(i_phases, u_phases):
                        i_phasor = sliding_window_fft(i_p, fft_window, 1)[:, 0]
                        u_phasor = sliding_window_fft(u_p, fft_window, 1)[:, 0]
                        _, p_act, q_react = calculate_power(u_phasor, i_phasor)
                        features.append(np.stack([p_act, q_react], axis=1))
                
                if features:
                    data = np.concatenate(features, axis=1)
                    collected_features.append(np.nan_to_num(data))
                else:
                    # Fallback or empty? If power requested but no U/I, maybe return zeros or raw?
                    # For consistency with other modes, we might fallback to raw or zeros.
                    # But mixing raw with others might be bad.
                    # Let's return zeros of same length if possible, or just skip.
                    # If we skip, the dimension might be wrong.
                    # Let's assume if 'power' is requested, U and I are expected.
                    # If not found, we append nothing? No, that breaks batching.
                    # Let's append zeros if we can't calculate.
                    # But we don't know the shape easily.
                    # Let's fallback to raw if nothing else works, but that's also dangerous.
                    # I'll stick to the pattern: if features found, use them. If not, fallback to raw (as in other modes).
                    collected_features.append(sample_df.select(self.feature_columns).to_numpy())

        if not collected_features:
             # Should not happen due to init checks, but safe fallback
             x_data = sample_df.select(self.feature_columns).to_numpy()
        else:
             x_data = np.concatenate(collected_features, axis=1)

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
