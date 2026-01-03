import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Union, Optional
from osc_tools.features.pdr_calculator import sliding_window_fft, calculate_symmetrical_components

class OscillogramDataset(Dataset):
    """
    Универсальный Dataset для осциллограмм.
    Поддерживает режимы классификации, сегментации и реконструкции.
    Поддерживает различные режимы формирования признаков (feature_mode).
    """
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        indices: pd.DataFrame, 
        window_size: int, 
        mode: str = 'classification',
        feature_mode: str = 'raw',
        sampling_rate: int = 1600,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        target_position: Optional[int] = None
    ):
        """
        Args:
            dataframe: Исходный DataFrame с данными.
            indices: DataFrame или список индексов начал окон.
            window_size: Размер окна (количество отсчетов).
            mode: Режим работы ('classification', 'segmentation', 'reconstruction').
            feature_mode: Режим признаков ('raw', 'symmetric', 'complex_channels').
            sampling_rate: Частота дискретизации (нужна для feature_mode != 'raw').
            feature_columns: Список колонок для входа (X). Если None, берутся все.
            target_columns: Колонка(и) для выхода (Y).
            target_position: Позиция целевого значения внутри окна (для classification). 
                             По умолчанию - последнее значение (window_size - 1).
        """
        self.data = dataframe
        self.indices = indices
        self.window_size = window_size
        self.mode = mode
        self.feature_mode = feature_mode
        self.sampling_rate = sampling_rate
        self.feature_columns = feature_columns if feature_columns is not None else dataframe.columns.tolist()
        self.target_columns = target_columns
        
        if target_position is not None:
            self.target_position = target_position
        else:
            self.target_position = window_size - 1

        # Проверка режима
        valid_modes = ['classification', 'segmentation', 'reconstruction']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")
            
        valid_feature_modes = ['raw', 'symmetric', 'complex_channels']
        if feature_mode not in valid_feature_modes:
            raise ValueError(f"Unknown feature_mode: {feature_mode}. Valid modes: {valid_feature_modes}")

    def __len__(self):
        return len(self.indices)

    def _get_phase_data(self, df: pd.DataFrame, prefix: str) -> List[np.ndarray]:
        """Helper to extract phase data (A, B, C) based on column names."""
        # Simple heuristic: look for columns containing prefix and 'A', 'B', 'C'
        # This assumes standard naming like 'IA', 'IB', 'IC' or 'UA BB', 'UB BB', 'UC BB'
        # We prioritize exact matches from constants if possible, but here we search dynamically
        
        cols = df.columns
        # Try to find columns that start with prefix (e.g. 'I') and contain 'A', 'B', 'C'
        # Or contain 'A', 'B', 'C' and the prefix is part of it.
        
        # Specific logic for this project's naming convention
        if prefix == 'I':
            candidates = ['IA', 'IB', 'IC']
        elif prefix == 'U':
            # Prefer BB (Bus Bar) voltages if available, else CL (Cable Line)
            candidates_bb = ['UA BB', 'UB BB', 'UC BB']
            if all(c in cols for c in candidates_bb):
                candidates = candidates_bb
            else:
                candidates = ['UA CL', 'UB CL', 'UC CL']
        else:
            return []

        found = []
        for c in candidates:
            if c in cols:
                found.append(df[c].to_numpy())
            else:
                # Fallback: search for any column containing prefix and phase
                # This is risky
                return []
        
        if len(found) == 3:
            return found
        return []

    def __getitem__(self, idx):
        # Получаем стартовый индекс из переданных indices
        # Предполагаем, что indices - это DataFrame с индексом, соответствующим индексу в self.data
        if isinstance(self.indices, pd.DataFrame):
            start_idx = self.indices.iloc[idx].name
        else:
            start_idx = self.indices[idx]

        # Проверка на выход за границы
        if start_idx + self.window_size > len(self.data):
             return None, None

        # Извлечение окна данных
        # Используем loc, так как indices привязаны к индексам DataFrame
        sample_df = self.data.loc[start_idx : start_idx + self.window_size - 1]
        
        # Формирование признаков (X)
        if self.feature_mode == 'raw':
            x_data = sample_df[self.feature_columns].to_numpy(dtype=np.float32)
            
        elif self.feature_mode == 'symmetric':
            # Расчет симметричных составляющих (I1, I2, I0, U1, U2, U0)
            fft_window = int(self.sampling_rate / 50) # 1 период пром. частоты
            features = []
            
            # Токи
            i_phases = self._get_phase_data(sample_df, 'I')
            if i_phases:
                phasors = [sliding_window_fft(p, fft_window, 1)[:, 0] for p in i_phases]
                i1, i2, i0 = calculate_symmetrical_components(*phasors)
                # Добавляем Real и Imag части
                features.append(np.stack([i1.real, i1.imag, i2.real, i2.imag, i0.real, i0.imag], axis=1))
                
            # Напряжения
            u_phases = self._get_phase_data(sample_df, 'U')
            if u_phases:
                phasors = [sliding_window_fft(p, fft_window, 1)[:, 0] for p in u_phases]
                u1, u2, u0 = calculate_symmetrical_components(*phasors)
                features.append(np.stack([u1.real, u1.imag, u2.real, u2.imag, u0.real, u0.imag], axis=1))
            
            if features:
                x_data = np.concatenate(features, axis=1)
                # Заполняем NaN (возникают в начале окна из-за FFT) нулями или ближайшими значениями
                x_data = np.nan_to_num(x_data)
            else:
                # Fallback
                x_data = sample_df[self.feature_columns].to_numpy(dtype=np.float32)

        elif self.feature_mode == 'complex_channels':
            # Комплексные фазоры (Re, Im) для каждой фазы
            fft_window = int(self.sampling_rate / 50)
            features = []
            
            for prefix in ['I', 'U']:
                phases = self._get_phase_data(sample_df, prefix)
                if phases:
                    for p in phases:
                        phasor = sliding_window_fft(p, fft_window, 1)[:, 0]
                        features.append(np.stack([phasor.real, phasor.imag], axis=1))
            
            if features:
                x_data = np.concatenate(features, axis=1)
                x_data = np.nan_to_num(x_data)
            else:
                x_data = sample_df[self.feature_columns].to_numpy(dtype=np.float32)
        
        else:
            x_data = sample_df[self.feature_columns].to_numpy(dtype=np.float32)

        x = torch.tensor(x_data, dtype=torch.float32)
        
        # Извлечение целевой переменной (Y)
        y = None
        if self.mode == 'classification':
            if self.target_columns:
                # Берем значение в target_position (обычно конец окна)
                # Нужно найти индекс в sample_df, соответствующий target_position
                target_idx = start_idx + self.target_position
                
                if isinstance(self.target_columns, list):
                    y_val = self.data.loc[target_idx, self.target_columns].values
                    y = torch.tensor(y_val.astype(np.float32), dtype=torch.float32)
                else:
                    y_val = self.data.loc[target_idx, self.target_columns]
                    y = torch.tensor(y_val, dtype=torch.long) # Обычно классы - int
                    
        elif self.mode == 'segmentation':
            if self.target_columns:
                # Берем все значения окна
                y_data = sample_df[self.target_columns].to_numpy()
                y = torch.tensor(y_data, dtype=torch.long) # (Time, Classes) or (Time)
                
        elif self.mode == 'reconstruction':
            y = x.clone()

        return x, y

        # Извлечение Y
        if self.mode == 'reconstruction':
            target = x.clone()
            
        elif self.mode == 'segmentation':
            if self.target_columns is None:
                raise ValueError("target_columns must be specified for segmentation")
            y_data = sample_df[self.target_columns].to_numpy(dtype=np.float32)
            target = torch.tensor(y_data, dtype=torch.float32)
            
        elif self.mode == 'classification':
            if self.target_columns is None:
                raise ValueError("target_columns must be specified for classification")
            
            # Целевое значение в конкретной точке окна
            target_idx_in_df = start_idx + self.target_position
            y_data = self.data.loc[target_idx_in_df, self.target_columns]
            
            # Если y_data - скаляр (одна колонка) или Series
            if isinstance(y_data, pd.Series):
                y_data = y_data.to_numpy(dtype=np.float32)
            else:
                y_data = np.float32(y_data)
                
            target = torch.tensor(y_data, dtype=torch.float32)

        return x, target
