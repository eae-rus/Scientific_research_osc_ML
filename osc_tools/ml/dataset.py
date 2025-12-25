import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Union, Optional

class OscillogramDataset(Dataset):
    """
    Универсальный Dataset для осциллограмм.
    Поддерживает режимы классификации, сегментации и реконструкции.
    """
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        indices: pd.DataFrame, 
        window_size: int, 
        mode: str = 'classification',
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
            feature_columns: Список колонок для входа (X). Если None, берутся все.
            target_columns: Колонка(и) для выхода (Y).
            target_position: Позиция целевого значения внутри окна (для classification). 
                             По умолчанию - последнее значение (window_size - 1).
        """
        self.data = dataframe
        self.indices = indices
        self.window_size = window_size
        self.mode = mode
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Получаем стартовый индекс из переданных indices
        # Предполагаем, что indices - это DataFrame с индексом, соответствующим индексу в self.data
        if isinstance(self.indices, pd.DataFrame):
            start_idx = self.indices.iloc[idx].name
        else:
            start_idx = self.indices[idx]

        # Проверка границ
        # Note: self.data.loc slicing is inclusive for labels, but we need fixed size
        # Assuming integer index in self.data
        
        # Используем iloc для надежности, если индекс не числовой, но здесь ожидается числовой
        # Для скорости лучше использовать numpy array, но пока оставим pandas interface как в оригинале
        
        # Вариант с loc (как в оригинале):
        # sample = self.data.loc[start_idx : start_idx + self.window_size - 1]
        
        # Вариант с iloc (более надежный для окон):
        # Нам нужно найти позицию start_idx в self.data
        # Если self.data имеет непрерывный RangeIndex, то loc и iloc совпадают.
        # В оригинале используется loc.
        
        end_idx = start_idx + self.window_size
        
        # Проверка на выход за границы (хотя indices должны быть корректными)
        if start_idx + self.window_size > len(self.data):
             # Вернуть нули или ошибку? В оригинале None
             return None, None

        # Извлечение X
        # Используем loc, так как indices привязаны к индексам DataFrame
        # Важно: loc включает конец, поэтому -1
        sample_df = self.data.loc[start_idx : start_idx + self.window_size - 1]
        
        x_data = sample_df[self.feature_columns].to_numpy(dtype=np.float32)
        
        # Транспонирование для CNN: (Time, Channels) -> (Channels, Time)
        # Для MLP обычно (Features), но если это временной ряд, то (Time*Features) или (Time, Features)
        # PyTorch Conv1d ожидает (Batch, Channels, Time)
        # Пока оставим (Time, Channels), модель сама может транспонировать или Dataset может иметь параметр transform
        
        # В оригинале: x = torch.tensor(sample.to_numpy(), dtype=torch.float32) -> (Time, Features)
        # PDR_MLP ожидает (Batch, Time, Features) или Flatten?
        # В train.py: x shape (Batch, Window, Features)
        
        x = torch.tensor(x_data, dtype=torch.float32)

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
