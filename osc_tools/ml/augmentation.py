import torch
import numpy as np
from typing import Dict, Optional, List, Union

class TimeSeriesAugmenter:
    """
    Модуль аугментации временных рядов для осциллограмм.
    Применяет физически обоснованные преобразования к тензорам данных.
    Ожидает входной тензор размерности (Time, Channels) или (Batch, Time, Channels).
    
    Предполагаемая структура каналов (для PhaseShuffling):
    [IA, IB, IC, In, UA, UB, UC, Un]
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Словарь с параметрами аугментации.
                Пример:
                {
                    "p_inversion": 0.5,
                    "p_noise": 0.5,
                    "noise_std_current": 0.01,
                    "noise_std_voltage": 0.1,
                    "p_scaling": 0.5,
                    "scaling_range_current": (0.8, 1.2),
                    "scaling_range_voltage": (0.95, 1.05),
                    "p_offset": 0.5,
                    "offset_range": (-0.01, 0.01),
                    "p_phase_shuffling": 0.5,
                    "p_drop_channel": 0.1
                }
        """
        self.config = config or {}
        
        # Вероятности (по умолчанию обновлены согласно запросу пользователя для Фазы 2.5)
        self.p_inversion = self.config.get("p_inversion", 0.5)
        self.p_noise = self.config.get("p_noise", 0)
        self.p_scaling = self.config.get("p_scaling", 0.2)
        self.p_offset = self.config.get("p_offset", 0.0)
        self.p_phase_shuffling = self.config.get("p_phase_shuffling", 0.33)
        self.p_drop_channel = self.config.get("p_drop_channel", 0.0)
        
        # Параметры (Уменьшена интенсивность шума и масштаб)
        self.noise_std_current = self.config.get("noise_std_current", 0.005) # ~0.5% от номинала
        self.noise_std_voltage = self.config.get("noise_std_voltage", 0.05)  # ~0.1-0.2% от номинала
        
        self.scaling_range_current = self.config.get("scaling_range_current", (0.9, 1.1))
        self.scaling_range_voltage = self.config.get("scaling_range_voltage", (0.98, 1.02))
        
        self.offset_range = self.config.get("offset_range", (-0.005, 0.005))
        
        # Индексы каналов (стандартные для Smart Selector)
        self.current_indices = [0, 1, 2] # IA, IB, IC
        self.voltage_indices = [4, 5, 6] # UA, UB, UC
        self.neutral_indices = [3, 7]    # In, Un

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Применяет аугментации к входным данным.
        """
        is_torch = isinstance(x, torch.Tensor)
        if not is_torch:
            x = torch.from_numpy(x).float()
        else:
            x = x.clone()
            
        # Если вход (Time, Channels), добавляем Batch dim -> (1, Time, Channels)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # x shape: (Batch, Time, Channels)
        batch_size, time_steps, channels = x.shape
        
        # 1. Инверсия (умножение на -1)
        if self.p_inversion > 0 and torch.rand(1).item() < self.p_inversion:
            x = x * -1
            
        # 2. Масштабирование (Scaling)
        if self.p_scaling > 0 and torch.rand(1).item() < self.p_scaling:
            # Отдельные факторы для токов и напряжений
            scale_i = torch.empty(batch_size, 1, 1).uniform_(*self.scaling_range_current)
            scale_u = torch.empty(batch_size, 1, 1).uniform_(*self.scaling_range_voltage)
            
            # print(f"DEBUG: Scaling applied. scale_i={scale_i.item()}, scale_u={scale_u.item()}")
            
            # Применяем к соответствующим каналам
            # Предполагаем, что каналов достаточно. Если нет - пропускаем.
            if channels >= 3:
                x[:, :, self.current_indices] *= scale_i
            if channels >= 7:
                x[:, :, self.voltage_indices] *= scale_u
            if channels >= 4: # In
                x[:, :, 3:4] *= scale_i
            if channels >= 8: # Un
                x[:, :, 7:8] *= scale_u

        # 3. Гауссов шум (Gaussian Noise)
        if self.p_noise > 0 and torch.rand(1).item() < self.p_noise:
            noise = torch.zeros_like(x)
            
            # Шум для токов
            if channels >= 4:
                noise[:, :, :4].normal_(mean=0, std=self.noise_std_current)
            elif channels >= 3:
                noise[:, :, :3].normal_(mean=0, std=self.noise_std_current)
                
            # Шум для напряжений
            if channels >= 8:
                noise[:, :, 4:].normal_(mean=0, std=self.noise_std_voltage)
            elif channels >= 7:
                noise[:, :, 4:7].normal_(mean=0, std=self.noise_std_voltage)
                
            x = x + noise

        # 4. Смещение (Offset)
        if self.p_offset > 0 and torch.rand(1).item() < self.p_offset:
            offset = torch.empty(batch_size, 1, channels).uniform_(*self.offset_range)
            x = x + offset

        # 5. Перетасовка фаз (Phase Shuffling)
        # Только если у нас есть все 3 фазы токов и напряжений
        if self.p_phase_shuffling > 0 and channels >= 7 and torch.rand(1).item() < self.p_phase_shuffling:
            shift = torch.randint(1, 3, (1,)).item() # 1 или 2 (0 нет смысла)
            
            # Сдвиг токов
            x[:, :, self.current_indices] = torch.roll(x[:, :, self.current_indices], shifts=shift, dims=2)
            # Сдвиг напряжений (синхронно с токами)
            x[:, :, self.voltage_indices] = torch.roll(x[:, :, self.voltage_indices], shifts=shift, dims=2)
            
        # 6. Drop Channel (имитация обрыва измерительного канала)
        if self.p_drop_channel > 0 and torch.rand(1).item() < self.p_drop_channel:
            # Выбираем случайный канал для зануления
            # Не трогаем In/Un, чаще всего пропадают фазные замеры
            target_indices = self.current_indices + self.voltage_indices
            if channels >= 7:
                idx_to_drop = target_indices[torch.randint(0, len(target_indices), (1,)).item()]
                x[:, :, idx_to_drop] = 0.0

        if squeeze_output:
            x = x.squeeze(0)
            
        if not is_torch:
            x = x.numpy()
            
        return x
