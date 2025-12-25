import torch
import torch.nn.functional as F
from osc_tools.core.constants import Features

def cubic_interpolate(tensor, output_size=64):
    """
    Интерполяция данных до фиксированного числа точек с использованием кубической интерполяции.
    
    Параметры:
    tensor: Тензор размера (батч, каналы, точки)
    output_size: Требуемое количество точек (по умолчанию 64)
    
    Возвращает:
    Интерполированный тензор размера (батч, каналы, output_size)
    """
    #batch_size, channels, input_points = tensor.shape
    
    # Применяем интерполяцию только по третьему измерению (точки)
    interpolated_tensor = F.interpolate(
        tensor, 
        size=(output_size,),  # Интерполируем по размеру точек
        mode='bicubic',  # Кубическая интерполяция
        align_corners=True
    )

    return interpolated_tensor

def fft_calc(input, count_harmonic=1):
        prev_signals = torch.fft.rfft(input[:, :, :32])
        current_signals = torch.fft.rfft(input[:, :, 32:])
        
        return (prev_signals[:, :, :count_harmonic+1], current_signals[:, :, :count_harmonic+1])

def fft_calc_abs_angle(input, count_harmonic=1):
        prev_signals = torch.fft.rfft(input[:, :, :32])
        current_signals = torch.fft.rfft(input[:, :, 32:])
        
        prev_signals = prev_signals[:, :, :count_harmonic+1]
        current_signals = current_signals[:, :, :count_harmonic+1]
        
        return (torch.abs(prev_signals), torch.abs(current_signals), torch.angle(current_signals), torch.angle(current_signals))

def create_signal_group(x, currents, voltages, device = None):
    if device is None:
        device = x.device
    group = torch.zeros(x.size(0), len(currents), x.size(2), device=device, dtype=torch.cfloat)
    for i, (ic, iv) in enumerate(zip(currents, voltages)):
        if ic != -1 and iv != -1:
            group[:, i, :] = x[:, ic, :] + 1j * x[:, iv, :]
        elif ic == -1:
            group[:, i, :] = 1j * x[:, iv, :]
        else:
            group[:, i, :] = x[:, ic, :]
    return group

def create_line_group(x, ic_L, voltages, device=None):
        if device is None:
            device = x.device
        group = torch.zeros(x.size(0), len(ic_L), x.size(2), device=device, dtype=torch.cfloat)
        for i, (ic_pair, iv) in enumerate(zip(ic_L, voltages)):
            group[:, i, :] = x[:, ic_pair[0], :] - x[:, ic_pair[1], :] + 1j * x[:, iv, :]
        return group
