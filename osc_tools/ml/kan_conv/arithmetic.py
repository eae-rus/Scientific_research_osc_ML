import torch
import torch.nn as nn

class MultiplicationLayer(nn.Module):
    """
    Слой, выполняющий поэлементное умножение двух частей входного тензора.
    Предполагается, что входной тензор содержит пары значений, которые нужно перемножить.
    Тензор разбивается на две равные половины по измерению каналов (dim=1).
    
    Output channels = Input channels / 2
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Channels, Length) or (Batch, Channels)
        """
        c = x.shape[1]
        if c % 2 != 0:
            raise ValueError(f"Input channels must be even for MultiplicationLayer, got {c}")
        
        half = c // 2
        x1 = x[:, :half, ...]
        x2 = x[:, half:, ...]
        
        return x1 * x2

class DivisionLayer(nn.Module):
    """
    Слой, выполняющий поэлементное деление двух частей входного тензора.
    Тензор разбивается на две равные половины по измерению каналов (dim=1).
    Первая половина делится на вторую.
    
    Output channels = Input channels / 2
    """
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c % 2 != 0:
            raise ValueError(f"Input channels must be even for DivisionLayer, got {c}")
        
        half = c // 2
        x1 = x[:, :half, ...]
        x2 = x[:, half:, ...]
        
        # Простая защита от деления на ноль
        # Можно улучшить, сохраняя знак, но пока так
        denom = x2.clone()
        # Заменяем слишком маленькие значения на epsilon (с сохранением знака если нужно, но тут просто add)
        # Лучше так: если abs(x) < eps, то x = eps * sign(x)
        
        mask = torch.abs(denom) < self.epsilon
        denom[mask] = self.epsilon * torch.sign(denom[mask] + 1e-9) # +1e-9 чтобы 0 стал +
        
        return x1 / denom
