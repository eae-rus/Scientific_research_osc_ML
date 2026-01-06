import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.kan_layers import KANLinear, KANConv1d
from osc_tools.ml.kan_conv.arithmetic import MultiplicationLayer, DivisionLayer

class SimpleKAN(BaseModel):
    """
    Простая полносвязная сеть на основе KAN (Kolmogorov-Arnold Network).
    Аналог SimpleMLP, но с использованием KANLinear слоев.
    """
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, grid_size=5, spline_order=3, dropout=0.0, base_activation=torch.nn.SiLU):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(
                KANLinear(
                    in_features=prev_size, 
                    out_features=size,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation
                )
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size
            
        # Выходной слой
        layers.append(
            KANLinear(
                in_features=prev_size, 
                out_features=output_size,
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=base_activation
            )
        )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)

class SafeMaxPool1d(nn.Module):
    """Pooling layer that handles small input sizes gracefully."""
    def __init__(self, kernel_size):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x):
        if x.shape[-1] < self.pool.kernel_size:
            return x
        return self.pool(x)

class ConvKAN(BaseModel):
    """
    Сверточная сеть на основе KAN (Convolutional KAN) с гибкой архитектурой.
    Поддерживает произвольное количество слоев через список channels.
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [8, 16, 32], 
                 kernel_size: int = 3, grid_size: int = 5, spline_order: int = 3,
                 dropout: float = 0.2, pool_every: int = 1, base_activation=torch.nn.SiLU):
        super().__init__()
        
        layers = []
        curr_channels = in_channels
        
        for i, out_channels in enumerate(channels):
            # KAN Convolutional block
            # grid_size может быть списком или числом. Если список - берем по индексу.
            curr_grid = grid_size[i] if isinstance(grid_size, list) else grid_size
            
            layers.append(
                KANConv1d(
                    curr_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    padding=kernel_size//2, 
                    grid_size=curr_grid, 
                    spline_order=spline_order,
                    base_activation=base_activation
                )
            )
            
            # Pooling
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            curr_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            KANLinear(curr_channels, curr_channels // 2, grid_size=grid_size[0] if isinstance(grid_size, list) else grid_size, base_activation=base_activation),
            KANLinear(curr_channels // 2, num_classes, grid_size=grid_size[0] if isinstance(grid_size, list) else grid_size, base_activation=base_activation)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class PhysicsKAN(BaseModel):
    """
    KAN модель с физически интерпретируемыми слоями (умножение/деление).
    Принимает на вход [Currents, Voltages].
    Вычисляет Power (I*U) и Admittance (I/U), объединяет с исходными сигналами
    и подает в ConvKAN.
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [8, 16, 32], 
                 kernel_size: int = 3, grid_size: int = 5, spline_order: int = 3,
                 dropout: float = 0.2, pool_every: int = 1, base_activation=torch.nn.SiLU):
        super().__init__()
        
        if in_channels % 2 != 0:
            raise ValueError(f"PhysicsKAN requires even number of input channels (I, U pairs), got {in_channels}")
            
        self.mult = MultiplicationLayer()
        self.div = DivisionLayer()
        
        # Input to ConvKAN: Original (C) + Mult (C/2) + Div (C/2) = 2 * C
        conv_in_channels = in_channels + (in_channels // 2) * 2
        
        self.conv_kan = ConvKAN(
            in_channels=conv_in_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_size=kernel_size,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
            pool_every=pool_every,
            base_activation=base_activation
        )

    def forward(self, x):
        # x: [Batch, Channels, Length]
        # Предполагается, что каналы упорядочены так, что первая половина - это I, вторая - U (или наоборот).
        # MultiplicationLayer делает x[:half] * x[half:]
        
        s = self.mult(x) # Power-like features
        z = self.div(x)  # Impedance-like features
        
        # Concatenate along channel dimension
        x_combined = torch.cat([x, s, z], dim=1)
        
        return self.conv_kan(x_combined)
