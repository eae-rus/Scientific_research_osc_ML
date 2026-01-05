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
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, grid_size=5, spline_order=3, base_activation=torch.nn.SiLU):
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
            # KANLinear уже содержит функцию активации (SiLU по умолчанию)
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

class ConvKAN(BaseModel):
    """
    Сверточная сеть на основе KAN (Convolutional KAN).
    Использует KANConv1d для извлечения признаков.
    """
    def __init__(self, in_channels, num_classes, base_filters=8, kernel_size=3, grid_size=5, base_activation=torch.nn.SiLU):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            KANConv1d(in_channels, base_filters, kernel_size=kernel_size, padding=kernel_size//2, grid_size=grid_size, base_activation=base_activation),
            nn.MaxPool1d(2),
            
            # Block 2
            KANConv1d(base_filters, base_filters*2, kernel_size=kernel_size, padding=kernel_size//2, grid_size=grid_size, base_activation=base_activation),
            nn.MaxPool1d(2),
            
            # Block 3
            KANConv1d(base_filters*2, base_filters*4, kernel_size=kernel_size, padding=kernel_size//2, grid_size=grid_size, base_activation=base_activation),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            KANLinear(base_filters*4, base_filters*2, grid_size=grid_size, base_activation=base_activation),
            KANLinear(base_filters*2, num_classes, grid_size=grid_size, base_activation=base_activation)
        )

    def forward(self, x):
        x = self.features(x)
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
    def __init__(self, in_channels, num_classes, base_filters=8, kernel_size=3, grid_size=5, base_activation=torch.nn.SiLU):
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
            base_filters=base_filters,
            kernel_size=kernel_size,
            grid_size=grid_size,
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
