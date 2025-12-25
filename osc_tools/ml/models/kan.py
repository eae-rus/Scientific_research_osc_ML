import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.kan_layers import KANLinear, KANConv1d

class SimpleKAN(BaseModel):
    """
    Простая полносвязная сеть на основе KAN (Kolmogorov-Arnold Network).
    Аналог SimpleMLP, но с использованием KANLinear слоев.
    """
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, grid_size=5, spline_order=3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(
                KANLinear(
                    in_features=prev_size, 
                    out_features=size,
                    grid_size=grid_size,
                    spline_order=spline_order
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
                spline_order=spline_order
            )
        )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvKAN(BaseModel):
    """
    Сверточная сеть на основе KAN (Convolutional KAN).
    Использует KANConv1d для извлечения признаков.
    """
    def __init__(self, in_channels, num_classes, base_filters=8, kernel_size=3, grid_size=5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            KANConv1d(in_channels, base_filters, kernel_size=kernel_size, padding=kernel_size//2, grid_size=grid_size),
            nn.MaxPool1d(2),
            
            # Block 2
            KANConv1d(base_filters, base_filters*2, kernel_size=kernel_size, padding=kernel_size//2, grid_size=grid_size),
            nn.MaxPool1d(2),
            
            # Block 3
            KANConv1d(base_filters*2, base_filters*4, kernel_size=kernel_size, padding=kernel_size//2, grid_size=grid_size),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            KANLinear(base_filters*4, base_filters*2, grid_size=grid_size),
            KANLinear(base_filters*2, num_classes, grid_size=grid_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
