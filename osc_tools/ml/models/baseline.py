import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel

class SimpleMLP(BaseModel):
    """
    Простой многослойный перцептрон (MLP) для базовых экспериментов.
    Поддерживает настраиваемое количество слоев, нейронов, Dropout и BatchNormalization.
    """
    def __init__(self, input_size, hidden_sizes=[128, 64], output_size=1, dropout=0.2, use_bn=True):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_bn:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_size))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SimpleCNN(BaseModel):
    """
    Простая 1D сверточная сеть (CNN) для базовых экспериментов.
    """
    def __init__(self, in_channels, num_classes, base_filters=16, kernel_size=3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, base_filters, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(base_filters, base_filters*2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(base_filters*2, base_filters*4, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(base_filters*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(base_filters*4, base_filters*2),
            nn.ReLU(),
            nn.Linear(base_filters*2, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
