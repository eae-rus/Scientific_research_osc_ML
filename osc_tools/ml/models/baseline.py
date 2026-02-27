import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.layers.temporal_pooling import TemporalPooling

class SimpleMLP(BaseModel):
    """
    Простой многослойный перцептрон (MLP) для базовых экспериментов.
    Поддерживает настраиваемое количество слоев, нейронов, Dropout и BatchNormalization.
    """
    def __init__(self, input_size: int, hidden_sizes: list = [128, 64], output_size: int = 1, dropout: float = 0.2, use_bn: bool = True):
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
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)

class SafeMaxPool1d(nn.Module):
    """Pooling layer that handles small input sizes gracefully."""
    def __init__(self, kernel_size):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x):
        # Если размер меньше кернела, пулинг вернет 0 размера (ошибку) или пустоту
        # Поэтому пропускаем пулинг для экстремально малых размерностей
        if x.shape[-1] < self.pool.kernel_size:
            return x
        return self.pool(x)

class SimpleCNN(BaseModel):
    """
    Сверточная сеть (CNN) с гибкой архитектурой.
    Поддерживает произвольное количество слоев через список channels.
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [16, 32, 64], 
                 kernel_size: int = 3, stride: int = 1, dropout: float = 0.2, use_bn: bool = True,
                 pool_every: int = 1, pooling_strategy: str = "global_avg"):
        super().__init__()
        
        layers = []
        curr_channels = in_channels
        
        for i, out_channels in enumerate(channels):
            # Convolutional block
            # Применяем stride только к первому слою, чтобы сильно не терять разрешение дальше
            s = stride if i == 0 else 1
            layers.append(nn.Conv1d(curr_channels, out_channels, kernel_size, stride=s, padding=kernel_size//2))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            
            # Pooling
            if (i + 1) % pool_every == 0:
                layers.append(SafeMaxPool1d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            curr_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        self.pool = TemporalPooling(channels=curr_channels, strategy=pooling_strategy)
        
        # Размер входа классификатора зависит от стратегии пулинга
        classifier_input = curr_channels * self.pool.output_scale
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, max(4, classifier_input // 2)),
            nn.ReLU(),
            nn.Linear(max(4, classifier_input // 2), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)       # (B, C * output_scale) — уже без временной оси
        x = self.classifier(x)
        return x
