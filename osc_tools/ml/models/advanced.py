import torch
import torch.nn as nn
from osc_tools.ml.models.base import BaseModel
from osc_tools.ml.models.baseline import SafeMaxPool1d
from osc_tools.ml.models.cnn import ResBlock1D, ResNet1D
from osc_tools.ml.models.baseline import SimpleCNN

class DepthwisePointwiseConv1d(nn.Module):
    """
    Блок раздельной свертки (Depthwise Separable Convolution).
    Сначала применяет свертку отдельно к каждому каналу (groups=in_channels),
    затем смешивает каналы точечной сверткой (kernel_size=1).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SimpleCNN_Depthwise(SimpleCNN):
    """
    Версия SimpleCNN с Depthwise сверткой в первом слое.
    Позволяет обрабатывать физические каналы (фазы, токи, напряжения) независимо 
    на начальном этапе.
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list = [16, 32, 64], 
                 kernel_size: int = 3, stride: int = 1, dropout: float = 0.2, use_bn: bool = True,
                 pool_every: int = 1):
        super().__init__(in_channels, num_classes, channels, kernel_size, stride, dropout, use_bn, pool_every)
        
        # Заменяем первый сверточный слой
        first_out = channels[0]
        # В родительском классе первый слой: self.features[0]
        
        layers = list(self.features.children())
        
        # Создаем новый первый слой
        # Note: Родительский класс использует stride только в первом слое
        new_first_layer = DepthwisePointwiseConv1d(
            in_channels, 
            first_out, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size//2
        )
        
        layers[0] = new_first_layer
        self.features = nn.Sequential(*layers)


class ResNet1D_Depthwise(ResNet1D):
    """
    Версия ResNet1D с Depthwise сверткой в первом слое (stem).
    """
    def __init__(self, in_channels, num_classes, layers=[2, 2, 2, 2], base_filters=64):
        super().__init__(in_channels, num_classes, layers, base_filters)
        
        # self.conv1 - это первая свертка
        # kernel_size=7, stride=2, padding=3
        self.conv1 = DepthwisePointwiseConv1d(
            in_channels, 
            base_filters, 
            kernel_size=7, 
            stride=2, 
            padding=3,
            bias=False
        )
