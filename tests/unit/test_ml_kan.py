import pytest
import torch
from osc_tools.ml.layers.kan_layers import KANConv1d, KANLinear

class TestKANLayers:
    
    def test_kan_linear_init(self):
        layer = KANLinear(in_features=10, out_features=5)
        assert isinstance(layer, KANLinear)
        
    def test_kan_linear_forward(self):
        batch_size = 4
        in_features = 10
        out_features = 5
        layer = KANLinear(in_features, out_features)
        
        x = torch.randn(batch_size, in_features)
        y = layer(x)
        
        assert y.shape == (batch_size, out_features)
        assert not torch.isnan(y).any()

    def test_kan_conv1d_init(self):
        layer = KANConv1d(in_channels=3, out_channels=4, kernel_size=3)
        assert isinstance(layer, KANConv1d)

    def test_kan_conv1d_forward(self):
        batch_size = 2
        in_channels = 3
        out_channels = 4
        length = 20
        kernel_size = 3
        
        layer = KANConv1d(in_channels, out_channels, kernel_size)
        
        x = torch.randn(batch_size, in_channels, length)
        y = layer(x)
        
        # Длина выхода для шага 1 и паддинга 0: L - K + 1
        expected_length = length - kernel_size + 1
        
        assert y.shape == (batch_size, out_channels, expected_length)
        assert not torch.isnan(y).any()

    def test_kan_conv1d_padding(self):
        batch_size = 2
        in_channels = 1
        out_channels = 1
        length = 10
        kernel_size = 3
        padding = 1
        
        layer = KANConv1d(in_channels, out_channels, kernel_size, padding=padding)
        x = torch.randn(batch_size, in_channels, length)
        y = layer(x)
        
        assert y.shape == (batch_size, out_channels, length)

    def test_kan_linear_masking(self):
        in_features = 4
        out_features = 2
        layer = KANLinear(in_features, out_features)
        
        # Устанавливаем маску в ноль для первого выходного нейрона
        layer.mask[0, :] = 0
        
        x = torch.randn(2, in_features)
        y = layer(x)
        
        # Выход для первого нейрона должен быть равен нулю
        assert torch.allclose(y[:, 0], torch.zeros_like(y[:, 0]))
        # Выход для второго нейрона должен быть ненулевым (вероятно)
        assert not torch.allclose(y[:, 1], torch.zeros_like(y[:, 1]))
