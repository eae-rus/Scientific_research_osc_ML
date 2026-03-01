import pytest
import torch
from osc_tools.ml.models.baseline import SimpleMLP, SimpleCNN
from osc_tools.ml.models.kan import SimpleKAN, ConvKAN

class TestDynamicModels:
    """Тесты для проверки моделей с динамической глубиной."""

    def test_simple_mlp_depth(self):
        """Проверка SimpleMLP с разной глубиной."""
        input_size = 320
        hidden_configs = [[64], [128, 64, 32], [256, 128, 64, 32, 16]]
        
        for hidden in hidden_configs:
            model = SimpleMLP(input_size=input_size, hidden_sizes=hidden, output_size=4)
            x = torch.randn(2, input_size)
            y = model(x)
            assert y.shape == (2, 4)

    def test_simple_cnn_depth(self):
        """Проверка SimpleCNN с разной глубиной."""
        in_channels = 8
        channel_configs = [[16, 32], [16, 32, 64, 128], [8, 16, 24, 32, 48, 64]]
        
        for channels in channel_configs:
            model = SimpleCNN(in_channels=in_channels, num_classes=4, channels=channels)
            # Вход (batch, channels, length)
            x = torch.randn(2, in_channels, 320)
            y = model(x)
            assert y.shape == (2, 4)

    def test_conv_kan_depth(self):
        """Проверка ConvKAN с разной глубиной."""
        in_channels = 8
        channel_configs = [[8, 16], [4, 8, 12, 16]]
        
        for channels in channel_configs:
            model = ConvKAN(in_channels=in_channels, num_classes=4, channels=channels)
            x = torch.randn(2, in_channels, 320)
            y = model(x)
            assert y.shape == (2, 4)

    def test_conv_kan_variable_grid(self):
        """Проверка ConvKAN с разными grid_size для слоев."""
        in_channels = 8
        channels = [8, 16, 32]
        grid_sizes = [10, 5, 3]
        
        model = ConvKAN(in_channels=in_channels, num_classes=4, channels=channels, grid_size=grid_sizes)
        x = torch.randn(2, in_channels, 320)
        y = model(x)
        assert y.shape == (2, 4)
