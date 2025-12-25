import pytest
import torch
from osc_tools.ml.models.kan import SimpleKAN, ConvKAN

class TestKANModels:
    """
    Тесты для KAN моделей (SimpleKAN, ConvKAN).
    """

    def test_simple_kan_init(self):
        model = SimpleKAN(input_size=10, hidden_sizes=[20, 10], output_size=2)
        assert isinstance(model, SimpleKAN)
        
    def test_simple_kan_forward(self):
        batch_size = 5
        input_size = 10
        output_size = 2
        model = SimpleKAN(input_size=input_size, output_size=output_size)
        
        x = torch.randn(batch_size, input_size)
        y = model(x)
        
        assert y.shape == (batch_size, output_size)
        assert not torch.isnan(y).any()

    def test_conv_kan_init(self):
        model = ConvKAN(in_channels=3, num_classes=2)
        assert isinstance(model, ConvKAN)

    def test_conv_kan_forward(self):
        batch_size = 4
        in_channels = 3
        seq_len = 100
        num_classes = 2
        
        model = ConvKAN(in_channels=in_channels, num_classes=num_classes)
        
        x = torch.randn(batch_size, in_channels, seq_len)
        y = model(x)
        
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()
