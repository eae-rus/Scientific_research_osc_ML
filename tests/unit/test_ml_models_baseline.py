import pytest
import torch
from osc_tools.ml.models.baseline import SimpleMLP, SimpleCNN

class TestBaselineModels:
    """
    Тесты для базовых моделей (SimpleMLP, SimpleCNN).
    Проверяем инициализацию и проход данных (forward pass).
    """

    def test_simple_mlp_init(self):
        model = SimpleMLP(input_size=10, hidden_sizes=[20, 10], output_size=2)
        assert isinstance(model, SimpleMLP)
        
    def test_simple_mlp_forward(self):
        batch_size = 5
        input_size = 10
        output_size = 2
        model = SimpleMLP(input_size=input_size, output_size=output_size)
        
        x = torch.randn(batch_size, input_size)
        y = model(x)
        
        assert y.shape == (batch_size, output_size)
        assert not torch.isnan(y).any()

    def test_simple_cnn_init(self):
        model = SimpleCNN(in_channels=3, num_classes=2)
        assert isinstance(model, SimpleCNN)

    def test_simple_cnn_forward(self):
        batch_size = 4
        in_channels = 3
        seq_len = 100
        num_classes = 2
        
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
        
        # Input shape for 1D CNN: (Batch, Channels, Length)
        x = torch.randn(batch_size, in_channels, seq_len)
        y = model(x)
        
        assert y.shape == (batch_size, num_classes)
        assert not torch.isnan(y).any()
