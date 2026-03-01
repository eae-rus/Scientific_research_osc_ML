import torch
import pytest
from osc_tools.ml.kan_conv.arithmetic import MultiplicationLayer, DivisionLayer

def test_multiplication_layer():
    layer = MultiplicationLayer()
    
    # Batch=1, Channels=4 (2 pairs), Length=3
    x = torch.tensor([
        [
            [1.0, 2.0, 3.0], # x1_0
            [4.0, 5.0, 6.0], # x1_1
            [2.0, 2.0, 2.0], # x2_0
            [0.5, 0.5, 0.5]  # x2_1
        ]
    ])
    
    # Expected:
    # Channel 0: [1*2, 2*2, 3*2] = [2, 4, 6]
    # Channel 1: [4*0.5, 5*0.5, 6*0.5] = [2, 2.5, 3]
    
    out = layer(x)
    
    assert out.shape == (1, 2, 3)
    assert torch.allclose(out[0, 0], torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(out[0, 1], torch.tensor([2.0, 2.5, 3.0]))

def test_division_layer():
    layer = DivisionLayer(epsilon=1e-6)
    
    x = torch.tensor([
        [
            [10.0, 20.0], # x1
            [2.0, 0.0]    # x2 (division by zero case)
        ]
    ])
    
    out = layer(x)
    
    assert out.shape == (1, 1, 2)
    # 10 / 2 = 5
    assert torch.abs(out[0, 0, 0] - 5.0) < 1e-5
    
    # 20 / epsilon (approx large number)
    # Check that it didn't return NaN or Inf
    assert torch.isfinite(out[0, 0, 1])
    assert out[0, 0, 1] > 1000.0

def test_odd_channels_error():
    layer = MultiplicationLayer()
    x = torch.randn(1, 3, 10)
    with pytest.raises(ValueError):
        layer(x)
