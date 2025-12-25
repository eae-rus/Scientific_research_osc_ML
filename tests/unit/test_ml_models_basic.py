"""
Tests for osc_tools.ml.models module.

Layer 3: Experimental Zone (Smoke Tests Only)
Focuses on:
- Model instantiation and basic functionality
- Forward pass without errors
- Output tensor shapes are correct
- NO value assertions or weight checks
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Setup sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.layers.complex_ops import (
    cLeakyReLU,
    cSigmoid,
    cMaxPool1d,
    cDropout1d
)
from osc_tools.ml.models.cnn import create_conv_block, Conv_3


class TestCLeakyReLU:
    """Test complex LeakyReLU activation."""
    
    def test_instantiation(self):
        """Test module can be instantiated."""
        layer = cLeakyReLU()
        assert isinstance(layer, nn.Module)
    
    def test_forward_pass_complex_input(self):
        """Test forward pass accepts complex tensor."""
        layer = cLeakyReLU()
        x = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        output = layer(x)
        assert output.is_complex()
        assert output.shape == x.shape
    
    def test_forward_pass_preserves_shape(self):
        """Test forward pass preserves input shape."""
        layer = cLeakyReLU(negative_slope=0.1)
        x = torch.complex(
            torch.randn(4, 16, 32),
            torch.randn(4, 16, 32)
        )
        output = layer(x)
        assert output.shape == x.shape
    
    def test_custom_negative_slope(self):
        """Test with custom negative slope parameter."""
        layer = cLeakyReLU(negative_slope=0.2)
        assert layer.negative_slope == 0.2
    
    def test_forward_no_nan(self):
        """Test forward pass doesn't produce NaN."""
        layer = cLeakyReLU()
        x = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        output = layer(x)
        assert not torch.isnan(output).any()


class TestCSigmoid:
    """Test complex sigmoid activation."""
    
    def test_instantiation(self):
        """Test module can be instantiated."""
        layer = cSigmoid()
        assert isinstance(layer, nn.Module)
    
    def test_forward_pass_real_input(self):
        """Test forward pass with real part focus."""
        layer = cSigmoid()
        x = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        output = layer(x)
        assert output.shape[0] == 2
    
    def test_output_is_tensor(self):
        """Test output is valid tensor."""
        layer = cSigmoid()
        x = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
        output = layer(x)
        assert isinstance(output, torch.Tensor)
    
    def test_forward_no_nan(self):
        """Test forward pass doesn't produce NaN."""
        layer = cSigmoid()
        x = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        output = layer(x)
        assert not torch.isnan(output).any()


class TestCMaxPool1d:
    """Test complex MaxPool1d."""
    
    def test_instantiation(self):
        """Test module can be instantiated."""
        pool = cMaxPool1d(kernel_size=2)
        assert isinstance(pool, nn.Module)
    
    def test_forward_pass_complex_input(self):
        """Test forward pass with complex input."""
        pool = cMaxPool1d(kernel_size=2, stride=2)
        # Shape: (batch, channels, length)
        x = torch.complex(
            torch.randn(2, 16, 32),
            torch.randn(2, 16, 32)
        )
        output = pool(x)
        assert output.is_complex()
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 16  # channels preserved
    
    def test_output_length_reduction(self):
        """Test pooling reduces sequence length."""
        pool = cMaxPool1d(kernel_size=2, stride=2)
        x = torch.complex(torch.randn(1, 8, 32), torch.randn(1, 8, 32))
        output = pool(x)
        assert output.shape[-1] < x.shape[-1]
    
    def test_with_padding(self):
        """Test pooling with padding."""
        pool = cMaxPool1d(kernel_size=2, stride=2, padding=1)
        x = torch.complex(torch.randn(2, 16, 33), torch.randn(2, 16, 33))
        output = pool(x)
        assert output.is_complex()
    
    def test_forward_no_nan(self):
        """Test forward pass doesn't produce NaN."""
        pool = cMaxPool1d(kernel_size=2)
        x = torch.complex(torch.randn(2, 16, 32), torch.randn(2, 16, 32))
        output = pool(x)
        assert not torch.isnan(output).any()


class TestCreateConvBlock:
    """Test convolution block creation."""
    
    def test_create_real_conv_block(self):
        """Test creating real-valued convolution block."""
        block = create_conv_block(16, 32, useComplex=False)
        assert isinstance(block, nn.Sequential)
    
    def test_create_complex_conv_block(self):
        """Test creating complex convolution block."""
        block = create_conv_block(16, 32, useComplex=True)
        assert isinstance(block, nn.Sequential)
    
    def test_real_block_forward_pass(self):
        """Test real convolution block forward pass."""
        block = create_conv_block(1, 16, useComplex=False)
        x = torch.randn(2, 1, 64)
        output = block(x)
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 16  # out channels
    
    def test_complex_block_forward_pass(self):
        """Test complex convolution block forward pass."""
        block = create_conv_block(1, 16, useComplex=True)
        x = torch.complex(torch.randn(2, 1, 64), torch.randn(2, 1, 64))
        output = block(x)
        assert output.is_complex()
        assert output.shape[0] == 2
        assert output.shape[1] == 16
    
    def test_custom_kernel_size(self):
        """Test block with custom kernel size."""
        block = create_conv_block(16, 32, kernel_size=5, useComplex=False)
        x = torch.randn(2, 16, 64)
        output = block(x)
        assert output.shape[0] == 2
    
    def test_custom_maxpool_size(self):
        """Test block with custom maxpool size."""
        block = create_conv_block(16, 32, maxPool_size=4, useComplex=False)
        x = torch.randn(2, 16, 64)
        output = block(x)
        # Pooling should reduce sequence length
        assert output.shape[-1] <= x.shape[-1]
    
    def test_block_no_nan_real(self):
        """Test real block doesn't produce NaN."""
        block = create_conv_block(1, 16, useComplex=False)
        x = torch.randn(2, 1, 64)
        output = block(x)
        assert not torch.isnan(output).any()
    
    def test_block_no_nan_complex(self):
        """Test complex block doesn't produce NaN."""
        block = create_conv_block(1, 16, useComplex=True)
        x = torch.complex(torch.randn(2, 1, 64), torch.randn(2, 1, 64))
        output = block(x)
        assert not torch.isnan(output).any()


class TestConv3Model:
    """Test Conv_3 model."""
    
    def test_instantiation_real(self):
        """Test Conv_3 real model instantiation."""
        model = Conv_3(useComplex=False)
        assert isinstance(model, nn.Module)
    
    def test_instantiation_complex(self):
        """Test Conv_3 complex model instantiation."""
        model = Conv_3(useComplex=True)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass_real(self):
        """Test real Conv_3 forward pass."""
        model = Conv_3(useComplex=False)
        x = torch.randn(2, 1, 256)
        output = model(x)
        assert output.shape[0] == 2  # batch size preserved
    
    def test_forward_pass_complex(self):
        """Test complex Conv_3 forward pass."""
        model = Conv_3(useComplex=True)
        x = torch.complex(torch.randn(2, 1, 256), torch.randn(2, 1, 256))
        output = model(x)
        assert output.is_complex()
        assert output.shape[0] == 2
    
    def test_model_has_layer_attribute(self):
        """Test model has layer attribute."""
        model = Conv_3()
        assert hasattr(model, 'layer')
        assert isinstance(model.layer, nn.Sequential)
    
    def test_real_model_no_nan(self):
        """Test real model doesn't produce NaN."""
        model = Conv_3(useComplex=False)
        x = torch.randn(2, 1, 256)
        output = model(x)
        assert not torch.isnan(output).any()
    
    def test_complex_model_no_nan(self):
        """Test complex model doesn't produce NaN."""
        model = Conv_3(useComplex=True)
        x = torch.complex(torch.randn(2, 1, 256), torch.randn(2, 1, 256))
        output = model(x)
        assert not torch.isnan(output).any()
    
    def test_eval_mode(self):
        """Test model can switch to eval mode."""
        model = Conv_3()
        model.eval()
        x = torch.randn(2, 1, 256)
        output = model(x)
        assert output is not None
    
    def test_train_mode(self):
        """Test model can switch to train mode."""
        model = Conv_3()
        model.train()
        x = torch.randn(2, 1, 256)
        output = model(x)
        assert output is not None


class TestCDropout1d:
    """Test complex Dropout1d."""
    
    def test_instantiation(self):
        """Test module can be instantiated."""
        dropout = cDropout1d(p=0.1)
        assert isinstance(dropout, nn.Module)
    
    def test_forward_pass_eval_mode(self):
        """Test forward pass in eval mode (no dropout)."""
        dropout = cDropout1d(p=0.5)
        dropout.eval()
        x = torch.complex(torch.ones(2, 16, 32), torch.ones(2, 16, 32))
        output = dropout(x)
        assert torch.allclose(output, x)
    
    def test_forward_pass_train_mode(self):
        """Test forward pass in train mode."""
        dropout = cDropout1d(p=0.1)
        dropout.train()
        x = torch.complex(torch.randn(2, 16, 32), torch.randn(2, 16, 32))
        output = dropout(x)
        assert output.is_complex()
        assert output.shape == x.shape
    
    def test_zero_dropout_probability(self):
        """Test with zero dropout probability."""
        dropout = cDropout1d(p=0.0)
        dropout.train()
        x = torch.complex(torch.randn(2, 16, 32), torch.randn(2, 16, 32))
        output = dropout(x)
        assert torch.allclose(output, x)
    
    def test_output_shape_preserved(self):
        """Test output shape matches input."""
        dropout = cDropout1d(p=0.3)
        x = torch.complex(torch.randn(4, 8, 16), torch.randn(4, 8, 16))
        dropout.eval()
        output = dropout(x)
        assert output.shape == x.shape
    
    def test_forward_no_nan(self):
        """Test forward pass doesn't produce NaN."""
        dropout = cDropout1d(p=0.1)
        dropout.train()
        x = torch.complex(torch.randn(2, 16, 32), torch.randn(2, 16, 32))
        output = dropout(x)
        assert not torch.isnan(output).any()


class TestMLModelsIntegration:
    """Integration tests for ML model components."""
    
    def test_stacking_blocks(self):
        """Test stacking multiple conv blocks."""
        blocks = nn.Sequential(
            create_conv_block(1, 16, useComplex=False),
            create_conv_block(16, 32, useComplex=False)
        )
        x = torch.randn(2, 1, 256)
        output = blocks(x)
        assert output.shape[0] == 2
    
    def test_complex_pipeline(self):
        """Test complex number pipeline."""
        model = nn.Sequential(
            create_conv_block(1, 16, useComplex=True),
            cDropout1d(p=0.1)
        )
        model.eval()
        x = torch.complex(torch.randn(2, 1, 256), torch.randn(2, 1, 256))
        output = model(x)
        assert output.is_complex()
    
    def test_model_device_flexibility(self):
        """Test models work with CPU (GPU optional)."""
        model = Conv_3(useComplex=False)
        x = torch.randn(2, 1, 256)
        output = model(x)
        assert output.device == x.device
