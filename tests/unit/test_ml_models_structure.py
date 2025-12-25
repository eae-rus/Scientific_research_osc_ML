import pytest
import torch
import torch.nn as nn
from osc_tools.ml import models as Model

def test_models_import():
    """Проверка, что все модели корректно импортируются из пакета."""
    assert hasattr(Model, 'PDR_MLP_v1')
    assert hasattr(Model, 'PDR_MLP_v2')
    assert hasattr(Model, 'Conv_3')
    assert hasattr(Model, 'FFT_MLP_COMPLEX_v1')
    assert hasattr(Model, 'CONV_MLP_v2')

def test_pdr_mlp_v2_instantiation():
    """Проверка создания PDR_MLP_v2."""
    model = Model.PDR_MLP_v2(input_features=12)
    assert isinstance(model, nn.Module)
    assert model.num_blocks == 3 # Default config

def test_fft_mlp_complex_v1_instantiation():
    """Проверка создания FFT_MLP_COMPLEX_v1."""
    model = Model.FFT_MLP_COMPLEX_v1(frame_size=64, channel_num=5, hidden_size=20)
    assert isinstance(model, nn.Module)

def test_base_model_methods():
    """Проверка методов BaseModel."""
    model = Model.PDR_MLP_v2(input_features=12)
    assert hasattr(model, 'save')
    assert hasattr(model, 'load')
    assert hasattr(model, 'predict')
    
    # Проверка predict
    x = torch.randn(2, 12, 1)
    output = model.predict(x)
    assert output.shape == (2, 1)
