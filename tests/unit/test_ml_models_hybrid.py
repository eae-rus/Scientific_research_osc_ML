import torch
from osc_tools.ml.models.hybrid import HybridMLP, HybridCNN


def test_hybrid_mlp_features_tail_len():
    model = HybridMLP(
        in_channels=24,
        num_classes=4,
        hidden_sizes=[64, 32],
        raw_channels=8,
        features_channels=16,
        seq_len=20,
        features_seq_len=1
    )
    assert model.features_input_size == 16

    x = torch.randn(2, 24, 20)
    y = model(x)
    assert y.shape == (2, 4)


def test_hybrid_cnn_features_tail_len():
    model = HybridCNN(
        in_channels=24,
        num_classes=4,
        channels=[8, 16],
        raw_channels=8,
        features_channels=16,
        features_seq_len=1
    )
    x = torch.randn(2, 24, 20)
    y = model(x)
    assert y.shape == (2, 4)
