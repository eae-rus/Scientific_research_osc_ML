import torch
from torch.utils.data import DataLoader, TensorDataset

from osc_tools.ml.kan_conv.KANLinear import KANLinear
from osc_tools.ml.kan_pruning import (
    collect_kan_inputs,
    calculate_kan_importance,
    apply_pruning_mask
)


def test_kan_pruning_pipeline_shapes() -> None:
    """Проверка формы важностей и базового пайплайна."""
    torch.manual_seed(42)

    model = torch.nn.Sequential(
        KANLinear(in_features=4, out_features=3, grid_size=3)
    )

    x = torch.randn(32, 4)
    y = torch.zeros(32)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    device = torch.device('cpu')
    inputs = collect_kan_inputs(model, loader, device, max_batches=2, max_samples=16)

    assert len(inputs) == 1
    layer_name = list(inputs.keys())[0]
    assert inputs[layer_name].shape[1] == 4

    importances = calculate_kan_importance(model, inputs, device)
    assert layer_name in importances
    assert importances[layer_name].shape == (3, 4)


def test_kan_pruning_mask_application() -> None:
    """Проверка применения маски прореживания."""
    torch.manual_seed(0)

    model = torch.nn.Sequential(
        KANLinear(in_features=2, out_features=2, grid_size=3)
    )

    x = torch.randn(16, 2)
    y = torch.zeros(16)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    device = torch.device('cpu')
    inputs = collect_kan_inputs(model, loader, device, max_batches=1, max_samples=8)
    importances = calculate_kan_importance(model, inputs, device)

    # Порог выше максимума — все связи должны обнулиться
    high_threshold = float(importances[list(importances.keys())[0]].max().item() + 1.0)
    active_edges, total_edges = apply_pruning_mask(model, importances, high_threshold)

    assert total_edges == 4
    assert active_edges == 0
