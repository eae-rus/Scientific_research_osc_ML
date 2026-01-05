import torch
from torch.utils.data import DataLoader, TensorDataset
from osc_tools.ml.class_weights import compute_pos_weight_from_loader


def test_compute_pos_weight_basic():
    # Создадим искусственные метки: 6 сэмплов, 3 класса
    # Положительные по классам: [1, 2, 3]
    labels = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=torch.float32)

    ds = TensorDataset(torch.zeros(len(labels), 1), labels)
    loader = DataLoader(ds, batch_size=2)

    pw = compute_pos_weight_from_loader(loader)
    # Ожидаем: 
    # Class 0: 1 pos, 5 neg -> 5/1 = 5.0
    # Class 1: 2 pos, 4 neg -> 4/2 = 2.0
    # Class 2: 3 pos, 3 neg -> 3/3 = 1.0
    expected = torch.tensor([5.0, 2.0, 1.0], dtype=torch.float32)

    assert torch.allclose(pw, expected, atol=1e-6)


def test_compute_pos_weight_with_zero_counts():
    # Один класс вообще не встречается
    labels = torch.tensor([
        [0, 1],
        [0, 1],
        [0, 1],
    ], dtype=torch.float32)

    ds = TensorDataset(torch.zeros(len(labels), 1), labels)
    loader = DataLoader(ds, batch_size=2)

    pw = compute_pos_weight_from_loader(loader)
    # Class 0: 0 pos, 3 neg -> safe_pos=1 -> 3/1 = 3.0
    # Class 1: 3 pos, 0 neg -> 0/3 = 0.0
    expected = torch.tensor([3.0, 0.0], dtype=torch.float32)

    assert torch.allclose(pw, expected, atol=1e-6)
