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
    # Ожидаем: total=6, n_classes=3 -> [6/(3*1)=2.0, 6/(3*2)=1.0, 6/(3*3)=0.666...]
    expected = torch.tensor([2.0, 1.0, 6.0/(3*3)], dtype=torch.float32)

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
    # total=3, n_classes=2, counts=[0,3] -> safe counts=[1,3]
    expected = torch.tensor([3.0/(2*1), 3.0/(2*3)], dtype=torch.float32)

    assert torch.allclose(pw, expected, atol=1e-6)
