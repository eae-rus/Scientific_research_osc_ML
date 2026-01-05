from typing import Optional
import torch
from torch.utils.data import DataLoader


def compute_pos_weight_from_loader(loader: DataLoader, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Вычисляет вектор `pos_weight` для многометочной задачи по данным из `DataLoader`.

    Формула (как в плане):
        w_i = N_total / (N_classes * N_i)

    Args:
        loader: DataLoader возвращающий пары (x, y), где y - тензор формы (batch, n_classes).
        device: устройство для результирующего тензора (если None, CPU).

    Returns:
        torch.Tensor размера (n_classes,) с dtype=float32
    """
    total = 0
    pos_counts = None

    for _, y in loader:
        if y is None:
            continue
        # Приводим к float
        y = y.float()
        if pos_counts is None:
            # Поддержка как векторных меток (batch, C) или (batch,) для single-label
            if y.dim() == 1:
                # Сделаем колонку
                y = y.unsqueeze(1)
            pos_counts = torch.zeros(y.shape[1], dtype=torch.float64)

        if y.dim() == 1:
            y = y.unsqueeze(1)

        pos_counts += y.sum(dim=0).double()
        total += y.shape[0]

    if pos_counts is None:
        raise ValueError("No labels found in loader to compute pos_weight")

    n_classes = pos_counts.shape[0]
    
    # Вычисляем отрицательные примеры
    neg_counts = total - pos_counts
    
    # Защита от деления на ноль
    pos_counts_safe = pos_counts.clone()
    pos_counts_safe[pos_counts_safe == 0.0] = 1.0

    # Формула для pos_weight в BCEWithLogitsLoss: количество отрицательных / количество положительных
    weights = neg_counts / pos_counts_safe

    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor
