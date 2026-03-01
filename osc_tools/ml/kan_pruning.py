from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from osc_tools.ml.kan_conv.KANLinear import KANLinear


def collect_kan_inputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
    max_samples: int = 1000
) -> Dict[str, torch.Tensor]:
    """
    Собирает входы в слои KANLinear с помощью forward-hook.

    Args:
        model: модель, содержащая KANLinear.
        loader: DataLoader для подачи примеров.
        device: устройство для инференса.
        max_batches: максимум батчей для сбора.
        max_samples: максимум сохранённых примеров на слой.

    Returns:
        Словарь {имя_слоя: тензор входов формы (N, in_features)}
    """
    inputs: Dict[str, list[torch.Tensor]] = {}

    def get_input_hook(name: str):
        def hook(_module: nn.Module, input: Tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if name not in inputs:
                inputs[name] = []
            if len(inputs[name]) < max_samples:
                inputs[name].append(input[0].detach().cpu())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            hooks.append(module.register_forward_hook(get_input_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            model(x)
            if i + 1 >= max_batches:
                break

    for h in hooks:
        h.remove()

    collected: Dict[str, torch.Tensor] = {}
    for name, chunks in inputs.items():
        if not chunks:
            continue
        x_cat = torch.cat(chunks, dim=0)
        if x_cat.shape[0] > max_samples:
            x_cat = x_cat[:max_samples]
        collected[name] = x_cat

    return collected


def calculate_kan_importance(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Рассчитывает важность связей KANLinear как среднее |phi(x)|.

    Args:
        model: модель с KANLinear слоями.
        inputs: входы слоёв, собранные collect_kan_inputs.
        device: устройство расчётов.

    Returns:
        Словарь {имя_слоя: тензор важностей (out_features, in_features)}
    """
    importances: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not isinstance(module, KANLinear) or name not in inputs:
            continue

        x_sample = inputs[name].to(device)
        scores = torch.zeros(module.out_features, module.in_features, device=device)

        # Предвычисляем сплайны
        bases = module.b_splines(x_sample)

        for out_i in range(module.out_features):
            for in_j in range(module.in_features):
                # Базовая часть
                y_base = module.base_weight[out_i, in_j] * module.base_activation(x_sample[:, in_j])

                # Сплайн-часть
                y_spline = torch.matmul(bases[:, in_j, :], module.spline_weight[out_i, in_j, :])
                if module.enable_standalone_scale_spline:
                    y_spline = y_spline * module.spline_scaler[out_i, in_j]

                y_total = y_base + y_spline
                scores[out_i, in_j] = torch.mean(torch.abs(y_total))

        importances[name] = scores.detach().cpu()

    return importances


def apply_pruning_mask(
    model: nn.Module,
    importances: Dict[str, torch.Tensor],
    threshold: float
) -> Tuple[int, int]:
    """
    Применяет маску прореживания по порогу важности.

    Args:
        model: модель с KANLinear слоями.
        importances: словарь важностей.
        threshold: порог отсечения.

    Returns:
        (active_edges, total_edges)
    """
    total_edges = 0
    active_edges = 0

    for name, module in model.named_modules():
        if not isinstance(module, KANLinear) or name not in importances:
            continue

        scores = importances[name]
        mask = (scores >= threshold).float().to(module.mask.device)
        module.mask.copy_(mask)

        n_links = module.in_features * module.out_features
        n_active = int(mask.sum().item())

        total_edges += n_links
        active_edges += n_active

    return active_edges, total_edges


def compute_importance_stats(importances: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Статистика распределений важности по слоям.

    Returns:
        {имя_слоя: {mean, median, p90, max}}
    """
    stats: Dict[str, Dict[str, float]] = {}
    for name, scores in importances.items():
        arr = scores.detach().cpu().numpy().reshape(-1)
        if arr.size == 0:
            continue
        stats[name] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr))
        }
    return stats
