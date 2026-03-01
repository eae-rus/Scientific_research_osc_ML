from __future__ import annotations

import importlib
import warnings
from typing import Any

import torch.nn as nn

from osc_tools.ml.layers.kan_layers import KANLinear as BaselineKANLinear


def _create_baseline_kan_linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    base_activation: type[nn.Module],
) -> nn.Module:
    return BaselineKANLinear(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order,
        base_activation=base_activation,
    )


def _resolve_efficient_kan_linear() -> Any | None:
    """Пытается найти класс линейного слоя из efficient-kan по нескольким вариантам импорта."""
    candidates = [
        ('efficient_kan', 'KANLinear'),
        ('efficient_kan', 'EfficientKANLinear'),
        ('efficient_kan.layers', 'KANLinear'),
        ('efficient_kan.layers', 'EfficientKANLinear'),
    ]

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            layer_cls = getattr(module, class_name, None)
            if layer_cls is not None:
                return layer_cls
        except Exception:
            continue

    return None


def _create_efficient_kan_linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    base_activation: type[nn.Module],
) -> nn.Module:
    """Создаёт слой efficient-kan; при несовместимости безопасно откатывается к baseline."""
    layer_cls = _resolve_efficient_kan_linear()
    if layer_cls is None:
        warnings.warn(
            'Не удалось импортировать efficient-kan, используется baseline KANLinear.',
            RuntimeWarning,
            stacklevel=2,
        )
        return _create_baseline_kan_linear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    constructor_attempts = [
        {
            'in_features': in_features,
            'out_features': out_features,
            'grid_size': grid_size,
            'spline_order': spline_order,
            'base_activation': base_activation,
        },
        {
            'in_features': in_features,
            'out_features': out_features,
            'grid_size': grid_size,
        },
        {
            'in_features': in_features,
            'out_features': out_features,
        },
        {
            'input_dim': in_features,
            'output_dim': out_features,
        },
    ]

    for kwargs in constructor_attempts:
        try:
            return layer_cls(**kwargs)
        except TypeError:
            continue
        except Exception:
            break

    warnings.warn(
        'Слой efficient-kan недоступен в текущей сигнатуре, используется baseline KANLinear.',
        RuntimeWarning,
        stacklevel=2,
    )
    return _create_baseline_kan_linear(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order,
        base_activation=base_activation,
    )


def build_kan_linear(
    backend: str,
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int = 3,
    base_activation: type[nn.Module] = nn.SiLU,
) -> nn.Module:
    """Фабрика KANLinear с выбором backend (`baseline` или `efficient`)."""
    backend_norm = backend.strip().lower()

    if backend_norm in {'baseline', 'pykan'}:
        return _create_baseline_kan_linear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    if backend_norm in {'efficient', 'efficient-kan'}:
        return _create_efficient_kan_linear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    raise ValueError(f'Неизвестный backend KAN: {backend}')
