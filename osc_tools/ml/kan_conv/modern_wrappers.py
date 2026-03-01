from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

import torch.nn as nn

from osc_tools.ml.layers.kan_layers import KANLinear as BaselineKANLinear


ML_DIR = Path(__file__).resolve().parents[1]
_MODULE_CACHE: dict[str, Any] = {}


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


def _load_module_from_file(module_key: str, file_path: Path) -> Any | None:
    """Загружает модуль по пути файла и кэширует его."""
    if module_key in _MODULE_CACHE:
        return _MODULE_CACHE[module_key]

    if not file_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location(module_key, file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE[module_key] = module
        return module
    except Exception:
        return None


def _resolve_local_efficient_kan_linear() -> Any | None:
    module = _load_module_from_file(
        'phase3_local_efficient_kan',
        ML_DIR / 'efficient-kan-master' / 'src' / 'efficient_kan' / 'kan.py',
    )
    if module is None:
        return None
    return getattr(module, 'KANLinear', None)


def _resolve_fast_kan_layer() -> Any | None:
    module = _load_module_from_file(
        'phase3_local_fast_kan',
        ML_DIR / 'fast-kan-master' / 'fastkan' / 'fastkan.py',
    )
    if module is None:
        return None
    return getattr(module, 'FastKANLayer', None)


def _resolve_cheby_kan_layer() -> Any | None:
    module = _load_module_from_file(
        'phase3_local_cheby_kan',
        ML_DIR / 'ChebyKAN-main' / 'ChebyKANLayer.py',
    )
    if module is None:
        return None
    return getattr(module, 'ChebyKANLayer', None)


def _resolve_wav_kan_linear() -> Any | None:
    module = _load_module_from_file(
        'phase3_local_wav_kan',
        ML_DIR / 'Wav-KAN-main' / 'KAN.py',
    )
    if module is None:
        return None
    return getattr(module, 'KANLinear', None)


def _resolve_torchconv_kans_layers_module() -> Any | None:
    """Пытается импортировать kans.layers из локальной копии torch-conv-kan-main."""
    repo_root = ML_DIR / 'torch-conv-kan-main'
    if not repo_root.exists():
        return None

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    try:
        return importlib.import_module('kans.layers')
    except Exception:
        return None


def _resolve_torch_wavelet_layer() -> Any | None:
    module = _resolve_torchconv_kans_layers_module()
    if module is None:
        return None
    return getattr(module, 'WavKANLayer', None)


def _create_efficient_kan_linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    base_activation: type[nn.Module],
) -> nn.Module:
    """Создаёт слой efficient-kan; при несовместимости безопасно откатывается к baseline."""
    layer_cls = _resolve_efficient_kan_linear() or _resolve_local_efficient_kan_linear()
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


def _create_fast_kan_linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    base_activation: type[nn.Module],
) -> nn.Module:
    """Создаёт слой FastKANLayer (локальная копия fast-kan)."""
    layer_cls = _resolve_fast_kan_layer()
    if layer_cls is None:
        warnings.warn(
            'Не удалось загрузить FastKANLayer, используется baseline KANLinear.',
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

    try:
        return layer_cls(
            input_dim=in_features,
            output_dim=out_features,
            num_grids=grid_size,
            use_base_update=True,
            use_layernorm=True,
        )
    except Exception:
        warnings.warn(
            'FastKANLayer несовместим с текущими параметрами, используется baseline KANLinear.',
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


def _create_cheby_kan_linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    base_activation: type[nn.Module],
) -> nn.Module:
    """Создаёт слой ChebyKANLayer (локальная копия ChebyKAN)."""
    layer_cls = _resolve_cheby_kan_layer()
    if layer_cls is None:
        warnings.warn(
            'Не удалось загрузить ChebyKANLayer, используется baseline KANLinear.',
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

    try:
        degree = max(2, int(grid_size))
        return layer_cls(
            input_dim=in_features,
            output_dim=out_features,
            degree=degree,
        )
    except Exception:
        warnings.warn(
            'ChebyKANLayer несовместим с текущими параметрами, используется baseline KANLinear.',
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


def _create_wav_kan_linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    base_activation: type[nn.Module],
) -> nn.Module:
    """Создаёт слой Wav-KAN (локальная копия Wav-KAN-main)."""
    layer_cls = _resolve_wav_kan_linear()
    if layer_cls is None:
        warnings.warn(
            'Не удалось загрузить Wav-KAN KANLinear, используется baseline KANLinear.',
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

    try:
        return layer_cls(
            in_features=in_features,
            out_features=out_features,
            wavelet_type='mexican_hat',
        )
    except Exception:
        warnings.warn(
            'Wav-KAN KANLinear несовместим с текущими параметрами, используется baseline KANLinear.',
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


def _create_torch_wavelet_kan_linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    base_activation: type[nn.Module],
) -> nn.Module:
    """Создаёт слой WavKANLayer из локальной копии torch-wavelet-kan/torch-conv-kan."""
    layer_cls = _resolve_torch_wavelet_layer()
    if layer_cls is None:
        warnings.warn(
            'Не удалось загрузить WavKANLayer из torch-conv-kan-main, используется baseline KANLinear.',
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

    try:
        return layer_cls(
            in_features=in_features,
            out_features=out_features,
            wavelet_type='mexican_hat',
        )
    except Exception:
        warnings.warn(
            'WavKANLayer (torch-wavelet-kan) несовместим с текущими параметрами, используется baseline KANLinear.',
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
    """Фабрика KANLinear с выбором backend (`baseline`, `efficient`, `fast`, `cheby`, `wav`, `torch_wavelet`)."""
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

    if backend_norm in {'fast', 'fast-kan'}:
        return _create_fast_kan_linear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    if backend_norm in {'cheby', 'cheby-kan', 'chebykan'}:
        return _create_cheby_kan_linear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    if backend_norm in {'wav', 'wav-kan', 'wavelet', 'wavelet-kan'}:
        return _create_wav_kan_linear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    if backend_norm in {'torch_wavelet', 'torch-wavelet', 'torch-wavelet-kan'}:
        return _create_torch_wavelet_kan_linear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    raise ValueError(f'Неизвестный backend KAN: {backend}')
