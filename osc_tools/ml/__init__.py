"""Инфраструктура ML-проекта.

Подпакеты не импортируются здесь eagerly: подготовка данных Phase 5 должна
запускаться на CPU-машине без установленного PyTorch. Импорт
``osc_tools.ml.models`` остаётся доступен через обычный импорт подпакета.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
	'models',
	'blocks',
	'complex_ops',
	'OscillogramDataset',
	'PrecomputedDataset',
	'create_precomputed_dataset',
]


def __getattr__(name: str) -> Any:
    """Лениво сохранить legacy exports, не требуя torch для data tools."""

    if name == "models":
        return import_module(".models", __name__)
    if name in {"blocks", "complex_ops"}:
        return import_module(f".layers.{name}", __name__)
    if name == "OscillogramDataset":
        return import_module(".dataset", __name__).OscillogramDataset
    if name in {"PrecomputedDataset", "create_precomputed_dataset"}:
        return getattr(import_module(".precomputed_dataset", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
