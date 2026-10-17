from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extract_thresholds_dict(payload: Any) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None

    if payload and all(isinstance(key, str) and _is_number(value)
                       for key, value in payload.items()):
        return {key: float(value) for key, value in payload.items()}

    direct = payload.get('thresholds')
    if isinstance(direct, dict):
        extracted = _extract_thresholds_dict(direct)
        if extracted:
            return extracted

    per_class = payload.get('per_class')
    if isinstance(per_class, dict):
        extracted: dict[str, float] = {}
        for key, value in per_class.items():
            if isinstance(value, dict) and _is_number(value.get('threshold')):
                extracted[key] = float(value['threshold'])
        if extracted:
            return extracted

    for nested_key in ('optimal_thresholds', 'metrics_window_level', 'metrics'):
        nested = payload.get(nested_key)
        extracted = _extract_thresholds_dict(nested)
        if extracted:
            return extracted

    return None


def load_thresholds_from_json(path: str | Path) -> dict[str, float]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f'JSON с уставками не найден: {json_path}')

    with open(json_path, encoding='utf-8') as f:
        payload = json.load(f)

    thresholds = _extract_thresholds_dict(payload)
    if not thresholds:
        raise ValueError(f'Не удалось извлечь пороги из JSON: {json_path}')
    return thresholds


def normalize_threshold_map(
    class_names: Sequence[str],
    threshold: float = 0.5,
    per_class_thresholds: Mapping[str, float] | None = None,
) -> dict[str, float]:
    default_threshold = float(threshold)
    if per_class_thresholds is None:
        return {name: default_threshold for name in class_names}
    return {
        name: float(per_class_thresholds.get(name, default_threshold))
        for name in class_names
    }


def resolve_threshold_config(
    class_names: Sequence[str],
    threshold: float | Mapping[str, float] = 0.5,
    thresholds_json: str | Path | None = None,
    per_class_thresholds: Mapping[str, float] | None = None,
    source_label: str | None = None,
) -> dict[str, Any]:
    if isinstance(threshold, Mapping):
        per_class_thresholds = threshold
        threshold = 0.5

    default_threshold = float(threshold)
    resolved_source = source_label or 'fixed'

    if thresholds_json is not None:
        per_class_thresholds = load_thresholds_from_json(thresholds_json)
        resolved_source = str(thresholds_json)

    threshold_map = normalize_threshold_map(class_names, default_threshold, per_class_thresholds)
    is_per_class = per_class_thresholds is not None

    return {
        'mode': 'per_class' if is_per_class else 'fixed',
        'default_threshold': default_threshold,
        'per_class_thresholds': threshold_map if is_per_class else None,
        'threshold_spec': threshold_map if is_per_class else default_threshold,
        'source': resolved_source,
    }


def threshold_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'mode': config.get('mode', 'fixed'),
        'default_threshold': float(config.get('default_threshold', 0.5)),
        'source': config.get('source', 'fixed'),
        'per_class_thresholds': config.get('per_class_thresholds'),
    }


def thresholds_for_classes(
    class_names: Sequence[str],
    threshold: float | Mapping[str, float] = 0.5,
) -> np.ndarray:
    if isinstance(threshold, Mapping):
        threshold_map = normalize_threshold_map(class_names, 0.5, threshold)
        return np.array([threshold_map[name] for name in class_names], dtype=np.float32)
    return np.full(len(class_names), float(threshold), dtype=np.float32)


def binarize_probabilities(
    probs: np.ndarray,
    class_names: Sequence[str],
    threshold: float | Mapping[str, float] = 0.5,
) -> np.ndarray:
    thr = thresholds_for_classes(class_names, threshold).reshape(1, -1)
    return (probs >= thr).astype(np.int8)


def any_positive_mask(
    probs: np.ndarray,
    class_names: Sequence[str],
    threshold: float | Mapping[str, float] = 0.5,
) -> np.ndarray:
    binary = binarize_probabilities(probs, class_names, threshold)
    valid = ~np.isnan(probs)
    return np.any((binary > 0) & valid, axis=1)


def threshold_label(config: Mapping[str, Any]) -> str:
    if config.get('mode') == 'per_class':
        source = str(config.get('source', 'per_class'))
        return f'per-class ({Path(source).name})' if source.endswith('.json') else source
    return f"fixed={float(config.get('default_threshold', 0.5)):.3f}"