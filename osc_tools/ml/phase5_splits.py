"""Воспроизводимые train/validation/never-seen splits Phase 5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .phase5_sources import DatasetSource


class IndexedDatasetSource(DatasetSource):
    """Read-only view источника по зафиксированному списку record indices."""

    def __init__(self, source: DatasetSource, indices: Sequence[int], split_name: str) -> None:
        self.source = source
        self.indices = tuple(int(index) for index in indices)
        self.name = source.name
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.indices)

    def get_metadata(self, idx: int) -> dict[str, object]:
        source_index = self.indices[idx]
        return self.source.get_metadata(source_index) | {
            "source_record_index": source_index,
            "split": self.split_name,
        }

    def load_signal(self, idx: int) -> np.ndarray:
        return self.source.load_signal(self.indices[idx])

    def get_provenance(self, idx: int) -> np.ndarray:
        return self.source.get_provenance(self.indices[idx])


def assign_keys(
    keys: Sequence[str],
    source_name: str,
    validation_fraction: float = 0.1,
    holdout_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Разделить записи, не разрывая одинаковые group keys между splits."""

    if validation_fraction < 0 or holdout_fraction < 0 or validation_fraction + holdout_fraction >= 1:
        raise ValueError("Некорректные доли validation/holdout")
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(str(key), []).append(index)
    ranked = sorted(groups, key=lambda key: _stable_rank(source_name, key, seed))
    n_groups = len(ranked)
    if n_groups <= 24:
        # Для малого числа сильно неравных файлов ищем subset, близкий к доле
        # записей. Полный перебор при 18 Open_EE CSV остаётся дешёвым.
        total_records = len(keys)
        holdout_keys = _closest_weight_subset(
            ranked, groups, round(total_records * holdout_fraction)
        ) if holdout_fraction else set()
        remaining = [key for key in ranked if key not in holdout_keys]
        validation_keys = _closest_weight_subset(
            remaining, groups, round(total_records * validation_fraction)
        ) if validation_fraction else set()
    else:
        n_holdout = max(1, round(n_groups * holdout_fraction)) if holdout_fraction else 0
        n_validation = max(1, round(n_groups * validation_fraction)) if validation_fraction else 0
        holdout_keys = set(ranked[:n_holdout])
        validation_keys = set(ranked[n_holdout:n_holdout + n_validation])
    result = {"train": [], "validation": [], "holdout": []}
    for key, indices in groups.items():
        split = "holdout" if key in holdout_keys else "validation" if key in validation_keys else "train"
        result[split].extend(indices)
    for indices in result.values():
        indices.sort()
    return result


def split_manifest_hash(manifest: dict[str, object]) -> str:
    payload = dict(manifest)
    payload.pop("sha256", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_split_manifest(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    result = dict(manifest)
    result["sha256"] = split_manifest_hash(result)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _stable_rank(source_name: str, key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source_name}:{key}".encode("utf-8")).hexdigest()


def _closest_weight_subset(
    ordered_keys: Sequence[str],
    groups: dict[str, list[int]],
    target_records: int,
) -> set[str]:
    """Найти детерминированный file subset с числом записей ближе к target."""

    if target_records <= 0 or not ordered_keys:
        return set()
    weights = [len(groups[key]) for key in ordered_keys]
    best_mask = 0
    best_distance = float("inf")
    for mask in range(1, 1 << len(ordered_keys)):
        total = sum(weight for index, weight in enumerate(weights) if mask & (1 << index))
        distance = abs(total - target_records)
        if distance < best_distance:
            best_distance = distance
            best_mask = mask
            if distance == 0:
                break
    return {key for index, key in enumerate(ordered_keys) if best_mask & (1 << index)}
