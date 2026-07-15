"""Единый lazy Dataset сырых 8-канальных окон для SSL Phase 5."""

from __future__ import annotations

import random
from typing import Mapping

import numpy as np

from .phase5_contracts import periods_to_samples
from .phase5_sources import DatasetSource

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Позволяет проверять CPU-only инфраструктуру без torch.
    class Dataset:  # type: ignore[no-redef]
        pass


class LazyMultiSourceDataset(Dataset):
    """Детерминированная по index/epoch выборка источника и 10-периодного окна.

    Спектральный builder намеренно не живёт здесь: этот Dataset отвечает только
    за unified raw contract и metadata, чтобы SSL и task adapters не дублировали
    загрузку источников.
    """

    def __init__(
        self,
        sources: Mapping[str, DatasetSource],
        source_weights: Mapping[str, float],
        samples_per_epoch: int,
        window_periods: float = 10.0,
        seed: int = 42,
        max_attempts: int = 32,
    ) -> None:
        if not sources or samples_per_epoch <= 0:
            raise ValueError("Нужны непустые sources и положительное samples_per_epoch")
        self.sources = dict(sources)
        self.names = tuple(self.sources)
        self.weights = tuple(float(source_weights.get(name, 0.0)) for name in self.names)
        if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("Веса источников должны быть неотрицательны и иметь положительную сумму")
        self.samples_per_epoch = samples_per_epoch
        self.window_periods = window_periods
        self.seed = seed
        self.epoch = 0
        self.max_attempts = max_attempts

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Сменить детерминированную последовательность выборки на новую эпоху."""

        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = random.Random((self.seed, self.epoch, index).__hash__())
        for _ in range(self.max_attempts):
            name = rng.choices(self.names, weights=self.weights, k=1)[0]
            source = self.sources[name]
            record_idx = rng.randrange(len(source))
            metadata = source.get_metadata(record_idx)
            raw = source.load_signal(record_idx)
            spp = int(metadata["spp"])
            window_samples = periods_to_samples(self.window_periods, spp)
            if raw.shape != (8, raw.shape[1]) or raw.shape[1] < window_samples:
                continue
            start = rng.randrange(raw.shape[1] - window_samples + 1)
            return {
                "raw": raw[:, start:start + window_samples],
                "metadata": dict(metadata) | {
                    "source": name, "record_index": record_idx, "window_start": start,
                    "window_samples": window_samples, "window_periods": self.window_periods,
                },
            }
        raise RuntimeError("Не удалось выбрать осциллограмму достаточной длины для окна")
