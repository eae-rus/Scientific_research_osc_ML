"""Единый lazy Dataset сырых 8-канальных окон для SSL Phase 5."""

from __future__ import annotations

import random
from typing import Mapping

import numpy as np

from .phase5_contracts import TemporalMode, periods_to_samples, spectral_positions
from .phase5_sources import DatasetSource
from .spectral_features import SpectralFeatureBuilder

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
        history_periods: float = 0.0,
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
        self.history_periods = history_periods
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
            channel_provenance = source.get_provenance(record_idx)
            spp = int(metadata["spp"])
            model_window_samples = periods_to_samples(self.window_periods, spp)
            history_samples = 0 if self.history_periods == 0 else periods_to_samples(self.history_periods, spp)
            window_samples = history_samples + model_window_samples
            if raw.shape != (8, raw.shape[1]) or raw.shape[1] < window_samples:
                continue
            start = rng.randrange(raw.shape[1] - window_samples + 1)
            return {
                "raw": raw[:, start:start + window_samples],
                "metadata": dict(metadata) | {
                    "source": name, "record_index": record_idx, "window_start": start,
                    "window_samples": window_samples, "window_periods": self.window_periods,
                    "history_samples": history_samples,
                    "history_periods": self.history_periods,
                    "model_window_samples": model_window_samples,
                    "channel_provenance": channel_provenance.tolist(),
                },
            }
        raise RuntimeError("Не удалось выбрать осциллограмму достаточной длины для окна")


class SpectralMultiSourceDataset(Dataset):
    """Спектральное представление lazy raw dataset по feature contract v2.

    Wrapper не выполняет SSL-маскирование: он возвращает неизменённый target и
    физическую missing-mask, а стратегия masked modeling добавляется отдельным
    воспроизводимым transform на этапе SSL.
    """

    def __init__(
        self,
        raw_dataset: LazyMultiSourceDataset,
        feature_builder: SpectralFeatureBuilder,
        temporal_mode: TemporalMode = "sequence_1_8",
    ) -> None:
        required_history = max(feature_builder.config.low_periods, default=1)
        if raw_dataset.history_periods < required_history:
            raise ValueError(
                f"Для causal low-period признаков нужна history_periods >= {required_history}"
            )
        self.raw_dataset = raw_dataset
        self.feature_builder = feature_builder
        self.temporal_mode = temporal_mode

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.raw_dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.raw_dataset[index]
        raw_channels_first = np.asarray(sample["raw"], dtype=np.float32)
        metadata = dict(sample["metadata"])
        spp = int(metadata["spp"])
        positions = spectral_positions(
            raw_channels_first.shape[1],
            spp,
            self.temporal_mode,
            history_periods=float(metadata["history_periods"]),
        )
        features, missing_mask, feature_metadata = self.feature_builder.build(
            raw_channels_first.T,
            spp=spp,
            positions=positions,
            voltage_basis=str(metadata.get("voltage_basis", "phase")),
            channel_provenance=metadata["channel_provenance"],
        )
        provenance = np.broadcast_to(
            np.asarray(feature_metadata["feature_provenance"], dtype=np.uint8),
            features.shape,
        ).copy()
        provenance[missing_mask] = 0
        return {
            "features": features,
            "target": features.copy(),
            "missing_mask": missing_mask,
            "provenance": provenance,
            "metadata": metadata | feature_metadata | {
                "temporal_mode": self.temporal_mode,
                "sequence_length": len(positions),
            },
        }
