"""Независимое от PyTorch masked modeling для Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral_features import FeatureSchema


@dataclass(frozen=True)
class SpectralMaskingConfig:
    """Доля маскируемых пар ``(время, физическая группа)``."""

    ratio: float = 0.25
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError("Mask ratio должен лежать строго между 0 и 1")


def apply_group_mask(
    features: np.ndarray,
    missing_mask: np.ndarray,
    schema: FeatureSchema,
    config: SpectralMaskingConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Скрыть целые физические группы в отдельных временных токенах.

    Missing-признаки никогда не выбираются как SSL target. Возвращаемая mask
    означает только позиции reconstruction loss, а не физическое отсутствие.
    """

    features = np.asarray(features, dtype=np.float32)
    missing_mask = np.asarray(missing_mask, dtype=bool)
    if features.ndim != 2 or missing_mask.shape != features.shape:
        raise ValueError("features/missing_mask должны иметь одинаковую форму (T, C)")
    if features.shape[1] != len(schema.names) or len(schema.groups) != len(schema.names):
        raise ValueError("FeatureSchema несовместима с размерностью features")

    groups = tuple(dict.fromkeys(schema.groups))
    indices_by_group = {
        group: np.asarray([i for i, value in enumerate(schema.groups) if value == group])
        for group in groups
    }
    candidates = [
        (time_index, group)
        for time_index in range(features.shape[0])
        for group in groups
        if np.any(~missing_mask[time_index, indices_by_group[group]])
    ]
    if not candidates:
        raise ValueError("Нет доступных признаков для masked modeling")

    count = max(1, int(np.floor(len(candidates) * config.ratio + 0.5)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    reconstruction_mask = np.zeros_like(missing_mask)
    for candidate_index in np.atleast_1d(chosen):
        time_index, group = candidates[int(candidate_index)]
        indices = indices_by_group[group]
        reconstruction_mask[time_index, indices] = ~missing_mask[time_index, indices]

    model_input = features.copy()
    model_input[reconstruction_mask] = config.mask_value
    return model_input, reconstruction_mask


class MaskedSpectralDataset:
    """Epoch-aware SSL transform поверх ``SpectralMultiSourceDataset``."""

    def __init__(
        self,
        dataset: object,
        config: SpectralMaskingConfig,
        seed: int = 42,
        max_attempts: int = 16,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts должен быть положительным")
        self.dataset = dataset
        self.config = config
        self.seed = seed
        self.max_attempts = max_attempts
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        setter = getattr(self.dataset, "set_epoch", None)
        if setter is not None:
            setter(epoch)

    def __getitem__(self, index: int) -> dict[str, object]:
        schema = self.dataset.feature_builder.schema
        dataset_length = len(self.dataset)  # type: ignore[arg-type]
        rejected: list[dict[str, object]] = []
        for attempt in range(self.max_attempts):
            # Нечётный шаг даёт полный цикл для типичных степеней двойки smoke-size.
            sampled_index = (int(index) + attempt * 7919) % dataset_length
            sample = self.dataset[sampled_index]  # type: ignore[index]
            metadata = dict(sample["metadata"])
            mask_seed = self.seed + self.epoch * 1_000_003 + sampled_index * 97
            try:
                model_input, reconstruction_mask = apply_group_mask(
                    sample["features"], sample["missing_mask"], schema, self.config, mask_seed
                )
            except ValueError as exc:
                if str(exc) != "Нет доступных признаков для masked modeling":
                    raise
                rejected.append({
                    "sampled_index": sampled_index,
                    "source": metadata.get("source"),
                    "source_record_index": metadata.get("source_record_index"),
                    "record_index": metadata.get("record_index"),
                })
                continue
            return dict(sample) | {
                "features": model_input,
                "target": np.asarray(sample["target"], dtype=np.float32),
                "reconstruction_mask": reconstruction_mask,
                "metadata": metadata | {
                    "ssl_mask_ratio": self.config.ratio,
                    "ssl_mask_seed": mask_seed,
                    "requested_dataset_index": int(index),
                    "sampled_dataset_index": sampled_index,
                    "ssl_resample_attempt": attempt,
                },
            }
        raise RuntimeError(
            "Не удалось найти запись с доступными признаками для masked modeling "
            f"за {self.max_attempts} попыток; rejected={rejected}"
        )
