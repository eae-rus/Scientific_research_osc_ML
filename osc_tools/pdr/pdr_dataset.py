"""PyTorch Task Dataset для задачи РНМ (PDRTaskDataset).

Объединяет спектральные признаки Phase 5 (Version A / Version B) с псевдометками РНМ,
сгенерированными PDRDatasetLabeler. Поддерживает маскирование прогрева (warmup_mask)
и сплиты согласно research_strict_splits.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # type: ignore

from osc_tools.ml.phase5_contracts import (
    TimebaseContract,
    TemporalMode,
    available_harmonics,
)
from osc_tools.ml.phase5_sources import DatasetSource
from osc_tools.ml.spectral_features import SpectralFeatureBuilder
from .base import PDRDirection


class PDRTaskDataset(Dataset):
    """PyTorch датасет для дообучения KAN-Transformer на задачу РНМ."""

    def __init__(
        self,
        source: DatasetSource,
        indices: List[int],
        labels_npz_path: Path,
        timebase: TimebaseContract,
        temporal_mode: TemporalMode = "snapshot_5",
        feature_version: str = "version_b",
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch не установлен в текущем окружении.")

        self.source = source
        self.indices = indices
        self.labels_path = Path(labels_npz_path)
        self.timebase = timebase
        self.temporal_mode = temporal_mode
        self.feature_version = feature_version

        # Загрузка меток из NPZ
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Файл меток РНМ не найден: {self.labels_path}")

        self.labels_npz = np.load(self.labels_path)
        self.feature_builder = SpectralFeatureBuilder(
            harmonics=available_harmonics(timebase.spp, requested=9),
            feature_version=feature_version,
        )

        # Индексация валидных (прошедших разогрев) окон для обучения
        self.samples: List[Tuple[int, int]] = []  # (record_idx, window_idx)
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        """Построение индекса образцов (исключая окна прогрева)."""
        for rec_idx in self.indices:
            prefix = f"rec_{rec_idx}"
            if f"{prefix}_dir" not in self.labels_npz:
                continue
            warmup = self.labels_npz[f"{prefix}_warmup"]
            # Находим все индексы окон, не входящие в зону прогрева
            valid_w_indices = np.where(~warmup)[0]
            for w_idx in valid_w_indices:
                self.samples.append((rec_idx, int(w_idx)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, w_idx = self.samples[idx]
        prefix = f"rec_{rec_idx}"

        # Загрузка исходных сигналов
        signal = self.source.load_signal(rec_idx)
        provenance = self.source.get_provenance(rec_idx)
        meta = self.source.get_metadata(rec_idx)
        voltage_basis = str(meta.get("voltage_basis", "phase"))

        # Извлечение меток РНМ для данного окна
        direction_val = int(self.labels_npz[f"{prefix}_dir"][w_idx])
        margin_val = float(self.labels_npz[f"{prefix}_margin"][w_idx])
        warmup_val = bool(self.labels_npz[f"{prefix}_warmup"][w_idx])
        sample_end_idx = int(self.labels_npz[f"{prefix}_samples"][w_idx])

        # Извлечение подмассива сигналов длины window_samples
        start_idx = max(0, sample_end_idx - self.timebase.window_samples + 1)
        sub_signal = signal[:, start_idx : sample_end_idx + 1]

        # Извлечение спектральных признаков KAN-Transformer
        spectral_feat = self.feature_builder.extract_features(
            sub_signal,
            provenance=provenance,
            spp=self.timebase.spp,
            voltage_basis=voltage_basis,
            mode=self.temporal_mode,
        )

        # Бинарная классификация направления: 0: REVERSE, 1: FORWARD
        target_class = 1 if direction_val == 1 else 0

        return {
            "features": torch.tensor(spectral_feat, dtype=torch.float32),
            "target_class": torch.tensor(target_class, dtype=torch.long),
            "pdr_direction": torch.tensor(direction_val, dtype=torch.int16),
            "pdr_margin": torch.tensor(margin_val, dtype=torch.float32),
            "warmup_mask": torch.tensor(warmup_val, dtype=torch.bool),
            "provenance": torch.tensor(provenance, dtype=torch.long),
            "record_id": rec_idx,
            "window_idx": w_idx,
        }
