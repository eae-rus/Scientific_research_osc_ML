"""PyTorch Task Dataset для задачи РНМ (PDRTaskDataset).

Объединяет спектральные признаки Phase 5 (Version A / Version B) с псевдометками РНМ,
сгенерированными PDRDatasetLabeler. Поддерживает маскирование неразмеченных зон (UNLABELED=-999)
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
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig
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
        feature_version: str = "B",
        include_warmup: bool = False,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch не установлен в текущем окружении.")

        self.source = source
        self.indices = indices
        self.labels_path = Path(labels_npz_path)
        self.timebase = timebase
        self.temporal_mode = temporal_mode
        self.feature_version = feature_version
        self.include_warmup = include_warmup

        # Загрузка меток из NPZ
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Файл меток РНМ не найден: {self.labels_path}")

        with np.load(self.labels_path) as data:
            self.labels_dict = {k: data[k] for k in data.files}

        ver = "B" if str(feature_version).upper().endswith("B") else "A"
        feat_config = SpectralFeatureConfig(version=ver)
        self.feature_builder = SpectralFeatureBuilder(config=feat_config)

        # Индексация валидных окон для обучения
        self.samples: List[Tuple[int, int]] = []  # (record_idx, window_idx)
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        """Построение индекса образцов (исключая неразмеченные окна UNLABELED)."""
        for rec_idx in self.indices:
            prefix = f"rec_{rec_idx}"
            if f"{prefix}_dir" not in self.labels_dict:
                continue
            dirs = self.labels_dict[f"{prefix}_dir"]
            warmup = self.labels_dict[f"{prefix}_warmup"]

            # Валидные окна: разметка не равна UNLABELED (-999)
            if self.include_warmup:
                valid_mask = (dirs != int(PDRDirection.UNLABELED))
            else:
                valid_mask = (dirs != int(PDRDirection.UNLABELED)) & (~warmup)

            valid_w_indices = np.where(valid_mask)[0]
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
        direction_val = int(self.labels_dict[f"{prefix}_dir"][w_idx])
        margin_val = float(self.labels_dict[f"{prefix}_margin"][w_idx])
        warmup_val = bool(self.labels_dict[f"{prefix}_warmup"][w_idx])
        sample_end_idx = int(self.labels_dict[f"{prefix}_samples"][w_idx])

        # Извлечение подмассива сигналов длины window_samples с дополнением слева при необходимости
        start_idx = max(0, sample_end_idx - self.timebase.window_samples + 1)
        sub_signal_raw = signal[:, start_idx : sample_end_idx + 1]

        target_len = self.timebase.window_samples
        if sub_signal_raw.shape[1] < target_len:
            pad_len = target_len - sub_signal_raw.shape[1]
            sub_signal = np.pad(sub_signal_raw, ((0, 0), (pad_len, 0)), mode="edge")
        else:
            sub_signal = sub_signal_raw

        # Извлечение спектральных признаков KAN-Transformer (передаем sub_signal формы (T, 8))
        spectral_feat, missing_mask, _meta_feat = self.feature_builder.build(
            sub_signal.T,
            spp=self.timebase.spp,
            voltage_basis=voltage_basis,
            channel_provenance=provenance,
        )

        # Бинарная классификация направления: 0: REVERSE, 1: FORWARD
        target_class = 1 if direction_val == 1 else 0

        return {
            "features": torch.tensor(spectral_feat, dtype=torch.float32),
            "missing_mask": torch.tensor(missing_mask, dtype=torch.bool),
            "target_class": torch.tensor(target_class, dtype=torch.long),
            "pdr_direction": torch.tensor(direction_val, dtype=torch.int16),
            "pdr_margin": torch.tensor(margin_val, dtype=torch.float32),
            "warmup_mask": torch.tensor(warmup_val, dtype=torch.bool),
            "provenance": torch.tensor(provenance, dtype=torch.long),
            "record_id": rec_idx,
            "window_idx": w_idx,
        }
