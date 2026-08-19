"""PyTorch Task Dataset для задачи РНМ (PDRTaskDataset).

Объединяет спектральные признаки Phase 5 (Version A / Version B) с псевдометками РНМ,
сгенерированными PDRDatasetLabeler. UNLABELED используется как отрицательная цель
применимости, но маскируется для направления/margin. Поддерживает strict splits.
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
    periods_to_samples,
    spectral_positions,
)
from osc_tools.ml.phase5_sources import DatasetSource
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig
from .base import PDRDirection
from .signal_analysis import derive_missing_currents
from .study import PDRStudyLabelStore


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
        teacher_algorithm_id: Optional[str] = None,
        include_unlabeled_for_applicability: bool = True,
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
        self.include_unlabeled_for_applicability = include_unlabeled_for_applicability

        # Legacy single-NPZ остаётся совместимым; новый массовый формат читается
        # лениво по shards и не загружает сотни миллионов меток в RAM.
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Файл меток РНМ не найден: {self.labels_path}")
        self.label_store: Optional[PDRStudyLabelStore] = None
        self.labels_dict: Dict[str, np.ndarray] = {}
        if self.labels_path.is_dir() or self.labels_path.name == "manifest.json":
            self.label_store = PDRStudyLabelStore(
                self.labels_path,
                algorithm_id=teacher_algorithm_id,
            )
        else:
            with np.load(self.labels_path) as data:
                self.labels_dict = {k: data[k] for k in data.files}

        ver = "B" if str(feature_version).upper().endswith("B") else "A"
        feat_config = SpectralFeatureConfig(version=ver)
        self.feature_builder = SpectralFeatureBuilder(config=feat_config)
        self.feature_history_periods = float(max(feat_config.low_periods, default=1))

        # Индексация валидных окон для обучения
        self.samples: List[Tuple[int, int]] = []  # (record_idx, window_idx)
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        """Построение индекса направления и применимости по causal-окнам."""
        for rec_idx in self.indices:
            record_labels = self._record_labels(rec_idx)
            if record_labels is None:
                continue
            dirs = record_labels["directions"]
            warmup = record_labels["warmup"]
            sample_indices = record_labels["samples"]
            record_timebase = self._record_timebase(rec_idx)
            required_samples = periods_to_samples(
                self.feature_history_periods + record_timebase.window_periods,
                record_timebase.spp,
            )

            # UNLABELED после полного causal-контекста — отдельная цель
            # применимости органа. Это не третий класс направления.
            if self.include_unlabeled_for_applicability:
                valid_mask = np.ones_like(dirs, dtype=bool)
            else:
                valid_mask = (dirs != int(PDRDirection.UNLABELED))
            if not self.include_warmup:
                valid_mask &= ~warmup
            # Для совместимости с SSL backbone нужен полный causal-фрагмент:
            # feature history (до 10 периодов для lp10) + 10 периодов модели.
            valid_mask &= sample_indices >= (required_samples - 1)

            valid_w_indices = np.where(valid_mask)[0]
            for w_idx in valid_w_indices:
                self.samples.append((rec_idx, int(w_idx)))

    def _record_timebase(self, rec_idx: int) -> TimebaseContract:
        """Получить временной контракт конкретной записи, а не всего источника."""
        meta = self.source.get_metadata(rec_idx)
        sampling_rate = meta.get("f_adc", meta.get("sampling_rate_hz"))
        network_frequency = meta.get("f_network", meta.get("network_frequency_hz"))
        if sampling_rate is None or network_frequency is None:
            return self.timebase
        return TimebaseContract.create(
            float(sampling_rate),
            float(network_frequency),
            window_periods=self.timebase.window_periods,
            stride_fraction=self.timebase.stride_fraction,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, w_idx = self.samples[idx]
        record_labels = self._record_labels(rec_idx)
        if record_labels is None:
            raise KeyError(f"Разметка записи {rec_idx} исчезла после построения индекса")

        # Загрузка исходных сигналов
        signal = self.source.load_signal(rec_idx)
        source_provenance = self.source.get_provenance(rec_idx)
        signal, provenance = derive_missing_currents(signal, source_provenance)
        meta = self.source.get_metadata(rec_idx)
        voltage_basis = str(meta.get("voltage_basis", "phase"))
        record_timebase = self._record_timebase(rec_idx)

        # Извлечение меток РНМ для данного окна
        direction_val = int(record_labels["directions"][w_idx])
        margin_val = float(record_labels["margins"][w_idx])
        warmup_val = bool(record_labels["warmup"][w_idx])
        confidence_val = float(record_labels["confidences"][w_idx])
        sample_end_idx = int(record_labels["samples"][w_idx])

        total_periods = self.feature_history_periods + record_timebase.window_periods
        total_samples = periods_to_samples(total_periods, record_timebase.spp)
        start_idx = sample_end_idx - total_samples + 1
        if start_idx < 0:
            raise RuntimeError("В индекс PDRTaskDataset попала точка без полного causal-контекста")
        sub_signal_raw = signal[:, start_idx : sample_end_idx + 1]
        if sub_signal_raw.shape[1] != total_samples:
            raise RuntimeError("Длина causal-фрагмента не совпадает с временным контрактом записи")
        positions = spectral_positions(
            total_samples,
            record_timebase.spp,
            self.temporal_mode,
            history_periods=self.feature_history_periods,
            stride_fraction=record_timebase.stride_fraction,
        )

        # Тот же temporal/feature contract, который использовался в SSL pretrain.
        spectral_feat, missing_mask, meta_feat = self.feature_builder.build(
            sub_signal_raw.T,
            spp=record_timebase.spp,
            positions=positions,
            voltage_basis=voltage_basis,
            channel_provenance=provenance,
        )
        feature_provenance = np.broadcast_to(
            np.asarray(meta_feat["feature_provenance"], dtype=np.uint8),
            spectral_feat.shape,
        ).copy()
        feature_provenance[missing_mask] = 0

        # Для UNLABELED target_class — лишь безопасное фиктивное значение:
        # direction loss обязательно маскируется через target_applicable.
        target_applicable = direction_val != int(PDRDirection.UNLABELED)
        target_class = 1 if direction_val == int(PDRDirection.FORWARD) else 0

        return {
            "features": torch.tensor(spectral_feat, dtype=torch.float32),
            "missing_mask": torch.tensor(missing_mask, dtype=torch.bool),
            "provenance": torch.tensor(feature_provenance, dtype=torch.long),
            "target_class": torch.tensor(target_class, dtype=torch.long),
            "target_applicable": torch.tensor(target_applicable, dtype=torch.bool),
            "pdr_direction": torch.tensor(direction_val, dtype=torch.int16),
            "pdr_margin": torch.tensor(margin_val, dtype=torch.float32),
            "pdr_confidence": torch.tensor(confidence_val, dtype=torch.float32),
            "warmup_mask": torch.tensor(warmup_val, dtype=torch.bool),
            "channel_provenance": torch.tensor(provenance, dtype=torch.long),
            "record_id": rec_idx,
            "window_idx": w_idx,
        }

    def _record_labels(self, rec_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Получить одну запись из legacy NPZ либо lazy sharded store."""

        if self.label_store is not None:
            if not self.label_store.has_record(rec_idx):
                return None
            return self.label_store.get_record(rec_idx)
        prefix = f"rec_{rec_idx}"
        if f"{prefix}_dir" not in self.labels_dict:
            return None
        margins = self.labels_dict[f"{prefix}_margin"]
        return {
            "directions": self.labels_dict[f"{prefix}_dir"],
            "margins": margins,
            "confidences": self.labels_dict.get(
                f"{prefix}_confidence",
                np.ones_like(margins, dtype=np.float32),
            ),
            "warmup": self.labels_dict[f"{prefix}_warmup"],
            "samples": self.labels_dict[f"{prefix}_samples"],
        }
