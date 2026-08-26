"""PyTorch Task Dataset для задачи РНМ (PDRTaskDataset).

Объединяет спектральные признаки Phase 5 (Version A / Version B) с псевдометками РНМ,
сгенерированными PDRDatasetLabeler. UNLABELED используется как отрицательная цель
применимости, но маскируется для направления/margin. Поддерживает strict splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple
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
from .signal_analysis import check_pdr_signal_sufficiency, derive_missing_currents
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
        exclude_structurally_insufficient: bool = True,
        sample_subset: str = "train",
        index_stride_samples: int = 1,
        max_samples_per_record: Optional[int] = None,
        augmentation_seed: Optional[int] = None,
        augmentation_probability: float = 0.0,
        index_progress_callback: Optional[Callable[[int, int], None]] = None,
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
        self.exclude_structurally_insufficient = exclude_structurally_insufficient
        if sample_subset not in ("train", "transition", "all"):
            raise ValueError("sample_subset должен быть train, transition или all")
        if index_stride_samples <= 0:
            raise ValueError("index_stride_samples должен быть положительным")
        if max_samples_per_record is not None and max_samples_per_record <= 0:
            raise ValueError("max_samples_per_record должен быть положительным")
        self.sample_subset = sample_subset
        self.index_stride_samples = int(index_stride_samples)
        self.max_samples_per_record = max_samples_per_record
        if not 0.0 <= augmentation_probability <= 1.0:
            raise ValueError("augmentation_probability должен быть в диапазоне [0, 1]")
        self.augmentation_seed = augmentation_seed
        self.augmentation_probability = float(augmentation_probability)
        self.augmentation_epoch = 0
        self.index_progress_callback = index_progress_callback

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
        # Callback нужен только в главном процессе при построении индекса.
        # Не переносим лишнее состояние ProgressReporter в DataLoader workers.
        self.index_progress_callback = None

    def _build_sample_index(self) -> None:
        """Построение индекса направления и применимости по causal-окнам."""
        total_records = len(self.indices)
        for position, rec_idx in enumerate(self.indices, start=1):
            record_labels = self._record_labels(rec_idx)
            if record_labels is None:
                if self.index_progress_callback is not None:
                    self.index_progress_callback(position, total_records)
                continue
            # Недостаток исходных каналов (<2I либо <2U) означает, что РНМ
            # невозможно рассчитать вообще. Такие записи не являются
            # отрицательными примерами головы применимости и не индексируются.
            if self.exclude_structurally_insufficient:
                if self.label_store is not None and not self.label_store.is_structurally_eligible(rec_idx):
                    if self.index_progress_callback is not None:
                        self.index_progress_callback(position, total_records)
                    continue
                provenance = np.asarray(
                    record_labels.get("provenance", self.source.get_provenance(rec_idx))
                )
                voltage_basis = str(self.source.get_metadata(rec_idx).get("voltage_basis", "phase"))
                sufficiency = check_pdr_signal_sufficiency(provenance, voltage_basis)
                if not sufficiency.can_run_phase_pdr:
                    if self.index_progress_callback is not None:
                        self.index_progress_callback(position, total_records)
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
            train_mask = np.asarray(
                record_labels.get("train_mask", np.ones_like(dirs, dtype=bool)),
                dtype=bool,
            )
            transition_mask = np.asarray(
                record_labels.get("transition_eval_mask", np.zeros_like(dirs, dtype=bool)),
                dtype=bool,
            )
            if train_mask.shape != dirs.shape or transition_mask.shape != dirs.shape:
                raise ValueError(f"Маски экспертного слоя имеют неверную форму: record={rec_idx}")
            if self.sample_subset == "train":
                valid_mask &= train_mask
            elif self.sample_subset == "transition":
                valid_mask &= transition_mask
            # Для совместимости с SSL backbone нужен полный causal-фрагмент:
            # feature history (до 10 периодов для lp10) + 10 периодов модели.
            valid_mask &= sample_indices >= (required_samples - 1)
            valid_mask &= (sample_indices % self.index_stride_samples) == 0

            valid_w_indices = np.where(valid_mask)[0]
            if (
                self.max_samples_per_record is not None
                and valid_w_indices.size > self.max_samples_per_record
            ):
                positions = np.linspace(
                    0,
                    valid_w_indices.size - 1,
                    self.max_samples_per_record,
                    dtype=np.int64,
                )
                valid_w_indices = valid_w_indices[np.unique(positions)]
            for w_idx in valid_w_indices:
                self.samples.append((rec_idx, int(w_idx)))
            if self.index_progress_callback is not None:
                self.index_progress_callback(position, total_records)

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

    def set_epoch(self, epoch: int) -> None:
        """Сделать детерминированную аугментацию различной между эпохами."""
        self.augmentation_epoch = int(epoch)

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
        augmentation_code = "none"
        if self.augmentation_seed is not None and self.augmentation_probability > 0:
            sub_signal_raw, provenance, augmentation_code = self._augment_invariant(
                sub_signal_raw, provenance, rec_idx, w_idx
            )
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
            "expert_train_mask": torch.tensor(
                bool(record_labels.get("train_mask", np.ones_like(record_labels["directions"], dtype=bool))[w_idx]),
                dtype=torch.bool,
            ),
            "expert_transition_eval_mask": torch.tensor(
                bool(record_labels.get("transition_eval_mask", np.zeros_like(record_labels["directions"], dtype=bool))[w_idx]),
                dtype=torch.bool,
            ),
            "channel_provenance": torch.tensor(provenance, dtype=torch.long),
            "source_id": 0 if self.source.name == "open_ee" else 1 if self.source.name == "french_rte" else 2,
            "record_id": rec_idx,
            "window_idx": w_idx,
            "augmentation_code": augmentation_code,
        }

    def _augment_invariant(
        self,
        signal: np.ndarray,
        provenance: np.ndarray,
        rec_idx: int,
        w_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Применить только физически инвариантные аугментации направления.

        Совместная циклическая перестановка фаз и одновременная смена полярности
        U/I не меняют взаимный угол и знак мощности. Масштабирование и шум здесь
        намеренно не используются: около уставки они способны изменить истинную
        целевую метку.
        """

        token = (
            int(self.augmentation_seed)
            + int(rec_idx) * 1_000_003
            + int(w_idx) * 97
            + int(self.augmentation_epoch) * 10_000_019
        ) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(token)
        if rng.random() >= self.augmentation_probability:
            return signal, provenance, "none"
        result = np.asarray(signal, dtype=np.float32).copy()
        result_provenance = np.asarray(provenance, dtype=np.uint8).copy()
        operations: list[str] = []
        shift = int(rng.integers(0, 3))
        if shift:
            for start in (0, 4):
                result[start : start + 3] = np.roll(result[start : start + 3], shift, axis=0)
                result_provenance[start : start + 3] = np.roll(
                    result_provenance[start : start + 3], shift
                )
            operations.append(f"phase_roll_{shift}")
        if bool(rng.integers(0, 2)):
            result *= -1.0
            operations.append("global_polarity")
        return result, result_provenance, "+".join(operations) if operations else "identity"

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
            "provenance": self.labels_dict.get(
                f"{prefix}_prov", self.source.get_provenance(rec_idx)
            ),
        }
