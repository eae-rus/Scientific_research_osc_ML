"""Модуль генерации псевдоразметки РНМ (PDRDatasetLabeler).

Прогоняет заданный teacher-алгоритм РНМ по скользящему окну осциллограмм датасета,
вычисляет фазоры 1-й гармоники и сохраняет результатирующие метки (pdr_direction,
pdr_margin, warmup_mask, provenance).

Публичные органы без памяти начинают разметку после первого полного периода.
Органы с ``requires_history=True`` получают реальный фазор в точке t-history и
до накопления этой предыстории возвращают UNLABELED; подмена истории первым
доступным фазором не допускается. Флаг warmup_mask отражает именно отсутствие
обязательной предыстории teacher, а полный causal-контекст нейросети проверяет
``PDRTaskDataset`` отдельно.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from osc_tools.ml.phase5_contracts import (
    CHANNEL_ORDER,
    ChannelProvenance,
    TimebaseContract,
    periods_to_samples,
    period_fraction_stride,
)
from osc_tools.ml.phase5_sources import DatasetSource
from .base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection
from .signal_analysis import derive_missing_currents

logger = logging.getLogger(__name__)


def compute_causal_h1_phasors(
    signal: np.ndarray,
    end_idx: int,
    spp: int,
) -> Dict[str, complex]:
    """Вычисление комплексного фазора 1-й гармоники по прямоугольному окну в 1 период.

    Args:
        signal: Форма (8, T) - каналы IA, IB, IC, IN, UA, UB, UC, UN
        end_idx: Завершающий отсчёт 1-периодного окна (включительно)
        spp: Число отсчётов на период сети

    Returns:
        Словарь комплексных фазоров по каналам {"IA": ..., "UA": ...}
    """
    start_idx = end_idx - spp + 1
    if start_idx < 0 or end_idx >= signal.shape[1]:
        return {}

    window_data = signal[:, start_idx : end_idx + 1]
    if not np.isfinite(window_data).any():
        return {}

    t_arr = np.arange(spp)
    cos_kernel = np.cos(2.0 * np.pi * t_arr / spp)
    sin_kernel = np.sin(2.0 * np.pi * t_arr / spp)

    phasors: Dict[str, complex] = {}
    for ch_idx, ch_name in enumerate(CHANNEL_ORDER):
        ch_raw = window_data[ch_idx]
        if not np.isfinite(ch_raw).all():
            continue
        real_part = (2.0 / spp) * np.sum(ch_raw * cos_kernel)
        imag_part = (2.0 / spp) * np.sum(ch_raw * sin_kernel)
        phasors[ch_name] = complex(real_part, -imag_part)

    return phasors


def precompute_causal_h1_phasors(
    signal: np.ndarray,
    end_indices: List[int],
    spp: int,
) -> Dict[int, Dict[str, complex]]:
    """Рассчитать h1 только для нужных окон без повторного прохода по каждому окну.

    Используется сумма глобально демодулированного сигнала на интервале. Это
    эквивалентно прямому одночастотному ДПФ, но требует O(C*T + C*N_windows),
    а не O(C*SPP*N_windows). Окна с NaN по конкретному каналу пропускаются.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[0] != len(CHANNEL_ORDER):
        raise ValueError(f"Ожидался signal формы (8, T), получено {signal.shape}")
    if spp <= 0:
        raise ValueError("SPP должен быть положительным")

    positions = np.asarray(sorted({int(value) for value in end_indices}), dtype=np.int64)
    positions = positions[(positions >= spp - 1) & (positions < signal.shape[1])]
    if positions.size == 0:
        return {}

    starts = positions - spp + 1
    omega = 2.0 * np.pi / spp
    global_rotation = np.exp(-1j * omega * np.arange(signal.shape[1]))
    phasor_matrix = np.full(
        (positions.size, len(CHANNEL_ORDER)),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    for channel_index in range(len(CHANNEL_ORDER)):
        channel = np.asarray(signal[channel_index], dtype=np.float64)
        finite = np.isfinite(channel)
        safe = np.where(finite, channel, 0.0)
        weighted_prefix = np.concatenate((
            np.zeros(1, dtype=np.complex128),
            np.cumsum(safe * global_rotation),
        ))
        valid_prefix = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.cumsum(finite, dtype=np.int64),
        ))
        window_sum = weighted_prefix[positions + 1] - weighted_prefix[starts]
        valid_count = valid_prefix[positions + 1] - valid_prefix[starts]
        local_rotation = np.exp(1j * omega * starts)
        values = (2.0 / spp) * window_sum * local_rotation
        phasor_matrix[valid_count == spp, channel_index] = values[valid_count == spp]

    result: Dict[int, Dict[str, complex]] = {}
    for row, end_idx in enumerate(positions.tolist()):
        result[end_idx] = {
            channel_name: complex(phasor_matrix[row, channel_index])
            for channel_index, channel_name in enumerate(CHANNEL_ORDER)
            if np.isfinite(phasor_matrix[row, channel_index])
        }
    return result


@dataclass
class LabelingRecordResult:
    """Результат псевдоразметки одной осциллограммы."""

    record_id: int | str
    directions: np.ndarray  # int16 массив длины N_windows (-999, 0, 1)
    margins: np.ndarray  # float32 массив длины N_windows
    confidences: np.ndarray  # float32 массив длины N_windows
    warmup_mask: np.ndarray  # bool массив длины N_windows
    sample_indices: np.ndarray  # int32 конечные индексы окон в сигналах
    provenance: np.ndarray  # uint8 массив (8,)
    input_sha256: str


class PDRDatasetLabeler:
    """Генератор псевдоразметки РНМ для датасетов Phase 5."""

    def __init__(
        self,
        teacher_algorithm: PDRAlgorithm,
        history_periods: float = 10.0,
        stride_fraction: int = 8,
        allow_teacher_fallback: bool = False,
    ) -> None:
        self.teacher = teacher_algorithm
        self.history_periods = history_periods
        self.stride_fraction = stride_fraction
        self.allow_teacher_fallback = allow_teacher_fallback
        if getattr(self.teacher, "fallback_applied", False) and not allow_teacher_fallback:
            raise ValueError(
                "Генерация pseudo-label запрещена: запрошенный teacher был заменён fallback-алгоритмом. "
                "Передайте allow_teacher_fallback=True только после явного подтверждения подмены."
            )

    def label_single_record(
        self,
        record_id: int | str,
        signals: np.ndarray,
        provenance: np.ndarray,
        timebase: TimebaseContract,
        voltage_basis: str = "phase",
    ) -> LabelingRecordResult:
        """Разметка одной осциллограммы.

        Args:
            record_id: Идентификатор записи
            signals: Сигналы формы (8, T)
            provenance: Массив provenance (8,)
            timebase: Параметры временной сетки (spp, sampling_rate_hz и т.д.)
            voltage_basis: Базис напряжений ('phase' или 'line')

        Returns:
            LabelingRecordResult с метками направления и запасов
        """
        # Сброс внутреннего состояния алгоритма перед началом новой осциллограммы
        self.teacher.reset_state()

        input_hash = hashlib.sha256(
            np.ascontiguousarray(signals).view(np.uint8).tobytes()
            + np.ascontiguousarray(provenance).view(np.uint8).tobytes()
        ).hexdigest()

        # Восстановление любого отсутствующего тока если возможно
        signals, provenance = derive_missing_currents(signals, provenance)

        spp = timebase.spp
        stride = period_fraction_stride(spp, self.stride_fraction)
        n_samples = signals.shape[1]
        history_samples = periods_to_samples(self.history_periods, spp)
        requires_history = bool(getattr(self.teacher, "requires_history", False))

        # Таймеры stateful-органа считаются в точках разметки, поэтому их
        # масштаб должен соответствовать фактическому stride этой группы SPP.
        if "steps_per_period" in self.teacher.params:
            self.teacher.params["steps_per_period"] = max(1, round(spp / stride))

        # Точки окон с шагом stride_fraction
        end_indices = list(range(spp - 1, n_samples, stride))
        n_windows = len(end_indices)

        directions = np.full(n_windows, int(PDRDirection.UNLABELED), dtype=np.int16)
        margins = np.zeros(n_windows, dtype=np.float32)
        confidences = np.zeros(n_windows, dtype=np.float32)
        warmup_mask = np.ones(n_windows, dtype=bool)
        sample_indices = np.array(end_indices, dtype=np.int32)

        history_end_indices = [
            end_idx - history_samples
            for end_idx in end_indices
            if end_idx - history_samples >= spp - 1
        ]
        phasor_table = precompute_causal_h1_phasors(
            signals,
            end_indices + history_end_indices,
            spp,
        )

        for w_idx, end_idx in enumerate(end_indices):
            hist_end_idx = end_idx - history_samples
            hist_phasors = phasor_table.get(hist_end_idx)
            is_warmup = requires_history and not hist_phasors
            warmup_mask[w_idx] = is_warmup
            if is_warmup:
                continue

            # Расчёт фазоров текущего момента (t)
            current_phasors = phasor_table.get(end_idx, {})
            if not current_phasors:
                directions[w_idx] = int(PDRDirection.UNLABELED)
                continue

            u_dict = {}
            for key, val in current_phasors.items():
                if key.startswith("U"):
                    u_dict[key[1:]] = val

            i_dict = {}
            for key, val in current_phasors.items():
                if key.startswith("I"):
                    i_dict[key[1:]] = val

            hist_u = {}
            if hist_phasors:
                for key, val in hist_phasors.items():
                    if key.startswith("U"):
                        hist_u[key[1:]] = val

            hist_i = {}
            if hist_phasors:
                for key, val in hist_phasors.items():
                    if key.startswith("I"):
                        hist_i[key[1:]] = val

            inp = PDRInputData(
                phasors_u=u_dict,
                phasors_i=i_dict,
                history_phasors_u=hist_u,
                history_phasors_i=hist_i,
                provenance=provenance,
                voltage_basis=voltage_basis,
                timestamp_sec=float(end_idx / timebase.sampling_rate_hz),
            )

            out = self.teacher.compute(inp)
            directions[w_idx] = int(out.direction)
            margins[w_idx] = float(out.margin)
            confidences[w_idx] = float(out.confidence)

        return LabelingRecordResult(
            record_id=record_id,
            directions=directions,
            margins=margins,
            confidences=confidences,
            warmup_mask=warmup_mask,
            sample_indices=sample_indices,
            provenance=provenance,
            input_sha256=input_hash,
        )

    def process_dataset_split(
        self,
        source: DatasetSource,
        indices: List[int],
        timebase: TimebaseContract,
        output_dir: Path,
        split_name: str = "train",
    ) -> Path:
        """Разметка всего сплита датасета и сохранение в сжатый файл NPZ.

        Args:
            source: Источник осциллограмм
            indices: Список индексов осциллограмм в сплите
            timebase: Контракт временной сетки
            output_dir: Каталог для сохранения
            split_name: Имя сплита ('train', 'val', 'holdout')

        Returns:
            Путь к сохранённому файлу меток
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records_data: Dict[str, Any] = {}
        metadata_records: List[Dict[str, Any]] = []

        logger.info(f"Запуск разметки PDR сплита '{split_name}' ({len(indices)} записей)...")

        for idx in indices:
            meta = source.get_metadata(idx)
            signal = source.load_signal(idx)
            provenance = source.get_provenance(idx)
            voltage_basis = str(meta.get("voltage_basis", "phase"))
            record_timebase = timebase
            sampling_rate = meta.get("f_adc", meta.get("sampling_rate_hz"))
            network_frequency = meta.get("f_network", meta.get("network_frequency_hz"))
            if sampling_rate is not None and network_frequency is not None:
                record_timebase = TimebaseContract.create(
                    float(sampling_rate),
                    float(network_frequency),
                    window_periods=timebase.window_periods,
                    stride_fraction=self.stride_fraction,
                )

            res = self.label_single_record(
                record_id=idx,
                signals=signal,
                provenance=provenance,
                timebase=record_timebase,
                voltage_basis=voltage_basis,
            )

            prefix = f"rec_{idx}"
            records_data[f"{prefix}_dir"] = res.directions
            records_data[f"{prefix}_margin"] = res.margins
            records_data[f"{prefix}_confidence"] = res.confidences
            records_data[f"{prefix}_warmup"] = res.warmup_mask
            records_data[f"{prefix}_samples"] = res.sample_indices
            records_data[f"{prefix}_prov"] = res.provenance

            metadata_records.append({
                "record_id": idx,
                "n_windows": len(res.directions),
                "n_valid_windows": int(np.sum(res.directions != int(PDRDirection.UNLABELED))),
                "forward_windows": int(np.sum(res.directions == int(PDRDirection.FORWARD))),
                "reverse_windows": int(np.sum(res.directions == int(PDRDirection.REVERSE))),
                "input_sha256": res.input_sha256,
                "timebase": record_timebase.to_metadata(),
            })

        npz_path = output_dir / f"pdr_labels_{split_name}.npz"
        np.savez_compressed(npz_path, **records_data)

        meta_json_path = output_dir / f"pdr_labels_{split_name}_meta.json"
        meta_payload = {
            "teacher_id": self.teacher.algorithm_id,
            "requested_teacher_id": getattr(self.teacher, "requested_algorithm_id", self.teacher.algorithm_id),
            "resolved_teacher_id": getattr(self.teacher, "resolved_algorithm_id", self.teacher.algorithm_id),
            "teacher_fallback_applied": bool(getattr(self.teacher, "fallback_applied", False)),
            "fallback_algorithm_id": getattr(self.teacher, "fallback_algorithm_id", None),
            "teacher_params": self.teacher.params,
            "split_name": split_name,
            "timebase": timebase.to_metadata(),
            "n_records": len(indices),
            "records": metadata_records,
        }
        meta_json_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Разметка сплита '{split_name}' завершена. Сохранено в {npz_path}")
        return npz_path
