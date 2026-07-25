"""Модуль генерации псевдоразметки РНМ (PDRDatasetLabeler).

Прогоняет заданный teacher-алгоритм РНМ по скользящему 10-периодному окну
осциллограмм датасета, вычисляет фазоры 1-й гармоники и сохраняет результатирующие
метки (pdr_direction, pdr_margin, warmup_mask, provenance).
"""

from __future__ import annotations

from dataclasses import dataclass
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
    # Ортогональные составляющие (1-я гармоника)
    cos_kernel = np.cos(2.0 * np.pi * t_arr / spp)
    sin_kernel = np.sin(2.0 * np.pi * t_arr / spp)

    phasors: Dict[str, complex] = {}
    for ch_idx, ch_name in enumerate(CHANNEL_ORDER):
        ch_raw = window_data[ch_idx]
        if not np.isfinite(ch_raw).all():
            continue
        # Синусная и косинусная составляющие Фурье (с коэффициентом 2/N)
        real_part = (2.0 / spp) * np.sum(ch_raw * cos_kernel)
        imag_part = (2.0 / spp) * np.sum(ch_raw * sin_kernel)
        # Действующее значение (амплитуда / sqrt(2)) и фазовый угол
        # Комплексный фазор: Re - j*Im
        phasors[ch_name] = complex(real_part, -imag_part)

    return phasors


@dataclass
class LabelingRecordResult:
    """Результат псевдоразметки одной осциллограммы."""

    record_id: int | str
    directions: np.ndarray  # int16 массив длины N_windows (-999, -1, 0, 1)
    margins: np.ndarray  # float32 массив длины N_windows
    warmup_mask: np.ndarray  # bool массив длины N_windows
    sample_indices: np.ndarray  # int32 конечные индексы окон в сигналах
    provenance: np.ndarray  # uint8 массив (8,)


class PDRDatasetLabeler:
    """Генератор псевдоразметки РНМ для датасетов Phase 5."""

    def __init__(
        self,
        teacher_algorithm: PDRAlgorithm,
        history_periods: float = 10.0,
        stride_fraction: int = 8,
    ) -> None:
        self.teacher = teacher_algorithm
        self.history_periods = history_periods
        self.stride_fraction = stride_fraction

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
        # Восстановление отсутствующего IB если возможно
        signals, provenance = derive_missing_currents(signals, provenance)

        spp = timebase.spp
        stride = timebase.stride_samples
        n_samples = signals.shape[1]

        # Минимально необходимое окно предыстории (10 периодов)
        warmup_samples = timebase.window_samples

        # Точки окон с шагом stride_fraction
        end_indices = list(range(spp - 1, n_samples, stride))
        n_windows = len(end_indices)

        directions = np.full(n_windows, int(PDRDirection.UNLABELED), dtype=np.int16)
        margins = np.zeros(n_windows, dtype=np.float32)
        warmup_mask = np.ones(n_windows, dtype=bool)
        sample_indices = np.array(end_indices, dtype=np.int32)

        for w_idx, end_idx in enumerate(end_indices):
            # Проверка зоны разогрева (первые 10 периодов)
            if end_idx < warmup_samples - 1:
                warmup_mask[w_idx] = True
                directions[w_idx] = int(PDRDirection.UNLABELED)
                margins[w_idx] = 0.0
                continue

            warmup_mask[w_idx] = False

            # Расчёт фазоров текущего момента (t)
            current_phasors = compute_causal_h1_phasors(signals, end_idx, spp)
            if not current_phasors:
                directions[w_idx] = int(PDRDirection.BLOCK)
                continue

            # Расчёт фазоров предыстории (t - 200 мс = t - 10 периодов)
            hist_end_idx = end_idx - warmup_samples
            hist_phasors = compute_causal_h1_phasors(signals, hist_end_idx, spp) if hist_end_idx >= spp - 1 else None

            # Если глубокая предыстория (t - 200 мс) недоступна, берем фазоры первого валидного БПФ-окна (spp - 1)
            if hist_phasors is None:
                first_valid_idx = spp - 1
                hist_phasors = compute_causal_h1_phasors(signals, first_valid_idx, spp) if end_idx >= first_valid_idx else None

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

        return LabelingRecordResult(
            record_id=record_id,
            directions=directions,
            margins=margins,
            warmup_mask=warmup_mask,
            sample_indices=sample_indices,
            provenance=provenance,
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

            res = self.label_single_record(
                record_id=idx,
                signals=signal,
                provenance=provenance,
                timebase=timebase,
                voltage_basis=voltage_basis,
            )

            prefix = f"rec_{idx}"
            records_data[f"{prefix}_dir"] = res.directions
            records_data[f"{prefix}_margin"] = res.margins
            records_data[f"{prefix}_warmup"] = res.warmup_mask
            records_data[f"{prefix}_samples"] = res.sample_indices
            records_data[f"{prefix}_prov"] = res.provenance

            metadata_records.append({
                "record_id": idx,
                "n_windows": len(res.directions),
                "n_valid_windows": int(np.sum(~res.warmup_mask)),
                "forward_windows": int(np.sum(res.directions == int(PDRDirection.FORWARD))),
                "reverse_windows": int(np.sum(res.directions == int(PDRDirection.REVERSE))),
            })

        npz_path = output_dir / f"pdr_labels_{split_name}.npz"
        np.savez_compressed(npz_path, **records_data)

        meta_json_path = output_dir / f"pdr_labels_{split_name}_meta.json"
        meta_payload = {
            "teacher_id": self.teacher.algorithm_id,
            "teacher_params": self.teacher.params,
            "split_name": split_name,
            "timebase": timebase.to_metadata(),
            "n_records": len(indices),
            "records": metadata_records,
        }
        meta_json_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Разметка сплита '{split_name}' завершена. Сохранено в {npz_path}")
        return npz_path
