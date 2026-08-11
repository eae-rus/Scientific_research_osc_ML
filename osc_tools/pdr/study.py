"""Потоковая multi-PDR разметка и поиск интересных осциллограмм.

Модуль не зависит от PyTorch. Фазоры h1 одной записи вычисляются один раз и
используются всеми выбранными измерительными органами. Stateful-алгоритмы идут
последовательно по времени, поэтому их внутренние таймеры сохраняют смысл.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from osc_tools.ml.phase5_contracts import TimebaseContract, periods_to_samples
from .base import PDRAlgorithm, PDRDirection, PDRInputData
from .labeler import precompute_causal_h1_phasors
from .signal_analysis import derive_missing_currents


@dataclass(frozen=True)
class MultiPDRRecordResult:
    """Результаты нескольких РНМ на общей временной сетке одной записи."""

    algorithm_ids: tuple[str, ...]
    sample_indices: np.ndarray
    directions: np.ndarray
    margins: np.ndarray
    confidences: np.ndarray
    warmup_mask: np.ndarray
    provenance: np.ndarray
    input_sha256: str


def label_record_multi(
    signals: np.ndarray,
    provenance: np.ndarray,
    timebase: TimebaseContract,
    voltage_basis: str,
    algorithms: Sequence[PDRAlgorithm],
    *,
    sample_step: int = 1,
    default_history_periods: float = 10.0,
    allow_fallback: bool = False,
) -> MultiPDRRecordResult:
    """Разметить запись несколькими органами, разделяя один расчёт h1.

    Публичный орган без обязательной истории работает после первого полного
    периода. По умолчанию решение рассчитывается для каждого следующего исходного
    отсчёта. Орган с ``requires_history=True`` до настоящей точки t-history
    оставляет метку ``UNLABELED``.
    """

    if not algorithms:
        raise ValueError("Нужен хотя бы один алгоритм РНМ")
    if signals.ndim != 2 or signals.shape[0] != 8:
        raise ValueError(f"Ожидались сигналы формы (8,T), получено {signals.shape}")
    if len(provenance) != 8:
        raise ValueError("Provenance должен содержать 8 элементов")
    if sample_step <= 0:
        raise ValueError("Шаг разметки в отсчётах должен быть положительным")

    algorithm_ids = tuple(str(algorithm.resolved_algorithm_id) for algorithm in algorithms)
    if len(set(algorithm_ids)) != len(algorithm_ids):
        raise ValueError("Идентификаторы алгоритмов в multi-PDR прогоне должны быть уникальными")
    fallback_ids = [
        algorithm.requested_algorithm_id
        for algorithm in algorithms
        if getattr(algorithm, "fallback_applied", False)
    ]
    if fallback_ids and not allow_fallback:
        raise RuntimeError(
            "Multi-PDR разметка запрещает неявный fallback: " + ", ".join(fallback_ids)
        )

    input_hash = hashlib.sha256(
        np.ascontiguousarray(signals).view(np.uint8).tobytes()
        + np.ascontiguousarray(provenance).view(np.uint8).tobytes()
    ).hexdigest()
    prepared_signals, prepared_provenance = derive_missing_currents(signals, provenance)
    spp = timebase.spp
    end_indices = list(range(spp - 1, prepared_signals.shape[1], sample_step))
    n_windows = len(end_indices)

    history_samples_by_algorithm: list[int] = []
    phasor_end_indices = list(end_indices)
    for algorithm in algorithms:
        history_periods = float(algorithm.params.get("history_periods", default_history_periods))
        history_samples = periods_to_samples(history_periods, spp)
        history_samples_by_algorithm.append(history_samples)
        phasor_end_indices.extend(
            end_idx - history_samples
            for end_idx in end_indices
            if end_idx - history_samples >= spp - 1
        )

    phasor_table = precompute_causal_h1_phasors(
        prepared_signals,
        phasor_end_indices,
        spp,
    )
    converted_phasors = {
        end_idx: _split_voltage_current(phasors)
        for end_idx, phasors in phasor_table.items()
    }

    n_algorithms = len(algorithms)
    directions = np.full(
        (n_algorithms, n_windows),
        int(PDRDirection.UNLABELED),
        dtype=np.int16,
    )
    margins = np.zeros((n_algorithms, n_windows), dtype=np.float32)
    confidences = np.zeros((n_algorithms, n_windows), dtype=np.float32)
    warmup_mask = np.zeros((n_algorithms, n_windows), dtype=bool)

    for algorithm_index, (algorithm, history_samples) in enumerate(
        zip(algorithms, history_samples_by_algorithm)
    ):
        algorithm.reset_state()
        if "steps_per_period" in algorithm.params:
            algorithm.params["steps_per_period"] = max(1, round(spp / sample_step))
        requires_history = bool(getattr(algorithm, "requires_history", False))

        for window_index, end_idx in enumerate(end_indices):
            current = converted_phasors.get(end_idx)
            if current is None:
                continue
            history = converted_phasors.get(end_idx - history_samples)
            if requires_history and history is None:
                warmup_mask[algorithm_index, window_index] = True
                continue

            history_u, history_i = history if history is not None else ({}, {})
            current_u, current_i = current
            output = algorithm.compute(PDRInputData(
                phasors_u=current_u,
                phasors_i=current_i,
                history_phasors_u=history_u,
                history_phasors_i=history_i,
                provenance=prepared_provenance,
                voltage_basis=voltage_basis,
                timestamp_sec=float(end_idx / timebase.sampling_rate_hz),
            ))
            directions[algorithm_index, window_index] = int(output.direction)
            margins[algorithm_index, window_index] = float(output.margin)
            confidences[algorithm_index, window_index] = float(output.confidence)

    return MultiPDRRecordResult(
        algorithm_ids=algorithm_ids,
        sample_indices=np.asarray(end_indices, dtype=np.int32),
        directions=directions,
        margins=margins,
        confidences=confidences,
        warmup_mask=warmup_mask,
        provenance=np.asarray(prepared_provenance, dtype=np.uint8),
        input_sha256=input_hash,
    )


def summarize_record_interest(
    result: MultiPDRRecordResult,
    timebase: TimebaseContract,
) -> dict[str, Any]:
    """Рассчитать статистику и динамический score интересности записи.

    Постоянное расхождение двух органов само по себе получает низкий score:
    pair-компонента равна ``4*d*(1-d)`` и обращается в ноль как при d=0, так и
    при d=1. Приоритет получают переключения, локальные расхождения и движение
    решения относительно границы срабатывания.
    """

    directions = np.asarray(result.directions)
    n_algorithms, n_windows = directions.shape
    duration_sec = (
        float((result.sample_indices[-1] - result.sample_indices[0]) / timebase.sampling_rate_hz)
        if n_windows > 1 else 0.0
    )
    algorithm_stats: dict[str, dict[str, Any]] = {}
    total_transitions = 0
    near_boundary_values: list[float] = []

    for algorithm_index, algorithm_id in enumerate(result.algorithm_ids):
        labels = directions[algorithm_index]
        valid = labels != int(PDRDirection.UNLABELED)
        valid_count = int(valid.sum())
        transitions = _transition_count(labels)
        total_transitions += transitions
        run_lengths = _run_lengths(labels)
        short_run_fraction = (
            float(sum(length for length in run_lengths if length <= 2) / valid_count)
            if valid_count else 0.0
        )
        forward_fraction = (
            float(np.mean(labels[valid] == int(PDRDirection.FORWARD)))
            if valid_count else 0.0
        )
        near_boundary = _near_boundary_fraction(result.margins[algorithm_index], valid)
        near_boundary_values.append(near_boundary)
        algorithm_stats[algorithm_id] = {
            "valid_windows": valid_count,
            "coverage_fraction": float(valid_count / max(1, n_windows)),
            "forward_fraction": forward_fraction,
            "transitions": transitions,
            "transitions_per_second": float(transitions / max(duration_sec, 1e-9)),
            "short_run_fraction": short_run_fraction,
            "near_boundary_fraction": near_boundary,
            "mean_confidence": (
                float(np.mean(result.confidences[algorithm_index, valid]))
                if valid_count else 0.0
            ),
        }

    pair_stats: list[dict[str, Any]] = []
    localized_values: list[float] = []
    disagreement_edges = 0
    for left in range(n_algorithms):
        for right in range(left + 1, n_algorithms):
            left_labels = directions[left]
            right_labels = directions[right]
            common = (
                (left_labels != int(PDRDirection.UNLABELED))
                & (right_labels != int(PDRDirection.UNLABELED))
            )
            common_count = int(common.sum())
            if common_count:
                disagreement = left_labels[common] != right_labels[common]
                fraction = float(np.mean(disagreement))
                localized = float(4.0 * fraction * (1.0 - fraction))
                edges = int(np.count_nonzero(disagreement[1:] != disagreement[:-1]))
            else:
                fraction = localized = 0.0
                edges = 0
            localized_values.append(localized)
            disagreement_edges += edges
            pair_stats.append({
                "left": result.algorithm_ids[left],
                "right": result.algorithm_ids[right],
                "common_windows": common_count,
                "disagreement_fraction": fraction,
                "localized_disagreement": localized,
                "disagreement_transitions": edges,
            })

    valid_matrix = directions != int(PDRDirection.UNLABELED)
    valid_votes = valid_matrix.sum(axis=0)
    comparable = valid_votes >= 2
    disagreement_fraction = 0.0
    vote_change_count = 0
    if np.any(comparable):
        forward_votes = np.sum(directions == int(PDRDirection.FORWARD), axis=0)
        vote_fraction = forward_votes[comparable] / valid_votes[comparable]
        disagreement_fraction = float(np.mean((vote_fraction > 0.0) & (vote_fraction < 1.0)))
        vote_change_count = int(np.count_nonzero(np.diff(vote_fraction) != 0.0))

    pair_count = max(1, len(pair_stats))
    localized_disagreement = float(np.mean(localized_values)) if localized_values else 0.0
    switching_component = float(1.0 - math.exp(-total_transitions / max(2.0, 2.0 * n_algorithms)))
    dynamic_disagreement = float(
        1.0 - math.exp(-(disagreement_edges / pair_count + vote_change_count) / 4.0)
    )
    boundary_component = float(max(near_boundary_values, default=0.0))
    interest_score = float(np.clip(
        0.40 * switching_component
        + 0.35 * localized_disagreement
        + 0.20 * dynamic_disagreement
        + 0.05 * boundary_component,
        0.0,
        1.0,
    ))
    static_threshold_disagreement = bool(
        disagreement_fraction >= 0.75
        and localized_disagreement <= 0.20
        and total_transitions <= 1
        and vote_change_count == 0
    )
    low_coverage = bool(
        min((item["coverage_fraction"] for item in algorithm_stats.values()), default=0.0) < 0.5
    )
    categories: list[str] = []
    if total_transitions:
        categories.append("switching")
    if localized_disagreement >= 0.25 or dynamic_disagreement >= 0.35:
        categories.append("localized_disagreement")
    if static_threshold_disagreement:
        categories.append("static_threshold_disagreement")
    if any(item["short_run_fraction"] >= 0.10 for item in algorithm_stats.values()):
        categories.append("chattering")
    if low_coverage:
        categories.append("low_coverage")
    if not categories:
        categories.append("stable_consensus")

    most_disagreeing_pair = max(
        pair_stats,
        key=lambda item: item["localized_disagreement"],
        default=None,
    )
    return {
        "n_windows": n_windows,
        "duration_sec": duration_sec,
        "interest_score": interest_score,
        "categories": categories,
        "total_transitions": total_transitions,
        "disagreement_fraction": disagreement_fraction,
        "localized_disagreement": localized_disagreement,
        "dynamic_disagreement": dynamic_disagreement,
        "vote_change_count": vote_change_count,
        "static_threshold_disagreement": static_threshold_disagreement,
        "low_coverage": low_coverage,
        "algorithms": algorithm_stats,
        "most_disagreeing_pair": most_disagreeing_pair,
    }


class PDRStudyLabelStore:
    """Lazy reader sharded-разметки для последующего PDR fine-tuning."""

    def __init__(self, manifest_path: Path, algorithm_id: str | None = None) -> None:
        path = Path(manifest_path)
        if path.is_dir():
            path = path / "manifest.json"
        self.manifest_path = path
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("kind") != "pdr_study_sharded":
            raise ValueError("Manifest не является pdr_study_sharded")
        self.algorithm_ids = tuple(self.manifest["algorithm_ids"])
        selected = algorithm_id or str(self.manifest["teacher_algorithm_id"])
        if selected not in self.algorithm_ids:
            raise KeyError(f"Алгоритм {selected!r} отсутствует в разметке")
        invalidation_path = next(
            (
                candidate
                for candidate in (
                    path.parent / "INVALIDATED_ALGORITHMS.json",
                    path.parent.parent / "INVALIDATED_ALGORITHMS.json",
                )
                if candidate.exists()
            ),
            None,
        )
        if invalidation_path is not None:
            invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
            invalid_ids = set(invalidation.get("invalid_algorithm_ids", ()))
            if selected in invalid_ids:
                raise RuntimeError(
                    f"Канал {selected!r} помечен недействительным в {invalidation_path}. "
                    "Используйте исправленную версию PDR-разметки."
                )
        self.algorithm_id = selected
        self.algorithm_index = self.algorithm_ids.index(selected)
        self._record_map: dict[int, tuple[Path, int]] = {}
        for shard in self.manifest["shards"]:
            shard_path = path.parent / str(shard["file"])
            for local_index, record_id in enumerate(shard["record_ids"]):
                self._record_map[int(record_id)] = (shard_path, local_index)
        self._cached_path: Path | None = None
        self._cached_npz: dict[str, np.ndarray] | None = None

    def has_record(self, record_id: int) -> bool:
        return int(record_id) in self._record_map

    def get_record(self, record_id: int) -> dict[str, np.ndarray]:
        shard_path, local_index = self._record_map[int(record_id)]
        shard = self._load_shard(shard_path)
        offsets = shard["offsets"]
        start = int(offsets[local_index])
        stop = int(offsets[local_index + 1])
        directions = shard["directions"][self.algorithm_index, start:stop]
        if "all_margins" in shard:
            margins = shard["all_margins"][self.algorithm_index, start:stop]
            confidences = shard["all_confidences"][self.algorithm_index, start:stop]
            warmup = shard["all_warmup"][self.algorithm_index, start:stop]
        elif self.algorithm_id == self.manifest["teacher_algorithm_id"]:
            margins = shard["teacher_margin"][start:stop]
            confidences = shard["teacher_confidence"][start:stop]
            warmup = shard["teacher_warmup"][start:stop]
        else:
            margins = np.zeros(stop - start, dtype=np.float32)
            confidences = np.ones(stop - start, dtype=np.float32)
            warmup = directions == int(PDRDirection.UNLABELED)
        return {
            "directions": np.asarray(directions),
            "margins": np.asarray(margins),
            "confidences": np.asarray(confidences),
            "warmup": np.asarray(warmup),
            "samples": np.asarray(shard["samples"][start:stop]),
        }

    def close(self) -> None:
        self._cached_npz = None
        self._cached_path = None

    def _load_shard(self, path: Path) -> dict[str, np.ndarray]:
        if self._cached_path == path and self._cached_npz is not None:
            return self._cached_npz
        self.close()
        with np.load(path, allow_pickle=False) as archive:
            self._cached_npz = {key: archive[key] for key in archive.files}
        self._cached_path = path
        return self._cached_npz


def stable_config_hash(payload: dict[str, Any]) -> str:
    """SHA-256 конфигурации длительного прогона без runtime-полей."""

    return hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _split_voltage_current(phasors: dict[str, complex]) -> tuple[dict[str, complex], dict[str, complex]]:
    voltages = {key[1:]: value for key, value in phasors.items() if key.startswith("U")}
    currents = {key[1:]: value for key, value in phasors.items() if key.startswith("I")}
    return voltages, currents


def _transition_count(labels: np.ndarray) -> int:
    if labels.size < 2:
        return 0
    valid = labels != int(PDRDirection.UNLABELED)
    adjacent = valid[1:] & valid[:-1]
    return int(np.count_nonzero(adjacent & (labels[1:] != labels[:-1])))


def _run_lengths(labels: np.ndarray) -> list[int]:
    result: list[int] = []
    current_label: int | None = None
    current_length = 0
    for raw_label in labels:
        label = int(raw_label)
        if label == int(PDRDirection.UNLABELED):
            if current_length:
                result.append(current_length)
            current_label = None
            current_length = 0
        elif label == current_label:
            current_length += 1
        else:
            if current_length:
                result.append(current_length)
            current_label = label
            current_length = 1
    if current_length:
        result.append(current_length)
    return result


def _near_boundary_fraction(margins: np.ndarray, valid: np.ndarray) -> float:
    values = np.abs(np.asarray(margins)[valid & np.isfinite(margins)])
    if values.size == 0:
        return 0.0
    scale = float(np.median(values))
    if scale <= 1e-12:
        return 0.0
    return float(np.mean(values <= 0.10 * scale))
