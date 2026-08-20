"""Статистический аудит и визуальная проверка multi-PDR разметки Phase 5.

Режим ``summary`` работает только с компактными JSONL и формирует таблицы,
кластеры и список репрезентативных случаев. ``agreement`` последовательно читает
готовые NPZ shards и считает точные попарные confusion/agreement. ``plots``
строит короткие диагностические окна с исходными сигналами и всеми РНМ.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.phase5_contracts import CHANNEL_ORDER
from osc_tools.ml.phase5_sources import FrenchRTESource, OpenEEShardedSource
from osc_tools.pdr.base import PDRDirection
from osc_tools.pdr.signal_analysis import check_pdr_signal_sufficiency
from scripts.phase5_experiments.progress import ProgressReporter


DEFAULT_LABEL_DIR = PROJECT_ROOT / "data/phase5/pdr_labels_v5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/phase5/pdr_analysis_v5"
DEFAULT_SOURCES = ("open_ee", "french_rte")
DEFAULT_LOW_CURRENT_RMS_THRESHOLD = 0.05 / 20.0
TEACHER_ID = "adaptive_pdr_mir"
TRANSITION_BINS = (
    "0_other_switch",
    "1_2",
    "3_10",
    "11_30",
    "31_plus",
    "static_disagreement",
    "low_coverage",
)
ALGORITHM_METRICS = (
    "valid_windows", "coverage_fraction", "forward_fraction", "transitions",
    "transitions_per_second", "short_run_fraction", "near_boundary_fraction",
    "mean_confidence",
)


@dataclass
class StudyArchive:
    root: Path

    def __post_init__(self) -> None:
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.algorithm_ids = tuple(self.manifest["algorithm_ids"])
        correction_path = self.root / "UNLABELED_RECORD_MASKS.json"
        correction = (
            json.loads(correction_path.read_text(encoding="utf-8"))
            if correction_path.exists() else {}
        )
        self.record_masks = {
            algorithm_id: {int(record_id) for record_id in record_ids}
            for algorithm_id, record_ids in correction.get("algorithm_record_masks", {}).items()
        }
        self.record_map: dict[int, tuple[Path, int]] = {}
        for shard in self.manifest["shards"]:
            path = self.root / str(shard["file"])
            for local_index, record_id in enumerate(shard["record_ids"]):
                self.record_map[int(record_id)] = (path, local_index)
        self._cached_path: Path | None = None
        self._cached: dict[str, np.ndarray] | None = None

    def get_record(self, record_id: int) -> dict[str, np.ndarray]:
        path, local_index = self.record_map[int(record_id)]
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as archive:
                self._cached = {name: archive[name] for name in archive.files}
            self._cached_path = path
        assert self._cached is not None
        offsets = self._cached["offsets"]
        start, stop = int(offsets[local_index]), int(offsets[local_index + 1])
        directions = self._cached["directions"][:, start:stop].copy()
        margins = self._cached["all_margins"][:, start:stop].copy()
        confidences = self._cached["all_confidences"][:, start:stop].copy()
        warmup = self._cached["all_warmup"][:, start:stop].copy()
        for algorithm_index, algorithm_id in enumerate(self.algorithm_ids):
            if int(record_id) in self.record_masks.get(algorithm_id, set()):
                directions[algorithm_index] = int(PDRDirection.UNLABELED)
                margins[algorithm_index] = 0.0
                confidences[algorithm_index] = 0.0
                warmup[algorithm_index] = True
        return {
            "samples": self._cached["samples"][start:stop],
            "directions": directions,
            "margins": margins,
            "confidences": confidences,
            "warmup": warmup,
            "provenance": self._cached["provenance"][local_index],
        }

    def get_provenance(self, record_id: int) -> np.ndarray:
        """Прочитать только компактный provenance, не копируя временные метки."""

        path, local_index = self.record_map[int(record_id)]
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as archive:
                self._cached = {name: archive[name] for name in archive.files}
            self._cached_path = path
        assert self._cached is not None
        return np.asarray(self._cached["provenance"][local_index], dtype=np.uint8)


def _load_record_masks(source_root: Path) -> dict[str, set[int]]:
    path = source_root / "UNLABELED_RECORD_MASKS.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        algorithm_id: {int(record_id) for record_id in record_ids}
        for algorithm_id, record_ids in payload.get("algorithm_record_masks", {}).items()
    }


def _apply_masks_to_record_summary(
    record: dict[str, Any], masks: dict[str, set[int]]
) -> dict[str, Any]:
    record_id = int(record["record_id"])
    masked = [algorithm_id for algorithm_id, ids in masks.items() if record_id in ids]
    if not masked:
        return record
    corrected = dict(record)
    corrected["algorithms"] = {
        algorithm_id: dict(stats) for algorithm_id, stats in record["algorithms"].items()
    }
    for algorithm_id in masked:
        stats = corrected["algorithms"][algorithm_id]
        stats.update({
            "valid_windows": 0,
            "coverage_fraction": 0.0,
            "forward_fraction": 0.0,
            "transitions": 0,
            "transitions_per_second": 0.0,
            "short_run_fraction": 0.0,
            "near_boundary_fraction": 0.0,
            "mean_confidence": 0.0,
        })
    corrected["low_coverage"] = True
    return corrected


def transition_group(record: dict[str, Any]) -> str:
    teacher = record["algorithms"][TEACHER_ID]
    transitions = int(teacher["transitions"])
    total = int(record["total_transitions"])
    if transitions == 0 and total > 0:
        return "0_other_switch"
    if transitions <= 2:
        return "1_2" if transitions else "0_stable"
    if transitions <= 10:
        return "3_10"
    if transitions <= 30:
        return "11_30"
    return "31_plus"


def audit_strata(record: dict[str, Any]) -> list[str]:
    strata = [transition_group(record)]
    if (
        int(record["total_transitions"]) == 0
        and float(record["disagreement_fraction"]) >= 0.25
    ):
        strata.append("static_disagreement")
    if bool(record["low_coverage"]):
        strata.append("low_coverage")
    return strata


def load_records(label_dir: Path, sources: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = [
        (source, path)
        for source in sources
        for path in sorted((label_dir / source).glob("shard_*.jsonl"))
    ]
    progress = ProgressReporter("Чтение PDR statistics", max(1, len(paths)), unit="файл")
    masks_by_source = {source: _load_record_masks(label_dir / source) for source in sources}
    for index, (source, path) in enumerate(paths, start=1):
        with path.open("r", encoding="utf-8") as stream:
            records.extend(
                _apply_masks_to_record_summary(json.loads(line), masks_by_source[source])
                for line in stream if line.strip()
            )
        progress.update(index)
    progress.finish()
    return records


def flatten_record(record: dict[str, Any], cluster_id: int | None = None) -> dict[str, Any]:
    row = {
        key: record.get(key)
        for key in (
            "source", "record_id", "split", "file_name", "source_csv", "input_sha256",
            "f_adc", "f_network", "spp", "voltage_basis", "duration_sec", "n_windows",
            "interest_score", "total_transitions", "vote_change_count",
            "disagreement_fraction", "localized_disagreement", "dynamic_disagreement",
            "static_threshold_disagreement", "low_coverage",
        )
    }
    row["categories"] = "|".join(record["categories"])
    row["adaptive_transition_group"] = transition_group(record)
    row["audit_strata"] = "|".join(audit_strata(record))
    row["cluster_id"] = cluster_id
    for algorithm_id, values in record["algorithms"].items():
        for metric in ALGORITHM_METRICS:
            row[f"{algorithm_id}__{metric}"] = values.get(metric)
    return row


def _feature_matrix(records: Sequence[dict[str, Any]]) -> np.ndarray:
    values: list[list[float]] = []
    for record in records:
        algorithms = record["algorithms"]
        forward = np.asarray([item["forward_fraction"] for item in algorithms.values()])
        transitions = np.asarray([item["transitions"] for item in algorithms.values()])
        teacher = algorithms[TEACHER_ID]
        values.append([
            math.log1p(float(record["duration_sec"])),
            math.log1p(float(record["total_transitions"])),
            float(record["disagreement_fraction"]),
            float(record["localized_disagreement"]),
            float(record["dynamic_disagreement"]),
            float(teacher["coverage_fraction"]),
            float(teacher["forward_fraction"]),
            math.log1p(float(teacher["transitions_per_second"])),
            float(teacher["near_boundary_fraction"]),
            float(np.std(forward)),
            float(np.std(np.log1p(transitions))),
        ])
    return np.asarray(values, dtype=np.float64)


def cluster_records(records: list[dict[str, Any]], output_dir: Path) -> dict[tuple[str, int], int]:
    """Exploratory KMeans per source; кластеры не считаются ground truth."""

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import RobustScaler
    except ImportError:
        print("sklearn отсутствует: exploratory clustering пропущен", file=sys.stderr)
        return {}

    assignments: dict[tuple[str, int], int] = {}
    diagnostics: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260811)
    feature_names = (
        "log_duration", "log_total_transitions", "disagreement", "localized_disagreement",
        "dynamic_disagreement", "teacher_coverage", "teacher_forward",
        "log_teacher_transition_rate", "teacher_near_boundary", "forward_dispersion",
        "transition_dispersion",
    )
    for source in sorted({str(record["source"]) for record in records}):
        subset = [record for record in records if record["source"] == source]
        raw = _feature_matrix(subset)
        scaled = RobustScaler(quantile_range=(10, 90)).fit_transform(raw)
        sample_indices = rng.choice(len(subset), min(5000, len(subset)), replace=False)
        best_k, best_score = 3, -1.0
        for k in range(3, min(9, len(subset))):
            model = KMeans(n_clusters=k, n_init=10, random_state=20260811).fit(scaled)
            score = float(silhouette_score(scaled[sample_indices], model.labels_[sample_indices]))
            diagnostics.append({"source": source, "k": k, "silhouette": score})
            if score > best_score:
                best_k, best_score = k, score
        model = KMeans(n_clusters=best_k, n_init=20, random_state=20260811).fit(scaled)
        for record, cluster_id in zip(subset, model.labels_):
            assignments[(source, int(record["record_id"]))] = int(cluster_id)
        for cluster_id in range(best_k):
            mask = model.labels_ == cluster_id
            profile: dict[str, Any] = {
                "source": source,
                "cluster_id": cluster_id,
                "records": int(mask.sum()),
                "fraction": float(mask.mean()),
                "selected_k": best_k,
                "silhouette": best_score,
            }
            for feature_index, name in enumerate(feature_names):
                profile[f"median__{name}"] = float(np.median(raw[mask, feature_index]))
            profiles.append(profile)
    _write_csv(output_dir / "cluster_diagnostics.csv", diagnostics)
    _write_csv(output_dir / "cluster_profiles.csv", profiles)
    return assignments


def build_summary(
    label_dir: Path,
    output_dir: Path,
    sources: Sequence[str],
    *,
    enable_clusters: bool,
    plots_per_group: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(label_dir, sources)
    eligibility_path = output_dir / "record_signal_eligibility.csv"
    eligible_keys: set[tuple[str, int]] | None = None
    if eligibility_path.exists():
        with eligibility_path.open("r", encoding="utf-8-sig", newline="") as stream:
            eligible_keys = {
                (str(row["source"]), int(row["record_id"]))
                for row in csv.DictReader(stream)
                if _as_bool(row["pdr_structurally_eligible"])
            }
    analysis_records = (
        [
            record for record in records
            if (str(record["source"]), int(record["record_id"])) in eligible_keys
        ]
        if eligible_keys is not None else records
    )
    assignments = cluster_records(analysis_records, output_dir) if enable_clusters else {}
    flat = [
        flatten_record(record, assignments.get((str(record["source"]), int(record["record_id"]))))
        for record in records
    ]
    _write_csv(output_dir / "record_statistics.csv", flat)

    transition_rows: list[dict[str, Any]] = []
    algorithm_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    analysis_groups = [("all", analysis_records)] + [
        (source, [record for record in analysis_records if record["source"] == source])
        for source in sources
    ]
    for source, subset in analysis_groups:
        groups = Counter(transition_group(record) for record in subset)
        for group in ("0_stable",) + TRANSITION_BINS[:5]:
            transition_rows.append({
                "source": source,
                "transition_group": group,
                "records": groups[group],
                "fraction": groups[group] / max(1, len(subset)),
            })
        dataset_rows.append(_dataset_summary(source, subset))
        algorithm_ids = tuple(subset[0]["algorithms"]) if subset else ()
        for algorithm_id in algorithm_ids:
            for metric in ALGORITHM_METRICS[1:]:
                included = [
                    record for record in subset
                    if metric == "coverage_fraction"
                    or int(record["algorithms"][algorithm_id]["valid_windows"]) > 0
                ]
                values = np.asarray([
                    float(record["algorithms"][algorithm_id][metric]) for record in included
                ])
                algorithm_rows.append({
                    "source": source,
                    "algorithm_id": algorithm_id,
                    "metric": metric,
                    "records_included": len(included),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                })
    _write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    _write_csv(output_dir / "transition_groups.csv", transition_rows)
    _write_csv(output_dir / "algorithm_record_distributions.csv", algorithm_rows)
    _write_duplicates(records, output_dir)
    _write_source_shift(analysis_records, sources, output_dir)
    candidates = select_candidates(analysis_records, plots_per_group)
    _write_csv(output_dir / "plot_candidates.csv", candidates)
    _plot_overview(analysis_records, sources, output_dir / "figures")
    _write_report(dataset_rows, transition_rows, output_dir)
    _atomic_json(output_dir / "summary.json", {
        "records": len(records),
        "pdr_eligible_records": len(analysis_records),
        "sources": list(sources),
        "label_dir": str(label_dir),
        "clusters_enabled": enable_clusters,
        "plot_candidates": len(candidates),
        "outputs": {
            "records": "record_statistics.csv",
            "datasets": "dataset_summary.csv",
            "transitions": "transition_groups.csv",
            "algorithms": "algorithm_record_distributions.csv",
            "duplicates": "duplicate_groups.csv",
            "source_shift": "source_shift.csv",
            "candidates": "plot_candidates.csv",
            "pointwise_agreement": "pairwise_pointwise_agreement.csv",
            "teacher_temporal": "teacher_temporal_statistics.csv",
            "state_patterns": "algorithm_state_patterns.csv",
        },
    })
    return candidates


def _dataset_summary(source: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hashes = Counter(str(record["input_sha256"]) for record in records)
    duplicate_records = sum(count for count in hashes.values() if count > 1)
    return {
        "population": "pdr_eligible_2i2u",
        "source": source,
        "records": len(records),
        "windows": sum(int(record["n_windows"]) for record in records),
        "duration_hours": sum(float(record["duration_sec"]) for record in records) / 3600.0,
        "median_duration_sec": float(np.median([record["duration_sec"] for record in records])),
        "switching_records": sum(int(record["total_transitions"]) > 0 for record in records),
        "teacher_switching_records": sum(
            int(record["algorithms"][TEACHER_ID]["transitions"]) > 0 for record in records
        ),
        "disagreement_records": sum(float(record["disagreement_fraction"]) > 0 for record in records),
        "low_coverage_records": sum(bool(record["low_coverage"]) for record in records),
        "unique_input_hashes": len(hashes),
        "duplicate_records": duplicate_records,
        "duplicate_fraction": duplicate_records / max(1, len(records)),
    }


def _write_duplicates(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["input_sha256"])].append(record)
    rows: list[dict[str, Any]] = []
    for digest, items in groups.items():
        if len(items) < 2:
            continue
        sources = sorted({str(item["source"]) for item in items})
        splits = sorted({str(item["split"]) for item in items})
        rows.append({
            "input_sha256": digest,
            "records": len(items),
            "sources": "|".join(sources),
            "splits": "|".join(splits),
            "cross_split": len(splits) > 1,
            "cross_source": len(sources) > 1,
            "record_keys": "|".join(f"{item['source']}:{item['record_id']}" for item in items),
        })
    rows.sort(key=lambda row: (-int(row["records"]), str(row["input_sha256"])))
    _write_csv(output_dir / "duplicate_groups.csv", rows, fallback_fields=(
        "input_sha256", "records", "sources", "splits", "cross_split", "cross_source",
        "record_keys",
    ))


def _write_source_shift(
    records: Sequence[dict[str, Any]], sources: Sequence[str], output_dir: Path
) -> None:
    if len(sources) != 2:
        return
    from scipy.stats import ks_2samp, wasserstein_distance

    left = [record for record in records if record["source"] == sources[0]]
    right = [record for record in records if record["source"] == sources[1]]
    if not left or not right:
        return
    extractors = {
        "duration_sec": lambda r: r["duration_sec"],
        "total_transitions": lambda r: r["total_transitions"],
        "disagreement_fraction": lambda r: r["disagreement_fraction"],
        "localized_disagreement": lambda r: r["localized_disagreement"],
        "teacher_coverage": lambda r: r["algorithms"][TEACHER_ID]["coverage_fraction"],
        "teacher_forward": lambda r: r["algorithms"][TEACHER_ID]["forward_fraction"],
        "teacher_transition_rate": lambda r: r["algorithms"][TEACHER_ID]["transitions_per_second"],
    }
    rows: list[dict[str, Any]] = []
    for name, extractor in extractors.items():
        a = np.asarray([float(extractor(record)) for record in left])
        b = np.asarray([float(extractor(record)) for record in right])
        pooled = np.concatenate((a, b))
        iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
        ks = ks_2samp(a, b)
        rows.append({
            "metric": name,
            f"median__{sources[0]}": float(np.median(a)),
            f"median__{sources[1]}": float(np.median(b)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "wasserstein": float(wasserstein_distance(a, b)),
            "wasserstein_over_pooled_iqr": float(wasserstein_distance(a, b) / max(iqr, 1e-12)),
        })
    _write_csv(output_dir / "source_shift.csv", rows)


def select_candidates(records: Sequence[dict[str, Any]], per_group: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in sorted({str(record["source"]) for record in records}):
        source_records = [record for record in records if record["source"] == source]
        for stratum in TRANSITION_BINS:
            subset = [record for record in source_records if stratum in audit_strata(record)]
            for rank, record in enumerate(_diverse_subset(subset, per_group), start=1):
                selected.append({
                    "source": source,
                    "record_id": int(record["record_id"]),
                    "stratum": stratum,
                    "selection_rank": rank,
                    "file_name": record.get("file_name"),
                    "split": record.get("split"),
                    "duration_sec": record.get("duration_sec"),
                    "teacher_transitions": record["algorithms"][TEACHER_ID]["transitions"],
                    "total_transitions": record["total_transitions"],
                    "disagreement_fraction": record["disagreement_fraction"],
                    "interest_score": record["interest_score"],
                    "focus_left": "",
                    "focus_right": "",
                })
    return selected


def _diverse_subset(records: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(records) <= count:
        return list(records)
    matrix = _feature_matrix(records)
    low = np.quantile(matrix, 0.10, axis=0)
    high = np.quantile(matrix, 0.90, axis=0)
    scaled = np.clip((matrix - low) / np.maximum(high - low, 1e-12), 0.0, 1.0)
    first = int(np.argmax([float(record["interest_score"]) for record in records]))
    chosen = [first]
    minimum_distance = np.linalg.norm(scaled - scaled[first], axis=1)
    while len(chosen) < count:
        minimum_distance[chosen] = -1.0
        next_index = int(np.argmax(minimum_distance))
        chosen.append(next_index)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(scaled - scaled[next_index], axis=1),
        )
    return [records[index] for index in chosen]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _agreement_metrics(confusion: np.ndarray) -> dict[str, float]:
    """Метрики двух бинарных органов без назначения одного из них ground truth."""

    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError("confusion должна иметь форму (2, 2)")
    n00, n01, n10, n11 = matrix.ravel()
    total = float(matrix.sum())
    observed = _safe_ratio(n00 + n11, total)
    row = matrix.sum(axis=1)
    column = matrix.sum(axis=0)
    expected = _safe_ratio(float(row @ column), total * total)
    kappa = _safe_ratio(observed - expected, 1.0 - expected)
    denominator = math.sqrt(float(row[0] * row[1] * column[0] * column[1]))
    mcc = _safe_ratio(n11 * n00 - n10 * n01, denominator)
    balanced_values = [
        _safe_ratio(n00, n00 + n01),
        _safe_ratio(n11, n10 + n11),
        _safe_ratio(n00, n00 + n10),
        _safe_ratio(n11, n01 + n11),
    ]
    finite_balanced = [value for value in balanced_values if np.isfinite(value)]
    balanced_symmetric = (
        float(np.mean(finite_balanced)) if finite_balanced else float("nan")
    )
    return {
        "agreement": observed,
        "disagreement": 1.0 - observed if np.isfinite(observed) else float("nan"),
        "cohen_kappa": kappa,
        "mcc": mcc,
        "balanced_agreement_symmetric": balanced_symmetric,
        "jaccard_reverse": _safe_ratio(n00, n00 + n01 + n10),
        "jaccard_forward": _safe_ratio(n11, n11 + n01 + n10),
        "forward_prevalence_left": _safe_ratio(n10 + n11, total),
        "forward_prevalence_right": _safe_ratio(n01 + n11, total),
    }


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _maximum_events_in_window(event_samples: np.ndarray, window_samples: int) -> int:
    if event_samples.size == 0:
        return 0
    right = np.searchsorted(event_samples, event_samples + window_samples, side="right")
    return int(np.max(right - np.arange(event_samples.size)))


def _temporal_metrics(
    labels: np.ndarray,
    samples: np.ndarray,
    f_adc: float,
    *,
    chatter_seconds: float = 0.10,
) -> dict[str, float | int]:
    """Record-level временная структура с сохранением разрывов UNLABELED."""

    labels = np.asarray(labels)
    samples = np.asarray(samples, dtype=np.int64)
    valid = (labels == 0) | (labels == 1)
    valid_count = int(valid.sum())
    result: dict[str, float | int] = {
        "valid_windows": valid_count,
        "state_entropy_bits": float("nan"),
        "transition_entropy_bits": float("nan"),
        "lag1_autocorrelation": float("nan"),
        "transitions": 0,
        "first_transition_sec": float("nan"),
        "last_transition_sec": float("nan"),
        "first_transition_edge_distance_sec": float("nan"),
        "last_transition_edge_distance_sec": float("nan"),
        "median_run_duration_sec": float("nan"),
        "p05_run_duration_sec": float("nan"),
        "p95_run_duration_sec": float("nan"),
        "short_runs_le_20ms": 0,
        "short_runs_le_100ms": 0,
        "chatter_returns_le_100ms": 0,
        "max_switches_in_0_5s": 0,
        "max_switches_in_1_0s": 0,
    }
    if valid_count == 0 or not np.isfinite(f_adc) or f_adc <= 0:
        return result

    state_counts = np.bincount(labels[valid].astype(np.int8), minlength=2)
    result["state_entropy_bits"] = _entropy_from_counts(state_counts)
    adjacent = valid[:-1] & valid[1:] & (np.diff(samples) == 1)
    left = labels[:-1][adjacent].astype(np.int8)
    right = labels[1:][adjacent].astype(np.int8)
    if left.size:
        transition_counts = np.bincount(2 * left + right, minlength=4)
        result["transition_entropy_bits"] = _entropy_from_counts(transition_counts)
        if np.std(left) > 0 and np.std(right) > 0:
            result["lag1_autocorrelation"] = float(np.corrcoef(left, right)[0, 1])

    transition_mask = adjacent & (labels[:-1] != labels[1:])
    transition_samples = samples[1:][transition_mask]
    result["transitions"] = int(transition_samples.size)
    if transition_samples.size:
        origin = int(samples[0])
        end = int(samples[-1])
        result["first_transition_sec"] = float((transition_samples[0] - origin) / f_adc)
        result["last_transition_sec"] = float((transition_samples[-1] - origin) / f_adc)
        result["first_transition_edge_distance_sec"] = float(
            min(transition_samples[0] - origin, end - transition_samples[0]) / f_adc
        )
        result["last_transition_edge_distance_sec"] = float(
            min(transition_samples[-1] - origin, end - transition_samples[-1]) / f_adc
        )
        chatter_samples = max(1, int(round(chatter_seconds * f_adc)))
        result["chatter_returns_le_100ms"] = int(
            np.count_nonzero(np.diff(transition_samples) <= chatter_samples)
        )
        result["max_switches_in_0_5s"] = _maximum_events_in_window(
            transition_samples, max(1, int(round(0.5 * f_adc)))
        )
        result["max_switches_in_1_0s"] = _maximum_events_in_window(
            transition_samples, max(1, int(round(1.0 * f_adc)))
        )

    run_starts = valid & np.r_[True, (~adjacent) | (labels[1:] != labels[:-1])]
    run_ends = valid & np.r_[(~adjacent) | (labels[:-1] != labels[1:]), True]
    starts = samples[run_starts]
    ends = samples[run_ends]
    if starts.size and starts.size == ends.size:
        durations = (ends - starts + 1) / f_adc
        result["median_run_duration_sec"] = float(np.median(durations))
        result["p05_run_duration_sec"] = float(np.quantile(durations, 0.05))
        result["p95_run_duration_sec"] = float(np.quantile(durations, 0.95))
        result["short_runs_le_20ms"] = int(np.count_nonzero(durations <= 0.020))
        result["short_runs_le_100ms"] = int(np.count_nonzero(durations <= 0.100))
    return result


def scan_exact_agreement(label_dir: Path, output_dir: Path, sources: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    focused_pairs = {
        frozenset(("phase_pdr_basic", "pos_seq_pdr_basic")),
        frozenset(("phase_power_pdr_basic", "pos_seq_power_pdr_basic")),
    }
    for source in sources:
        manifest = json.loads((label_dir / source / "manifest.json").read_text(encoding="utf-8"))
        algorithm_ids = tuple(manifest["algorithm_ids"])
        record_masks = _load_record_masks(label_dir / source)
        pair_counts = {
            (left, right): np.zeros((2, 2), dtype=np.int64)
            for left in range(len(algorithm_ids))
            for right in range(left + 1, len(algorithm_ids))
        }
        point_pattern_counts = np.zeros(1 << len(algorithm_ids), dtype=np.int64)
        record_pattern_sums = np.zeros_like(point_pattern_counts, dtype=np.float64)
        record_pattern_presence = np.zeros_like(point_pattern_counts, dtype=np.int64)
        records_with_all_algorithms = 0
        teacher_index = algorithm_ids.index(TEACHER_ID)
        summary_by_id = {
            int(record["record_id"]): record
            for record in load_records(label_dir, (source,))
        }
        progress = ProgressReporter(f"Agreement {source}", len(manifest["shards"]), unit="shard")
        for shard_number, shard in enumerate(manifest["shards"], start=1):
            with np.load(label_dir / source / shard["file"], allow_pickle=False) as archive:
                directions = archive["directions"].copy()
                samples = archive["samples"]
                offsets = archive["offsets"]
                record_ids = archive["record_ids"]
                for algorithm_index, algorithm_id in enumerate(algorithm_ids):
                    masked_ids = record_masks.get(algorithm_id, set())
                    if not masked_ids:
                        continue
                    for local_index, record_id in enumerate(record_ids):
                        if int(record_id) in masked_ids:
                            start = int(offsets[local_index])
                            stop = int(offsets[local_index + 1])
                            directions[algorithm_index, start:stop] = int(PDRDirection.UNLABELED)
                for local_index, record_id_raw in enumerate(record_ids):
                    record_id = int(record_id_raw)
                    start = int(offsets[local_index])
                    stop = int(offsets[local_index + 1])
                    local_directions = directions[:, start:stop]
                    local_samples = samples[start:stop]
                    all_valid = np.all(
                        local_directions != int(PDRDirection.UNLABELED), axis=0
                    )
                    if np.any(all_valid):
                        codes = np.sum(
                            local_directions[:, all_valid].astype(np.int64)
                            * (1 << np.arange(len(algorithm_ids), dtype=np.int64))[:, None],
                            axis=0,
                        )
                        counts = np.bincount(codes, minlength=len(point_pattern_counts))
                        point_pattern_counts += counts
                        fractions = counts / counts.sum()
                        record_pattern_sums += fractions
                        record_pattern_presence += counts > 0
                        records_with_all_algorithms += 1
                    summary = summary_by_id.get(record_id, {})
                    temporal_rows.append({
                        "source": source,
                        "record_id": record_id,
                        "algorithm_id": TEACHER_ID,
                        "f_adc": summary.get("f_adc", float("nan")),
                        "duration_sec": summary.get("duration_sec", float("nan")),
                        **_temporal_metrics(
                            local_directions[teacher_index],
                            local_samples,
                            float(summary.get("f_adc", float("nan"))),
                        ),
                    })
                for (left, right), confusion in pair_counts.items():
                    a, b = directions[left], directions[right]
                    common = (a != int(PDRDirection.UNLABELED)) & (b != int(PDRDirection.UNLABELED))
                    for av in (0, 1):
                        for bv in (0, 1):
                            confusion[av, bv] += int(np.count_nonzero(common & (a == av) & (b == bv)))
                    pair_name = frozenset((algorithm_ids[left], algorithm_ids[right]))
                    if pair_name in focused_pairs:
                        for local_index, record_id in enumerate(record_ids):
                            start = int(offsets[local_index])
                            stop = int(offsets[local_index + 1])
                            local_common = common[start:stop]
                            local_count = int(local_common.sum())
                            local_disagreement = (a[start:stop] != b[start:stop]) & local_common
                            disagreement_change = (
                                local_common[:-1]
                                & local_common[1:]
                                & (local_disagreement[1:] != local_disagreement[:-1])
                            )
                            record_rows.append({
                                "source": source,
                                "record_id": int(record_id),
                                "left": algorithm_ids[left],
                                "right": algorithm_ids[right],
                                "common_windows": local_count,
                                "disagreement_fraction": (
                                    float(local_disagreement.sum() / local_count) if local_count else 0.0
                                ),
                                "disagreement_transitions": int(
                                    np.count_nonzero(disagreement_change)
                                ),
                            })
            progress.update(shard_number)
        progress.finish()
        total_patterns = int(point_pattern_counts.sum())
        for code, count in enumerate(point_pattern_counts):
            if count == 0:
                continue
            pattern_rows.append({
                "source": source,
                "state_pattern": format(code, f"0{len(algorithm_ids)}b")[::-1],
                "algorithm_order": "|".join(algorithm_ids),
                "point_count": int(count),
                "point_fraction": float(count / max(1, total_patterns)),
                "records_with_pattern": int(record_pattern_presence[code]),
                "record_presence_fraction": float(
                    record_pattern_presence[code] / max(1, records_with_all_algorithms)
                ),
                "mean_record_fraction": float(
                    record_pattern_sums[code] / max(1, records_with_all_algorithms)
                ),
                "records_with_all_algorithms": records_with_all_algorithms,
            })
        for (left, right), confusion in pair_counts.items():
            total = int(confusion.sum())
            all_rows.append({
                "source": source,
                "left": algorithm_ids[left],
                "right": algorithm_ids[right],
                "common_windows": total,
                **_agreement_metrics(confusion),
                "n_reverse_reverse": int(confusion[0, 0]),
                "n_reverse_forward": int(confusion[0, 1]),
                "n_forward_reverse": int(confusion[1, 0]),
                "n_forward_forward": int(confusion[1, 1]),
            })
    _write_csv(output_dir / "pairwise_pointwise_agreement.csv", all_rows)
    _write_csv(output_dir / "pairwise_record_agreement.csv", record_rows)
    _write_csv(output_dir / "teacher_temporal_statistics.csv", temporal_rows)
    _write_csv(output_dir / "algorithm_state_patterns.csv", pattern_rows)
    _append_pair_candidates(output_dir, record_rows, per_group=4)


def scan_signal_statistics(
    label_dir: Path,
    output_dir: Path,
    sources: Sequence[str],
    *,
    low_current_rms_threshold: float,
) -> None:
    """Проверить гипотезы о нагрузке и качестве сигналов без пересчёта РНМ."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source_factories = {
        "open_ee": lambda: OpenEEShardedSource(
            PROJECT_ROOT / "data/phase5/open_ee_shards/manifest.json"
        ),
        "french_rte": lambda: FrenchRTESource(
            PROJECT_ROOT / "data/phase5/french_rte/DATA_S.npy"
        ),
    }
    rows: list[dict[str, Any]] = []
    for source_name in sources:
        source = source_factories[source_name]()
        manifest = json.loads(
            (Path(label_dir) / source_name / "manifest.json").read_text(encoding="utf-8")
        )
        selected_ids = [
            int(record_id)
            for shard in manifest.get("shards", [])
            for record_id in shard.get("record_ids", [])
        ]
        if len(selected_ids) != len(set(selected_ids)):
            raise RuntimeError(f"Manifest {source_name} содержит повторные record_id")
        checkpoint_path = output_dir / f"signal_records__{source_name}.csv"
        source_rows: list[dict[str, Any]] = []
        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8-sig", newline="") as stream:
                source_rows = list(csv.DictReader(stream))
            completed_ids = [int(row["record_id"]) for row in source_rows]
            if completed_ids != selected_ids[:len(completed_ids)]:
                raise RuntimeError(f"Signal checkpoint не совпадает с manifest: {checkpoint_path}")
        initial_completed = len(source_rows)
        progress = ProgressReporter(
            f"Signal audit {source_name}", len(selected_ids), unit="зап.",
            initial_completed=initial_completed,
        )
        progress.update(initial_completed)
        try:
            for progress_index, record_id in enumerate(
                selected_ids[initial_completed:], start=initial_completed + 1
            ):
                signal = np.asarray(source.load_signal(record_id), dtype=np.float64)
                metadata = source.get_metadata(record_id)
                current = signal[:3]
                voltage = signal[4:7]
                current_rms_phases = _channel_rms(current)
                voltage_rms_phases = _channel_rms(voltage)
                current_rms = _finite_median(current_rms_phases)
                voltage_rms = _finite_median(voltage_rms_phases)
                edge = max(1, signal.shape[1] // 5)
                first_current_rms = _global_rms(current[:, :edge])
                last_current_rms = _global_rms(current[:, -edge:])
                finite_current = np.abs(current[np.isfinite(current)])
                crest = (
                    float(np.quantile(finite_current, 0.99) / max(current_rms, 1e-12))
                    if finite_current.size else float("nan")
                )
                voltage_basis = str(metadata.get("voltage_basis", "phase"))
                power_proxy = (
                    float(np.nanmean(np.nansum(current * voltage, axis=0)))
                    if voltage_basis == "phase" else float("nan")
                )
                current_present = bool(np.isfinite(current_rms))
                voltage_present = bool(np.isfinite(voltage_rms))
                source_rows.append({
                    "source": source_name,
                    "record_id": record_id,
                    "file_name": metadata.get("file_name", record_id),
                    "f_adc": metadata.get("f_adc", metadata.get("sampling_rate_hz")),
                    "voltage_basis": voltage_basis,
                    "samples": signal.shape[1],
                    "current_rms": current_rms,
                    "voltage_rms": voltage_rms,
                    "current_to_voltage_rms": (
                        current_rms / max(voltage_rms, 1e-12)
                        if current_present and voltage_present else float("nan")
                    ),
                    "current_phase_unbalance_cv": _finite_cv(current_rms_phases),
                    "voltage_phase_unbalance_cv": _finite_cv(voltage_rms_phases),
                    "current_rms_last_over_first": (
                        last_current_rms / max(first_current_rms, 1e-12)
                        if np.isfinite(first_current_rms) and np.isfinite(last_current_rms)
                        else float("nan")
                    ),
                    "current_crest_p99_over_rms": crest,
                    "mean_three_phase_power_proxy": power_proxy,
                    "missing_current_group": not current_present,
                    "missing_voltage_group": not voltage_present,
                    "low_current_rms": current_present and current_rms < low_current_rms_threshold,
                })
                progress.update(progress_index)
                if progress_index % 1000 == 0:
                    _write_csv(checkpoint_path, source_rows)
        finally:
            if source_rows:
                _write_csv(checkpoint_path, source_rows)
            close = getattr(source, "close", None)
            if callable(close):
                close()
        _write_csv(checkpoint_path, source_rows)
        rows.extend(source_rows)
        progress.finish()
    _write_csv(output_dir / "signal_record_statistics.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    groups = [("all", rows)] + [
        (source, [row for row in rows if row["source"] == source]) for source in sources
    ]
    metrics = (
        "current_rms", "voltage_rms", "current_to_voltage_rms",
        "current_phase_unbalance_cv", "voltage_phase_unbalance_cv",
        "current_rms_last_over_first", "current_crest_p99_over_rms",
        "mean_three_phase_power_proxy",
    )
    for source, subset in groups:
        low_current_count = sum(
            np.isfinite(float(row["current_rms"]))
            and float(row["current_rms"]) < low_current_rms_threshold
            for row in subset
        )
        base = {
            "source": source,
            "records": len(subset),
            "missing_current_records": sum(_as_bool(row["missing_current_group"]) for row in subset),
            "missing_voltage_records": sum(_as_bool(row["missing_voltage_group"]) for row in subset),
            "low_current_records": low_current_count,
            "low_current_fraction": low_current_count / max(1, len(subset)),
            "low_current_rms_threshold": low_current_rms_threshold,
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            values = values[np.isfinite(values)]
            base[f"{metric}__median"] = float(np.median(values)) if values.size else float("nan")
            base[f"{metric}__p10"] = float(np.quantile(values, 0.10)) if values.size else float("nan")
            base[f"{metric}__p90"] = float(np.quantile(values, 0.90)) if values.size else float("nan")
        summary_rows.append(base)
    _write_csv(output_dir / "signal_summary.csv", summary_rows)


def scan_structural_eligibility(
    label_dir: Path,
    output_dir: Path,
    sources: Sequence[str],
) -> None:
    """Отделить невозможность расчёта записи от временного UNLABELED.

    Provenance хранится по восемь байт на запись. Если уже выполнен signal
    audit, дополнительно учитываются группы, в которых фактически нет ни
    одного конечного значения несмотря на формальный provenance.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    signal_quality: dict[tuple[str, int], tuple[bool, bool]] = {}
    signal_path = output_dir / "signal_record_statistics.csv"
    if signal_path.exists():
        with signal_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                signal_quality[(str(row["source"]), int(row["record_id"]))] = (
                    _as_bool(row["missing_current_group"]),
                    _as_bool(row["missing_voltage_group"]),
                )
    channel_groups = {
        "current": ("IA", "IB", "IC"),
        "voltage": ("UA", "UB", "UC"),
    }
    for source in sources:
        archive = StudyArchive(label_dir / source)
        reporter = ProgressReporter(f"Eligibility {source}", len(archive.record_map), unit="зап.")
        for processed, record_id in enumerate(sorted(archive.record_map), start=1):
            provenance = archive.get_provenance(record_id)
            counts = {
                name: sum(
                    int(provenance[CHANNEL_ORDER.index(channel)]) != 0
                    for channel in channels
                )
                for name, channels in channel_groups.items()
            }
            audit = check_pdr_signal_sufficiency(provenance)
            missing_current_group, missing_voltage_group = signal_quality.get(
                (source, record_id), (False, False)
            )
            structurally_eligible = bool(
                audit.can_run_phase_pdr
                and not missing_current_group
                and not missing_voltage_group
            )
            if missing_current_group:
                counts["current"] = 0
            if missing_voltage_group:
                counts["voltage"] = 0
            rows.append({
                "source": source,
                "record_id": record_id,
                "available_current_channels": counts["current"],
                "available_voltage_channels": counts["voltage"],
                "pdr_structurally_eligible": structurally_eligible,
                "missing_finite_current_group": missing_current_group,
                "missing_finite_voltage_group": missing_voltage_group,
                "missing_channels": "|".join(audit.missing_channels),
                "derived_channels": "|".join(audit.derived_channels),
            })
            reporter.update(processed)
        reporter.finish()
        ineligible = [
            int(row["record_id"])
            for row in rows
            if row["source"] == source and not row["pdr_structurally_eligible"]
        ]
        _atomic_json(label_dir / source / "STRUCTURALLY_INELIGIBLE_RECORDS.json", {
            "schema_version": 1,
            "policy": "exclude_record_if_fewer_than_2I_or_2U_are_available",
            "source": source,
            "record_ids": ineligible,
            "record_count": len(ineligible),
            "evidence": "stored provenance plus signal_record_statistics finite-group audit",
        })
    _write_csv(output_dir / "record_signal_eligibility.csv", rows)

def _channel_rms(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    sums = np.where(finite, values * values, 0.0).sum(axis=1)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    present = counts > 0
    result[present] = np.sqrt(sums[present] / counts[present])
    return result


def _global_rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _finite_cv(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    return float(np.std(finite) / max(np.mean(finite), 1e-12))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _append_pair_candidates(
    output_dir: Path, record_rows: Sequence[dict[str, Any]], *, per_group: int
) -> None:
    candidate_path = output_dir / "plot_candidates.csv"
    if not candidate_path.exists():
        print(
            "plot_candidates.csv отсутствует: pair-кандидаты будут добавлены после MODE='summary'",
            file=sys.stderr,
        )
        return
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidates = [row for row in csv.DictReader(stream) if not row["stratum"].startswith("pair_")]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in record_rows:
        fraction = float(row["disagreement_fraction"])
        transitions = int(row["disagreement_transitions"])
        if fraction <= 0.0:
            continue
        mode = "dynamic" if transitions > 0 else "static"
        key = (str(row["source"]), str(row["left"]), str(row["right"]), mode)
        grouped[key].append(row)
    for (source, left, right, mode), rows in sorted(grouped.items()):
        if mode == "dynamic":
            rows.sort(key=lambda row: (
                -int(row["disagreement_transitions"]),
                -float(row["disagreement_fraction"]),
                int(row["record_id"]),
            ))
        else:
            rows.sort(key=lambda row: (-float(row["disagreement_fraction"]), int(row["record_id"])))
        # Берём разные квантили списка, чтобы не получить только почти одинаковый extreme-tail.
        indices = np.linspace(0, len(rows) - 1, min(per_group, len(rows)), dtype=int)
        stratum = f"pair_{mode}__{left}__vs__{right}"
        for rank, row_index in enumerate(indices, start=1):
            row = rows[int(row_index)]
            candidates.append({
                "source": source,
                "record_id": int(row["record_id"]),
                "stratum": stratum,
                "selection_rank": rank,
                "file_name": "",
                "split": "",
                "duration_sec": "",
                "teacher_transitions": "",
                "total_transitions": "",
                "disagreement_fraction": row["disagreement_fraction"],
                "interest_score": "",
                "focus_left": left,
                "focus_right": right,
            })
    _write_csv(candidate_path, candidates)
    _update_summary_count(output_dir, "plot_candidates", len(candidates))


def _append_forced_candidates(
    output_dir: Path,
    forced_cases: Sequence[tuple[str, int]],
) -> None:
    """Добавить заданные исследователем осциллограммы независимо от score."""
    if not forced_cases:
        return
    candidate_path = output_dir / "plot_candidates.csv"
    if not candidate_path.exists():
        raise FileNotFoundError("Сначала нужен MODE='summary' для plot_candidates.csv")
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidates = [
            row for row in csv.DictReader(stream) if row["stratum"] != "forced_review"
        ]
    for rank, (source, record_id) in enumerate(forced_cases, start=1):
        candidates.append({
            "source": source,
            "record_id": int(record_id),
            "stratum": "forced_review",
            "selection_rank": rank,
            "file_name": "",
            "split": "",
            "duration_sec": "",
            "teacher_transitions": "",
            "total_transitions": "",
            "disagreement_fraction": "",
            "interest_score": "",
            "focus_left": "",
            "focus_right": "",
        })
    _write_csv(candidate_path, candidates)
    _update_summary_count(output_dir, "plot_candidates", len(candidates))


def _update_summary_count(output_dir: Path, key: str, value: int) -> None:
    """Keep the lightweight run manifest consistent after staged analysis modes."""
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload[key] = int(value)
    _atomic_json(summary_path, payload)


def build_diagnostic_plots(
    label_dir: Path,
    output_dir: Path,
    sources: Sequence[str],
    *,
    window_seconds: float,
    export_csv: bool,
) -> None:
    candidate_path = output_dir / "plot_candidates.csv"
    if not candidate_path.exists():
        raise FileNotFoundError("Сначала выполните MODE='summary': plot_candidates.csv отсутствует")
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidates = [row for row in csv.DictReader(stream) if row["source"] in sources]
    source_factories = {
        "open_ee": lambda: OpenEEShardedSource(
            PROJECT_ROOT / "data/phase5/open_ee_shards/manifest.json"
        ),
        "french_rte": lambda: FrenchRTESource(
            PROJECT_ROOT / "data/phase5/french_rte/DATA_S.npy"
        ),
    }
    sources_by_name = {source: source_factories[source]() for source in sources}
    archives = {source: StudyArchive(label_dir / source) for source in sources}
    plot_root = output_dir / "diagnostic_cases"
    plot_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    progress = ProgressReporter("Диагностические графики", max(1, len(candidates)), unit="случай")
    try:
        for index, candidate in enumerate(candidates, start=1):
            source_name = candidate["source"]
            record_id = int(candidate["record_id"])
            source = sources_by_name[source_name]
            metadata = source.get_metadata(record_id)
            signal = source.load_signal(record_id)
            labels = archives[source_name].get_record(record_id)
            f_adc = float(metadata.get("f_adc", metadata.get("sampling_rate_hz", 1.0)))
            start, stop, local_transitions, local_disagreement = _best_window(
                labels,
                signal.shape[1],
                f_adc,
                window_seconds,
                algorithm_ids=archives[source_name].algorithm_ids,
                focus_pair=(candidate.get("focus_left", ""), candidate.get("focus_right", "")),
            )
            # Stratum уже хранится в имени папки. Не дублируем его в
            # filename, чтобы не превышать Windows MAX_PATH для длинных pair-ID.
            stem = (
                f"{source_name}__record_{record_id:05d}__"
                f"rank_{int(candidate.get('selection_rank') or 0):03d}"
            )
            case_dir = plot_root / candidate["stratum"]
            case_dir.mkdir(parents=True, exist_ok=True)
            png_path = case_dir / f"{stem}.png"
            _plot_case(
                png_path, signal, labels, archives[source_name].algorithm_ids, metadata,
                source_name, record_id, start, stop, f_adc,
            )
            csv_path: Path | None = None
            if export_csv:
                csv_path = case_dir / f"{stem}.csv"
                _export_case_csv(csv_path, signal, labels, archives[source_name].algorithm_ids,
                                 start, stop, f_adc)
            manifest_rows.append(dict(candidate) | {
                "window_start_sec": start / f_adc,
                "window_stop_sec": stop / f_adc,
                "local_total_transitions": local_transitions,
                "local_disagreement_fraction": local_disagreement,
                "png": str(png_path.relative_to(output_dir)),
                "csv": str(csv_path.relative_to(output_dir)) if csv_path else "",
            })
            progress.update(index)
    finally:
        for source in sources_by_name.values():
            close = getattr(source, "close", None)
            if callable(close):
                close()
    progress.finish()
    _write_csv(output_dir / "diagnostic_cases.csv", manifest_rows)
    _update_summary_count(output_dir, "diagnostic_cases", len(manifest_rows))


def _best_window(
    labels: dict[str, np.ndarray],
    signal_length: int,
    f_adc: float,
    window_seconds: float,
    *,
    algorithm_ids: Sequence[str] = (),
    focus_pair: tuple[str, str] = ("", ""),
) -> tuple[int, int, int, float]:
    samples = labels["samples"].astype(np.int64)
    directions = labels["directions"]
    width = min(signal_length, max(2, int(round(window_seconds * f_adc))))
    transitions = np.zeros(directions.shape[1], dtype=np.float64)
    transitions[1:] = np.sum(
        (directions[:, 1:] != directions[:, :-1])
        & (directions[:, 1:] != int(PDRDirection.UNLABELED))
        & (directions[:, :-1] != int(PDRDirection.UNLABELED)),
        axis=0,
    )
    valid = directions != int(PDRDirection.UNLABELED)
    votes = np.sum(directions == int(PDRDirection.FORWARD), axis=0)
    valid_count = valid.sum(axis=0)
    disagreement = (valid_count >= 2) & (votes > 0) & (votes < valid_count)
    score = 3.0 * transitions + disagreement.astype(np.float64)
    if all(focus_pair) and all(algorithm_id in algorithm_ids for algorithm_id in focus_pair):
        left = algorithm_ids.index(focus_pair[0])
        right = algorithm_ids.index(focus_pair[1])
        pair_valid = valid[left] & valid[right]
        pair_disagreement = pair_valid & (directions[left] != directions[right])
        pair_transition = np.zeros(len(samples), dtype=np.float64)
        pair_transition[1:] = (
            pair_valid[1:] & pair_valid[:-1]
            & (pair_disagreement[1:] != pair_disagreement[:-1])
        )
        score += 5.0 * pair_transition + 2.0 * pair_disagreement.astype(np.float64)
    label_width = max(1, int(round(width * len(samples) / max(signal_length, 1))))
    cumulative = np.concatenate(([0.0], np.cumsum(score)))
    if len(score) <= label_width:
        center_sample = int(samples[len(samples) // 2])
    else:
        rolling = cumulative[label_width:] - cumulative[:-label_width]
        best = int(np.argmax(rolling))
        center_sample = int(samples[min(len(samples) - 1, best + label_width // 2)])
    start = min(max(0, center_sample - width // 2), max(0, signal_length - width))
    stop = min(signal_length, start + width)
    inside = (samples >= start) & (samples < stop)
    local_transitions = int(np.sum(transitions[inside]))
    local_disagreement = float(np.mean(disagreement[inside])) if np.any(inside) else 0.0
    return start, stop, local_transitions, local_disagreement


def _plot_case(
    path: Path,
    signal: np.ndarray,
    labels: dict[str, np.ndarray],
    algorithm_ids: Sequence[str],
    metadata: dict[str, Any],
    source: str,
    record_id: int,
    start: int,
    stop: int,
    f_adc: float,
) -> None:
    import matplotlib.pyplot as plt

    time_axis = np.arange(start, stop) / f_adc
    samples = labels["samples"]
    mask = (samples >= start) & (samples < stop)
    label_time = samples[mask] / f_adc
    figure, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                                gridspec_kw={"height_ratios": (2, 2, 2.4, 1.7)})
    for channel, name in zip(range(3), CHANNEL_ORDER[:3]):
        axes[0].plot(time_axis, signal[channel, start:stop], linewidth=0.8, label=name)
    voltage_names = ("UA", "UB", "UC") if metadata.get("voltage_basis", "phase") == "phase" else ("UAB", "UBC", "UCA")
    for channel, name in zip(range(4, 7), voltage_names):
        axes[1].plot(time_axis, signal[channel, start:stop], linewidth=0.8, label=name)
    axes[0].set_ylabel("Ток, p.u.")
    axes[1].set_ylabel("Напряжение, p.u.")
    axes[0].legend(ncol=3, loc="upper right")
    axes[1].legend(ncol=3, loc="upper right")

    directions = labels["directions"][:, mask]
    lane_height = 0.72
    for algorithm_index, algorithm_id in enumerate(algorithm_ids):
        values = directions[algorithm_index].astype(float)
        values[values == int(PDRDirection.UNLABELED)] = np.nan
        axes[2].step(
            label_time,
            algorithm_index + lane_height * values,
            where="post",
            linewidth=1.15,
        )
        axes[2].axhline(algorithm_index, color="0.82", linewidth=0.45, zorder=0)
        axes[2].axhline(
            algorithm_index + lane_height, color="0.90", linewidth=0.45, zorder=0
        )
    axes[2].set_yticks(
        np.arange(len(algorithm_ids)) + lane_height / 2.0,
        labels=algorithm_ids,
    )
    axes[2].set_ylim(-0.15, len(algorithm_ids) - 1 + lane_height + 0.15)
    axes[2].set_ylabel("Решение РНМ")
    axes[2].text(
        0.995,
        0.985,
        "верх дорожки = 1 / FORWARD / блокировка\n"
        "низ дорожки = 0 / REVERSE / разрешение; пробел = UNLABELED",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    axes[2].grid(axis="x", alpha=0.25)

    margins = labels["margins"][:, mask]
    for algorithm_index, algorithm_id in enumerate(algorithm_ids):
        valid = directions[algorithm_index] != int(PDRDirection.UNLABELED)
        plotted_margin = np.where(valid, margins[algorithm_index], np.nan)
        finite = np.abs(plotted_margin[np.isfinite(plotted_margin)])
        scale = float(np.quantile(finite, 0.90)) if finite.size else 1.0
        axes[3].plot(label_time, np.arcsinh(plotted_margin / max(scale, 1e-12)),
                     linewidth=0.8, label=algorithm_id)
    axes[3].axhline(0.0, color="black", linewidth=0.7)
    axes[3].set_ylabel("Норм. запас\n(asinh, /P90)")
    axes[3].text(
        0.005,
        0.04,
        "+ внутри зоны FORWARD; − вне зоны / ниже порога. "
        "Масштаб каждого РНМ отдельный — сравнивать знак и динамику, не высоту.",
        transform=axes[3].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    axes[3].set_xlabel("Время, с")
    axes[3].legend(ncol=3, fontsize=8, loc="upper right")
    title = (
        f"{source}, record={record_id}, {metadata.get('file_name', record_id)}; "
        f"f_adc={f_adc:g} Гц, U={metadata.get('voltage_basis', 'phase')}"
    )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _export_case_csv(
    path: Path,
    signal: np.ndarray,
    labels: dict[str, np.ndarray],
    algorithm_ids: Sequence[str],
    start: int,
    stop: int,
    f_adc: float,
) -> None:
    label_lookup = {int(sample): index for index, sample in enumerate(labels["samples"])}
    fields = ["sample", "time_sec", *CHANNEL_ORDER]
    for algorithm_id in algorithm_ids:
        fields.extend((f"{algorithm_id}__direction", f"{algorithm_id}__margin",
                       f"{algorithm_id}__confidence", f"{algorithm_id}__warmup"))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in range(start, stop):
            row: dict[str, Any] = {"sample": sample, "time_sec": sample / f_adc}
            row.update({name: float(signal[index, sample]) for index, name in enumerate(CHANNEL_ORDER)})
            label_index = label_lookup.get(sample)
            for algorithm_index, algorithm_id in enumerate(algorithm_ids):
                if label_index is None:
                    row[f"{algorithm_id}__direction"] = int(PDRDirection.UNLABELED)
                    row[f"{algorithm_id}__margin"] = ""
                    row[f"{algorithm_id}__confidence"] = ""
                    row[f"{algorithm_id}__warmup"] = True
                else:
                    row[f"{algorithm_id}__direction"] = int(labels["directions"][algorithm_index, label_index])
                    row[f"{algorithm_id}__margin"] = float(labels["margins"][algorithm_index, label_index])
                    row[f"{algorithm_id}__confidence"] = float(labels["confidences"][algorithm_index, label_index])
                    row[f"{algorithm_id}__warmup"] = bool(labels["warmup"][algorithm_index, label_index])
            writer.writerow(row)


def _plot_overview(records: Sequence[dict[str, Any]], sources: Sequence[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    groups = ("0_stable", "0_other_switch", "1_2", "3_10", "11_30", "31_plus")
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(sources))
    figure, axis = plt.subplots(figsize=(11, 5))
    for source_index, source in enumerate(sources):
        subset = [record for record in records if record["source"] == source]
        counts = Counter(transition_group(record) for record in subset)
        fractions = [counts[group] / max(1, len(subset)) for group in groups]
        axis.bar(x + (source_index - (len(sources) - 1) / 2) * width, fractions,
                 width=width, label=source)
    axis.set_xticks(x, labels=groups)
    axis.set_ylabel("Доля осциллограмм")
    axis.set_title("Переходы адаптивного РНМ по осциллограммам")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "adaptive_transition_groups.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(sources), figsize=(14, 5), sharey=True)
    if len(sources) == 1:
        axes = [axes]
    for axis, source in zip(axes, sources):
        subset = [record for record in records if record["source"] == source]
        algorithm_ids = tuple(subset[0]["algorithms"])
        record_means = []
        window_means = []
        any_forward = []
        for algorithm_id in algorithm_ids:
            stats = [record["algorithms"][algorithm_id] for record in subset]
            valid_stats = [item for item in stats if int(item["valid_windows"]) > 0]
            fractions = np.asarray([item["forward_fraction"] for item in valid_stats], dtype=float)
            valid_windows = np.asarray([item["valid_windows"] for item in valid_stats], dtype=float)
            record_means.append(float(np.mean(fractions)))
            window_means.append(float(np.sum(fractions * valid_windows) / max(1.0, np.sum(valid_windows))))
            any_forward.append(float(np.mean(fractions > 0.0)))
        x_alg = np.arange(len(algorithm_ids))
        bar_width = 0.25
        axis.bar(x_alg - bar_width, window_means, width=bar_width,
                 label="доля FORWARD по всем валидным точкам")
        axis.bar(x_alg, record_means, width=bar_width,
                 label="средняя доля FORWARD по осциллограммам")
        axis.bar(x_alg + bar_width, any_forward, width=bar_width,
                 label="доля записей, где FORWARD встречался")
        axis.set_xticks(x_alg, labels=algorithm_ids)
        axis.tick_params(axis="x", rotation=35)
        axis.set_title(source)
        axis.set_ylabel("Доля")
        axis.set_ylim(0.0, 1.04)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    figure.suptitle("FORWARD: три разных способа агрегации")
    figure.tight_layout()
    figure.savefig(output_dir / "algorithm_forward_fraction.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(sources), figsize=(14, 5), sharey=True)
    if len(sources) == 1:
        axes = [axes]
    for axis, source in zip(axes, sources):
        subset = [record for record in records if record["source"] == source]
        algorithm_ids = tuple(subset[0]["algorithms"])
        values = [[
            record["algorithms"][algorithm_id]["forward_fraction"]
            for record in subset
            if int(record["algorithms"][algorithm_id]["valid_windows"]) > 0
        ] for algorithm_id in algorithm_ids]
        axis.boxplot(values, tick_labels=algorithm_ids, showfliers=False)
        axis.tick_params(axis="x", rotation=35)
        axis.set_title(source)
        axis.set_ylabel("Доля FORWARD в одной записи")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Распределение доли FORWARD по осциллограммам")
    figure.tight_layout()
    figure.savefig(output_dir / "algorithm_forward_fraction_boxplot.png", dpi=160)
    plt.close(figure)


def _write_report(
    dataset_rows: Sequence[dict[str, Any]], transition_rows: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    lines = [
        "# Первичный статистический аудит multi-PDR разметки", "",
        "Отчёт разделяет источники: различия между Open_EE и French/RTE нельзя трактовать "
        "как качество алгоритма без ручной проверки и учёта состава режимов.", "",
        "## Полнота и дубликаты", "",
    ]
    for row in dataset_rows:
        lines.append(
            f"- **{row['source']}**: {row['records']:,} записей, "
            f"{row['windows']:,} решений, {row['duration_hours']:.2f} ч сигнала; "
            f"дубликатами затронуто {row['duplicate_records']:,} записей "
            f"({100 * row['duplicate_fraction']:.2f}%)."
        )
    lines.extend([
        "", "## Интерпретация", "",
        "- `transition_groups.csv` — стратификация по числу переходов адаптивного РНМ.",
        "- `algorithm_record_distributions.csv` — распределения считаются по осциллограммам, "
        "а не по автокоррелированным временным точкам.",
        "- `source_shift.csv` — описательная диагностика доменного сдвига; малые p-value при "
        "таком объёме данных сами по себе не означают практическую значимость.",
        "- `cluster_profiles.csv` — только exploratory-архетипы для покрытия ручным аудитом, "
        "не физические классы и не целевые метки.",
        "- `pairwise_pointwise_agreement.csv` появляется после режима `agreement`; для статьи "
        "основной единицей bootstrap/доверительных интервалов должна оставаться осциллограмма.",
        "- `teacher_temporal_statistics.csv` описывает серии, энтропию, локальную плотность "
        "переключений и chatter адаптивного teacher на уровне осциллограмм.",
        "- `algorithm_state_patterns.csv` хранит комбинации пяти РНМ одновременно как "
        "point-weighted и record-weighted доли.",
    ])
    (output_dir / "PRIMARY_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fallback_fields: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else list(fallback_fields)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_analysis(
    *,
    mode: str,
    label_dir: Path = DEFAULT_LABEL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sources: Sequence[str] = DEFAULT_SOURCES,
    enable_clusters: bool = True,
    plots_per_group: int = 4,
    window_seconds: float = 0.8,
    export_csv: bool = True,
    low_current_rms_threshold: float = DEFAULT_LOW_CURRENT_RMS_THRESHOLD,
    forced_cases: Sequence[tuple[str, int]] = (),
) -> None:
    if mode in {"signals", "all"}:
        scan_signal_statistics(
            label_dir,
            output_dir,
            sources,
            low_current_rms_threshold=low_current_rms_threshold,
        )
    if mode in {"eligibility", "all"}:
        scan_structural_eligibility(label_dir, output_dir, sources)
    if mode in {"summary", "all"}:
        build_summary(label_dir, output_dir, sources, enable_clusters=enable_clusters,
                      plots_per_group=plots_per_group)
    if mode in {"agreement", "all"}:
        scan_exact_agreement(label_dir, output_dir, sources)
    if mode in {"summary", "agreement", "plots", "all"}:
        _append_forced_candidates(output_dir, forced_cases)
    if mode in {"plots", "all"}:
        build_diagnostic_plots(label_dir, output_dir, sources,
                               window_seconds=window_seconds, export_csv=export_csv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("summary", "eligibility", "signals", "agreement", "plots", "all"), default="summary"
    )
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sources", nargs="+", choices=DEFAULT_SOURCES, default=list(DEFAULT_SOURCES))
    parser.add_argument("--plots-per-group", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=0.8)
    parser.add_argument("--no-clusters", action="store_true")
    parser.add_argument("--no-window-csv", action="store_true")
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="SOURCE:ID",
        help="Принудительно добавить запись в PNG/CSV, например open_ee:44574",
    )
    parser.add_argument(
        "--low-current-rms-threshold",
        type=float,
        default=DEFAULT_LOW_CURRENT_RMS_THRESHOLD,
        help=(
            "Порог для RMS мгновенного нормированного тока; по умолчанию "
            "0.05 Iном / current_reserve=20 = 0.0025"
        ),
    )
    args = parser.parse_args()
    forced_cases = []
    for value in args.record:
        source, separator, record_id = value.partition(":")
        if not separator or source not in DEFAULT_SOURCES:
            parser.error(f"Ожидалось SOURCE:ID, получено {value!r}")
        forced_cases.append((source, int(record_id)))
    run_analysis(
        mode=args.mode,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        sources=args.sources,
        enable_clusters=not args.no_clusters,
        plots_per_group=args.plots_per_group,
        window_seconds=args.window_seconds,
        export_csv=not args.no_window_csv,
        low_current_rms_threshold=args.low_current_rms_threshold,
        forced_cases=forced_cases,
    )
    return 0


def run_manual() -> None:
    # MODE="all": summary -> agreement/temporal/patterns -> signals -> plots.
    MODE = "all"               # summary | eligibility | signals | agreement | plots | all
    LABEL_DIR = DEFAULT_LABEL_DIR
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    SOURCES = DEFAULT_SOURCES
    ENABLE_CLUSTERS = True          # Exploratory KMeans отдельно внутри каждого источника.
    PLOTS_PER_GROUP = 20             # На источник и audit-группу; отбор разнообразный, не только top.
    WINDOW_SECONDS = 0.8            # Короткое окно 800 мс вокруг максимума локальной динамики.
    EXPORT_WINDOW_CSV = True        # Сигналы + все решения/margins для ручной перепроверки.
    # RMS исходной волны: 0.05 Iном / current_reserve=20. В отличие от DFT-порогов,
    # здесь sqrt(2) не нужен, поскольку сравниваются RMS с RMS.
    LOW_CURRENT_RMS_THRESHOLD = DEFAULT_LOW_CURRENT_RMS_THRESHOLD
    # Всегда построить проверочный случай из прежнего аудита, даже если
    # после исправления он больше не попадает в рейтинг disagreement.
    # Два ранее разобранных контрольных случая всегда попадают в PNG,
    # даже если после исправления они выпадут из вершины рейтинга.
    FORCED_CASES = (("open_ee", 44574), ("french_rte", 44))

    run_analysis(
        mode=MODE,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        sources=SOURCES,
        enable_clusters=ENABLE_CLUSTERS,
        plots_per_group=PLOTS_PER_GROUP,
        window_seconds=WINDOW_SECONDS,
        export_csv=EXPORT_WINDOW_CSV,
        low_current_rms_threshold=LOW_CURRENT_RMS_THRESHOLD,
        forced_cases=FORCED_CASES,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
