"""Потоковая multi-PDR разметка Phase 5, статистика и manual-review candidates.

Сценарий рассчитан на полный Open_EE + French/RTE архив: читает по одной записи,
сохраняет небольшие атомарные shards, автоматически продолжает готовые shards и
пишет ``progress.json`` с количеством записей, скоростью, ETA и объёмом результата.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.phase5_contracts import TimebaseContract
from osc_tools.ml.phase5_sources import DatasetSource, FrenchRTESource, OpenEEShardedSource
from osc_tools.pdr.base import PDRAlgorithm
from osc_tools.pdr.registry import get_pdr_algorithm
from osc_tools.pdr.study import (
    MultiPDRRecordResult,
    label_record_multi,
    stable_config_hash,
    summarize_record_interest,
)
from scripts.phase5_experiments.progress import ProgressReporter


DEFAULT_ALGORITHMS = (
    "adaptive_pdr_mir",
    "phase_pdr_basic",
    "pos_seq_pdr_basic",
    "phase_power_pdr_basic",
    "pos_seq_power_pdr_basic",
)
DEFAULT_TEACHER = "adaptive_pdr_mir"


def create_algorithms(algorithm_ids: Sequence[str]) -> list[PDRAlgorithm]:
    """Создать органы без допустимого fallback и проверить уникальность ID."""

    algorithms = [get_pdr_algorithm(algorithm_id, fallback_id=None) for algorithm_id in algorithm_ids]
    missing = [
        algorithm.requested_algorithm_id
        for algorithm in algorithms
        if algorithm.fallback_applied or algorithm.resolved_algorithm_id != algorithm.requested_algorithm_id
    ]
    if missing:
        raise RuntimeError(
            "Запрошенные алгоритмы недоступны или разрешились неявно: " + ", ".join(missing)
        )
    resolved = [algorithm.resolved_algorithm_id for algorithm in algorithms]
    if len(set(resolved)) != len(resolved):
        raise ValueError("После разрешения ID список алгоритмов содержит дубликаты")
    return algorithms


def run_study(
    *,
    output_dir: Path,
    source_names: Sequence[str],
    algorithm_ids: Sequence[str] = DEFAULT_ALGORITHMS,
    teacher_id: str = DEFAULT_TEACHER,
    split_scope: str = "all",
    sample_step: int = 1,
    records_per_shard: int = 64,
    max_records_per_source: int | None = None,
    lossless_compression: bool = True,
) -> dict[str, Any]:
    """Выполнить возобновляемую разметку выбранных источников."""

    if records_per_shard <= 0:
        raise ValueError("records_per_shard должен быть положительным")
    if sample_step <= 0:
        raise ValueError("sample_step должен быть положительным")
    algorithms = create_algorithms(algorithm_ids)
    resolved_ids = tuple(algorithm.resolved_algorithm_id for algorithm in algorithms)
    if teacher_id not in resolved_ids:
        raise ValueError(f"Teacher {teacher_id!r} должен входить в список алгоритмов")
    parameter_fingerprints = {
        algorithm.resolved_algorithm_id: stable_config_hash(_jsonable(algorithm.params))
        for algorithm in algorithms
    }
    run_config = {
        "schema_version": 1,
        "kind": "pdr_dataset_study",
        "source_names": list(source_names),
        "algorithm_ids": list(resolved_ids),
        "teacher_algorithm_id": teacher_id,
        "parameter_fingerprints": parameter_fingerprints,
        "split_scope": split_scope,
        "sample_step": sample_step,
        "records_per_shard": records_per_shard,
        "max_records_per_source": max_records_per_source,
        "lossless_compression": lossless_compression,
        "stores_all_algorithm_outputs": True,
    }
    config_hash = stable_config_hash(run_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise RuntimeError(
                "Каталог содержит другой PDR-контракт. Выберите новый OUTPUT_DIR; "
                "существующая разметка не будет перезаписана."
            )
    else:
        _atomic_write_json(config_path, run_config | {
            "config_hash": config_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    split_manifest = json.loads(
        (PROJECT_ROOT / "data/phase5/research_strict_splits.json").read_text(encoding="utf-8")
    )
    source_summaries: dict[str, Any] = {}
    for source_name in source_names:
        source = _create_source(source_name)
        try:
            selected, split_lookup = _select_indices(
                split_manifest,
                source_name,
                split_scope,
                max_records_per_source,
            )
            source_summaries[source_name] = _process_source(
                source=source,
                selected_indices=selected,
                split_lookup=split_lookup,
                algorithms=algorithms,
                teacher_id=teacher_id,
                output_dir=output_dir / source_name,
                config_hash=config_hash,
                sample_step=sample_step,
                records_per_shard=records_per_shard,
                lossless_compression=lossless_compression,
            )
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()

    summary = {
        "config_hash": config_hash,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_summaries,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _process_source(
    *,
    source: DatasetSource,
    selected_indices: list[int],
    split_lookup: dict[int, str],
    algorithms: Sequence[PDRAlgorithm],
    teacher_id: str,
    output_dir: Path,
    config_hash: str,
    sample_step: int,
    records_per_shard: int,
    lossless_compression: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        selected_indices[start:start + records_per_shard]
        for start in range(0, len(selected_indices), records_per_shard)
    ]
    existing: dict[int, dict[str, Any]] = {}
    for sidecar_path in sorted(output_dir.glob("shard_*.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        shard_index = int(sidecar["shard_index"])
        npz_path = output_dir / str(sidecar["file"])
        stats_path = output_dir / str(sidecar["statistics_file"])
        if sidecar.get("config_hash") != config_hash or not npz_path.exists() or not stats_path.exists():
            raise RuntimeError(f"Неполный или несовместимый shard: {sidecar_path}")
        if shard_index >= len(batches) or sidecar["record_ids"] != batches[shard_index]:
            raise RuntimeError(f"Состав resume-shard не совпадает с текущим выбором: {sidecar_path}")
        existing[shard_index] = sidecar

    completed_before = sum(len(item["record_ids"]) for item in existing.values())
    progress = ProgressReporter(
        f"PDR {source.name}",
        len(selected_indices),
        unit="зап.",
        initial_completed=completed_before,
    )
    progress.update(completed_before)
    started = time.monotonic()
    processed_now = 0
    estimated_windows = _estimate_windows(source, selected_indices, sample_step)
    n_algorithms = len(algorithms)
    # На каждый алгоритм: direction int16 + margin/confidence float32 + warmup bool.
    # Плюс общий sample index int32.
    bytes_per_window = 11 * n_algorithms + 4
    print(
        f"\n{source.name}: {len(selected_indices):,} записей; примерно {estimated_windows:,} окон; "
        f"сырой объём целевой разметки около {estimated_windows * bytes_per_window / 2**30:.2f} ГБ."
    )

    for shard_index, record_ids in enumerate(batches):
        if shard_index in existing:
            continue
        record_results: list[MultiPDRRecordResult] = []
        statistics: list[dict[str, Any]] = []
        try:
            for record_id in record_ids:
                metadata = source.get_metadata(record_id)
                timebase = _record_timebase(metadata)
                result = label_record_multi(
                    source.load_signal(record_id),
                    source.get_provenance(record_id),
                    timebase,
                    str(metadata.get("voltage_basis", "phase")),
                    algorithms,
                    sample_step=sample_step,
                )
                interest = summarize_record_interest(result, timebase)
                statistics.append({
                    "source": source.name,
                    "record_id": record_id,
                    "split": split_lookup.get(record_id, "unknown"),
                    "file_name": metadata.get("file_name", str(record_id)),
                    "source_csv": metadata.get("source_csv"),
                    "f_adc": timebase.sampling_rate_hz,
                    "f_network": timebase.network_frequency_hz,
                    "spp": timebase.spp,
                    "voltage_basis": metadata.get("voltage_basis", "phase"),
                    "input_sha256": result.input_sha256,
                } | interest)
                record_results.append(result)
                processed_now += 1
                completed = completed_before + processed_now
                progress.update(completed)
                _atomic_write_json(output_dir / "progress.json", progress.snapshot(completed) | {
                    "source": source.name,
                    "current_record_id": record_id,
                    "completed_shards": len(existing),
                    "total_shards": len(batches),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            completed = completed_before + processed_now
            _atomic_write_json(output_dir / "progress.json", progress.snapshot(completed) | {
                "source": source.name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_shard": shard_index,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise

        sidecar = _write_shard(
            output_dir=output_dir,
            shard_index=shard_index,
            record_ids=record_ids,
            results=record_results,
            statistics=statistics,
            teacher_id=teacher_id,
            config_hash=config_hash,
            lossless_compression=lossless_compression,
        )
        existing[shard_index] = sidecar
        _write_source_manifest(
            output_dir,
            source.name,
            algorithms,
            teacher_id,
            config_hash,
            selected_indices,
            existing,
            sample_step,
        )

    progress.finish()
    elapsed_now = max(time.monotonic() - started, 1e-9)
    aggregate = _aggregate_statistics(output_dir)
    result_bytes = sum((output_dir / item["file"]).stat().st_size for item in existing.values())
    summary = {
        "records": len(selected_indices),
        "windows": sum(int(item["n_windows"]) for item in existing.values()),
        "shards": len(existing),
        "result_bytes": result_bytes,
        "result_gib": result_bytes / 2**30,
        "records_processed_this_run": processed_now,
        "records_per_second_this_run": processed_now / elapsed_now,
        "statistics": aggregate,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_json(output_dir / "progress.json", {
        "source": source.name,
        "status": "complete",
        "completed": len(selected_indices),
        "total": len(selected_indices),
        "percent": 100.0,
        "result_gib": summary["result_gib"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return summary


def _write_shard(
    *,
    output_dir: Path,
    shard_index: int,
    record_ids: list[int],
    results: list[MultiPDRRecordResult],
    statistics: list[dict[str, Any]],
    teacher_id: str,
    config_hash: str,
    lossless_compression: bool,
) -> dict[str, Any]:
    if not results:
        raise ValueError("Нельзя сохранить пустой PDR shard")
    algorithm_ids = results[0].algorithm_ids
    if teacher_id not in algorithm_ids:
        raise ValueError("Teacher отсутствует среди результатов shard")
    lengths = [len(result.sample_indices) for result in results]
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    directions = np.concatenate([result.directions for result in results], axis=1)
    margins = np.concatenate([result.margins for result in results], axis=1)
    confidences = np.concatenate([result.confidences for result in results], axis=1)
    warmup = np.concatenate([result.warmup_mask for result in results], axis=1)
    payload: dict[str, np.ndarray] = {
        "record_ids": np.asarray(record_ids, dtype=np.int32),
        "offsets": offsets,
        "samples": np.concatenate([result.sample_indices for result in results]),
        "directions": directions,
        "all_margins": margins.astype(np.float32, copy=False),
        "all_confidences": confidences.astype(np.float32, copy=False),
        "all_warmup": warmup,
        "provenance": np.stack([result.provenance for result in results]),
    }

    npz_name = f"shard_{shard_index:05d}.npz"
    npz_path = output_dir / npz_name
    temporary_npz = output_dir / f".{npz_name}.tmp"
    with temporary_npz.open("wb") as stream:
        if lossless_compression:
            np.savez_compressed(stream, **payload)
        else:
            np.savez(stream, **payload)
    temporary_npz.replace(npz_path)

    stats_name = f"shard_{shard_index:05d}.jsonl"
    stats_path = output_dir / stats_name
    temporary_stats = output_dir / f".{stats_name}.tmp"
    with temporary_stats.open("w", encoding="utf-8") as stream:
        for record in statistics:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_stats.replace(stats_path)

    sidecar = {
        "config_hash": config_hash,
        "shard_index": shard_index,
        "file": npz_name,
        "statistics_file": stats_name,
        "record_ids": record_ids,
        "n_windows": int(offsets[-1]),
        "bytes": npz_path.stat().st_size,
    }
    _atomic_write_json(output_dir / f"shard_{shard_index:05d}.json", sidecar)
    return sidecar


def _write_source_manifest(
    output_dir: Path,
    source_name: str,
    algorithms: Sequence[PDRAlgorithm],
    teacher_id: str,
    config_hash: str,
    selected_indices: list[int],
    shards: dict[int, dict[str, Any]],
    sample_step: int,
) -> None:
    ordered = [shards[index] for index in sorted(shards)]
    manifest = {
        "schema_version": 1,
        "kind": "pdr_study_sharded",
        "source": source_name,
        "config_hash": config_hash,
        "algorithm_ids": [algorithm.resolved_algorithm_id for algorithm in algorithms],
        "teacher_algorithm_id": teacher_id,
        "sample_step": sample_step,
        "stores_all_algorithm_outputs": True,
        "selected_records": len(selected_indices),
        "completed_records": sum(len(item["record_ids"]) for item in ordered),
        "shards": ordered,
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)


def _aggregate_statistics(output_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    algorithm_totals: dict[str, dict[str, float]] = {}
    for stats_path in sorted(output_dir.glob("shard_*.jsonl")):
        with stats_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                records.append(record)
                for category in record["categories"]:
                    category_counts[category] = category_counts.get(category, 0) + 1
                for algorithm_id, stats in record["algorithms"].items():
                    total = algorithm_totals.setdefault(algorithm_id, {
                        "valid_windows": 0.0,
                        "forward_windows": 0.0,
                        "transitions": 0.0,
                    })
                    total["valid_windows"] += stats["valid_windows"]
                    total["forward_windows"] += stats["valid_windows"] * stats["forward_fraction"]
                    total["transitions"] += stats["transitions"]

    ranked = sorted(records, key=lambda item: item["interest_score"], reverse=True)
    csv_path = output_dir / "interesting_records.csv"
    algorithm_ids = list(algorithm_totals)
    algorithm_metric_names = (
        "valid_windows",
        "coverage_fraction",
        "forward_fraction",
        "transitions",
        "transitions_per_second",
        "short_run_fraction",
        "near_boundary_fraction",
        "mean_confidence",
    )
    base_fields = [
        "rank", "source", "record_id", "split", "file_name", "source_csv",
        "f_adc", "f_network", "spp", "voltage_basis", "duration_sec", "n_windows",
        "interest_score", "categories", "total_transitions", "vote_change_count",
        "disagreement_fraction", "localized_disagreement", "dynamic_disagreement",
        "static_threshold_disagreement", "low_coverage",
        "most_disagreeing_pair", "most_disagreeing_pair_fraction",
        "most_disagreeing_pair_transitions",
    ]
    algorithm_fields = [
        f"{algorithm_id}__{metric_name}"
        for algorithm_id in algorithm_ids
        for metric_name in algorithm_metric_names
    ]
    fieldnames = base_fields + algorithm_fields
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(ranked, start=1):
            pair = record.get("most_disagreeing_pair") or {}
            row = {key: record.get(key) for key in base_fields}
            row["rank"] = rank
            row["categories"] = "|".join(record["categories"])
            row["most_disagreeing_pair"] = (
                f"{pair.get('left')}|{pair.get('right')}" if pair else ""
            )
            row["most_disagreeing_pair_fraction"] = pair.get("disagreement_fraction")
            row["most_disagreeing_pair_transitions"] = pair.get("disagreement_transitions")
            for algorithm_id in algorithm_ids:
                stats = record["algorithms"].get(algorithm_id, {})
                for metric_name in algorithm_metric_names:
                    row[f"{algorithm_id}__{metric_name}"] = stats.get(metric_name)
            writer.writerow(row)

    def candidates(predicate: Any) -> list[dict[str, Any]]:
        return [_candidate_view(record) for record in ranked if predicate(record)]

    candidate_payload = {
        "selection_note": (
            "Списки не ограничены top-N. Полная числовая статистика всех записей "
            "находится в interesting_records.csv. overall_dynamic исключает записи, "
            "где единственная причина расхождения — постоянный сдвиг порога."
        ),
        "overall_dynamic": candidates(
            lambda record: (
                record["interest_score"] > 0.0
                and not record["static_threshold_disagreement"]
                and not record["low_coverage"]
            )
        ),
        "switching": candidates(lambda record: "switching" in record["categories"]),
        "localized_disagreement": candidates(
            lambda record: "localized_disagreement" in record["categories"]
        ),
        "chattering": candidates(lambda record: "chattering" in record["categories"]),
        "static_threshold_disagreement": candidates(
            lambda record: record["static_threshold_disagreement"]
        ),
        "low_coverage": candidates(lambda record: record["low_coverage"]),
    }
    _atomic_write_json(output_dir / "review_groups.json", candidate_payload)

    scores = np.asarray([record["interest_score"] for record in records], dtype=np.float64)
    for stats in algorithm_totals.values():
        valid = stats["valid_windows"]
        stats["forward_fraction"] = stats.pop("forward_windows") / max(valid, 1.0)
        stats["valid_windows"] = int(valid)
        stats["transitions"] = int(stats["transitions"])
    return {
        "records": len(records),
        "category_counts": category_counts,
        "interest_score_quantiles": {
            "p50": float(np.quantile(scores, 0.50)) if scores.size else 0.0,
            "p90": float(np.quantile(scores, 0.90)) if scores.size else 0.0,
            "p95": float(np.quantile(scores, 0.95)) if scores.size else 0.0,
            "p99": float(np.quantile(scores, 0.99)) if scores.size else 0.0,
        },
        "algorithms": algorithm_totals,
        "ranked_csv": csv_path.name,
        "review_groups": "review_groups.json",
    }


def _candidate_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "source", "record_id", "split", "file_name", "source_csv", "interest_score",
            "categories", "total_transitions", "disagreement_fraction",
            "localized_disagreement", "dynamic_disagreement", "most_disagreeing_pair",
        )
    }


def _create_source(source_name: str) -> DatasetSource:
    if source_name == "open_ee":
        return OpenEEShardedSource(PROJECT_ROOT / "data/phase5/open_ee_shards/manifest.json")
    if source_name == "french_rte":
        return FrenchRTESource(PROJECT_ROOT / "data/phase5/french_rte/DATA_S.npy")
    raise KeyError(f"Неизвестный источник Phase 5: {source_name}")


def _select_indices(
    split_manifest: dict[str, Any],
    source_name: str,
    split_scope: str,
    max_records: int | None,
) -> tuple[list[int], dict[int, str]]:
    splits = split_manifest["sources"][source_name]["splits"]
    lookup = {
        int(record_id): split_name
        for split_name, record_ids in splits.items()
        for record_id in record_ids
    }
    if split_scope == "all":
        selected = sorted(lookup)
    elif split_scope in splits:
        selected = [int(value) for value in splits[split_scope]]
    else:
        raise KeyError(f"Неизвестный split scope: {split_scope}")
    if max_records is not None:
        selected = selected[:max_records]
    return selected, lookup


def _record_timebase(metadata: dict[str, Any]) -> TimebaseContract:
    sampling_rate = metadata.get("f_adc", metadata.get("sampling_rate_hz"))
    network_frequency = metadata.get("f_network", metadata.get("network_frequency_hz"))
    if sampling_rate is None or network_frequency is None:
        raise ValueError("В metadata записи отсутствуют f_adc/f_network")
    return TimebaseContract.create(float(sampling_rate), float(network_frequency))


def _estimate_windows(source: DatasetSource, indices: Sequence[int], sample_step: int) -> int:
    total = 0
    default_samples = int(getattr(getattr(source, "data", None), "shape", (0, 0, 0))[-1])
    for record_id in indices:
        metadata = source.get_metadata(record_id)
        spp = int(metadata.get("spp") or round(float(metadata["f_adc"]) / float(metadata["f_network"])))
        n_samples = int(metadata.get("n_samples", default_samples))
        if n_samples >= spp:
            total += (n_samples - spp) // sample_step + 1
    return total


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="По 8 записей каждого источника")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/phase5/pdr_labels_v1")
    parser.add_argument("--sources", nargs="+", choices=("open_ee", "french_rte"), default=["open_ee", "french_rte"])
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--split", choices=("all", "train", "validation", "holdout"), default="all")
    parser.add_argument("--sample-step", type=int, default=1, help="Шаг решений в исходных отсчётах")
    parser.add_argument("--records-per-shard", type=int, default=64)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--no-compression", action="store_true", help="Отключить lossless ZIP-сжатие shards")
    args = parser.parse_args()
    max_records = 8 if args.smoke else args.max_records_per_source
    run_study(
        output_dir=args.output_dir,
        source_names=args.sources,
        algorithm_ids=args.algorithms,
        teacher_id=args.teacher,
        split_scope=args.split,
        sample_step=args.sample_step,
        records_per_shard=(4 if args.smoke else args.records_per_shard),
        max_records_per_source=max_records,
        lossless_compression=not args.no_compression,
    )
    return 0


def run_manual() -> None:
    # Сначала обязательно выполнить SMOKE=True. После проверки артефактов заменить на False.
    SMOKE = True
    OUTPUT_DIR = PROJECT_ROOT / "data/phase5/pdr_labels_v1_smoke" if SMOKE else PROJECT_ROOT / "data/phase5/pdr_labels_v1"
    SOURCES = ("open_ee", "french_rte")
    ALGORITHMS = DEFAULT_ALGORITHMS
    TEACHER = DEFAULT_TEACHER
    SPLIT_SCOPE = "all"           # Размечаем всё, но исходный train/validation/holdout сохраняется в статистике.
    SAMPLE_STEP = 1                # Решение каждого РНМ для каждого исходного отсчёта.
    RECORDS_PER_SHARD = 64         # RAM/checkpoint-компромисс; не меняет физическую разметку.
    LOSSLESS_COMPRESSION = True    # Только упаковка без потери точности значений.

    run_study(
        output_dir=OUTPUT_DIR,
        source_names=SOURCES,
        algorithm_ids=ALGORITHMS,
        teacher_id=TEACHER,
        split_scope=SPLIT_SCOPE,
        sample_step=SAMPLE_STEP,
        records_per_shard=(4 if SMOKE else RECORDS_PER_SHARD),
        max_records_per_source=(8 if SMOKE else None),
        lossless_compression=LOSSLESS_COMPRESSION,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
