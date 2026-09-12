"""Отобрать разнообразные случаи фазы 5 и экспортировать полные записи в COMTRADE.

Отбор воспроизводим: каждый экспортированный случай получает сопроводительный
JSON-файл. Последующие запуски рекурсивно проверяют весь каталог ручного обзора,
поэтому записи остаются исключёнными даже после сортировки по произвольным
подпапкам.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.io.comtrade_ascii import (
    AnalogChannel,
    DigitalChannel,
    ExportRecord,
    write_comtrade_ascii,
)
from osc_tools.ml.phase5_sources import FrenchRTESource, OpenEEShardedSource
from osc_tools.pdr.base import PDRDirection
from scripts.phase5_experiments.progress import ProgressReporter


ALGORITHMS = (
    "adaptive_pdr_mir",
    "phase_pdr_basic",
    "pos_seq_pdr_basic",
    "phase_power_pdr_basic",
    "pos_seq_power_pdr_basic",
    "pdr_sivokobylenko_2pt",
    "pdr_sivokobylenko_5pt",
    "pdr_bmrz_q_assisted",
    "pdr_bavr072_crosspol",
)
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "data/phase5/pdr_analysis_v6"
DEFAULT_LABEL_DIR = PROJECT_ROOT / "data/phase5/pdr_labels_v6"
DEFAULT_REVIEW_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_review_v6"
DEFAULT_MANUAL_LABEL_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_labels"
DEFAULT_BLIND_AUDIT_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_blind_audit"
DEFAULT_STANDARD_NEW_ROOT = (
    DEFAULT_REVIEW_ROOT / "batch_20260827_standard_diverse_takt3"
)
DEFAULT_NORM_CSV = PROJECT_ROOT / "data/norm_coef_all_v1.4.csv"
DEFAULT_QUOTAS = {
    "persistent_disagreement": 10,
    "phase_vs_sequence": 10,
    "adaptive_1_2": 10,
    "adaptive_3_10": 10,
    "adaptive_11_30": 10,
    "adaptive_31_plus": 5,
    "low_current_threshold_region": 10,
    "high_current": 10,
    "multivariate_anomaly": 10,
    "blind_control": 10,
}
DEFAULT_SOURCE_QUOTA_MULTIPLIERS = {
    # Open_EE ближе к целевой задаче БАВР. French/RTE сохраняется как внешний
    # домен и источник нетипичных режимов, но не занимает половину новой сессии.
    "open_ee": 1.0,
    "french_rte": 0.5,
}


@dataclass(frozen=True)
class SelectedCase:
    source: str
    record_id: int
    stratum: str
    rank: int
    selection_score: float
    reasons: tuple[str, ...]
    row: dict[str, Any]


class MultiAlgorithmLabelArchive:
    """Ленивое чтение результатов всех алгоритмов из одного архива РНМ."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.algorithm_ids = tuple(manifest["algorithm_ids"])
        if not manifest.get("stores_all_algorithm_outputs"):
            raise RuntimeError("Для экспорта COMTRADE нужны результаты всех алгоритмов")
        self._map: dict[int, tuple[Path, int]] = {}
        for shard in manifest["shards"]:
            path = self.root / str(shard["file"])
            for local_index, record_id in enumerate(shard["record_ids"]):
                self._map[int(record_id)] = (path, local_index)
        self._cached_path: Path | None = None
        self._cached: dict[str, np.ndarray] | None = None

    def get(self, record_id: int) -> dict[str, np.ndarray]:
        path, local_index = self._map[int(record_id)]
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as archive:
                self._cached = {name: archive[name] for name in archive.files}
            self._cached_path = path
        assert self._cached is not None
        offsets = self._cached["offsets"]
        start, stop = int(offsets[local_index]), int(offsets[local_index + 1])
        return {
            "samples": np.asarray(self._cached["samples"][start:stop]),
            "directions": np.asarray(self._cached["directions"][:, start:stop]),
            "warmup": np.asarray(self._cached["all_warmup"][:, start:stop]),
        }


class PhysicalScaleResolver:
    """Обратить документированную нормировку фазы 5 без угадывания базисов."""

    def __init__(self, norm_csv: Path) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        with Path(norm_csv).open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                self.rows[str(row["name"])] = row

    def scales(self, source: str, metadata: dict[str, Any]) -> tuple[list[float], list[float]]:
        if source == "french_rte":
            return [300.0 * 20.0] * 3, [90_000.0 * 3.0] * 3
        match = re.fullmatch(r"([0-9a-fA-F]+)_Bus (\d+)", str(metadata.get("file_name", "")))
        if not match:
            raise ValueError(f"Не удалось извлечь hash/Bus из {metadata.get('file_name')!r}")
        file_hash, bus = match.groups()
        row = self.rows.get(file_hash)
        if row is None or "YES" not in str(row.get("norm", "")):
            raise ValueError(f"Нет разрешённого профиля нормировки для {file_hash}")
        current = _positive_float(row.get(f"{bus}Ip_base"), f"{bus}Ip_base") * 20.0
        columns = list(metadata.get("source_columns", ()))
        voltage_scales: list[float] = []
        for column in columns[4:7]:
            suffix = "Ub_base" if " BB" in str(column) else "Uc_base" if " CL" in str(column) else None
            if suffix is None:
                raise ValueError(f"Неясный базис напряжения {column!r}")
            voltage_scales.append(_positive_float(row.get(f"{bus}{suffix}"), f"{bus}{suffix}") * 3.0)
        if len(voltage_scales) != 3:
            raise ValueError("В metadata нет трёх voltage source_columns")
        return [current] * 3, voltage_scales


def select_review_cases(
    analysis_dir: Path,
    review_root: Path,
    quotas: dict[str, int],
    sources: Sequence[str] = ("open_ee", "french_rte"),
    additional_exclusion_roots: Sequence[Path] = (),
    source_quota_multipliers: dict[str, float] | None = None,
    source_quotas: dict[str, dict[str, int]] | None = None,
    include_only_keys: set[tuple[str, int]] | None = None,
) -> tuple[list[SelectedCase], list[dict[str, Any]]]:
    """Построить полный каталог и разнообразный пакет без повторов записей."""

    records = _read_keyed_csv(Path(analysis_dir) / "record_statistics.csv")
    eligibility = _read_keyed_csv(Path(analysis_dir) / "record_signal_eligibility.csv")
    signals = _read_keyed_csv(Path(analysis_dir) / "signal_record_statistics.csv")
    anomalies = _read_keyed_csv(Path(analysis_dir) / "research_anomaly_candidates.csv")
    temporal = _read_keyed_csv(
        Path(analysis_dir) / "teacher_temporal_statistics.csv",
        predicate=lambda row: row.get("algorithm_id") == "adaptive_pdr_mir",
    )
    excluded = discover_previously_exported(review_root)
    for root in additional_exclusion_roots:
        excluded.update(discover_previously_exported(root))
    multipliers = source_quota_multipliers or {source: 1.0 for source in sources}
    rows: list[dict[str, Any]] = []
    for key, record in records.items():
        if key[0] not in sources or _as_bool(eligibility.get(key, {}).get("pdr_structurally_eligible")) is not True:
            continue
        if include_only_keys is not None and key not in include_only_keys:
            continue
        merged = dict(record)
        merged.update({f"signal__{k}": v for k, v in signals.get(key, {}).items()})
        merged.update({f"temporal__{k}": v for k, v in temporal.get(key, {}).items()})
        merged.update({f"anomaly__{k}": v for k, v in anomalies.get(key, {}).items()})
        merged["already_exported"] = key in excluded
        rows.append(merged)

    catalog: list[dict[str, Any]] = []
    memberships: dict[tuple[str, int], list[str]] = {}
    candidates_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["source"]), int(row["record_id"]))
        strata = _strata(row)
        memberships[key] = strata
        for stratum in strata:
            score = _score(row, stratum)
            catalog.append({
                "source": key[0], "record_id": key[1], "stratum": stratum,
                "selection_score": score, "already_exported": row["already_exported"],
                "file_name": row.get("file_name", ""), "input_sha256": row.get("input_sha256", ""),
                "adaptive_transitions": row.get("adaptive_pdr_mir__transitions", ""),
                "disagreement_fraction": row.get("disagreement_fraction", ""),
                "current_rms_physical_pu": row.get("signal__current_rms", ""),
                "anomaly_score": row.get("anomaly__anomaly_score", ""),
            })
            if not row["already_exported"]:
                candidates_by_group.setdefault((key[0], stratum), []).append(row)

    selected: list[SelectedCase] = []
    used = set(excluded)
    for source in sources:
        for stratum, quota in quotas.items():
            if source_quotas is not None:
                source_quota = max(0, int(source_quotas.get(source, {}).get(stratum, 0)))
            else:
                source_quota = int(math.ceil(max(0.0, float(multipliers.get(source, 1.0))) * quota))
            pool = [row for row in candidates_by_group.get((source, stratum), ())
                    if (source, int(row["record_id"])) not in used]
            chosen = _diverse_subset(pool, source_quota, stratum)
            for rank, row in enumerate(chosen, start=1):
                key = (source, int(row["record_id"]))
                used.add(key)
                selected.append(SelectedCase(
                    source, key[1], stratum, rank, _score(row, stratum),
                    tuple(memberships[key]), row,
                ))
    return selected, catalog


def export_batch(
    selected: Sequence[SelectedCase],
    *,
    label_dir: Path,
    review_root: Path,
    norm_csv: Path,
    batch_name: str | None = None,
) -> Path:
    batch_name = batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_dir = Path(review_root) / batch_name
    if batch_dir.exists():
        raise FileExistsError(f"Пакет уже существует: {batch_dir}")
    batch_dir.mkdir(parents=True)
    resolver = PhysicalScaleResolver(norm_csv)
    source_objects = {
        "open_ee": OpenEEShardedSource(PROJECT_ROOT / "data/phase5/open_ee_shards/manifest.json"),
        "french_rte": FrenchRTESource(PROJECT_ROOT / "data/phase5/french_rte/DATA_S.npy"),
    }
    archives = {name: MultiAlgorithmLabelArchive(Path(label_dir) / name) for name in source_objects}
    for name, archive in archives.items():
        if archive.algorithm_ids != ALGORITHMS:
            raise RuntimeError(
                f"Порядок алгоритмов {name} отличается от экспортного контракта: "
                f"{archive.algorithm_ids!r}"
            )
    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    progress = ProgressReporter("COMTRADE manual-review export", max(1, len(selected)), unit="файл")
    try:
        for index, case in enumerate(selected, start=1):
            try:
                source = source_objects[case.source]
                metadata = source.get_metadata(case.record_id)
                signal = np.asarray(source.load_signal(case.record_id), dtype=np.float64)
                labels = archives[case.source].get(case.record_id)
                record, scaling = _build_export_record(case, signal, metadata, labels, resolver)
                case_dir = batch_dir / case.stratum / case.source
                stem = f"{case.source}__record_{case.record_id:05d}"
                cfg_path, dat_path = case_dir / f"{stem}.cfg", case_dir / f"{stem}.dat"
                write_comtrade_ascii(record, cfg_path, dat_path)
                sidecar = {
                    "kind": "phase5_pdr_manual_review_case", "schema_version": 1,
                    "source": case.source, "record_id": case.record_id,
                    "input_sha256": case.row.get("input_sha256"), "file_name": metadata.get("file_name"),
                    "stratum": case.stratum, "all_selection_reasons": list(case.reasons),
                    "selection_rank": case.rank, "selection_score": case.selection_score,
                    "full_record": True, "analog_units": "physical_A_V",
                    "physical_scaling": scaling, "voltage_basis": metadata.get("voltage_basis", "phase"),
                    "algorithm_ids": list(archives[case.source].algorithm_ids),
                    "expert_channels": {"direction": 0, "applicable": 0},
                    "cfg_sha256": _sha256(cfg_path), "dat_sha256": _sha256(dat_path),
                    "cfg": cfg_path.name, "dat": dat_path.name,
                }
                sidecar_path = case_dir / f"{stem}.json"
                _atomic_json(sidecar_path, sidecar)
                manifest_rows.append({
                    "source": case.source, "record_id": case.record_id, "stratum": case.stratum,
                    "rank": case.rank, "selection_score": case.selection_score,
                    "all_reasons": "|".join(case.reasons),
                    "cfg": str(cfg_path.relative_to(batch_dir)), "dat": str(dat_path.relative_to(batch_dir)),
                    "sidecar": str(sidecar_path.relative_to(batch_dir)),
                })
                _write_csv(batch_dir / "batch_manifest.csv", manifest_rows)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                failure = {
                    "source": case.source,
                    "record_id": case.record_id,
                    "stratum": case.stratum,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failure_rows.append(failure)
                failure_path = batch_dir / "export_failures" / f"{case.source}__record_{case.record_id:05d}.json"
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_json(failure_path, {
                    "kind": "phase5_pdr_manual_review_export_failure",
                    "schema_version": 1,
                    **failure,
                })
                _write_csv(batch_dir / "export_failures.csv", failure_rows)
                print(f"\nПропущен {case.source}:{case.record_id}: {exc}")
            finally:
                progress.update(index)
    finally:
        for source in source_objects.values():
            close = getattr(source, "close", None)
            if callable(close):
                close()
    progress.finish()
    _write_csv(batch_dir / "batch_manifest.csv", manifest_rows)
    _write_csv(batch_dir / "export_failures.csv", failure_rows)
    _atomic_json(batch_dir / "batch_summary.json", {
        "kind": "phase5_pdr_manual_review_batch", "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "selected_records": len(selected),
        "exported_records": len(manifest_rows),
        "failed_records": len(failure_rows),
        "label_dir": str(Path(label_dir)), "selection_is_without_replacement": True,
    })
    return batch_dir


def discover_previously_exported(review_root: Path) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    root = Path(review_root)
    if not root.exists():
        return result
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("kind") in (
            "phase5_pdr_manual_review_case",
            "phase5_pdr_manual_review_export_failure",
        ):
            result.add((str(payload["source"]), int(payload["record_id"])))
    pattern = re.compile(r"(open_ee|french_rte)__record_(\d+)\.cfg$", re.IGNORECASE)
    for path in root.rglob("*.cfg"):
        match = pattern.fullmatch(path.name)
        if match:
            result.add((match.group(1).lower(), int(match.group(2))))
    # Манифесты сохраняют историю отбора даже после переноса самих комплектов
    # CFG/DAT/JSON в рабочее дерево ручной разметки.
    for pattern in ("batch_manifest.csv", "blind_audit_assignment*.csv"):
        for path in root.rglob(pattern):
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("source") and row.get("record_id"):
                            result.add((str(row["source"]), int(row["record_id"])))
            except (OSError, UnicodeDecodeError, csv.Error, ValueError):
                continue
    return result


def recover_incomplete_batches(review_root: Path) -> list[Path]:
    """Восстановить манифесты пакетов, прерванных после записи отдельных пар."""

    recovered: list[Path] = []
    for batch_dir in sorted(Path(review_root).glob("batch_*")):
        if not batch_dir.is_dir() or (batch_dir / "batch_summary.json").exists():
            continue
        manifest_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        for path in sorted(batch_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("kind") == "phase5_pdr_manual_review_case":
                cfg = path.with_name(str(payload["cfg"]))
                dat = path.with_name(str(payload["dat"]))
                if not cfg.exists() or not dat.exists():
                    continue
                manifest_rows.append({
                    "source": payload["source"], "record_id": payload["record_id"],
                    "stratum": payload.get("stratum", path.parent.parent.name),
                    "rank": payload.get("selection_rank", ""),
                    "selection_score": payload.get("selection_score", ""),
                    "all_reasons": "|".join(payload.get("all_selection_reasons", ())),
                    "cfg": str(cfg.relative_to(batch_dir)),
                    "dat": str(dat.relative_to(batch_dir)),
                    "sidecar": str(path.relative_to(batch_dir)),
                })
            elif payload.get("kind") == "phase5_pdr_manual_review_export_failure":
                failure_rows.append({
                    key: payload.get(key, "")
                    for key in ("source", "record_id", "stratum", "error_type", "error")
                })
        if not manifest_rows and not failure_rows:
            continue
        _write_csv(batch_dir / "batch_manifest.csv", manifest_rows)
        _write_csv(batch_dir / "export_failures.csv", failure_rows)
        _atomic_json(batch_dir / "batch_summary.json", {
            "kind": "phase5_pdr_manual_review_batch",
            "schema_version": 1,
            "recovered_at": datetime.now().isoformat(),
            "recovered_after_interruption": True,
            "exported_records": len(manifest_rows),
            "failed_records": len(failure_rows),
            "selection_is_without_replacement": True,
        })
        recovered.append(batch_dir)
    return recovered


def _build_export_record(case, signal, metadata, labels, resolver):
    current, current_valid = _complete_three(signal[:3])
    voltage, voltage_valid = _complete_three(signal[4:7])
    current_scales, voltage_scales = resolver.scales(case.source, metadata)
    current = current * np.asarray(current_scales)[:, None]
    voltage = voltage * np.asarray(voltage_scales)[:, None]
    basis = str(metadata.get("voltage_basis", "phase"))
    voltage_names = ("UA", "UB", "UC") if basis == "phase" else ("UAB", "UBC", "UCA")
    analog = tuple(
        AnalogChannel(name, "A", values, phase=name[-1])
        for name, values in zip(("IA", "IB", "IC"), current)
    ) + tuple(
        AnalogChannel(name, "V", values, phase=name.replace("U", ""))
        for name, values in zip(voltage_names, voltage)
    )
    n_samples = signal.shape[1]
    digital: list[DigitalChannel] = []
    samples = labels["samples"].astype(np.int64)
    if samples.size and (np.any(np.diff(samples) <= 0) or samples[0] < 0 or samples[-1] >= n_samples):
        raise ValueError(f"Некорректные sample_indices: {case.source}:{case.record_id}")
    for alg_index, algorithm_id in enumerate(ALGORITHMS):
        direction = np.full(n_samples, int(PDRDirection.UNLABELED), dtype=np.int16)
        direction[samples] = labels["directions"][alg_index]
        if not np.isin(direction, (int(PDRDirection.UNLABELED), 0, 1)).all():
            raise ValueError(f"Недопустимая метка {algorithm_id}")
        digital.append(DigitalChannel(f"{algorithm_id}__FWD", (direction == 1).astype(np.uint8)))
        digital.append(DigitalChannel(f"{algorithm_id}__VALID", (direction != int(PDRDirection.UNLABELED)).astype(np.uint8)))
    digital.extend((
        DigitalChannel("expert__FWD", np.zeros(n_samples, dtype=np.uint8)),
        DigitalChannel("expert__VALID", np.zeros(n_samples, dtype=np.uint8)),
    ))
    for name, valid in zip(("IABC", "UABC"), (current_valid, voltage_valid)):
        if not valid.all():
            digital.append(DigitalChannel(f"{name}__VALID", valid.astype(np.uint8)))
    f_adc = float(metadata["f_adc"])
    start = datetime(2000, 1, 1)
    record = ExportRecord(
        station_name=f"Phase5_{case.source}", recorder_id=f"record_{case.record_id}",
        sample_rate_hz=f_adc, network_frequency_hz=float(metadata.get("f_network", 50.0)),
        start_datetime=start, trigger_datetime=start, analog=analog, digital=tuple(digital),
    )
    scale_metadata = {
        "formula": "physical = stored_phase5_value * channel_scale",
        "current_scales_a": current_scales, "voltage_scales_v": voltage_scales,
        "absolute_time_is_synthetic": True,
    }
    return record, scale_metadata


def _complete_three(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(result)
    valid = finite.sum(axis=0) >= 2
    for missing in range(3):
        mask = valid & ~finite[missing]
        others = [index for index in range(3) if index != missing]
        result[missing, mask] = -(result[others[0], mask] + result[others[1], mask])
    result[:, ~valid] = 0.0
    if not np.isfinite(result).all():
        raise ValueError("Не удалось однозначно восстановить три аналоговых сигнала")
    return result, valid


def _strata(row: dict[str, Any]) -> list[str]:
    transitions = int(_number(row.get("adaptive_pdr_mir__transitions"), 0))
    disagreement = _number(row.get("disagreement_fraction"), 0.0)
    phase_gap = max(
        abs(_number(row.get("phase_pdr_basic__forward_fraction"), 0.0) - _number(row.get("pos_seq_pdr_basic__forward_fraction"), 0.0)),
        abs(_number(row.get("phase_power_pdr_basic__forward_fraction"), 0.0) - _number(row.get("pos_seq_power_pdr_basic__forward_fraction"), 0.0)),
    )
    current = _number(row.get("signal__current_rms"), math.nan) * 20.0 / math.sqrt(2.0)
    result = []
    if disagreement >= 0.5 and transitions <= 2: result.append("persistent_disagreement")
    if phase_gap >= 0.20: result.append("phase_vs_sequence")
    if 1 <= transitions <= 2: result.append("adaptive_1_2")
    if 3 <= transitions <= 10: result.append("adaptive_3_10")
    if 11 <= transitions <= 30: result.append("adaptive_11_30")
    if transitions >= 31: result.append("adaptive_31_plus")
    if math.isfinite(current) and 0.01 <= current < 0.05: result.append("low_current_threshold_region")
    if math.isfinite(current) and current >= 2.0: result.append("high_current")
    if row.get("anomaly__anomaly_score") not in (None, ""): result.append("multivariate_anomaly")
    result.append("blind_control")
    return result


def _score(row: dict[str, Any], stratum: str) -> float:
    transitions = _number(row.get("adaptive_pdr_mir__transitions"), 0.0)
    disagreement = _number(row.get("disagreement_fraction"), 0.0)
    anomaly = _number(row.get("anomaly__anomaly_score"), 0.0)
    entropy = _number(row.get("temporal__state_entropy_bits"), 0.0)
    if stratum == "persistent_disagreement": return disagreement - min(transitions, 2) * 0.01
    if stratum == "multivariate_anomaly": return anomaly
    if stratum == "blind_control":
        token = f"{row.get('source')}:{row.get('record_id')}:{row.get('input_sha256')}".encode()
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return math.log1p(max(transitions, 0.0)) + disagreement + entropy


def _diverse_subset(rows: Sequence[dict[str, Any]], count: int, stratum: str) -> list[dict[str, Any]]:
    if count <= 0 or not rows: return []
    ordered = sorted(rows, key=lambda row: _score(row, stratum), reverse=True)
    if len(ordered) <= count: return ordered
    features = np.asarray([[
        math.log1p(max(_number(row.get("signal__current_rms"), 0.0), 0.0)),
        math.log1p(max(_number(row.get("signal__voltage_rms"), 0.0), 0.0)),
        _number(row.get("signal__current_phase_unbalance_cv"), 0.0),
        _number(row.get("signal__voltage_phase_unbalance_cv"), 0.0),
        _number(row.get("disagreement_fraction"), 0.0),
        math.log1p(max(_number(row.get("adaptive_pdr_mir__transitions"), 0.0), 0.0)),
        _number(row.get("temporal__state_entropy_bits"), 0.0),
    ] for row in ordered], dtype=np.float64)
    features = np.nan_to_num(features)
    low, high = np.quantile(features, (0.1, 0.9), axis=0)
    scaled = np.clip((features - low) / np.maximum(high - low, 1e-12), 0, 1)
    chosen = [0]
    distance = np.linalg.norm(scaled - scaled[0], axis=1)
    while len(chosen) < count:
        distance[chosen] = -1
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(scaled - scaled[nxt], axis=1))
    return [ordered[index] for index in chosen]


def _read_keyed_csv(path: Path, predicate=None) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if predicate is None or predicate(row):
                result[(str(row["source"]), int(row["record_id"]))] = row
    return result


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0: raise ValueError(f"Некорректный {name}: {value!r}")
    return number


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool | None:
    if str(value).lower() == "true": return True
    if str(value).lower() == "false": return False
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _selection_profile(
    profile: str,
    count: int,
) -> tuple[dict[str, int], dict[str, float], dict[str, dict[str, int]] | None]:
    """Вернуть воспроизводимые квоты обычного или слепого пакета."""

    if profile != "standard":
        raise ValueError(f"Неизвестный обычный профиль отбора: {profile!r}")
    return dict(DEFAULT_QUOTAS), dict(DEFAULT_SOURCE_QUOTA_MULTIPLIERS), None


def select_blind_audit_cases(
    analysis_dir: Path,
    review_root: Path,
    new_cases_root: Path,
    count: int,
) -> tuple[list[SelectedCase], list[dict[str, Any]]]:
    """Случайно выбрать без повторов из объединения старых и новых записей."""

    if count <= 0:
        raise ValueError("Размер контрольного пакета должен быть положительным")
    old_keys = discover_previously_exported(DEFAULT_MANUAL_LABEL_ROOT / "completed")
    new_keys = discover_previously_exported(new_cases_root)
    overlap = old_keys & new_keys
    if overlap:
        raise ValueError(f"Новые и прежние completed пересекаются: {sorted(overlap)[:5]}")
    origins = {key: "previously_labeled" for key in old_keys}
    origins.update({key: "new_standard" for key in new_keys})
    pool_keys = set(origins)
    if count > len(pool_keys):
        raise ValueError(f"Запрошено {count}, но в объединённом пуле только {len(pool_keys)} записей")

    source_quotas = {
        source: {"blind_control": sum(key[0] == source for key in pool_keys)}
        for source in ("open_ee", "french_rte")
    }
    quotas = {name: 0 for name in DEFAULT_QUOTAS}
    quotas["blind_control"] = len(pool_keys)
    audit_history_roots = tuple(
        root for root in DEFAULT_BLIND_AUDIT_ROOT.parent.glob("pdr_manual_blind_audit_v*")
        if root.resolve() != Path(review_root).parent.resolve()
    )
    all_cases, catalog = select_review_cases(
        analysis_dir,
        review_root,
        quotas,
        additional_exclusion_roots=audit_history_roots,
        source_quota_multipliers={"open_ee": 1.0, "french_rte": 1.0},
        source_quotas=source_quotas,
        include_only_keys=pool_keys,
    )
    # blind_control score является детерминированным SHA-256 псевдослучайным
    # числом. Сортировка по нему даёт воспроизводимую простую случайную
    # подвыборку, не зависящую от сложности, источника или прежней метки.
    chosen = sorted(
        all_cases,
        key=lambda case: (case.selection_score, case.source, case.record_id),
        reverse=True,
    )[:count]
    selected: list[SelectedCase] = []
    for rank, case in enumerate(chosen, start=1):
        key = (case.source, case.record_id)
        selected.append(SelectedCase(
            source=case.source,
            record_id=case.record_id,
            stratum="blind_control",
            rank=rank,
            selection_score=case.selection_score,
            reasons=case.reasons,
            row={**case.row, "blind_audit_origin": origins[key]},
        ))
    return selected, catalog


def run_export(
    *,
    profile: str,
    count: int = 50,
    export_comtrade: bool,
    batch_name: str | None = None,
    audit_new_root: Path = DEFAULT_STANDARD_NEW_ROOT,
    blind_audit_root: Path = DEFAULT_BLIND_AUDIT_ROOT,
) -> Path | None:
    """Построить предпросмотр и при явном запросе экспортировать COMTRADE."""

    analysis_dir = DEFAULT_ANALYSIS_DIR
    label_dir = DEFAULT_LABEL_DIR
    review_root = Path(blind_audit_root) / "reference" if profile == "audit" else DEFAULT_REVIEW_ROOT

    recovered = recover_incomplete_batches(review_root)
    if recovered:
        print("Восстановлены прерванные пакеты: " + ", ".join(path.name for path in recovered))
    if profile == "audit":
        selected, catalog = select_blind_audit_cases(
            analysis_dir, review_root, audit_new_root, count
        )
    else:
        quotas, multipliers, source_quotas = _selection_profile(profile, count)
        selected, catalog = select_review_cases(
            analysis_dir,
            review_root,
            quotas,
            additional_exclusion_roots=(DEFAULT_MANUAL_LABEL_ROOT,),
            source_quota_multipliers=multipliers,
            source_quotas=source_quotas,
        )
    review_root.mkdir(parents=True, exist_ok=True)
    _write_csv(review_root / "candidate_catalog.csv", catalog)
    _write_csv(review_root / "selection_preview.csv", [{
        "source": case.source, "record_id": case.record_id, "stratum": case.stratum,
        "rank": case.rank, "selection_score": case.selection_score,
        "all_reasons": "|".join(case.reasons), "file_name": case.row.get("file_name", ""),
    } for case in selected])
    if profile == "audit" and export_comtrade:
        batch_name = batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
        assignment_suffix = re.sub(r"[^0-9A-Za-z_-]+", "_", batch_name)
        _write_csv(review_root / "_service" / f"blind_audit_assignment_{assignment_suffix}.csv", [{
            "source": case.source,
            "record_id": case.record_id,
            "origin": case.row["blind_audit_origin"],
        } for case in selected])
    print(f"Выбрано новых записей: {len(selected)}; полный каталог: {len(catalog)} строк")
    estimated_samples = sum(
        int(_number(case.row.get("signal__samples"), 0.0)) for case in selected
    )
    # Размер ASCII зависит от длины чисел. Оценка 180 байт/отсчёт намеренно
    # приблизительна: 6 аналоговых, 12 дискретных каналов и разделители CSV.
    estimated_gib = estimated_samples * 180.0 / (1024.0 ** 3)
    print(f"Грубая оценка объёма выбранных ASCII DAT: {estimated_gib:.2f} ГиБ")
    if export_comtrade:
        output = export_batch(selected, label_dir=label_dir, review_root=review_root,
                              norm_csv=DEFAULT_NORM_CSV, batch_name=batch_name)
        print(f"COMTRADE-пакет: {output}")
        return output
    else:
        print("Предпросмотр готов. Для экспорта установите EXPORT_COMTRADE=True.")
        return None


def run_manual() -> None:
    """Ручной запуск F5: сначала предпросмотр, затем явное включение экспорта."""

    PROFILE = "standard"  # "standard" либо "audit"
    AUDIT_COUNT = 100
    EXPORT_COMTRADE = False
    BATCH_NAME: str | None = None
    run_export(
        profile=PROFILE,
        count=AUDIT_COUNT,
        export_comtrade=EXPORT_COMTRADE,
        batch_name=BATCH_NAME,
        blind_audit_root=DEFAULT_BLIND_AUDIT_ROOT,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("standard", "audit"),
        default="standard",
    )
    parser.add_argument("--count", type=int, default=100,
                        help="Точное общее число записей для профиля audit")
    parser.add_argument("--export", action="store_true",
                        help="Создать CFG/DAT; без флага строится только предпросмотр")
    parser.add_argument("--batch-name", default=None,
                        help="Необязательное уникальное имя каталога пакета")
    parser.add_argument("--audit-new-root", type=Path, default=DEFAULT_STANDARD_NEW_ROOT,
                        help="Стандартный пакет новых записей для профиля audit")
    parser.add_argument("--blind-audit-root", type=Path, default=DEFAULT_BLIND_AUDIT_ROOT,
                        help="Корень независимого раунда слепого аудита")
    args = parser.parse_args()
    run_export(
        profile=args.profile,
        count=args.count,
        export_comtrade=args.export,
        batch_name=args.batch_name,
        audit_new_root=args.audit_new_root,
        blind_audit_root=args.blind_audit_root,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_manual()
    else:
        _main()
