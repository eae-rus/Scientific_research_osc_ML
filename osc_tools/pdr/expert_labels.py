"""Импорт и аудит экспертной разметки РНМ из COMTRADE 1999 ASCII.

Модуль намеренно читает только два эталонных экспертных дискрета
``expert__FWD``/``expert__VALID``. Дополнительные каналы, созданные во внешней
программе для диагностики, регистрируются в журнале, но не используются ни как
признаки, ни как метки.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .base import PDRDirection


EXPERT_FWD = "expert__FWD"
EXPERT_VALID = "expert__VALID"
ALGORITHM_IDS = (
    "adaptive_pdr_mir",
    "phase_pdr_basic",
    "pos_seq_pdr_basic",
    "phase_power_pdr_basic",
    "pos_seq_power_pdr_basic",
)
TRAINABLE_STATUSES = ("completed", "invalid_quality")
KNOWN_STATUSES = ("completed", "ambiguous", "invalid_quality", "deferred", "in_progress")


@dataclass(frozen=True)
class ParsedAsciiComtrade:
    analog: Mapping[str, np.ndarray]
    digital: Mapping[str, np.ndarray]
    sample_numbers: np.ndarray
    timestamps_us: np.ndarray
    sample_rate_hz: float
    analog_names: tuple[str, ...]
    digital_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.sample_numbers.size)


@dataclass(frozen=True)
class ImportedExpertRecord:
    source: str
    record_id: int
    status: str
    stratum: str
    input_sha256: str
    f_adc: float
    directions: np.ndarray
    applicable: np.ndarray
    train_mask: np.ndarray
    transition_eval_mask: np.ndarray
    automatic_directions: Mapping[str, np.ndarray]
    ignored_analog_channels: tuple[str, ...]
    ignored_digital_channels: tuple[str, ...]
    cfg_path: Path


def read_comtrade_1999_ascii(cfg_path: Path, dat_path: Path | None = None) -> ParsedAsciiComtrade:
    """Строго прочитать одночастотный COMTRADE 1999 ASCII.

    Для проверки неизменности сигналов возвращаются именно числовые поля DAT.
    Коэффициенты ``a``/``b`` из CFG не применяются: экспорт фазы 5 записывает
    обязательные IA..UC с ``a=1, b=0``, а внешняя программа может добавлять
    расчётные аналоговые каналы с собственным масштабом.
    """

    cfg_path = Path(cfg_path)
    dat_path = Path(dat_path) if dat_path is not None else cfg_path.with_suffix(".dat")
    lines = _read_text(cfg_path).splitlines()
    if len(lines) < 9:
        raise ValueError(f"Слишком короткий CFG: {cfg_path}")
    counts = [part.strip() for part in lines[1].split(",")]
    if len(counts) < 3:
        raise ValueError(f"Некорректная строка числа каналов: {cfg_path}")
    analog_count = _count_with_suffix(counts[1], "A")
    digital_count = _count_with_suffix(counts[2], "D")
    expected_total = analog_count + digital_count
    if int(counts[0]) != expected_total:
        raise ValueError(f"AA+DD не совпадает с общим числом каналов: {cfg_path}")

    analog_lines = lines[2 : 2 + analog_count]
    digital_lines = lines[2 + analog_count : 2 + analog_count + digital_count]
    analog_names = tuple(_channel_name(line, cfg_path) for line in analog_lines)
    digital_names = tuple(_channel_name(line, cfg_path) for line in digital_lines)
    if len(set((*analog_names, *digital_names))) != expected_total:
        raise ValueError(f"Имена каналов не уникальны: {cfg_path}")

    tail = 2 + analog_count + digital_count
    rate_count = int(lines[tail + 1].strip())
    if rate_count != 1:
        raise ValueError(f"Поддерживается ровно одна частота дискретизации: {cfg_path}")
    rate_fields = [part.strip() for part in lines[tail + 2].split(",")]
    sample_rate_hz = float(rate_fields[0])
    expected_samples = int(rate_fields[1])
    if lines[tail + 5].strip().upper() != "ASCII":
        raise ValueError(f"Поддерживается только ASCII DAT: {cfg_path}")

    try:
        matrix = np.loadtxt(dat_path, delimiter=",", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать DAT {dat_path}: {exc}") from exc
    expected_columns = 2 + expected_total
    if matrix.shape != (expected_samples, expected_columns):
        raise ValueError(
            f"Форма DAT {matrix.shape} не совпадает с CFG "
            f"({expected_samples}, {expected_columns}): {dat_path}"
        )
    sample_numbers = matrix[:, 0].astype(np.int64)
    timestamps = matrix[:, 1].astype(np.int64)
    if not np.array_equal(sample_numbers, np.arange(1, expected_samples + 1)):
        raise ValueError(f"Нарушена последовательность номеров отсчётов: {dat_path}")
    if expected_samples > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError(f"Временные метки DAT не монотонны: {dat_path}")

    analog_matrix = matrix[:, 2 : 2 + analog_count]
    digital_matrix = matrix[:, 2 + analog_count :]
    if not np.isfinite(digital_matrix).all() or not np.isin(digital_matrix, (0.0, 1.0)).all():
        raise ValueError(f"Дискреты должны содержать только 0/1: {dat_path}")
    return ParsedAsciiComtrade(
        analog={name: analog_matrix[:, index] for index, name in enumerate(analog_names)},
        digital={name: digital_matrix[:, index].astype(np.uint8) for index, name in enumerate(digital_names)},
        sample_numbers=sample_numbers,
        timestamps_us=timestamps,
        sample_rate_hz=sample_rate_hz,
        analog_names=analog_names,
        digital_names=digital_names,
    )


def build_transition_masks(
    fwd: np.ndarray,
    valid: np.ndarray,
    sample_rate_hz: float,
    transition_ms: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Построить train/eval-маски границы и следующих ``transition_ms``."""

    fwd = np.asarray(fwd, dtype=np.uint8)
    valid = np.asarray(valid, dtype=np.uint8)
    if fwd.shape != valid.shape or fwd.ndim != 1:
        raise ValueError("expert FWD/VALID должны быть одномерны и иметь одинаковую длину")
    if transition_ms < 0 or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Некорректный временной контракт переходной маски")
    changed_valid = valid[1:] != valid[:-1]
    changed_direction = (valid[1:] == 1) & (valid[:-1] == 1) & (fwd[1:] != fwd[:-1])
    transition_points = np.flatnonzero(changed_valid | changed_direction) + 1
    eval_mask = np.zeros(fwd.size, dtype=bool)
    span = int(np.ceil(transition_ms * sample_rate_hz / 1000.0))
    for index in transition_points:
        eval_mask[index : min(fwd.size, index + span + 1)] = True
    return ~eval_mask, eval_mask


def discover_reference_cases(review_root: Path) -> dict[tuple[str, int], tuple[Path, dict[str, object]]]:
    """Найти неизменённые исходные экспорты по sidecar и контрольным хэшам."""

    result: dict[tuple[str, int], tuple[Path, dict[str, object]]] = {}
    for sidecar_path in Path(review_root).rglob("*.json"):
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("kind") != "phase5_pdr_manual_review_case":
            continue
        cfg_path = sidecar_path.with_name(str(payload.get("cfg", sidecar_path.with_suffix(".cfg").name)))
        dat_path = sidecar_path.with_name(str(payload.get("dat", sidecar_path.with_suffix(".dat").name)))
        if not cfg_path.exists() or not dat_path.exists():
            continue
        cfg_hash = str(payload.get("cfg_sha256", ""))
        dat_hash = str(payload.get("dat_sha256", ""))
        if cfg_hash and _sha256(cfg_path) != cfg_hash:
            continue
        if dat_hash and _sha256(dat_path) != dat_hash:
            continue
        key = (str(payload["source"]), int(payload["record_id"]))
        if key in result:
            raise ValueError(f"Несколько эталонных экспортов для {key}")
        result[key] = (cfg_path, payload)
    return result


def import_expert_tree(
    labels_root: Path,
    reference_review_root: Path,
    *,
    statuses: Sequence[str] = KNOWN_STATUSES,
    transition_ms: float = 5.0,
) -> tuple[list[ImportedExpertRecord], list[dict[str, object]]]:
    """Проверить дерево ручной работы и извлечь экспертный слой."""

    labels_root = Path(labels_root)
    references = discover_reference_cases(reference_review_root)
    imported: list[ImportedExpertRecord] = []
    audit: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    allowed = set(statuses)
    for sidecar_path in sorted(labels_root.rglob("*.json")):
        relative = sidecar_path.relative_to(labels_root)
        if len(relative.parts) < 4 or relative.parts[0] not in allowed:
            continue
        status, stratum, source_from_path = relative.parts[:3]
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "phase5_pdr_manual_review_case":
            continue
        source, record_id = str(payload["source"]), int(payload["record_id"])
        key = (source, record_id)
        if source != source_from_path:
            raise ValueError(f"Источник в пути и JSON различается: {sidecar_path}")
        if key in seen:
            raise ValueError(f"Одна запись встречается в нескольких статусах: {key}")
        seen.add(key)
        cfg_path = sidecar_path.with_suffix(".cfg")
        dat_path = sidecar_path.with_suffix(".dat")
        if not cfg_path.exists() or not dat_path.exists():
            raise FileNotFoundError(f"Неполный комплект ручной разметки: {sidecar_path}")
        if key not in references:
            raise FileNotFoundError(f"Не найден неизменённый исходный экспорт для {key}")
        reference_cfg, reference_payload = references[key]
        if str(payload.get("input_sha256")) != str(reference_payload.get("input_sha256")):
            raise ValueError(f"input_sha256 не совпал для {key}")

        edited = read_comtrade_1999_ascii(cfg_path, dat_path)
        reference = read_comtrade_1999_ascii(reference_cfg)
        _assert_preserved(reference, edited, key)
        if EXPERT_FWD not in edited.digital or EXPERT_VALID not in edited.digital:
            raise ValueError(f"Нет экспертных каналов в {cfg_path}")
        fwd = edited.digital[EXPERT_FWD]
        valid = edited.digital[EXPERT_VALID]
        if status == "invalid_quality":
            # Статус всей записи сильнее содержимого рабочих дискретов: он
            # обучает голову применимости и не задаёт направление.
            fwd = np.zeros_like(fwd)
            valid = np.zeros_like(valid)
        train_mask, transition_mask = build_transition_masks(
            fwd, valid, edited.sample_rate_hz, transition_ms
        )
        directions = np.where(
            valid == 1,
            np.where(fwd == 1, int(PDRDirection.FORWARD), int(PDRDirection.REVERSE)),
            int(PDRDirection.UNLABELED),
        ).astype(np.int16)
        automatic = {
            algorithm_id: np.where(
                edited.digital[f"{algorithm_id}__VALID"] == 1,
                edited.digital[f"{algorithm_id}__FWD"],
                int(PDRDirection.UNLABELED),
            ).astype(np.int16)
            for algorithm_id in ALGORITHM_IDS
        }
        required_analog = set(reference.analog_names)
        required_digital = set(reference.digital_names)
        record = ImportedExpertRecord(
            source=source,
            record_id=record_id,
            status=status,
            stratum=stratum,
            input_sha256=str(payload.get("input_sha256", "")),
            f_adc=edited.sample_rate_hz,
            directions=directions,
            applicable=valid.astype(bool),
            train_mask=train_mask,
            transition_eval_mask=transition_mask,
            automatic_directions=automatic,
            ignored_analog_channels=tuple(name for name in edited.analog_names if name not in required_analog),
            ignored_digital_channels=tuple(name for name in edited.digital_names if name not in required_digital),
            cfg_path=cfg_path,
        )
        imported.append(record)
        audit.append(_audit_row(record))
    return imported, audit


def write_expert_archive(
    output_root: Path,
    records: Sequence[ImportedExpertRecord],
    *,
    transition_ms: float,
    split_lookup: Mapping[tuple[str, int], str] | None = None,
) -> None:
    """Записать небольшой lossless-архив экспертных меток отдельно от v5."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    global_rows: list[dict[str, object]] = []
    for source in sorted({record.source for record in records}):
        source_records = sorted(
            (record for record in records if record.source == source and record.status in TRAINABLE_STATUSES),
            key=lambda record: record.record_id,
        )
        source_dir = output_root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        offsets = [0]
        samples: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        train_masks: list[np.ndarray] = []
        transition_masks: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        for record in source_records:
            n_samples = record.directions.size
            samples.append(np.arange(n_samples, dtype=np.int32))
            directions.append(record.directions.astype(np.int16, copy=False))
            train_masks.append(record.train_mask.astype(bool, copy=False))
            transition_masks.append(record.transition_eval_mask.astype(bool, copy=False))
            offsets.append(offsets[-1] + n_samples)
            split = split_lookup.get((source, record.record_id), "unknown") if split_lookup else "unknown"
            row = {
                "source": source,
                "record_id": record.record_id,
                "status": record.status,
                "stratum": record.stratum,
                "split": split,
                "input_sha256": record.input_sha256,
                "n_samples": n_samples,
                "duration_sec": n_samples / record.f_adc,
            }
            rows.append(row)
            global_rows.append(row)
        archive_path = source_dir / "expert_labels.npz"
        _atomic_npz(
            archive_path,
            offsets=np.asarray(offsets, dtype=np.int64),
            samples=np.concatenate(samples) if samples else np.asarray([], dtype=np.int32),
            directions=np.concatenate(directions) if directions else np.asarray([], dtype=np.int16),
            train_mask=np.concatenate(train_masks) if train_masks else np.asarray([], dtype=bool),
            transition_eval_mask=(
                np.concatenate(transition_masks) if transition_masks else np.asarray([], dtype=bool)
            ),
        )
        _write_csv(source_dir / "records.csv", rows)
        _atomic_json(source_dir / "manifest.json", {
            "kind": "pdr_expert_labels_sharded",
            "schema_version": 1,
            "source": source,
            "transition_exclusion_ms": transition_ms,
            "record_count": len(source_records),
            "trainable_statuses": list(TRAINABLE_STATUSES),
            "shards": [{"file": archive_path.name, "record_ids": [row["record_id"] for row in rows]}],
        })
    _write_csv(output_root / "records.csv", global_rows)
    _atomic_json(output_root / "summary.json", {
        "kind": "pdr_expert_labels_collection",
        "schema_version": 1,
        "transition_exclusion_ms": transition_ms,
        "record_count": len(global_rows),
        "sources": {
            source: sum(row["source"] == source for row in global_rows)
            for source in sorted({str(row["source"]) for row in global_rows})
        },
        "statuses": {
            status: sum(row["status"] == status for row in global_rows)
            for status in TRAINABLE_STATUSES
        },
    })


def aggregate_algorithm_comparison(records: Sequence[ImportedExpertRecord]) -> list[dict[str, object]]:
    """Сравнить пять автоматических органов с завершённым экспертным слоем."""

    rows: list[dict[str, object]] = []
    groups: list[tuple[str, Sequence[ImportedExpertRecord]]] = [("all", records)]
    groups.extend((source, [record for record in records if record.source == source])
                  for source in sorted({record.source for record in records}))
    for group, group_records in groups:
        for algorithm_id in ALGORITHM_IDS:
            comparable_total = agree_total = 0
            expert_forward = expert_reverse = auto_forward = 0
            app_tp = app_tn = app_fp = app_fn = 0
            per_record_accuracy: list[float] = []
            per_record_counts: list[tuple[int, int]] = []
            for record in group_records:
                if record.status not in TRAINABLE_STATUSES:
                    continue
                automatic = record.automatic_directions[algorithm_id]
                expert_app = record.applicable
                auto_app = automatic != int(PDRDirection.UNLABELED)
                eval_mask = record.train_mask
                app_tp += int(np.count_nonzero(eval_mask & expert_app & auto_app))
                app_tn += int(np.count_nonzero(eval_mask & ~expert_app & ~auto_app))
                app_fp += int(np.count_nonzero(eval_mask & ~expert_app & auto_app))
                app_fn += int(np.count_nonzero(eval_mask & expert_app & ~auto_app))
                comparable = eval_mask & expert_app & auto_app
                count = int(np.count_nonzero(comparable))
                if count:
                    agree = int(np.count_nonzero(automatic[comparable] == record.directions[comparable]))
                    comparable_total += count
                    agree_total += agree
                    per_record_accuracy.append(agree / count)
                    per_record_counts.append((agree, count))
                    expert_forward += int(np.count_nonzero(record.directions[comparable] == 1))
                    expert_reverse += int(np.count_nonzero(record.directions[comparable] == 0))
                    auto_forward += int(np.count_nonzero(automatic[comparable] == 1))
            app_count = app_tp + app_tn + app_fp + app_fn
            sample_ci = _cluster_bootstrap_accuracy(per_record_counts, macro=False, seed=_stable_seed(group, algorithm_id))
            macro_ci = _cluster_bootstrap_accuracy(per_record_counts, macro=True, seed=_stable_seed(group, algorithm_id) + 1)
            record_accuracy_array = np.asarray(per_record_accuracy, dtype=np.float64)
            rows.append({
                "group": group,
                "algorithm_id": algorithm_id,
                "records_compared": len(per_record_accuracy),
                "comparable_samples": comparable_total,
                "sample_accuracy": agree_total / comparable_total if comparable_total else "",
                "sample_accuracy_ci95_low": sample_ci[0] if sample_ci else "",
                "sample_accuracy_ci95_high": sample_ci[1] if sample_ci else "",
                "record_macro_accuracy": float(np.mean(record_accuracy_array)) if per_record_accuracy else "",
                "record_macro_accuracy_ci95_low": macro_ci[0] if macro_ci else "",
                "record_macro_accuracy_ci95_high": macro_ci[1] if macro_ci else "",
                "record_accuracy_std": (
                    float(np.std(record_accuracy_array, ddof=1))
                    if len(per_record_accuracy) > 1 else 0.0 if per_record_accuracy else ""
                ),
                "record_accuracy_q25": (
                    float(np.quantile(record_accuracy_array, 0.25)) if per_record_accuracy else ""
                ),
                "record_accuracy_median": (
                    float(np.median(record_accuracy_array)) if per_record_accuracy else ""
                ),
                "record_accuracy_q75": (
                    float(np.quantile(record_accuracy_array, 0.75)) if per_record_accuracy else ""
                ),
                "expert_forward_fraction": expert_forward / comparable_total if comparable_total else "",
                "automatic_forward_fraction": auto_forward / comparable_total if comparable_total else "",
                "applicability_accuracy": (app_tp + app_tn) / app_count if app_count else "",
                "applicability_false_positive_fraction": app_fp / app_count if app_count else "",
                "applicability_false_negative_fraction": app_fn / app_count if app_count else "",
                "app_tp": app_tp, "app_tn": app_tn, "app_fp": app_fp, "app_fn": app_fn,
            })
    return rows


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _cluster_bootstrap_accuracy(
    counts: Sequence[tuple[int, int]],
    *,
    macro: bool,
    seed: int,
    iterations: int = 2000,
) -> tuple[float, float] | None:
    """95% cluster-bootstrap CI, где кластером служит целая осциллограмма.

    Каждая бутстрэп-выборка содержит столько же целых записей, сколько
    было в исходном наборе, но записи выбираются с возвращением. Отсчёты
    внутри выбранной осциллограммы не перевыбираются: используются их полные
    счётчики совпадений и сопоставимых точек.
    """

    if not counts:
        return None
    values = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        chosen = values[rng.integers(0, len(values), len(values))]
        estimates[iteration] = (
            np.mean(chosen[:, 0] / chosen[:, 1])
            if macro else chosen[:, 0].sum() / chosen[:, 1].sum()
        )
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def record_algorithm_comparison(records: Sequence[ImportedExpertRecord]) -> list[dict[str, object]]:
    """Сформировать полный, не усечённый рейтинг расхождений по записям."""

    rows: list[dict[str, object]] = []
    for record in records:
        if record.status not in TRAINABLE_STATUSES:
            continue
        for algorithm_id in ALGORITHM_IDS:
            automatic = record.automatic_directions[algorithm_id]
            auto_applicable = automatic != int(PDRDirection.UNLABELED)
            comparable = record.train_mask & record.applicable & auto_applicable
            count = int(np.count_nonzero(comparable))
            agreement = (
                float(np.mean(automatic[comparable] == record.directions[comparable]))
                if count else ""
            )
            expert_forward = (
                float(np.mean(record.directions[comparable] == int(PDRDirection.FORWARD)))
                if count else ""
            )
            automatic_forward = (
                float(np.mean(automatic[comparable] == int(PDRDirection.FORWARD)))
                if count else ""
            )
            rows.append({
                "source": record.source,
                "record_id": record.record_id,
                "status": record.status,
                "stratum": record.stratum,
                "algorithm_id": algorithm_id,
                "comparable_samples": count,
                "sample_accuracy": agreement,
                "expert_forward_fraction": expert_forward,
                "automatic_forward_fraction": automatic_forward,
                "expert_applicable_fraction": float(np.mean(record.applicable)),
                "automatic_applicable_fraction": float(np.mean(auto_applicable)),
                "opposite_stable_candidate": bool(
                    count
                    and float(agreement) <= 0.10
                    and (
                        (float(expert_forward) >= 0.90 and float(automatic_forward) <= 0.10)
                        or (float(expert_forward) <= 0.10 and float(automatic_forward) >= 0.90)
                    )
                ),
            })
    return rows


def expert_quality_flags(records: Sequence[ImportedExpertRecord]) -> list[dict[str, object]]:
    """Найти случаи для повторного слепого контроля, не изменяя метки."""

    rows: list[dict[str, object]] = []
    for record in records:
        if record.status != "completed":
            continue
        flags: list[str] = []
        applicable_fraction = float(np.mean(record.applicable))
        if applicable_fraction == 0.0:
            flags.append("whole_record_0_0")
        if 0.0 < applicable_fraction < 0.05:
            flags.append("very_low_applicable_fraction")
        transition_count = _direction_transition_count(record.directions, record.applicable)
        if transition_count >= 20:
            flags.append("many_expert_transitions")
        minimum_run_ms = _minimum_internal_run_ms(
            record.directions, record.applicable, record.f_adc
        )
        if minimum_run_ms is not None and minimum_run_ms < 5.0:
            flags.append("expert_internal_run_shorter_than_5ms")
        adaptive = record.automatic_directions["adaptive_pdr_mir"]
        comparable = (
            record.train_mask
            & record.applicable
            & (adaptive != int(PDRDirection.UNLABELED))
        )
        adaptive_accuracy = (
            float(np.mean(adaptive[comparable] == record.directions[comparable]))
            if np.any(comparable) else None
        )
        if adaptive_accuracy is not None and adaptive_accuracy < 0.50:
            flags.append("adaptive_disagreement_over_50pct")
        if flags:
            rows.append({
                "source": record.source,
                "record_id": record.record_id,
                "stratum": record.stratum,
                "flags": "|".join(flags),
                "applicable_fraction": applicable_fraction,
                "expert_transitions": transition_count,
                "minimum_internal_expert_run_ms": (
                    minimum_run_ms if minimum_run_ms is not None else ""
                ),
                "adaptive_accuracy": adaptive_accuracy if adaptive_accuracy is not None else "",
                "cfg": str(record.cfg_path),
            })
    return rows


def expert_group_summary(
    records: Sequence[ImportedExpertRecord],
    split_lookup: Mapping[tuple[str, int], str] | None = None,
) -> list[dict[str, object]]:
    """Сводка объёма и состава по статусу, источнику, split и страте."""

    definitions: list[tuple[str, str, Sequence[ImportedExpertRecord]]] = [("all", "all", records)]
    for source in sorted({record.source for record in records}):
        definitions.append(("source", source, [record for record in records if record.source == source]))
    for status in sorted({record.status for record in records}):
        definitions.append(("status", status, [record for record in records if record.status == status]))
    for stratum in sorted({record.stratum for record in records}):
        definitions.append(("stratum", stratum, [record for record in records if record.stratum == stratum]))
    if split_lookup:
        for split in ("train", "validation", "holdout", "unknown"):
            subset = [
                record for record in records
                if split_lookup.get((record.source, record.record_id), "unknown") == split
            ]
            if subset:
                definitions.append(("split", split, subset))
    rows: list[dict[str, object]] = []
    for group_type, group_value, subset in definitions:
        samples = sum(record.directions.size for record in subset)
        applicable = sum(int(np.count_nonzero(record.applicable)) for record in subset)
        forward = sum(int(np.count_nonzero(record.directions == int(PDRDirection.FORWARD))) for record in subset)
        transition = sum(int(np.count_nonzero(record.transition_eval_mask)) for record in subset)
        rows.append({
            "group_type": group_type,
            "group_value": group_value,
            "records": len(subset),
            "samples": samples,
            "duration_sec": sum(record.directions.size / record.f_adc for record in subset),
            "applicable_samples": applicable,
            "applicable_fraction": applicable / samples if samples else "",
            "forward_fraction_when_applicable": forward / applicable if applicable else "",
            "expert_transitions": sum(
                _direction_transition_count(record.directions, record.applicable)
                for record in subset
            ),
            "transition_eval_samples": transition,
            "transition_eval_fraction": transition / samples if samples else "",
        })
    return rows


def compare_expert_repeatability(
    original_records: Sequence[ImportedExpertRecord],
    repeated_records: Sequence[ImportedExpertRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Сравнить две независимые разметки одних и тех же осциллограмм.

    Повторная оценка не выбирается автоматически как более правильная. Функция
    возвращает пофайловые метрики, агрегаты и очередь инженерного согласования.
    Переходные зоны обеих версий исключаются только из steady-state метрик, но
    полное сырое согласие и сдвиги границ сохраняются отдельно.
    """

    originals = {(record.source, record.record_id): record for record in original_records}
    repeated = {(record.source, record.record_id): record for record in repeated_records}
    if len(originals) != len(original_records) or len(repeated) != len(repeated_records):
        raise ValueError("В одной версии повторяются source/record_id")

    rows: list[dict[str, object]] = []
    for key in sorted(repeated):
        if key not in originals:
            raise ValueError(f"Для повторной разметки нет исходной версии: {key}")
        first, second = originals[key], repeated[key]
        if first.input_sha256 != second.input_sha256:
            raise ValueError(f"input_sha256 различается между версиями: {key}")
        if first.directions.shape != second.directions.shape or not np.isclose(first.f_adc, second.f_adc):
            raise ValueError(f"Временная сетка различается между версиями: {key}")

        steady = ~(first.transition_eval_mask | second.transition_eval_mask)
        both_valid = steady & first.applicable & second.applicable
        raw_equal = first.directions == second.directions
        state_equal = steady & raw_equal
        valid_equal = steady & (first.applicable == second.applicable)
        direction_equal = both_valid & raw_equal
        first_valid_count = int(np.count_nonzero(steady & first.applicable))
        second_valid_count = int(np.count_nonzero(steady & second.applicable))
        common_valid_count = int(np.count_nonzero(both_valid))
        steady_count = int(np.count_nonzero(steady))

        first_transitions = _state_transition_indices(first.directions)
        second_transitions = _state_transition_indices(second.directions)
        first_valid_transitions = _state_transition_indices(first.applicable.astype(np.int8))
        second_valid_transitions = _state_transition_indices(second.applicable.astype(np.int8))
        first_direction_transitions = _direction_transition_count(
            first.directions, first.applicable
        )
        second_direction_transitions = _direction_transition_count(
            second.directions, second.applicable
        )
        boundary_distances = _symmetric_nearest_distances(first_transitions, second_transitions)
        boundary_ms = boundary_distances * 1000.0 / first.f_adc
        maximum_boundary_shift_ms = (
            float(np.max(boundary_ms)) if boundary_ms.size else 0.0
        )
        transition_count_difference = abs(
            int(first_transitions.size) - int(second_transitions.size)
        )
        disagreement = steady & ~raw_equal
        disagreement_runs = _true_run_lengths(disagreement)

        first_forward_fraction = _masked_forward_fraction(first.directions, steady & first.applicable)
        second_forward_fraction = _masked_forward_fraction(second.directions, steady & second.applicable)
        valid_agreement = float(np.count_nonzero(valid_equal) / steady_count) if steady_count else np.nan
        direction_agreement = (
            float(np.count_nonzero(direction_equal) / common_valid_count)
            if common_valid_count else np.nan
        )
        state_agreement = float(np.count_nonzero(state_equal) / steady_count) if steady_count else np.nan
        state_kappa = _cohen_kappa(
            first.directions,
            second.directions,
            steady,
            (int(PDRDirection.UNLABELED), int(PDRDirection.REVERSE), int(PDRDirection.FORWARD)),
        )
        valid_kappa = _cohen_kappa(
            first.applicable.astype(np.int8),
            second.applicable.astype(np.int8),
            steady,
            (0, 1),
        )
        direction_kappa = _cohen_kappa(
            first.directions,
            second.directions,
            both_valid,
            (int(PDRDirection.REVERSE), int(PDRDirection.FORWARD)),
        )
        stable_opposite = bool(
            common_valid_count
            and (
                (first_forward_fraction >= 0.90 and second_forward_fraction <= 0.10)
                or (first_forward_fraction <= 0.10 and second_forward_fraction >= 0.90)
            )
        )
        if stable_opposite:
            priority, action = "critical", "повторно изучить физику и принять третье согласованное решение"
        elif transition_count_difference:
            priority, action = "high", "согласовать число переключений и лишние/пропущенные границы"
        elif maximum_boundary_shift_ms > 20.0:
            priority, action = "high", "согласовать границы, сдвинутые более чем на 20 мс"
        elif np.isfinite(valid_agreement) and valid_agreement < 0.90:
            priority, action = "high", "согласовать критерий VALID и границы невалидности"
        elif np.isfinite(direction_agreement) and direction_agreement < 0.90:
            priority, action = "high", "согласовать направление на общих валидных участках"
        elif not bool(np.all(raw_equal)):
            priority, action = "low", "локальное расхождение либо сдвиг границы не более 20 мс"
        else:
            priority, action = "none", "согласование не требуется"

        rows.append({
            "source": first.source,
            "record_id": first.record_id,
            "stratum": first.stratum,
            "first_status": first.status,
            "repeat_status": second.status,
            "samples": first.directions.size,
            "duration_sec": first.directions.size / first.f_adc,
            "raw_exact_match": bool(np.all(raw_equal)),
            "raw_state_agreement": float(np.mean(raw_equal)),
            "steady_samples": steady_count,
            "steady_state_agreement": state_agreement,
            "steady_state_cohen_kappa": state_kappa,
            "valid_agreement": valid_agreement,
            "valid_cohen_kappa": valid_kappa,
            "common_valid_samples": common_valid_count,
            "direction_agreement_when_both_valid": direction_agreement,
            "direction_cohen_kappa_when_both_valid": direction_kappa,
            "first_applicable_fraction": first_valid_count / steady_count if steady_count else np.nan,
            "repeat_applicable_fraction": second_valid_count / steady_count if steady_count else np.nan,
            "first_forward_fraction_when_valid": first_forward_fraction,
            "repeat_forward_fraction_when_valid": second_forward_fraction,
            "first_transitions": int(first_transitions.size),
            "repeat_transitions": int(second_transitions.size),
            "transition_count_difference": transition_count_difference,
            "transition_count_match": transition_count_difference == 0,
            "first_valid_transitions": int(first_valid_transitions.size),
            "repeat_valid_transitions": int(second_valid_transitions.size),
            "first_direction_transitions_when_valid": first_direction_transitions,
            "repeat_direction_transitions_when_valid": second_direction_transitions,
            "boundary_nearest_median_ms": float(np.median(boundary_ms)) if boundary_ms.size else np.nan,
            "boundary_nearest_max_ms": maximum_boundary_shift_ms,
            "large_boundary_shift_over_20ms": maximum_boundary_shift_ms > 20.0,
            "disagreement_runs": int(disagreement_runs.size),
            "longest_disagreement_ms": (
                float(np.max(disagreement_runs) * 1000.0 / first.f_adc)
                if disagreement_runs.size else 0.0
            ),
            "stable_opposite_candidate": stable_opposite,
            "adjudication_priority": priority,
            "recommended_action": action,
        })

    summaries = _expert_repeatability_summaries(rows)
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
    queue = []
    for row in sorted(rows, key=lambda item: (
        order[str(item["adjudication_priority"])],
        _finite_sort_value(item["steady_state_agreement"]),
        str(item["source"]),
        int(item["record_id"]),
    )):
        if row["adjudication_priority"] not in ("critical", "high"):
            continue
        queue.append({
            **row,
            "adjudication_status": "pending",
            "accepted_version": "",
            "engineering_reason": "",
            "reviewer_comment": "",
        })
    return rows, summaries, queue


def write_expert_repeatability_audit(
    output_root: Path,
    record_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    adjudication_rows: Sequence[dict[str, object]],
) -> None:
    """Сохранить аудит самосогласованности без создания обучающего архива."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "repeatability_record_comparison.csv", record_rows)
    _write_csv(output_root / "repeatability_summary.csv", summary_rows)
    _write_csv(output_root / "adjudication_queue.csv", adjudication_rows)
    _atomic_json(output_root / "repeatability_summary.json", {
        "kind": "pdr_expert_repeatability_audit",
        "schema_version": 1,
        "paired_records": len(record_rows),
        "adjudication_records": len(adjudication_rows),
        "does_not_modify_training_labels": True,
        "files": {
            "record_comparison": "repeatability_record_comparison.csv",
            "summary": "repeatability_summary.csv",
            "adjudication_queue": "adjudication_queue.csv",
        },
    })


def _expert_repeatability_summaries(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[tuple[str, Sequence[dict[str, object]]]] = [("all", rows)]
    groups.extend((source, [row for row in rows if row["source"] == source])
                  for source in sorted({str(row["source"]) for row in rows}))
    result: list[dict[str, object]] = []
    for group, subset in groups:
        steady = sum(int(row["steady_samples"]) for row in subset)
        common = sum(int(row["common_valid_samples"]) for row in subset)
        state_agree = sum(
            float(row["steady_state_agreement"]) * int(row["steady_samples"])
            for row in subset if np.isfinite(float(row["steady_state_agreement"]))
        )
        direction_agree = sum(
            float(row["direction_agreement_when_both_valid"]) * int(row["common_valid_samples"])
            for row in subset if np.isfinite(float(row["direction_agreement_when_both_valid"]))
        )
        result.append({
            "group": group,
            "records": len(subset),
            "exact_record_agreement_fraction": (
                sum(bool(row["raw_exact_match"]) for row in subset) / len(subset) if subset else np.nan
            ),
            "steady_state_micro_agreement": state_agree / steady if steady else np.nan,
            "steady_state_record_macro_agreement": _nanmean_rows(subset, "steady_state_agreement"),
            "steady_state_record_macro_cohen_kappa": _nanmean_rows(
                subset, "steady_state_cohen_kappa"
            ),
            "valid_record_macro_agreement": _nanmean_rows(subset, "valid_agreement"),
            "valid_record_macro_cohen_kappa": _nanmean_rows(subset, "valid_cohen_kappa"),
            "direction_micro_agreement_when_both_valid": direction_agree / common if common else np.nan,
            "direction_record_macro_agreement_when_both_valid": _nanmean_rows(
                subset, "direction_agreement_when_both_valid"
            ),
            "direction_record_macro_cohen_kappa_when_both_valid": _nanmean_rows(
                subset, "direction_cohen_kappa_when_both_valid"
            ),
            "stable_opposite_records": sum(bool(row["stable_opposite_candidate"]) for row in subset),
            "matching_transition_count_records": sum(
                bool(row["transition_count_match"]) for row in subset
            ),
            "records_with_boundary_shift_over_20ms": sum(
                bool(row["large_boundary_shift_over_20ms"]) for row in subset
            ),
            "records_requiring_adjudication": sum(
                row["adjudication_priority"] in ("critical", "high") for row in subset
            ),
            "critical_records": sum(row["adjudication_priority"] == "critical" for row in subset),
            "high_priority_records": sum(
                row["adjudication_priority"] == "high" for row in subset
            ),
            "low_local_difference_records": sum(
                row["adjudication_priority"] == "low" for row in subset
            ),
        })
    return result


def _state_transition_indices(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states)
    return np.flatnonzero(states[1:] != states[:-1]) + 1 if states.size > 1 else np.asarray([], dtype=int)


def _symmetric_nearest_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.size == 0 and second.size == 0:
        return np.asarray([], dtype=np.float64)
    if first.size == 0 or second.size == 0:
        return np.asarray([np.inf], dtype=np.float64)
    first_to_second = np.min(np.abs(first[:, None] - second[None, :]), axis=1)
    second_to_first = np.min(np.abs(second[:, None] - first[None, :]), axis=1)
    return np.concatenate((first_to_second, second_to_first)).astype(np.float64)


def _true_run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return (edges[1::2] - edges[::2]).astype(np.int64)


def _masked_forward_fraction(states: np.ndarray, mask: np.ndarray) -> float:
    count = int(np.count_nonzero(mask))
    return float(np.count_nonzero((states == int(PDRDirection.FORWARD)) & mask) / count) if count else np.nan


def _cohen_kappa(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    classes: Sequence[int],
) -> float:
    count = int(np.count_nonzero(mask))
    if not count:
        return np.nan
    first_values = np.asarray(first)[mask]
    second_values = np.asarray(second)[mask]
    observed = float(np.mean(first_values == second_values))
    expected = sum(
        float(np.mean(first_values == value)) * float(np.mean(second_values == value))
        for value in classes
    )
    if np.isclose(expected, 1.0):
        return 1.0 if np.isclose(observed, 1.0) else 0.0
    return float((observed - expected) / (1.0 - expected))


def _nanmean_rows(rows: Sequence[dict[str, object]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _finite_sort_value(value: object) -> float:
    number = float(value)
    return number if np.isfinite(number) else float("inf")


def _assert_preserved(
    reference: ParsedAsciiComtrade,
    edited: ParsedAsciiComtrade,
    key: tuple[str, int],
) -> None:
    if reference.n_samples != edited.n_samples:
        raise ValueError(f"Изменилось число отсчётов для {key}")
    if not np.array_equal(reference.sample_numbers, edited.sample_numbers):
        raise ValueError(f"Изменились номера отсчётов для {key}")
    if not np.isclose(reference.sample_rate_hz, edited.sample_rate_hz, rtol=0, atol=1e-9):
        raise ValueError(f"Изменилась частота дискретизации для {key}")
    # APSilloscоpe при повторном сохранении использует усечение вместо
    # округления для дробного шага в микросекундах (например, 833 вместо 833/834
    # при 1200 Гц). Это безопасное текстовое переформатирование: индекс,
    # частота и число отсчётов сохраняются, а расхождение не превышает 1 мкс.
    timestamp_delta = np.abs(reference.timestamps_us - edited.timestamps_us)
    ideal = np.arange(reference.n_samples, dtype=np.float64) * 1_000_000.0 / reference.sample_rate_hz
    if (
        np.any(timestamp_delta > 1)
        or np.any(np.abs(edited.timestamps_us - ideal) > 1.0)
    ):
        raise ValueError(f"Изменились временные метки для {key}")
    for name, expected in reference.analog.items():
        if name not in edited.analog:
            raise ValueError(f"Удалён обязательный аналоговый канал {name!r} для {key}")
        if not np.allclose(expected, edited.analog[name], rtol=1e-9, atol=1e-6, equal_nan=True):
            raise ValueError(f"Изменён обязательный аналоговый канал {name!r} для {key}")
    for name, expected in reference.digital.items():
        if name in (EXPERT_FWD, EXPERT_VALID):
            continue
        if name not in edited.digital or not np.array_equal(expected, edited.digital[name]):
            raise ValueError(f"Изменён автоматический дискрет {name!r} для {key}")


def _audit_row(record: ImportedExpertRecord) -> dict[str, object]:
    applicable = record.applicable
    valid_directions = record.directions[applicable]
    transitions = int(np.count_nonzero(
        (record.applicable[1:] != record.applicable[:-1])
        | (
            record.applicable[1:] & record.applicable[:-1]
            & (record.directions[1:] != record.directions[:-1])
        )
    )) if record.directions.size > 1 else 0
    return {
        "status": record.status,
        "stratum": record.stratum,
        "source": record.source,
        "record_id": record.record_id,
        "input_sha256": record.input_sha256,
        "n_samples": record.directions.size,
        "duration_sec": record.directions.size / record.f_adc,
        "applicable_fraction": float(np.mean(applicable)),
        "forward_fraction_when_applicable": (
            float(np.mean(valid_directions == int(PDRDirection.FORWARD)))
            if valid_directions.size else ""
        ),
        "transitions": transitions,
        "transition_eval_fraction": float(np.mean(record.transition_eval_mask)),
        "ignored_analog_channels": "|".join(record.ignored_analog_channels),
        "ignored_digital_channels": "|".join(record.ignored_digital_channels),
        "cfg": str(record.cfg_path),
    }


def write_audit_tables(
    output_root: Path,
    audit_rows: Iterable[dict[str, object]],
    comparison_rows: Iterable[dict[str, object]],
    record_comparison_rows: Iterable[dict[str, object]] = (),
    quality_flag_rows: Iterable[dict[str, object]] = (),
    group_summary_rows: Iterable[dict[str, object]] = (),
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "record_audit.csv", audit_rows)
    _write_csv(output_root / "algorithm_comparison.csv", comparison_rows)
    _write_csv(output_root / "record_algorithm_comparison.csv", record_comparison_rows)
    _write_csv(output_root / "expert_quality_flags.csv", quality_flag_rows)
    _write_csv(output_root / "expert_group_summary.csv", group_summary_rows)


def _direction_transition_count(directions: np.ndarray, applicable: np.ndarray) -> int:
    if directions.size < 2:
        return 0
    return int(np.count_nonzero(
        (applicable[1:] != applicable[:-1])
        | (applicable[1:] & applicable[:-1] & (directions[1:] != directions[:-1]))
    ))


def _minimum_internal_run_ms(
    directions: np.ndarray,
    applicable: np.ndarray,
    sample_rate_hz: float,
) -> float | None:
    """Минимальная длительность состояния между двумя реальными границами.

    Первый и последний сегменты не учитываются: короткий край записи часто
    возникает из-за положения курсора редактора и уже исключается переходной
    маской. Он не свидетельствует о дребезге экспертного решения внутри файла.
    """

    states = np.where(applicable, directions, int(PDRDirection.UNLABELED))
    if states.size == 0:
        return None
    boundaries = np.flatnonzero(states[1:] != states[:-1]) + 1
    lengths = np.diff(np.concatenate(([0], boundaries, [states.size])))
    internal = lengths[1:-1]
    return float(internal.min() * 1000.0 / sample_rate_hz) if internal.size else None


def _read_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, f"Не удалось определить кодировку {path}")


def _count_with_suffix(value: str, suffix: str) -> int:
    value = value.strip()
    if not value.upper().endswith(suffix):
        raise ValueError(f"Ожидался суффикс {suffix}: {value!r}")
    return int(value[:-1].strip())


def _channel_name(line: str, path: Path) -> str:
    fields = [field.strip() for field in next(csv.reader([line]))]
    if len(fields) < 2 or not fields[1]:
        raise ValueError(f"У канала отсутствует имя: {path}")
    return fields[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
