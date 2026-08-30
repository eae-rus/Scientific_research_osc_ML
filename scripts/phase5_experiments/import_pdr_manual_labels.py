"""Проверить COMTRADE ручной разметки и создать отдельный экспертный слой."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.pdr.expert_labels import (
    aggregate_algorithm_comparison,
    compare_expert_repeatability,
    expert_quality_flags,
    expert_group_summary,
    import_expert_tree,
    record_algorithm_comparison,
    write_audit_tables,
    write_expert_archive,
    write_expert_repeatability_audit,
)


DEFAULT_LABELS_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_labels_v1"
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_review_v5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1"
DEFAULT_BLIND_AUDIT_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_blind_audit_v1"
DEFAULT_BLIND_AUDIT_LABELS_ROOT = DEFAULT_BLIND_AUDIT_ROOT / "labels"
DEFAULT_BLIND_AUDIT_REFERENCE_ROOT = DEFAULT_BLIND_AUDIT_ROOT / "reference"
DEFAULT_BLIND_AUDIT_OUTPUT_ROOT = DEFAULT_BLIND_AUDIT_ROOT / "analysis"
DEFAULT_SPLITS = PROJECT_ROOT / "data/phase5/research_strict_splits.json"


def _split_lookup(path: Path) -> dict[tuple[str, int], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[tuple[str, int], str] = {}
    for source, source_payload in payload["sources"].items():
        for split, record_ids in source_payload["splits"].items():
            for record_id in record_ids:
                key = (str(source), int(record_id))
                if key in result:
                    raise ValueError(f"Запись попала в несколько частей split: {key}")
                result[key] = str(split)
    return result


def run(
    labels_root: Path = DEFAULT_LABELS_ROOT,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    transition_ms: float = 5.0,
) -> None:
    records, audit_rows = import_expert_tree(
        labels_root,
        reference_root,
        transition_ms=transition_ms,
    )
    split_lookup = _split_lookup(DEFAULT_SPLITS)
    write_expert_archive(
        output_root,
        records,
        transition_ms=transition_ms,
        split_lookup=split_lookup,
    )
    comparisons = aggregate_algorithm_comparison(records)
    record_comparisons = record_algorithm_comparison(records)
    quality_flags = expert_quality_flags(records)
    group_summary = expert_group_summary(records, split_lookup)
    write_audit_tables(
        output_root,
        audit_rows,
        comparisons,
        record_comparisons,
        quality_flags,
        group_summary,
    )
    status_counts = {
        status: sum(record.status == status for record in records)
        for status in sorted({record.status for record in records})
    }
    source_counts = {
        source: sum(record.source == source for record in records)
        for source in sorted({record.source for record in records})
    }
    print(json.dumps({
        "records_checked": len(records),
        "statuses": status_counts,
        "sources": source_counts,
        "output": str(output_root),
        "ambiguous_is_not_trainable": True,
        "invalid_quality_trains_applicability_only": True,
    }, ensure_ascii=False, indent=2))


def _blind_audit_origins(reference_root: Path) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    path = Path(reference_root) / "_service" / "blind_audit_assignment.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            origin = str(row.get("origin", ""))
            if origin not in {"previously_labeled", "new_standard"}:
                raise ValueError(f"Нет корректного blind audit origin: {row}")
            key = (str(row["source"]), int(row["record_id"]))
            if key in result:
                raise ValueError(f"Повтор source/record_id в blind audit: {key}")
            result[key] = origin
    return result


def run_blind_audit(
    audit_labels_root: Path = DEFAULT_BLIND_AUDIT_LABELS_ROOT,
    audit_reference_root: Path = DEFAULT_BLIND_AUDIT_REFERENCE_ROOT,
    output_root: Path = DEFAULT_BLIND_AUDIT_OUTPUT_ROOT,
    transition_ms: float = 5.0,
) -> None:
    """Разобрать единый слепой пакет, не меняя основной train-архив."""

    original, _ = import_expert_tree(
        DEFAULT_LABELS_ROOT,
        DEFAULT_REFERENCE_ROOT,
        transition_ms=transition_ms,
    )
    # Неизменяемые исходные экспорты всех слепых случаев также хранятся в
    # общем дереве manual-review. Рабочий blind/reference может содержать
    # только манифест и скрытое назначение после переноса файлов экспертом.
    audited, audit_rows = import_expert_tree(
        audit_labels_root,
        DEFAULT_REFERENCE_ROOT,
        transition_ms=transition_ms,
    )
    origins = _blind_audit_origins(audit_reference_root)
    # ``origin`` описывает состояние записи в момент формирования слепого
    # пакета. К моменту импорта новый стандартный пакет уже может быть размечен
    # независимо и добавлен в основное дерево. Тогда это тоже полноценная пара
    # для проверки повторяемости, а не новый одиночный случай.
    original_keys = {(record.source, record.record_id) for record in original}
    old_records = [
        record for record in audited
        if (record.source, record.record_id) in original_keys
    ]
    new_records = [
        record for record in audited
        if (record.source, record.record_id) not in original_keys
    ]
    record_rows, summary_rows, adjudication_rows = compare_expert_repeatability(
        original,
        old_records,
    )
    write_expert_repeatability_audit(
        output_root / "repeatability",
        record_rows,
        summary_rows,
        adjudication_rows,
    )

    split_lookup = _split_lookup(DEFAULT_SPLITS)
    new_root = output_root / "new_cases"
    new_keys = {(record.source, record.record_id) for record in new_records}
    write_expert_archive(
        new_root / "expert_labels_candidate",
        new_records,
        transition_ms=transition_ms,
        split_lookup=split_lookup,
    )
    write_audit_tables(
        new_root,
        [row for row in audit_rows if (str(row["source"]), int(row["record_id"])) in new_keys],
        aggregate_algorithm_comparison(new_records),
        record_algorithm_comparison(new_records),
        expert_quality_flags(new_records),
        expert_group_summary(new_records, split_lookup),
    )
    print(json.dumps({
        "audited_records": len(audited),
        "repeatability_records": len(old_records),
        "new_records": len(new_records),
        "selection_origins": {
            origin: sum(value == origin for value in origins.values())
            for origin in sorted(set(origins.values()))
        },
        "adjudication_records": len(adjudication_rows),
        "output": str(output_root),
        "training_labels_modified": False,
    }, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--transition-ms", type=float, default=5.0)
    parser.add_argument("--blind-audit", action="store_true",
                        help="Разобрать единый смешанный слепой пакет")
    parser.add_argument("--audit-labels-root", type=Path, default=DEFAULT_BLIND_AUDIT_LABELS_ROOT)
    parser.add_argument("--audit-reference-root", type=Path, default=DEFAULT_BLIND_AUDIT_REFERENCE_ROOT)
    parser.add_argument("--audit-output-root", type=Path, default=DEFAULT_BLIND_AUDIT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.blind_audit:
        run_blind_audit(
            args.audit_labels_root,
            args.audit_reference_root,
            args.audit_output_root,
            args.transition_ms,
        )
    else:
        run(args.labels_root, args.reference_root, args.output_root, args.transition_ms)
    return 0


def run_manual() -> None:
    """Ручной запуск первого импорта экспертной разметки."""
    LABELS_ROOT = DEFAULT_LABELS_ROOT
    REFERENCE_ROOT = DEFAULT_REFERENCE_ROOT
    OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    TRANSITION_MS = 5.0
    run(LABELS_ROOT, REFERENCE_ROOT, OUTPUT_ROOT, TRANSITION_MS)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
