"""Проверить COMTRADE ручной разметки и создать отдельный экспертный слой."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.pdr.expert_labels import (
    aggregate_algorithm_comparison,
    expert_quality_flags,
    expert_group_summary,
    import_expert_tree,
    record_algorithm_comparison,
    write_audit_tables,
    write_expert_archive,
)


DEFAULT_LABELS_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_labels_v1"
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "data/phase5/pdr_manual_review_v5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/phase5/pdr_expert_labels_v1"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--transition-ms", type=float, default=5.0)
    args = parser.parse_args()
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
