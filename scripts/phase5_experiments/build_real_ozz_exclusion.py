"""Построить file-level индекс Open_EE без известных real_OZZ записей."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5_experiments.progress import ProgressReporter

def _normalise(value: str) -> str:
    """Нормализовать идентификатор, не меняя его семантику."""

    return Path(value.strip()).stem.lower()


def load_known_names(report_path: Path) -> set[str]:
    """Извлечь базовые имена осциллограмм из CSV отчёта real_OZZ."""

    with report_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if "filename" not in (reader.fieldnames or ()):
            raise ValueError("В отчёте real_OZZ отсутствует колонка filename")
        return {_normalise(row["filename"]) for row in reader if row.get("filename", "").strip()}


def match_name(file_name: str, known_names: set[str], prefix_index: dict[str, set[str]]) -> tuple[str, list[str]]:
    """Вернуть тип совпадения и список подходящих known real_OZZ имён."""

    normalized = _normalise(file_name)
    if normalized in known_names:
        return "exact", [normalized]
    candidates = prefix_index.get(normalized[:6], set())
    matches = sorted(name for name in candidates if name in normalized or normalized in name)
    if len(matches) == 1:
        return "soft", matches
    if len(matches) > 1:
        return "ambiguous", matches
    return "none", []


def build_index(open_ee_dir: Path, report_path: Path) -> dict[str, object]:
    """Потоково сформировать split-группы по целым осциллограммам."""

    known_names = load_known_names(report_path)
    prefix_index: dict[str, set[str]] = {}
    for name in known_names:
        prefix_index.setdefault(name[:6], set()).add(name)

    entries: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    paths = sorted(open_ee_dir.glob("unlabeled_*.csv"))
    progress = ProgressReporter("real_OZZ exclusion", sum(path.stat().st_size for path in paths))
    completed_bytes = 0
    for csv_path in paths:
        row_count = 0
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                row_count += 1
                file_name = (row.get("file_name") or "").strip()
                key = (csv_path.name, file_name)
                if not file_name or key in seen:
                    continue
                seen.add(key)
                match_type, matches = match_name(file_name, known_names, prefix_index)
                entries.append(
                    {
                        "source_csv": csv_path.name,
                        "file_name": file_name,
                        "is_known_real_ozz": match_type in {"exact", "soft"},
                        "match_type": match_type,
                        "matched_report_names": matches,
                        "split_group": f"open_ee:{csv_path.name}:{file_name}",
                    }
                )
                if row_count % 100_000 == 0:
                    progress.update(completed_bytes + min(csv_path.stat().st_size, stream.buffer.tell()))
        completed_bytes += csv_path.stat().st_size
        progress.update(completed_bytes)
    progress.finish()

    excluded = [item for item in entries if item["is_known_real_ozz"]]
    ambiguous = [item for item in entries if item["match_type"] == "ambiguous"]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matching_policy": "exact; then containment only among identifiers with the same first 6 characters",
        "known_report_names": len(known_names),
        "records_total": len(entries),
        "records_excluded": len(excluded),
        "records_ambiguous": len(ambiguous),
        "entries": entries,
        "ambiguous_matches": ambiguous,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-ee-dir", type=Path, default=PROJECT_ROOT / "data/Open_EE_Dataset_v1_3_osc_CSV")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/phase5")
    args = parser.parse_args(argv)
    result = build_index(args.open_ee_dir, args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "real_ozz_exclusion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    no_ozz = [item for item in result["entries"] if not item["is_known_real_ozz"] and item["match_type"] != "ambiguous"]
    (args.output_dir / "open_ee_real_no_ozz_index.json").write_text(json.dumps({"schema_version": 1, "entries": no_ozz}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Всего: {result['records_total']}; исключено: {result['records_excluded']}; неоднозначно: {result['records_ambiguous']}")
    return 0


def run_manual() -> None:
    """Ручной запуск file-level exclusion через F5."""

    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА F5
    # Для CLI: python -m scripts.phase5_experiments.build_real_ozz_exclusion
    # =================================================================
    OPEN_EE_DIR = PROJECT_ROOT / "data/Open_EE_Dataset_v1_3_osc_CSV"
    REAL_OZZ_REPORT = PROJECT_ROOT / "data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv"
    OUTPUT_DIR = PROJECT_ROOT / "data/phase5"

    result = build_index(OPEN_EE_DIR, REAL_OZZ_REPORT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "real_ozz_exclusion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    no_ozz = [item for item in result["entries"] if not item["is_known_real_ozz"] and item["match_type"] != "ambiguous"]
    (OUTPUT_DIR / "open_ee_real_no_ozz_index.json").write_text(json.dumps({"schema_version": 1, "entries": no_ozz}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Всего: {result['records_total']}; исключено: {result['records_excluded']}; неоднозначно: {result['records_ambiguous']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
