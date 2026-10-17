"""Тесты file-level исключения известных real_OZZ."""

import csv
from pathlib import Path

from scripts.phase5_experiments.build_real_ozz_exclusion import build_index


def test_exclusion_uses_exact_and_safe_soft_matching(tmp_path: Path) -> None:
    report = tmp_path / "report.csv"
    report.write_text("filename;overvoltage\nabc123def;1\nzzz999;1\n", encoding="utf-8")
    source = tmp_path / "unlabeled_50_1600.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "file_name", "IA"])
        writer.writerow([0, "abc123def", 0])
        writer.writerow([1, "abc123def", 0])
        writer.writerow([0, "abc123def_section2", 0])
        writer.writerow([0, "ordinary", 0])

    result = build_index(tmp_path, report)

    assert result["records_total"] == 3
    assert result["records_excluded"] == 2
    assert [entry["match_type"] for entry in result["entries"]] == ["exact", "soft", "none"]
