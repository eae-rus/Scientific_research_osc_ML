"""Тесты потокового сканера Open_EE."""

import csv
from pathlib import Path

from scripts.phase5_experiments.scan_open_ee_dataset import scan_csv, scan_dataset


def _write_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "file_name", "IA", "UA BB", "UN BB"])
        writer.writerow([0, "record_a", 1.0, 10.0, ""])
        writer.writerow([1, "record_a", 3.0, 12.0, 0.0])
        writer.writerow([0, "record_b", -2.0, 8.0, "nan"])


def test_scan_csv_preserves_zero_and_counts_missing(tmp_path: Path) -> None:
    path = tmp_path / "unlabeled_50_600.csv"
    _write_csv(path)
    report = scan_csv(path, reservoir_size=10)

    assert report["row_count"] == 3
    assert report["record_count"] == 2
    assert report["record_lengths"] == {"record_a": 2, "record_b": 1}
    assert report["timebase"]["spp"] == 12
    assert report["available_harmonics"] == [1, 2, 3, 4, 5, 6]
    assert report["channel_stats"]["UN BB"]["valid_count"] == 1
    assert report["channel_stats"]["UN BB"]["min"] == 0.0


def test_scan_dataset_smoke_is_marked_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "unlabeled_60_1200.csv"
    _write_csv(path)
    report = scan_dataset(tmp_path, max_rows_per_file=2, reservoir_size=5)

    assert report["scan_complete"] is False
    assert report["files"][0]["row_count"] == 2
    assert report["files"][0]["timebase"]["spp"] == 20
