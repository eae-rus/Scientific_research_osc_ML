"""Потоковая инвентаризация Open_EE без загрузки CSV целиком в память."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Iterable

# Поддерживаем прямой F5 независимо от cwd, заданного IDE.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.ml.phase5_contracts import TimebaseContract, available_harmonics


FILE_PATTERN = re.compile(r"^unlabeled_(?P<network>\d+)_(?P<sampling>\d+)\.csv$")
IDENTIFIER_COLUMNS = {"sample", "file_name"}


@dataclass
class StreamingStats:
    """Численно устойчивые статистики и ограниченный reservoir для квантилей."""

    reservoir_size: int = 20_000
    seed: int = 42
    count: int = 0
    missing: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    reservoir: list[float] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def update(self, raw_value: str | None) -> None:
        if raw_value is None or not raw_value.strip():
            self.missing += 1
            return
        try:
            value = float(raw_value)
        except ValueError:
            self.missing += 1
            return
        if not math.isfinite(value):
            self.missing += 1
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(value)
        else:
            index = self._rng.randrange(self.count)
            if index < self.reservoir_size:
                self.reservoir[index] = value

    def as_dict(self, total_rows: int) -> dict[str, object]:
        values = sorted(self.reservoir)

        def quantile(q: float) -> float | None:
            if not values:
                return None
            pos = math.floor((len(values) - 1) * q + 0.5)
            return values[pos]

        return {
            "valid_count": self.count,
            "missing_count": self.missing,
            "nonempty_fraction": self.count / total_rows if total_rows else 0.0,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
            "mean": self.mean if self.count else None,
            "std": math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0,
            "quantiles_approx": {str(q): quantile(q) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "quantile_sample_size": len(values),
        }


def parse_frequencies(path: Path) -> tuple[int, int]:
    """Извлечь частоты сети и дискретизации из имени Open_EE CSV."""

    match = FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Неожиданное имя Open_EE файла: {path.name}")
    return int(match.group("network")), int(match.group("sampling"))


def scan_csv(
    path: Path,
    max_rows: int | None = None,
    reservoir_size: int = 20_000,
) -> dict[str, object]:
    """Просканировать один CSV последовательно, сохранив только агрегаты."""

    network_hz, sampling_hz = parse_frequencies(path)
    timebase = TimebaseContract.create(sampling_hz, network_hz)
    lengths: dict[str, int] = {}
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        signal_columns = tuple(column for column in columns if column not in IDENTIFIER_COLUMNS)
        stats = {
            column: StreamingStats(reservoir_size=reservoir_size, seed=42 + idx)
            for idx, column in enumerate(signal_columns)
        }
        for row in reader:
            if max_rows is not None and row_count >= max_rows:
                break
            row_count += 1
            file_name = (row.get("file_name") or "").strip()
            if file_name:
                lengths[file_name] = lengths.get(file_name, 0) + 1
            for column, accumulator in stats.items():
                accumulator.update(row.get(column))

    length_values = sorted(lengths.values())
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "scan_complete": max_rows is None,
        "max_rows": max_rows,
        "row_count": row_count,
        "record_count": len(lengths),
        "record_lengths": lengths,
        "record_length_min": min(length_values) if length_values else None,
        "record_length_max": max(length_values) if length_values else None,
        "columns": list(columns),
        "signal_columns": list(signal_columns),
        "network_frequency_hz": network_hz,
        "sampling_rate_hz": sampling_hz,
        "timebase": timebase.to_metadata(),
        "available_harmonics": list(available_harmonics(timebase.spp)),
        "normalized": True,
        "normalization_note": "Open_EE already normalized; norm_coef_all_v1.4.csv must not be applied again",
        "channel_stats": {name: value.as_dict(row_count) for name, value in stats.items()},
    }


def scan_dataset(
    dataset_dir: Path,
    max_rows_per_file: int | None = None,
    reservoir_size: int = 20_000,
) -> dict[str, object]:
    """Просканировать все Open_EE CSV в детерминированном порядке."""

    paths = sorted(dataset_dir.glob("unlabeled_*.csv"))
    if not paths:
        raise FileNotFoundError(f"В {dataset_dir} не найдены unlabeled_*.csv")
    files = [scan_csv(path, max_rows_per_file, reservoir_size) for path in paths]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Open_EE_Dataset_v1_3_osc_CSV",
        "scan_complete": max_rows_per_file is None,
        "file_count": len(files),
        "files": files,
    }


def render_markdown(report: dict[str, object]) -> str:
    """Сформировать компактный человекочитаемый отчёт."""

    lines = [
        "# Инвентаризация Open_EE",
        "",
        f"Файлов: {report['file_count']}",
        "",
        "| Файл | Строк | Записей | SPP | Гармоники |",
        "|---|---:|---:|---:|---|",
    ]
    for item in report["files"]:
        harmonics = ",".join(map(str, item["available_harmonics"]))
        lines.append(f"| {item['file']} | {item['row_count']} | {item['record_count']} | {item['timebase']['spp']} | {harmonics} |")
    lines.extend(["", "> Open_EE уже нормализован; повторное применение norm_coef_all_v1.4.csv запрещено.", ""])
    return "\n".join(lines)


def write_report(report: dict[str, object], json_path: Path, markdown_path: Path) -> None:
    """Атомарно на уровне файла сохранить JSON и Markdown результаты."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/Open_EE_Dataset_v1_3_osc_CSV")
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "data/Open_EE_Dataset_v1_3_osc_CSV/open_ee_scan.json")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "reports/phase5/open_ee_scan.md")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--reservoir-size", type=int, default=20_000)
    parser.add_argument("--smoke", action="store_true", help="Прочитать не более 1000 строк каждого файла")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_rows = 1000 if args.smoke else args.max_rows_per_file
    report = scan_dataset(args.dataset_dir, max_rows, args.reservoir_size)
    write_report(report, args.json, args.markdown)
    print(f"Просканировано файлов: {report['file_count']}; полный проход: {report['scan_complete']}")
    return 0


def run_manual() -> None:
    """Ручной запуск полного/ограниченного Open_EE scan через F5."""

    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА F5
    # Для CLI: python -m scripts.phase5_experiments.scan_open_ee_dataset
    # =================================================================
    DATASET_DIR = PROJECT_ROOT / "data/Open_EE_Dataset_v1_3_osc_CSV"
    OUTPUT_JSON = DATASET_DIR / "open_ee_scan.json"
    OUTPUT_MARKDOWN = PROJECT_ROOT / "reports/phase5/open_ee_scan.md"
    # None = полный проход; 1000 = короткая проверка каждого CSV.
    MAX_ROWS_PER_FILE: int | None = None
    RESERVOIR_SIZE = 20_000

    report = scan_dataset(DATASET_DIR, MAX_ROWS_PER_FILE, RESERVOIR_SIZE)
    write_report(report, OUTPUT_JSON, OUTPUT_MARKDOWN)
    print(f"Просканировано файлов: {report['file_count']}; полный проход: {report['scan_complete']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
