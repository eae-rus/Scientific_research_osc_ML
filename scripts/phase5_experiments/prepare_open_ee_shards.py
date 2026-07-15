"""Собрать ограниченный или полный Open_EE shard layout для benchmark Phase 5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from osc_tools.ml.phase5_contracts import CHANNEL_ORDER, TimebaseContract
from osc_tools.ml.phase5_sources import adapt_open_ee_rows
from scripts.phase5_experiments.progress import ProgressReporter
from scripts.phase5_experiments.scan_open_ee_dataset import parse_frequencies


def iter_records(path: Path) -> Iterator[tuple[str, list[dict[str, str]]]]:
    """Последовательно выдать цельные осциллограммы; file_name обязан быть сгруппирован."""

    completed: set[str] = set()
    current_name: str | None = None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("file_name") or "").strip()
            if not name:
                continue
            if current_name is None:
                current_name = name
            if name != current_name:
                if name in completed:
                    raise ValueError(f"file_name {name!r} повторно встретился в {path.name}")
                completed.add(current_name)
                yield current_name, rows
                current_name, rows = name, []
            rows.append(row)
    if current_name is not None:
        yield current_name, rows


def write_shard(records: list[dict[str, object]], path: Path, compressed: bool) -> list[dict[str, object]]:
    """Записать flat signals + offsets без padding переменной длины."""

    offsets = [0]
    signal_parts: list[np.ndarray] = []
    manifest_entries: list[dict[str, object]] = []
    for local_index, record in enumerate(records):
        signal = record["signals"]
        signal_parts.append(signal)
        offsets.append(offsets[-1] + len(signal))
        manifest_entries.append(
            {key: value for key, value in record.items() if key not in {"signals", "provenance"}}
            | {"local_index": local_index}
        )
    saver = np.savez_compressed if compressed else np.savez
    saver(
        path,
        signals=np.concatenate(signal_parts, axis=0),
        offsets=np.asarray(offsets, dtype=np.int64),
        provenance=np.stack([record["provenance"] for record in records]),
        metadata_json=np.asarray(json.dumps(manifest_entries, ensure_ascii=False)),
        channel_order=np.asarray(CHANNEL_ORDER),
    )
    return manifest_entries


def prepare_shards(
    source_dir: Path, output_dir: Path, shard_records: int = 100,
    max_records: int | None = None, compressed: bool = False,
) -> dict[str, object]:
    """Создать shards; ``max_records`` предназначен для обязательного benchmark prototype."""

    if shard_records <= 0:
        raise ValueError("shard_records должен быть положительным")
    paths = sorted(source_dir.glob("unlabeled_*.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = max_records if max_records is not None else 44_773
    progress = ProgressReporter("Open_EE shards", target)
    buffer: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    written = 0
    shard_index = 0
    for csv_path in paths:
        f_network, f_adc = parse_frequencies(csv_path)
        timebase = TimebaseContract.create(f_adc, f_network)
        for file_name, rows in iter_records(csv_path):
            adapted = adapt_open_ee_rows(rows)
            buffer.append({
                "signals": adapted.signals, "provenance": adapted.provenance,
                "file_name": file_name, "source_csv": csv_path.name,
                "n_samples": len(rows), "f_network": f_network, "f_adc": f_adc,
                "spp": timebase.spp, "voltage_basis": adapted.voltage_basis,
                "source_columns": adapted.source_columns, "normalized": True,
            })
            if len(buffer) == shard_records or (max_records is not None and written + len(buffer) >= max_records):
                if max_records is not None:
                    buffer = buffer[:max_records - written]
                shard_path = output_dir / f"open_ee_{shard_index:05d}.npz"
                shard_entries = write_shard(buffer, shard_path, compressed)
                for entry in shard_entries:
                    entries.append(entry | {"shard_path": str(shard_path.relative_to(output_dir.parent))})
                written += len(buffer)
                progress.update(written)
                shard_index += 1
                buffer = []
                if max_records is not None and written >= max_records:
                    progress.finish()
                    return _manifest(output_dir, entries, compressed, shard_records)
    if buffer:
        shard_path = output_dir / f"open_ee_{shard_index:05d}.npz"
        shard_entries = write_shard(buffer, shard_path, compressed)
        entries.extend(entry | {"shard_path": str(shard_path.relative_to(output_dir.parent))} for entry in shard_entries)
        written += len(buffer)
        progress.update(written)
    progress.finish()
    return _manifest(output_dir, entries, compressed, shard_records)


def _manifest(output_dir: Path, entries: list[dict[str, object]], compressed: bool, shard_records: int) -> dict[str, object]:
    result = {"version": 1, "kind": "open_ee_sharded", "path": str(output_dir), "compressed": compressed,
              "shard_records": shard_records, "channel_order": list(CHANNEL_ORDER), "records": entries}
    (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data/Open_EE_Dataset_v1_3_osc_CSV")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/phase5/open_ee_shards_prototype")
    parser.add_argument("--shard-records", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()
    result = prepare_shards(args.source_dir, args.output_dir, args.shard_records, args.max_records, args.compressed)
    print(f"Подготовлено записей: {len(result['records'])}; shards: {len(list(args.output_dir.glob('*.npz')))}")
    return 0


if __name__ == "__main__":
    main()
