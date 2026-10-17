"""Малый benchmark чтения Open_EE prototype shards с реального диска."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def benchmark(directory: Path, repeats: int = 3) -> dict[str, object]:
    """Измерить cold-ish последовательное открытие и извлечение всех записей."""

    shards = sorted(directory.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"В {directory} не найдены shards")
    timings: list[float] = []
    samples = 0
    records = 0
    for _ in range(repeats):
        started = time.perf_counter()
        local_samples = 0
        local_records = 0
        for shard_path in shards:
            with np.load(shard_path, allow_pickle=False) as shard:
                signals = shard["signals"]
                offsets = shard["offsets"]
                for index in range(len(offsets) - 1):
                    # Материализуем срез как это сделает DatasetSource.
                    _ = signals[int(offsets[index]):int(offsets[index + 1])]
                    local_samples += int(offsets[index + 1] - offsets[index])
                    local_records += 1
        timings.append(time.perf_counter() - started)
        samples, records = local_samples, local_records
    size_bytes = sum(path.stat().st_size for path in shards)
    best = min(timings)
    return {
        "directory": str(directory), "shards": len(shards), "records": records,
        "samples": samples, "size_bytes": size_bytes, "repeats": repeats,
        "seconds": timings, "best_seconds": best,
        "records_per_second": records / best if best else None,
        "samples_per_second": samples / best if best else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=Path("reports/phase5/storage_benchmark.json"))
    args = parser.parse_args()
    result = {"results": [benchmark(directory, args.repeats) for directory in args.directories]}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in result["results"]:
        print(f"{item['directory']}: {item['records_per_second']:.1f} records/s, {item['size_bytes'] / 2**20:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
