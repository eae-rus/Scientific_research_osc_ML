"""Инспекция French/RTE DATA_S и RMS-скан mmap-доступной копии NPY."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

# При F5 VS Code может назначить cwd каталогом самого скрипта.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.lib import format as npy_format

from osc_tools.ml.phase5_contracts import TimebaseContract


def inspect_npz(path: Path) -> dict[str, object]:
    """Прочитать metadata NPY внутри ZIP без распаковки массива."""

    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1:
            raise ValueError(f"Ожидался один массив в {path}, найдено {len(members)}")
        member = members[0]
        with archive.open(member) as stream:
            version = npy_format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_1_0(stream)
            else:
                shape, fortran_order, dtype = npy_format.read_array_header_2_0(stream)
    compressed = member.compress_type != zipfile.ZIP_STORED
    return {
        "path": str(path),
        "member": member.filename,
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": fortran_order,
        "compression": "deflate" if compressed else "stored",
        "compressed_bytes": member.compress_size,
        "uncompressed_bytes": member.file_size,
        "true_mmap_available": not compressed,
        "warning": (
            "np.load(..., mmap_mode='r') не обеспечивает mmap для сжатого NPZ; "
            "для RMS-скана сначала нужен контролируемый extraction/benchmark"
            if compressed else None
        ),
    }


def scan_npy_rms(
    path: Path,
    max_records: int | None = None,
    batch_records: int = 8,
    quant_voltage: float = 18.310,
    quant_current: float = 4.314,
) -> dict[str, object]:
    """Посчитать периодные RMS по mmap-доступному массиву формы (N, 6, T)."""

    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 3 or data.shape[1] != 6:
        raise ValueError(f"Ожидалась форма (N, 6, T), получена {data.shape}")
    count = min(data.shape[0], max_records) if max_records is not None else data.shape[0]
    spp = 128
    period_count = data.shape[2] // spp
    collected: list[np.ndarray] = []
    scales = np.asarray([quant_voltage] * 3 + [quant_current] * 3, dtype=np.float64)
    for start in range(0, count, batch_records):
        chunk = np.asarray(data[start:min(start + batch_records, count), :, :period_count * spp])
        chunk = chunk.reshape(chunk.shape[0], 6, period_count, spp)
        rms = np.sqrt(np.mean(np.square(chunk, dtype=np.float64), axis=-1))
        collected.append(rms * scales[None, :, None])
    values = np.concatenate(collected, axis=(0)) if collected else np.empty((0, 6, period_count))
    quantiles = (0.1, 0.2, 0.5, 0.9, 0.95, 0.99, 1.0)
    return {
        "records_scanned": count,
        "periods_per_record": period_count,
        "units": ["V", "V", "V", "A", "A", "A"],
        "period_rms_quantiles": {
            f"channel_{idx}": {str(q): float(np.quantile(values[:, idx], q)) for q in quantiles}
            for idx in range(6)
        },
    }


def build_report(npz_path: Path, extracted_npy: Path | None = None, max_records: int | None = None) -> dict[str, object]:
    """Собрать структурный отчёт и, при наличии NPY, статистику RMS."""

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "French_RTE_DATA_S",
        "container": inspect_npz(npz_path),
        "timebase": TimebaseContract.create(6400.0, 50.0).to_metadata(),
        "channel_order_source": ["v1", "v2", "v3", "i1", "i2", "i3"],
        "quantization": {"voltage_v": 18.310, "current_a": 4.314},
        "voltage_nominal_v": 90000.0,
        "current_nominal_a": None,
        "normalization_status": "blocked_pending_researcher_current_nominal",
    }
    if extracted_npy is not None:
        report["rms_scan"] = scan_npy_rms(extracted_npy, max_records=max_records)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=PROJECT_ROOT / "data/digital-fault-recording-database/DATA_S.npz")
    parser.add_argument("--extracted-npy", type=Path)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "data/digital-fault-recording-database/french_scan.json")
    args = parser.parse_args()
    report = build_report(args.npz, args.extracted_npy, args.max_records)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["container"], ensure_ascii=False, indent=2))
    return 0


def run_manual() -> None:
    """Ручной запуск через F5: параметры редактируются только в этом блоке."""

    # =================================================================
    # РЕЖИМ РУЧНОГО ЗАПУСКА F5 (VS Code / PyCharm)
    # Для CLI: python -m scripts.phase5_experiments.scan_french_dataset
    #          --extracted-npy data/phase5/french_rte/DATA_S.npy
    # =================================================================
    SOURCE_NPZ = PROJECT_ROOT / "data/digital-fault-recording-database/DATA_S.npz"
    EXTRACTED_NPY = PROJECT_ROOT / "data/phase5/french_rte/DATA_S.npy"
    OUTPUT_JSON = PROJECT_ROOT / "data/digital-fault-recording-database/french_scan.json"
    # None = все 12053 записей. Для первой короткой проверки: 100.
    MAX_RECORDS: int | None = None

    report = build_report(SOURCE_NPZ, EXTRACTED_NPY, MAX_RECORDS)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["container"], ensure_ascii=False, indent=2))
    print(f"RMS рассчитан для записей: {report['rms_scan']['records_scanned']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    run_manual()
