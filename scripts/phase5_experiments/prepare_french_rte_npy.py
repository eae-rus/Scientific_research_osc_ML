"""Подготовить mmap-доступную копию French/RTE DATA_S без изменения архива."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
from numpy.lib import format as npy_format

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5_experiments.progress import ProgressReporter

def inspect_npz(path: Path) -> dict[str, object]:
    """Прочитать metadata единственного NPY member без импорта ML-пакета."""

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
    return {
        "member": member.filename,
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": fortran_order,
        "uncompressed_bytes": member.file_size,
    }


def _free_bytes(path: Path) -> int:
    """Вернуть доступное место на диске целевой директории."""

    return shutil.disk_usage(path).free


def extract_npy(source_npz: Path, destination: Path, overwrite: bool = False) -> dict[str, object]:
    """Извлечь единственный NPY member из NPZ в указанное место.

    Запись идёт во временный файл рядом с назначением. Исходный архив остаётся
    неизменным; при прерывании готовый путь не появляется.
    """

    metadata = inspect_npz(source_npz)
    if destination.exists() and not overwrite:
        existing = np.load(destination, mmap_mode="r", allow_pickle=False)
        try:
            valid = list(existing.shape) == metadata["shape"] and str(existing.dtype) == metadata["dtype"]
        finally:
            del existing
        if not valid:
            raise FileExistsError(
                "Подготовленный массив уже существует, но не совпадает с исходным NPZ. "
                "Укажите --overwrite только после проверки пути."
            )
        return {
            "source_npz": str(source_npz),
            "prepared_npy": str(destination),
            "bytes": destination.stat().st_size,
            "shape": metadata["shape"],
            "dtype": metadata["dtype"],
            "random_access": "numpy_mmap",
            "reused_existing": True,
        }
    required = int(metadata["uncompressed_bytes"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _free_bytes(destination.parent) < required * 2:
        raise OSError(
            "Недостаточно свободного места для безопасного extraction через временный файл: "
            f"нужно не менее {required * 2} байт"
        )

    with zipfile.ZipFile(source_npz) as archive:
        member = str(metadata["member"])
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
            with archive.open(member) as compressed:
                progress = ProgressReporter("French/RTE extraction", required)
                copied = 0
                while chunk := compressed.read(8 * 1024 * 1024):
                    temporary.write(chunk)
                    copied += len(chunk)
                    progress.update(copied)
                progress.finish()
        try:
            shutil.copystat(source_npz, temporary_path, follow_symlinks=True)
            loaded = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
            if list(loaded.shape) != metadata["shape"] or str(loaded.dtype) != metadata["dtype"]:
                raise ValueError("Извлечённый NPY не совпадает с metadata исходного NPZ")
            del loaded
            if destination.exists():
                destination.unlink()
            temporary_path.replace(destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    return {
        "source_npz": str(source_npz),
        "prepared_npy": str(destination),
        "bytes": destination.stat().st_size,
        "shape": metadata["shape"],
        "dtype": metadata["dtype"],
        "random_access": "numpy_mmap",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/digital-fault-recording-database/DATA_S.npz")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "data/phase5/french_rte/DATA_S.npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/phase5/french_rte/preparation_manifest.json")
    args = parser.parse_args()
    result = extract_npy(args.source, args.destination, args.overwrite)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
