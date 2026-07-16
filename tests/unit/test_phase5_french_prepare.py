"""Тесты воспроизводимого extraction French/RTE NPY."""

from pathlib import Path

import numpy as np
import pytest

from scripts.phase5_experiments.prepare_french_rte_npy import extract_npy


def test_extracts_mmap_readable_npy_without_changing_archive(tmp_path: Path) -> None:
    source = tmp_path / "DATA_S.npz"
    original = np.arange(24, dtype=np.float64).reshape(1, 6, 4)
    np.savez_compressed(source, DATA_S=original)
    archive_size = source.stat().st_size
    destination = tmp_path / "prepared" / "DATA_S.npy"

    result = extract_npy(source, destination)

    assert source.stat().st_size == archive_size
    assert result["random_access"] == "numpy_mmap"
    restored = np.load(destination, mmap_mode="r", allow_pickle=False)
    assert np.array_equal(restored, original)


def test_reuses_compatible_prepared_array(tmp_path: Path) -> None:
    source = tmp_path / "DATA_S.npz"
    np.savez_compressed(source, DATA_S=np.zeros((1, 6, 4)))
    destination = tmp_path / "DATA_S.npy"
    first = extract_npy(source, destination)
    second = extract_npy(source, destination)

    assert "reused_existing" not in first
    assert second["reused_existing"] is True


def test_refuses_to_overwrite_incompatible_prepared_array(tmp_path: Path) -> None:
    source = tmp_path / "DATA_S.npz"
    np.savez_compressed(source, DATA_S=np.zeros((1, 6, 4)))
    destination = tmp_path / "DATA_S.npy"
    np.save(destination, np.zeros((2, 6, 4)))

    with pytest.raises(FileExistsError):
        extract_npy(source, destination)
