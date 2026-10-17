"""Тесты безопасной инспекции French/RTE контейнера."""

from pathlib import Path

import numpy as np

from scripts.phase5_experiments.scan_french_dataset import inspect_npz, scan_npy_rms


def test_compressed_npz_is_reported_as_not_mmap_safe(tmp_path: Path) -> None:
    path = tmp_path / "DATA_S.npz"
    np.savez_compressed(path, DATA_S=np.zeros((2, 6, 256), dtype=np.float64))
    metadata = inspect_npz(path)
    assert metadata["shape"] == [2, 6, 256]
    assert metadata["true_mmap_available"] is False
    assert "не обеспечивает mmap" in metadata["warning"]


def test_rms_scan_converts_quantized_values_to_physical_units(tmp_path: Path) -> None:
    path = tmp_path / "DATA_S.npy"
    data = np.ones((1, 6, 256), dtype=np.float64)
    np.save(path, data)
    report = scan_npy_rms(path)
    assert report["periods_per_record"] == 2
    assert report["period_rms_quantiles"]["channel_0"]["0.5"] == 18.310
    assert report["period_rms_quantiles"]["channel_3"]["0.5"] == 4.314
