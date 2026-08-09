"""Контракт PDRTaskDataset с Phase 5 spectral backbone."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from osc_tools.ml.phase5_contracts import ChannelProvenance, TimebaseContract
from osc_tools.ml.phase5_sources import DatasetSource
from osc_tools.pdr.pdr_dataset import PDRTaskDataset


class _SingleRecordSource(DatasetSource):
    name = "test"

    def __init__(self, signal: np.ndarray, spp: int) -> None:
        self.signal = signal
        self.spp = spp

    def __len__(self) -> int:
        return 1

    def get_metadata(self, idx: int) -> dict[str, object]:
        return {
            "f_adc": self.spp * 50,
            "f_network": 50,
            "spp": self.spp,
            "voltage_basis": "phase",
        }

    def load_signal(self, idx: int) -> np.ndarray:
        return self.signal.copy()

    def get_provenance(self, idx: int) -> np.ndarray:
        return np.asarray([
            ChannelProvenance.MEASURED,
            ChannelProvenance.MISSING,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MISSING,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MISSING,
        ], dtype=np.uint8)


def test_dataset_uses_full_context_last_point_and_derived_current(tmp_path: Path) -> None:
    spp = 24
    n_samples = 20 * spp
    t = np.arange(n_samples)
    signal = np.full((8, n_samples), np.nan, dtype=np.float32)
    amplitude = np.where(t < 10 * spp, 1.0, 5.0)
    a_phase = 2.0 * np.pi * t / spp
    signal[0] = amplitude * np.sin(a_phase)
    signal[2] = amplitude * np.sin(a_phase + 2.0 * np.pi / 3.0)
    signal[4] = np.sin(a_phase)
    signal[5] = np.sin(a_phase - 2.0 * np.pi / 3.0)
    signal[6] = np.sin(a_phase + 2.0 * np.pi / 3.0)

    labels = tmp_path / "labels.npz"
    np.savez_compressed(
        labels,
        rec_0_dir=np.asarray([1], dtype=np.int16),
        rec_0_margin=np.asarray([1.0], dtype=np.float32),
        rec_0_confidence=np.asarray([0.8], dtype=np.float32),
        rec_0_warmup=np.asarray([False]),
        rec_0_samples=np.asarray([n_samples - 1], dtype=np.int32),
        rec_0_prov=np.asarray(_SingleRecordSource(signal, spp).get_provenance(0)),
    )
    timebase = TimebaseContract.create(spp * 50, 50)
    dataset = PDRTaskDataset(
        _SingleRecordSource(signal, spp),
        [0],
        labels,
        timebase,
        temporal_mode="snapshot_5",
        feature_version="A",
    )

    sample = dataset[0]
    assert sample["features"].shape == (5, 220)
    assert sample["provenance"].shape == sample["features"].shape
    assert sample["channel_provenance"][1] == int(ChannelProvenance.DERIVED)
    # Первый snapshot относится к началу модельного окна, последний — к target point.
    assert sample["features"][0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert sample["features"][-1, 0].item() == pytest.approx(5.0, abs=1e-5)
    assert sample["pdr_confidence"].item() == pytest.approx(0.8)
