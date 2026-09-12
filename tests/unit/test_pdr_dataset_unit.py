"""Контракт PDRTaskDataset с Phase 5 spectral backbone."""

import json
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
    assert sample["target_applicable"].item() is True


def test_dataset_reads_new_sharded_teacher_format(tmp_path: Path) -> None:
    spp = 24
    # Обе последние точки должны иметь полные 20 периодов причинного контекста.
    n_samples = 21 * spp
    t = np.arange(n_samples)
    signal = np.full((8, n_samples), np.nan, dtype=np.float32)
    phase = 2.0 * np.pi * t / spp
    for index, shift in zip((0, 2), (0.0, 2.0 * np.pi / 3.0)):
        signal[index] = np.sin(phase + shift)
    for index, shift in zip((4, 5, 6), (0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0)):
        signal[index] = np.sin(phase + shift)

    labels_dir = tmp_path / "sharded"
    labels_dir.mkdir()
    np.savez_compressed(
        labels_dir / "shard_00000.npz",
        record_ids=np.asarray([0], dtype=np.int32),
        offsets=np.asarray([0, 2], dtype=np.int64),
        samples=np.asarray([n_samples - 2, n_samples - 1], dtype=np.int32),
        directions=np.asarray([[-999, 1], [0, 0]], dtype=np.int16),
        teacher_margin=np.asarray([np.nan, 0.75], dtype=np.float32),
        teacher_confidence=np.asarray([0.0, 0.625], dtype=np.float16),
        teacher_warmup=np.asarray([False, False]),
        provenance=np.ones((1, 8), dtype=np.uint8),
    )
    (labels_dir / "manifest.json").write_text(json.dumps({
        "kind": "pdr_study_sharded",
        "algorithm_ids": ["adaptive_teacher", "comparison"],
        "teacher_algorithm_id": "adaptive_teacher",
        "shards": [{"file": "shard_00000.npz", "record_ids": [0]}],
    }), encoding="utf-8")

    dataset = PDRTaskDataset(
        _SingleRecordSource(signal, spp),
        [0],
        labels_dir,
        TimebaseContract.create(spp * 50, 50),
        temporal_mode="snapshot_5",
        feature_version="A",
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["pdr_direction"].item() == -999
    assert sample["target_applicable"].item() is False
    sample = dataset[1]
    assert sample["pdr_direction"].item() == 1
    assert sample["target_applicable"].item() is True
    assert sample["pdr_margin"].item() == pytest.approx(0.75)
    assert sample["pdr_confidence"].item() == pytest.approx(0.625)


def test_dataset_excludes_record_with_fewer_than_two_currents(tmp_path: Path) -> None:
    spp = 24
    signal = np.zeros((8, 20 * spp), dtype=np.float32)
    labels = tmp_path / "labels.npz"
    provenance = np.asarray([
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
        ChannelProvenance.MISSING,
        ChannelProvenance.MISSING,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
    ], dtype=np.uint8)
    np.savez_compressed(
        labels,
        rec_0_dir=np.asarray([-999], dtype=np.int16),
        rec_0_margin=np.asarray([0.0], dtype=np.float32),
        rec_0_warmup=np.asarray([False]),
        rec_0_samples=np.asarray([20 * spp - 1], dtype=np.int32),
        rec_0_prov=provenance,
    )

    source = _SingleRecordSource(signal, spp)
    source.get_provenance = lambda idx: provenance.copy()  # type: ignore[method-assign]
    dataset = PDRTaskDataset(source, [0], labels, TimebaseContract.create(spp * 50, 50))

    assert len(dataset) == 0
@pytest.mark.parametrize("mode", ["snapshot_2", "snapshot_5", "sequence_1_8"])
@pytest.mark.parametrize("spp", [12, 128])
def test_gallery_cached_features_match_training_windows(mode, spp):
    """Полнофайловый кеш галереи обязан совпадать с причинным окном обучения."""
    from scripts.visualization.generate_pdr_article_figures import _record_spectral_cache
    from osc_tools.ml.phase5_contracts import TimebaseContract, spectral_positions
    from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig
    rng = np.random.default_rng(20260910)
    raw = rng.normal(size=(8, 22 * spp)).astype(np.float32)
    raw[3] = np.nan
    provenance = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=np.uint8)
    tb = TimebaseContract.create(spp * 50, 50)
    ends = np.array([20 * spp - 1, 20 * spp + 2, 21 * spp - 1])
    cached, prov, lookup = _record_spectral_cache(raw, provenance, "phase", tb, mode, "B", ends)
    builder = SpectralFeatureBuilder(SpectralFeatureConfig("B"))
    for i, end in enumerate(ends):
        direct, mask, metadata = builder.build(
            raw[:, end - 20 * spp + 1:end + 1].T, spp,
            spectral_positions(20 * spp, spp, mode, history_periods=10),
            channel_provenance=provenance,
        )
        np.testing.assert_allclose(cached[lookup[i]], direct, atol=0, rtol=0)
        expected = np.broadcast_to(metadata["feature_provenance"], direct.shape).copy()
        expected[mask] = 0
        np.testing.assert_array_equal(prov[lookup[i]], expected)
def test_expert_gallery_preserves_unlabeled_for_uint8_comtrade():
    from scripts.visualization.generate_pdr_article_figures import _expert_state_track
    import numpy as np
    states = _expert_state_track(np.array([0, 1, 1], dtype=np.uint8), np.array([0, 0, 1], dtype=np.uint8))
    np.testing.assert_array_equal(states, [-999, 0, 1])
