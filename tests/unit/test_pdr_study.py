import json

import numpy as np
import pytest

from osc_tools.ml.phase5_contracts import ChannelProvenance, TimebaseContract
from osc_tools.pdr.base import PDRAlgorithm, PDRDirection, PDROutput
from osc_tools.pdr.study import (
    MultiPDRRecordResult,
    PDRStudyLabelStore,
    label_record_multi,
    summarize_record_interest,
)


class _AlwaysForward(PDRAlgorithm):
    algorithm_id = "always_forward"

    def compute(self, input_data):
        return PDROutput(PDRDirection.FORWARD, True, 1.0)


class _HistoryReverse(PDRAlgorithm):
    algorithm_id = "history_reverse"
    requires_history = True
    tunable_parameters = {"history_periods": 10.0}

    def compute(self, input_data):
        assert input_data.history_phasors_i
        return PDROutput(PDRDirection.REVERSE, False, -1.0)


def test_multi_labeler_separates_public_start_and_required_history() -> None:
    timebase = TimebaseContract.create(800.0, 50.0)
    samples = 12 * timebase.spp
    t = np.arange(samples) / timebase.sampling_rate_hz
    signal = np.full((8, samples), np.nan, dtype=np.float32)
    for index, shift in zip((0, 1, 2), (0.0, -2 * np.pi / 3, 2 * np.pi / 3)):
        signal[index] = np.sin(2 * np.pi * 50 * t + shift)
    for index, shift in zip((4, 5, 6), (0.0, -2 * np.pi / 3, 2 * np.pi / 3)):
        signal[index] = np.sin(2 * np.pi * 50 * t + shift)
    provenance = np.asarray([
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MEASURED,
        ChannelProvenance.MISSING,
    ], dtype=np.uint8)

    result = label_record_multi(
        signal,
        provenance,
        timebase,
        "phase",
        [_AlwaysForward(), _HistoryReverse()],
    )

    assert len(result.sample_indices) == samples - timebase.spp + 1
    assert np.all(np.diff(result.sample_indices) == 1)
    assert np.all(result.directions[0] == int(PDRDirection.FORWARD))
    assert np.any(result.warmup_mask[1])
    assert np.all(result.directions[1, result.warmup_mask[1]] == int(PDRDirection.UNLABELED))
    assert np.any(result.directions[1] == int(PDRDirection.REVERSE))


def test_interest_score_downranks_constant_threshold_disagreement() -> None:
    timebase = TimebaseContract.create(800.0, 50.0)
    samples = np.arange(40, dtype=np.int32)
    constant = MultiPDRRecordResult(
        algorithm_ids=("a", "b"),
        sample_indices=samples,
        directions=np.stack((np.zeros(40, dtype=np.int16), np.ones(40, dtype=np.int16))),
        margins=np.ones((2, 40), dtype=np.float32),
        confidences=np.ones((2, 40), dtype=np.float32),
        warmup_mask=np.zeros((2, 40), dtype=bool),
        provenance=np.ones(8, dtype=np.uint8),
        input_sha256="constant",
    )
    localized_directions = constant.directions.copy()
    localized_directions[1, :20] = 0
    localized = MultiPDRRecordResult(
        algorithm_ids=constant.algorithm_ids,
        sample_indices=samples,
        directions=localized_directions,
        margins=constant.margins,
        confidences=constant.confidences,
        warmup_mask=constant.warmup_mask,
        provenance=constant.provenance,
        input_sha256="localized",
    )

    constant_stats = summarize_record_interest(constant, timebase)
    localized_stats = summarize_record_interest(localized, timebase)

    assert constant_stats["static_threshold_disagreement"] is True
    assert localized_stats["localized_disagreement"] > constant_stats["localized_disagreement"]
    assert localized_stats["interest_score"] > constant_stats["interest_score"]


def test_sharded_label_store_reads_teacher_and_other_algorithm(tmp_path) -> None:
    np.savez(
        tmp_path / "shard_00000.npz",
        record_ids=np.asarray([7], dtype=np.int32),
        offsets=np.asarray([0, 3], dtype=np.int64),
        samples=np.asarray([10, 12, 14], dtype=np.int32),
        directions=np.asarray([[1, 0, 1], [0, 0, 1]], dtype=np.int16),
        all_margins=np.asarray([[0.5, -0.2, 0.8], [-0.4, -0.3, 0.2]], dtype=np.float32),
        all_confidences=np.asarray([[1.0, 0.5, 1.0], [0.7, 0.8, 0.9]], dtype=np.float32),
        all_warmup=np.zeros((2, 3), dtype=bool),
        provenance=np.ones((1, 8), dtype=np.uint8),
    )
    manifest = {
        "kind": "pdr_study_sharded",
        "algorithm_ids": ["teacher", "comparison"],
        "teacher_algorithm_id": "teacher",
        "shards": [{"file": "shard_00000.npz", "record_ids": [7]}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    teacher = PDRStudyLabelStore(tmp_path)
    teacher_record = teacher.get_record(7)
    assert teacher_record["directions"].tolist() == [1, 0, 1]
    assert teacher_record["margins"].tolist() == pytest.approx([0.5, -0.2, 0.8])
    teacher.close()

    comparison = PDRStudyLabelStore(tmp_path, algorithm_id="comparison")
    comparison_record = comparison.get_record(7)
    assert comparison_record["directions"].tolist() == [0, 0, 1]
    assert comparison_record["margins"].tolist() == pytest.approx([-0.4, -0.3, 0.2])
    assert comparison_record["confidences"].tolist() == pytest.approx([0.7, 0.8, 0.9])
    comparison.close()
