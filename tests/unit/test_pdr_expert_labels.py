from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from osc_tools.io.comtrade_ascii import AnalogChannel, DigitalChannel, ExportRecord, write_comtrade_ascii
from osc_tools.pdr.base import PDRDirection
from osc_tools.pdr.expert_labels import (
    ALGORITHM_IDS,
    ImportedExpertRecord,
    aggregate_algorithm_comparison,
    build_transition_masks,
    import_expert_tree,
    read_comtrade_1999_ascii,
    write_expert_archive,
)
from osc_tools.pdr.study import PDRStudyLabelStore


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(n: int, expert_fwd: np.ndarray, expert_valid: np.ndarray, extra: bool) -> ExportRecord:
    x = np.arange(n, dtype=np.float64)
    analog = tuple(AnalogChannel(name, unit, x + index) for index, (name, unit) in enumerate(
        (("IA", "A"), ("IB", "A"), ("IC", "A"), ("UA", "V"), ("UB", "V"), ("UC", "V"))
    ))
    if extra:
        analog += (AnalogChannel("diagnostic", "V", x * 2),)
    digital = []
    for algorithm_id in ALGORITHM_IDS:
        digital.extend((
            DigitalChannel(f"{algorithm_id}__FWD", (x >= n // 2).astype(np.uint8)),
            DigitalChannel(f"{algorithm_id}__VALID", np.ones(n, dtype=np.uint8)),
        ))
    digital.extend((
        DigitalChannel("expert__FWD", expert_fwd),
        DigitalChannel("expert__VALID", expert_valid),
    ))
    return ExportRecord(
        "test", "record_1", 1000.0, 50.0,
        datetime(2000, 1, 1), datetime(2000, 1, 1), analog, tuple(digital),
    )


def test_transition_mask_includes_boundary_and_next_five_ms() -> None:
    fwd = np.zeros(20, dtype=np.uint8)
    valid = np.ones(20, dtype=np.uint8)
    fwd[10:] = 1
    train, transition = build_transition_masks(fwd, valid, 1000.0, 5.0)
    assert np.flatnonzero(transition).tolist() == [10, 11, 12, 13, 14, 15]
    assert np.array_equal(train, ~transition)


def test_manual_import_ignores_extra_channels_and_preserves_00(tmp_path: Path) -> None:
    review = tmp_path / "review" / "batch" / "+blind_control" / "open_ee"
    labels = tmp_path / "labels" / "completed" / "blind_control" / "open_ee"
    review.mkdir(parents=True)
    labels.mkdir(parents=True)
    stem = "open_ee__record_00001"
    n = 30
    zeros = np.zeros(n, dtype=np.uint8)
    reference_cfg, reference_dat = review / f"{stem}.cfg", review / f"{stem}.dat"
    write_comtrade_ascii(_record(n, zeros, zeros, False), reference_cfg, reference_dat)
    sidecar = {
        "kind": "phase5_pdr_manual_review_case", "schema_version": 1,
        "source": "open_ee", "record_id": 1, "input_sha256": "abc",
        "cfg": reference_cfg.name, "dat": reference_dat.name,
        "cfg_sha256": _hash(reference_cfg), "dat_sha256": _hash(reference_dat),
    }
    (review / f"{stem}.json").write_text(json.dumps(sidecar), encoding="utf-8")

    fwd = np.zeros(n, dtype=np.uint8)
    valid = np.zeros(n, dtype=np.uint8)
    fwd[10:] = 1
    valid[10:] = 1
    write_comtrade_ascii(_record(n, fwd, valid, True), labels / f"{stem}.cfg", labels / f"{stem}.dat")
    (labels / f"{stem}.json").write_text(json.dumps(sidecar), encoding="utf-8")

    records, audit = import_expert_tree(tmp_path / "labels", tmp_path / "review")
    assert len(records) == len(audit) == 1
    record = records[0]
    assert record.ignored_analog_channels == ("diagnostic",)
    assert np.all(record.directions[:10] == int(PDRDirection.UNLABELED))
    assert np.all(record.directions[10:] == int(PDRDirection.FORWARD))
    assert not record.train_mask[10:16].any()

    output = tmp_path / "expert"
    write_expert_archive(output, records, transition_ms=5.0)
    store = PDRStudyLabelStore(output / "open_ee")
    loaded = store.get_record(1)
    assert np.array_equal(loaded["directions"], record.directions)
    assert np.array_equal(loaded["train_mask"], record.train_mask)
    assert np.isnan(loaded["margins"]).all()


def test_invalid_quality_overrides_accidental_direction(tmp_path: Path) -> None:
    review = tmp_path / "review" / "batch" / "+phase_vs_sequence" / "open_ee"
    labels = tmp_path / "labels" / "invalid_quality" / "phase_vs_sequence" / "open_ee"
    review.mkdir(parents=True)
    labels.mkdir(parents=True)
    stem = "open_ee__record_00002"
    n = 12
    zeros = np.zeros(n, dtype=np.uint8)
    ones = np.ones(n, dtype=np.uint8)
    ref_cfg, ref_dat = review / f"{stem}.cfg", review / f"{stem}.dat"
    write_comtrade_ascii(_record(n, zeros, zeros, False), ref_cfg, ref_dat)
    sidecar = {
        "kind": "phase5_pdr_manual_review_case", "source": "open_ee", "record_id": 2,
        "input_sha256": "def", "cfg": ref_cfg.name, "dat": ref_dat.name,
        "cfg_sha256": _hash(ref_cfg), "dat_sha256": _hash(ref_dat),
    }
    (review / f"{stem}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    write_comtrade_ascii(_record(n, ones, ones, False), labels / f"{stem}.cfg", labels / f"{stem}.dat")
    (labels / f"{stem}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    records, _ = import_expert_tree(tmp_path / "labels", tmp_path / "review")
    assert not records[0].applicable.any()
    assert np.all(records[0].directions == int(PDRDirection.UNLABELED))


def test_ascii_reader_accepts_one_microsecond_timestamp_reformat(tmp_path: Path) -> None:
    n = 8
    zeros = np.zeros(n, dtype=np.uint8)
    cfg, dat = tmp_path / "x.cfg", tmp_path / "x.dat"
    record = _record(n, zeros, zeros, False)
    record = ExportRecord(
        record.station_name, record.recorder_id, 1200.0, record.network_frequency_hz,
        record.start_datetime, record.trigger_datetime, record.analog, record.digital,
    )
    write_comtrade_ascii(record, cfg, dat)
    parsed = read_comtrade_1999_ascii(cfg)
    assert parsed.n_samples == n


def test_algorithm_comparison_reports_record_distribution() -> None:
    def imported(record_id: int, automatic: np.ndarray) -> ImportedExpertRecord:
        n = len(automatic)
        expert = np.ones(n, dtype=np.int8)
        return ImportedExpertRecord(
            source="open_ee",
            record_id=record_id,
            status="completed",
            stratum="test",
            input_sha256=f"hash-{record_id}",
            f_adc=1000.0,
            directions=expert,
            applicable=np.ones(n, dtype=bool),
            train_mask=np.ones(n, dtype=bool),
            transition_eval_mask=np.zeros(n, dtype=bool),
            automatic_directions={algorithm_id: automatic for algorithm_id in ALGORITHM_IDS},
            ignored_analog_channels=(),
            ignored_digital_channels=(),
            cfg_path=Path(f"record-{record_id}.cfg"),
        )

    rows = aggregate_algorithm_comparison([
        imported(1, np.ones(4, dtype=np.int8)),
        imported(2, np.zeros(4, dtype=np.int8)),
    ])
    row = next(item for item in rows if item["group"] == "all")
    assert row["sample_accuracy"] == 0.5
    assert row["record_macro_accuracy"] == 0.5
    assert row["record_accuracy_std"] == np.sqrt(0.5)
    assert row["record_accuracy_q25"] == 0.25
    assert row["record_accuracy_median"] == 0.5
    assert row["record_accuracy_q75"] == 0.75
