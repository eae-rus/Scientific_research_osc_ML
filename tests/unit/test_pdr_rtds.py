"""Контракты предварительной RTDS-разметки и безопасного экспорта."""
from datetime import datetime
import numpy as np

from scripts.phase5_experiments.run_pdr_rtds import expert_seed, section_signals
from osc_tools.io.comtrade_ascii import AnalogChannel, DigitalChannel, ExportRecord, write_comtrade_ascii
from osc_tools.pdr.expert_labels import read_comtrade_1999_ascii, _read_text


def test_external_fault_uses_own_episode_and_breaker():
    fault = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    breaker = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    current = np.full((3, 9), 30.)
    state, _ = expert_seed(current, fault, breaker, internal_fault=False)
    assert state.tolist() == [1, 0, 0, 0, 1, 1, 1, 1, 1]
    internal, _ = expert_seed(current, fault, breaker, internal_fault=True)
    assert internal.all()
    current[:, 2] = 29.99
    current[:, 3] = np.nan
    internal, reason = expert_seed(current, fault, breaker, internal_fault=True)
    assert internal[2] == 0 and reason[2] == 1
    assert internal[3] == 1


def test_nominals_and_section_bus_voltage_not_input_voltage():
    physical = np.full((18, 4), 9000.)
    physical[:3] = 3000.
    physical[9:12] = 6000.
    physical[6:9] = 6000.
    physical[15:18] = 3000.
    first, prov = section_signals(physical, 1)
    second, _ = section_signals(physical, 2)
    np.testing.assert_allclose(first[:3], .05)
    np.testing.assert_allclose(second[:3], .1)
    np.testing.assert_allclose(first[4:7], 1/3)
    np.testing.assert_allclose(second[4:7], 1/6)
    assert np.isnan(first[3]).all() and prov.tolist() == [1, 1, 1, 0, 1, 1, 1, 0]


def test_cyrillic_comtrade_roundtrip(tmp_path):
    now = datetime(2026, 7, 2)
    rec = ExportRecord("RTDS", "0", 20000, 50, now, now,
        (AnalogChannel("Ток через ВВ1 ф.А", "A", np.array([1., 2.]), "A"),),
        (DigitalChannel("Положение ВВ1", np.array([0, 1])),), "cp1251")
    cfg = tmp_path / "example.cfg"
    write_comtrade_ascii(rec, cfg, cfg.with_suffix(".dat"))
    assert "Ток через ВВ1 ф.А,A," in _read_text(cfg)
    loaded = read_comtrade_1999_ascii(cfg)
    np.testing.assert_array_equal(loaded.analog["Ток через ВВ1 ф.А"], [1., 2.])
    np.testing.assert_array_equal(loaded.digital["Положение ВВ1"], [0, 1])


def test_gallery_atomic_save_preserves_previous_on_render_failure(tmp_path):
    from scripts.visualization.generate_pdr_article_figures import _save_gallery_png
    import pytest
    path = tmp_path / "image.png"
    path.write_bytes(b"old")
    class BrokenFigure:
        def savefig(self, *args, **kwargs):
            raise OSError("render failed")
    with pytest.raises(OSError):
        _save_gallery_png(BrokenFigure(), path)
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def test_verified_metrics_ignore_extra_signals_and_protect_automatic(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from dataclasses import replace
    import scripts.phase5_experiments.run_pdr_rtds as module
    now = datetime(2026, 7, 2)
    target = np.array([0, 1], dtype=np.uint8)
    digital = []
    for section in (1, 2):
        names = ["expert"] + list(module.DEFAULT_ALGORITHMS) + [f"nn_{s}_{m}_{k}" for s in ("weak", "expert") for m in module.MODES for k in ("last", "best")]
        for name in names:
            digital.extend((DigitalChannel(f"S{section}__{name}__FWD", target), DigitalChannel(f"S{section}__{name}__VALID", np.ones(2, dtype=np.uint8))))
    source = ExportRecord("RTDS", "0", 20000, 50, now, now, (AnalogChannel("IA", "A", np.array([1., 2.])),), tuple(digital))
    exported = replace(source, analog=source.analog + (AnalogChannel("MY_EXTRA", "A", np.array([99., 99.])),))
    cfg = tmp_path / "Oscilogramma1.1.cfg"
    write_comtrade_ascii(exported, cfg, cfg.with_suffix(".dat"))
    # Внешний редактор добавляет пробелы в CFG; единицы и имена не меняются.
    cfg.write_text(_read_text(cfg).replace(",", ", "), encoding="utf-8", newline="")
    cfg.with_suffix(".json").write_text(json.dumps({"automatic_digital_hashes": {
        ch.name: hashlib.sha256(ch.values.tobytes()).hexdigest() for ch in digital if "__expert__" not in ch.name}}))
    monkeypatch.setattr(module, "VERIFIED_ROOT", tmp_path)
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "review")
    monkeypatch.setattr(module, "read_rtds", lambda path: (source, None, None, None))
    result = module.evaluate_verified()
    assert result["n_verified_records"] == 1
    assert len(result["record_section_metrics"]) == 42
    assert all(r["state_accuracy"] == 1 for r in result["record_section_metrics"])
    changed = list(digital)
    changed[2] = replace(changed[2], values=np.zeros(2, dtype=np.uint8))
    write_comtrade_ascii(replace(exported, digital=tuple(changed)), cfg, cfg.with_suffix(".dat"))
    with pytest.raises(ValueError, match="Изменён неэкспертный"):
        module.evaluate_verified()


def test_comparison_metrics_do_not_reward_abstention_or_invalid_direction():
    from scripts.phase5_experiments.run_pdr_rtds import _comparison_metrics
    result = _comparison_metrics(np.array([0, 1, 1, 0]), np.array([1, 1, 0, 0]),
                                 np.array([0, 0, 1, 0]), np.array([1, 0, 0, 1]), 1000)
    assert result["direction_accuracy"] == 1
    assert result["direction_coverage"] == .5
    assert result["state_accuracy"] == .5
    assert result["state_confusion"] == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
    assert result["expert_transitions"] == 1


def test_comparison_metrics_long_errors_and_post_transition_exclusion():
    from scripts.phase5_experiments.run_pdr_rtds import _comparison_metrics
    target = np.r_[np.zeros(10), np.ones(30)].astype(np.uint8)
    pred = np.zeros(40, dtype=np.uint8)
    valid = np.ones(40, dtype=bool)
    result = _comparison_metrics(target, valid, pred, valid, 1000)
    assert result["max_error_episode_ms"] == 30
    assert result["error_episodes_over_20ms"] == 1
    assert result["stable_state_accuracy"] == 10/34


def test_h123_augmentation_preserves_reference_and_resumes(tmp_path, monkeypatch):
    import hashlib
    import json
    from dataclasses import replace
    from types import SimpleNamespace
    import pytest
    import scripts.phase5_experiments.run_pdr_rtds as m
    import scripts.visualization.generate_pdr_article_figures as gallery
    from osc_tools.ml.phase5_contracts import TimebaseContract
    source = tmp_path / "labels"
    output = tmp_path / "augmented"
    source.mkdir()
    cfg = source / "Oscilogramma1.1.cfg"
    now = datetime(2026, 9, 18)
    digital = [DigitalChannel("old_formula", np.array([0, 1, 1, 0], dtype=np.uint8))]
    for s in (1, 2):
        digital.extend([DigitalChannel(f"S{s}__expert__FWD", np.array([1, 0, 1, 0], dtype=np.uint8)),
                        DigitalChannel(f"S{s}__expert__VALID", np.array([1, 0, 0, 1], dtype=np.uint8))])
    record = ExportRecord("RTDS", "0", 100, 50, now, now,
                           (AnalogChannel("IA", "A", np.arange(4.)),), tuple(digital))
    write_comtrade_ascii(record, cfg, cfg.with_suffix(".dat"))
    cfg.with_suffix(".json").write_text(json.dumps({"automatic_digital_hashes": {
        "old_formula": hashlib.sha256(digital[0].values.tobytes()).hexdigest()}}))
    before = (cfg.read_bytes(), cfg.with_suffix(".dat").read_bytes())
    calls = []
    monkeypatch.setattr(m, "read_rtds", lambda p: (record, np.zeros((18, 4)), None, TimebaseContract.create(100, 50)))
    monkeypatch.setattr(m, "load_models", lambda **kw: {"new_model": (SimpleNamespace(temporal_mode="snapshot_5", feature_version="B_H123"), None, None, "cpu", {"sha256": "test"})})
    monkeypatch.setattr(gallery, "_record_spectral_cache", lambda raw, prov, basis, tb, mode, version, ends, **kw: (None, None, ends))
    def predict(model, head, f, p, lookup, *args):
        calls.append(len(lookup))
        return np.ones(len(lookup), dtype=np.int8), np.ones(len(lookup))
    monkeypatch.setattr(gallery, "_predict_cached", predict)
    m.augment_h123(source, output)
    edited = read_comtrade_1999_ascii(output / cfg.name)
    for ch in digital:
        np.testing.assert_array_equal(edited.digital[ch.name], ch.values)
    assert before == (cfg.read_bytes(), cfg.with_suffix(".dat").read_bytes())
    assert len(calls) == 2
    m.augment_h123(source, output)
    assert len(calls) == 2  # Уже готово: сеть не вызывается.
    with (output / cfg.name).open("a") as stream:
        stream.write("manual edit")
    with pytest.raises(ValueError, match="перезапись запрещена"):
        m.augment_h123(source, output)
