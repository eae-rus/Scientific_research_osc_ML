from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from osc_tools.io.comtrade_ascii import (
    AnalogChannel,
    DigitalChannel,
    ExportRecord,
    write_comtrade_ascii,
)


def _record() -> ExportRecord:
    start = datetime(2026, 8, 20, 12, 30)
    return ExportRecord(
        station_name="test",
        recorder_id="r1",
        sample_rate_hz=600.0,
        network_frequency_hz=50.0,
        start_datetime=start,
        trigger_datetime=start,
        analog=(AnalogChannel("IA", "A", np.asarray([1.0, -2.5, 3.0])),),
        digital=(DigitalChannel("PDR__VALID", np.asarray([0, 1, 1], dtype=np.uint8)),),
    )


def test_ascii_writer_uses_crlf_and_index_based_timestamps(tmp_path: Path) -> None:
    cfg, dat = tmp_path / "x.cfg", tmp_path / "x.dat"
    write_comtrade_ascii(_record(), cfg, dat)
    assert b"\r\n" in cfg.read_bytes()
    assert b"\r\n" in dat.read_bytes()
    rows = dat.read_text(encoding="ascii").splitlines()
    assert rows == ["1,0,1,0", "2,1667,-2.5,1", "3,3333,3,1"]
    assert "2,1A,1D" in cfg.read_text(encoding="ascii")


def test_ascii_writer_is_deterministic(tmp_path: Path) -> None:
    cfg, dat = tmp_path / "x.cfg", tmp_path / "x.dat"
    write_comtrade_ascii(_record(), cfg, dat)
    first = (cfg.read_bytes(), dat.read_bytes())
    write_comtrade_ascii(_record(), cfg, dat)
    assert first == (cfg.read_bytes(), dat.read_bytes())


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_ascii_writer_rejects_nonfinite_analog(tmp_path: Path, bad: float) -> None:
    record = _record()
    invalid = ExportRecord(**{
        **record.__dict__,
        "analog": (AnalogChannel("IA", "A", np.asarray([1.0, bad, 3.0])),),
    })
    with pytest.raises(ValueError, match="NaN/Inf"):
        write_comtrade_ascii(invalid, tmp_path / "x.cfg", tmp_path / "x.dat")
