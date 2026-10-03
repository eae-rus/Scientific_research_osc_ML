import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from osc_tools.io.oscillogram import (
    NativeOscillogram, OscillogramReadError, UnsupportedFormatError,
    _read_bridge_output, detect_format, iter_oscillograms, load_oscillogram,
)


@pytest.mark.parametrize('name,kind', [
    ('record.CFG', 'comtrade'), ('record.DAT', 'comtrade'), ('record.cff', 'comtrade'),
    ('record.DO', 'parma'), ('record.d01', 'parma'), ('record.d02', 'parma'),
    ('record.dfr', 'ekra-dfr'), ('record.BRS', 'bresler'), ('record.bb', 'blackbox'),
    ('record.sg2', 'res3'), ('record.os32', 'neva'), ('record.OS', 'neva'),
    ('DR177F1.046', 'ekra-ndr'), ('dr650f0.017', 'ekra-ndr'),
])
def test_detect(name, kind):
    assert detect_format(name) == kind


@pytest.mark.parametrize('name', ['WConfig.046', 'any.046', 'x.to', 'x.t0a', 'x.zip', 'x.csv'])
def test_non_oscillograms(name):
    with pytest.raises(UnsupportedFormatError):
        detect_format(name)


def test_discovery_keeps_entries_only(tmp_path):
    names = ['a.CFG', 'a.DAT', 'b.do', 'b.d01', 'DR177F0.046', 'WConfig.046', 'trend.to', 'c.OS32']
    for name in names:
        (tmp_path / name).touch()
    assert {p.name for p in iter_oscillograms(tmp_path)} == {'a.CFG', 'b.do', 'DR177F0.046', 'c.OS32'}


def test_ndr_requires_config_before_backend(tmp_path):
    source = tmp_path / 'DR177F0.046'
    source.touch()
    with pytest.raises(FileNotFoundError, match='WConfig.046'):
        load_oscillogram(source)


def test_missing_backend_is_actionable(tmp_path):
    source = tmp_path / 'record.bb'
    source.touch()
    with pytest.raises(FileNotFoundError, match='OSCILLOGRAM_FORMATS'):
        load_oscillogram(source, bridge=tmp_path / 'absent.dll')


def test_protocol_precision_and_corruption(tmp_path):
    metadata = dict(protocol=1, format='blackbox', total_samples=3,
                    analog_channels=[dict(name='IA', a=2, b=3, pors='P')],
                    status_channels=[dict(name='Trip')], frequency=50,
                    station_name='ПС', rec_dev_id='1', sample_rates=[[1000, 3]],
                    start_timestamp='2026-10-02T12:00:00', trigger_timestamp='2026-10-02T12:00:00.001')
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    time = np.array([0, .001, .002], dtype='<f8')
    values = np.array([1.000000000001, -3.5, 1e10 + .125], dtype='<f8')
    payload = time.tobytes() + values.tobytes() + bytes([0, 1, 1])
    binary = tmp_path / 'samples.bin'
    binary.write_bytes(payload)
    record = _read_bridge_output(tmp_path, Path('sample.bb'), False)
    np.testing.assert_array_equal(record.time, time)
    np.testing.assert_array_equal(record.analog[0], values)  # Без преобразования в float32 и повторной калибровки.
    np.testing.assert_array_equal(record.status[0], [0, 1, 1])
    assert record.to_dataframe().shape == (3, 3)
    assert record.cfg.analog_channels[0].pors == 'P'
    binary.write_bytes(payload[:-1])
    with pytest.raises(OscillogramReadError, match='length'):
        _read_bridge_output(tmp_path, Path('sample.bb'), False)


def test_duplicate_names_do_not_overwrite_samples():
    names = ['Time', 'Time_analog_0_0', 'Time']
    cfg = SimpleNamespace(analog_channels=[SimpleNamespace(name=n) for n in names],
                          status_channels=[SimpleNamespace(name='Time')])
    rec = NativeOscillogram('x', 'test', cfg, np.arange(2),
                            [np.full(2, i) for i in range(3)], [np.ones(2)])
    frame = rec.to_dataframe()
    assert frame.width == 5
    assert len(set(frame.columns)) == 5
    assert frame.to_numpy()[0].tolist() == [0, 0, 1, 2, 1]


def test_comtrade_pair_case_and_values(tmp_path):
    cfg = tmp_path / 'record.CFG'
    cfg.write_text('station,device,1999\n2,1A,1D\n1,IA,A,,A,2,1,0,-32767,32767,1,1,P\n'
                   '1,Trip,,,0\n50\n1\n1000,3\n02/10/2026,12:00:00.000000\n'
                   '02/10/2026,12:00:00.001000\nASCII\n1\n', encoding='ascii')
    dat = tmp_path / 'record.DAT'
    dat.write_text('1,0,2,0\n2,1000,-2,1\n3,2000,0,1\n', encoding='ascii')
    record = load_oscillogram(dat)
    np.testing.assert_allclose(record.analog[0], [5, -3, 1])
    np.testing.assert_allclose(record.time, [0, .001, .002])
    np.testing.assert_array_equal(record.status[0], [0, 1, 1])
