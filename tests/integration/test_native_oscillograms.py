"""Необязательные интеграционные проверки по независимым сохранённым экспортам производителей.

Укажите APScilloscope/Tests/ExternalFormats.Parity/Corpus в APS_FORMAT_CORPUS
и предварительно соберите адаптер. Исходные пользовательские записи и архивы
не копируются в git.
"""
import os
import csv
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from osc_tools.io.oscillogram import iter_oscillograms, load_oscillogram, OscillogramReadError


def vendor_reference(folder, kind):
    """Независимо читает сохранённые экспорты с учётом особенностей CFG производителей.

    Вспомогательная функция не использует двоичные декодеры форматов производителей.
    РЭС-3 экспортирует в OEM866, остальные производители — в CP1251.
    В прежних вариантах CFG строки аналоговых и дискретных каналов содержат
    соответственно 10 и 3 поля. Etc.exe может использовать десятичную запятую
    в строке единственной частоты дискретизации.
    """
    with zipfile.ZipFile(folder / 'Reference.zip') as archive:
        cfg_name, = [n for n in archive.namelist() if n.lower().endswith('.cfg')]
        dat_name, = [n for n in archive.namelist() if n.lower().endswith('.dat')]
        text = archive.read(cfg_name).decode('cp866' if kind == 'res3' else 'cp1251')
        payload = archive.read(dat_name)
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    na, nd = int(rows[1][1][:-1]), int(rows[1][2][:-1])
    analog = rows[2:2 + na]
    digital = rows[2 + na:2 + na + nd]
    # WinBres записывает запятую без кавычек внутри имени одного дискретного канала BRS.
    if kind == 'bresler':
        digital = [r if len(r) == 5 else [r[0], ','.join(r[1:-3]), *r[-3:]] for r in digital]
    offset = 2 + na + nd
    assert rows[offset + 1] == ['1']
    rate = rows[offset + 2]
    if kind == 'ekra-ndr' and len(rate) == 3:
        assert len(rate[1]) == 6 and all(p.isdigit() for p in rate)
        rate = [rate[0] + '.' + rate[1], rate[2]]
    frequency, count = float(rate[0]), int(rate[1])
    encoding = rows[offset + 5][0].strip()
    if encoding == 'ASCII':
        values = np.loadtxt(io.BytesIO(payload.rstrip(b'\x1a\r\n \t')), delimiter=',', ndmin=2)
        raw, status = values[:, 2:2 + na].T, values[:, 2 + na:].T
    else:
        assert encoding == 'BINARY'
        dtype = np.dtype([('n', '<u4'), ('time', '<u4'), ('analog', '<i2', na),
                          ('digital', '<u2', (nd + 15) // 16)])
        values = np.frombuffer(payload, dtype=dtype)
        raw = values['analog'].T
        status = np.array([(values['digital'][:, i // 16] >> (i % 16)) & 1 for i in range(nd)])
    assert raw.shape == (na, count) and status.shape == (nd, count)
    calibrated = np.array([v * float(row[5]) + float(row[6]) for v, row in zip(raw, analog)])
    return analog, digital, np.arange(count) / frequency, calibrated, status


@pytest.fixture
def corpus():
    value = os.environ.get('APS_FORMAT_CORPUS')
    if not value:
        pytest.skip('Set APS_FORMAT_CORPUS to run independent vendor comparisons')
    path = Path(value)
    assert path.is_dir(), f'Missing corpus: {path}'
    return path


@pytest.mark.parametrize('case', [
    'parma-17204f0f27', 'ekra-dfr-226626dd23', 'bresler-a504a36c6a',
    'ekra-ndr-065c8f74bb', 'res3-451ad72380', 'blackbox-bd42857269',
])
def test_frozen_vendor_samples(corpus, case):
    folder = corpus / case
    source = next(iter_oscillograms(folder / 'Input'))
    native = load_oscillogram(source)
    analog, digital, time, samples, status = vendor_reference(folder, native.format)
    assert native.total_samples == len(time)
    assert native.analog_channel_ids == [row[1] for row in analog]
    assert native.status_channel_ids == [row[1] for row in digital]
    np.testing.assert_allclose(native.time, time, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(native.status, status)
    for actual, expected, meta, ref in zip(native.analog, samples,
                                          native.cfg.analog_channels, analog):
        # Сравниваем значения в одной шкале первичных или вторичных величин.
        # Сохранённые экспорты BRS/DFR могут менять шкалу.
        # Эталонный экспорт дополнительно квантует отсчёты АЦП.
        factor = 1.0
        ps = ref[12] if len(ref) == 13 else 'S'
        if meta.pors != ps:
            factor = meta.primary / meta.secondary if meta.pors == 'S' else meta.secondary / meta.primary
        if native.format == 'blackbox':
            # bb2cmt повторно квантует отсчёты АЦП и округляет A/B до шести значащих цифр.
            # Парсер формата производителя сохраняет исходные значения с плавающей точкой.
            low, high = min(0, meta.min), max(.000001, meta.max)
            adc = np.floor((actual - low) * (900000 / (high - low)) + .5)
            exported = adc * float(ref[5]) + float(ref[6])
            np.testing.assert_allclose(exported, expected, rtol=0, atol=abs(float(ref[5])) * 1.1)
            continue
        np.testing.assert_allclose(actual * factor, expected,
                                   rtol=3e-6, atol=abs(float(ref[5])) * 1.1 + 1e-6)


def test_dfr_any_part_is_same_record(corpus):
    parts = list(iter_oscillograms(corpus / 'ekra-dfr-226626dd23' / 'Input'))
    assert len(parts) == 2
    first, second = map(load_oscillogram, parts)
    np.testing.assert_array_equal(first.time, second.time)
    np.testing.assert_array_equal(first.analog, second.analog)
    np.testing.assert_array_equal(first.status, second.status)


def test_res3_recovery_is_explicit(corpus):
    source = next(iter_oscillograms(corpus / 'res3-1192eef1aa' / 'Input'))
    with pytest.raises(OscillogramReadError):
        load_oscillogram(source)
    with pytest.warns(UserWarning, match='truncated'):
        record = load_oscillogram(source, recover_incomplete=True)
    assert record.total_samples > 0
    assert record.recovery_requested


def test_neva_recovery_keeps_whole_frames(corpus):
    source = next(iter_oscillograms(corpus / 'neva-35ca3dfe29' / 'Input'))
    with pytest.warns(UserWarning):
        record = load_oscillogram(source, recover_incomplete=True)
    assert record.total_samples > 0
    assert np.all(np.diff(record.time) > 0)
    assert all(len(a) == record.total_samples for a in record.analog + record.status)


def test_legacy_reader_accepts_native_record(corpus):
    from osc_tools.data_management.comtrade_processing import ReadComtrade
    source = next(iter_oscillograms(corpus / 'bresler-a504a36c6a' / 'Input'))
    record, frame = ReadComtrade().read_comtrade(source)
    assert record.format == 'bresler'
    assert frame.shape == (record.total_samples, 1 + record.channels_count)


def test_corrupt_native_file_is_rejected(corpus, tmp_path):
    source = next(iter_oscillograms(corpus / 'blackbox-bd42857269' / 'Input'))
    broken = tmp_path / 'broken.bb'
    broken.write_bytes(source.read_bytes()[:100])
    with pytest.raises(OscillogramReadError):
        load_oscillogram(broken)
