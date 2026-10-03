"""Единая загрузка осциллограмм со строгой проверкой ошибок.

Время задаётся в секундах; аналоговые значения уже откалиброваны.
Форматы производителей используют необязательный адаптер OscFormats на .NET.
Для COMTRADE достаточно существующих зависимостей Python в проекте.
Подробности приведены в docs/OSCILLOGRAM_FORMATS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np
import polars as pl


class UnsupportedFormatError(ValueError):
    """Файл не относится к поддерживаемым осциллограммам; тренды обрабатываются отдельно."""


class OscillogramReadError(ValueError):
    """Парсер формата производителя отклонил запись или вернул некорректные данные."""


class MissingCompanionError(FileNotFoundError):
    """Связанный файл записи или конфигурации отсутствует либо найден неоднозначно."""


def detect_format(path: str | Path) -> str:
    """Определяет формат по имени; содержимое и связанные файлы проверяются при чтении."""
    path = Path(path)
    ext = path.suffix.lower()
    formats = {'.cfg': 'comtrade', '.dat': 'comtrade', '.cff': 'comtrade',
               '.do': 'parma', '.dfr': 'ekra-dfr', '.brs': 'bresler',
               '.bb': 'blackbox', '.sg2': 'res3'}
    if ext in formats:
        return formats[ext]
    if re.fullmatch(r'\.d0\d+', ext):
        return 'parma'
    if re.fullmatch(r'\.os\d*', ext):
        return 'neva'
    if re.fullmatch(r'DR\d+F[0-3]\.\d+', path.name, re.IGNORECASE):
        return 'ekra-ndr'
    raise UnsupportedFormatError(f'Unsupported oscillogram: {path.name}')


def _companion(path: Path, name: str) -> Path:
    matches = [p for p in path.parent.iterdir() if p.is_file() and p.name.casefold() == name.casefold()]
    if len(matches) != 1:
        raise MissingCompanionError(f'Expected one companion {name!r} beside {path}; found {len(matches)}')
    return matches[0]


def iter_oscillograms(root: str | Path):
    """Рекурсивно возвращает входные файлы без дублей DAT и продолжений ПАРМА.

    Части DFR/NDR остаются кандидатами: парсеры объединяют их по двоичным
    заголовкам, а не по именам файлов. Для устранения дублей составных записей
    используйте содержимое или идентификатор записи.
    Файлы WConfig и суточных трендов исключаются из входных файлов.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        try:
            detect_format(path)
        except UnsupportedFormatError:
            continue
        if path.suffix.lower() == '.dat' or re.fullmatch(r'\.d0\d+', path.suffix.lower()):
            continue
        yield path


@dataclass
class NativeOscillogram:
    """Запись производителя с интерфейсом чтения данных для существующих обработчиков.

    Метаданные каналов cfg сохраняют исходные коэффициенты A/B и признак P/S.
    Коэффициенты A/B уже применены к аналоговым массивам; повторная калибровка
    не требуется.
    """
    file_path: str
    format: str
    cfg: SimpleNamespace
    time: np.ndarray
    analog: list[np.ndarray]
    status: list[np.ndarray]
    recovery_requested: bool = False

    @property
    def total_samples(self):
        return len(self.time)

    @property
    def analog_channel_ids(self):
        return [c.name for c in self.cfg.analog_channels]

    @property
    def status_channel_ids(self):
        return [c.name for c in self.cfg.status_channels]

    @property
    def analog_count(self):
        return len(self.analog)

    @property
    def status_count(self):
        return len(self.status)

    @property
    def channels_count(self):
        return self.analog_count + self.status_count

    @property
    def frequency(self):
        return self.cfg.frequency

    @property
    def station_name(self):
        return self.cfg.station_name

    @property
    def rec_dev_id(self):
        return self.cfg.rec_dev_id

    @property
    def start_timestamp(self):
        return self.cfg.start_timestamp

    @property
    def trigger_timestamp(self):
        return self.cfg.trigger_timestamp

    def to_dataframe(self) -> pl.DataFrame:
        data = {'Time': self.time}
        for names, channels, label in ((self.analog_channel_ids, self.analog, 'analog'),
                                      (self.status_channel_ids, self.status, 'status')):
            for i, (name, values) in enumerate(zip(names, channels)):
                unique = name
                suffix = 0
                while unique in data:
                    unique = f'{name}_{label}_{i}_{suffix}'
                    suffix += 1
                data[unique] = values
        return pl.DataFrame(data)


def _read_bridge_output(folder: Path, source: Path, recovery: bool) -> NativeOscillogram:
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8-sig'))
    if metadata.pop('protocol') != 1:
        raise OscillogramReadError('Unsupported OscFormats protocol')
    count = metadata.pop('total_samples')
    kind = metadata.pop('format')
    na, nd = len(metadata['analog_channels']), len(metadata['status_channels'])
    if not isinstance(count, int) or count <= 0 or na + nd == 0:
        raise OscillogramReadError('Empty or invalid record')
    binary = folder / 'samples.bin'
    if binary.stat().st_size != count * (8 + 8 * na + nd):
        raise OscillogramReadError('Invalid bridge sample payload length')
    with binary.open('rb') as stream:
        time = np.fromfile(stream, dtype='<f8', count=count)
        analog = np.fromfile(stream, dtype='<f8', count=na * count).reshape(na, count)
        status = np.fromfile(stream, dtype='u1', count=nd * count).reshape(nd, count)
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise OscillogramReadError('Time must be finite and strictly increasing')
    if not np.all(np.isfinite(analog)) or np.any(status > 1):
        raise OscillogramReadError('Invalid channel samples')
    for key in ('start_timestamp', 'trigger_timestamp'):
        metadata[key] = datetime.fromisoformat(metadata[key])
    for key in ('analog_channels', 'status_channels'):
        metadata[key] = [SimpleNamespace(**channel) for channel in metadata[key]]
    metadata.update(analog_count=na, status_count=nd, channels_count=na + nd)
    return NativeOscillogram(str(source), kind, SimpleNamespace(**metadata), time,
                             list(analog), list(status), recovery)


def load_oscillogram(path: str | Path, *, wconfig: str | Path | None = None,
                     recover_incomplete: bool = False, bridge: str | Path | None = None,
                     timeout: float = 120):
    """Возвращает Comtrade или NativeOscillogram; при ошибке выбрасывает исключение.

    Путь к адаптеру по умолчанию: build/osc_formats/OscFormats.dll.
    Его можно переопределить через OSC_FORMATS_BRIDGE или аргумент ``bridge``.
    Во время чтения сборка и загрузка зависимостей не выполняются.
    NDR требует явно указанного WConfig либо WConfig.<номер> рядом с записью.
    Восстановление доступно только для НЕВА и РЭС-3 и сопровождается предупреждением.
    """
    source = Path(path).resolve(strict=True)
    kind = detect_format(source)
    if recover_incomplete and kind not in ('neva', 'res3'):
        raise ValueError('recover_incomplete is supported only for NEVA and RES-3')
    if kind == 'comtrade':
        from osc_tools.core.comtrade_custom import Comtrade
        rec = Comtrade()
        if source.suffix.lower() == '.cff':
            rec.load(str(source))
        else:
            cfg = _companion(source, source.stem + '.cfg')
            dat = _companion(source, source.stem + '.dat')
            rec.load(str(cfg), str(dat))
        return rec
    if kind == 'parma' and source.suffix.lower() != '.do':
        source = _companion(source, source.stem + '.do')
    config = None
    if kind == 'ekra-ndr':
        config = Path(wconfig).resolve(strict=True) if wconfig is not None else _companion(source, 'WConfig' + source.suffix)
    default = Path(__file__).resolve().parents[2] / 'build' / 'osc_formats' / 'OscFormats.dll'
    executable = Path(bridge or os.environ.get('OSC_FORMATS_BRIDGE', default)).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f'OscFormats bridge not built: {executable}. See docs/OSCILLOGRAM_FORMATS.md')
    command = [str(executable)]
    if executable.suffix.lower() == '.dll':
        dotnet = shutil.which('dotnet')
        if not dotnet:
            raise RuntimeError('Native formats require .NET 8 runtime (dotnet)')
        command.insert(0, dotnet)
    if recover_incomplete:
        warnings.warn('Incomplete final block recovery enabled; returned record may be truncated.',
                      UserWarning, stacklevel=2)
    with tempfile.TemporaryDirectory(prefix='osc-formats-') as temporary:
        command.extend([kind, str(source), temporary, str(recover_incomplete).lower()])
        if config is not None:
            command.append(str(config))
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=timeout, check=False)
        if result.returncode:
            raise OscillogramReadError(f'{source.name}: {result.stderr.strip()}')
        return _read_bridge_output(Path(temporary), source, recover_incomplete)


class ReadOscillogram:
    """Строгий загрузчик с возвратом кортежа для работы совместно с прежним ReadComtrade."""

    def read_oscillogram(self, file_name, **kwargs):
        record = load_oscillogram(file_name, **kwargs)
        return record, record.to_dataframe()
