"""Строгий компактный модуль записи COMTRADE 1999 ASCII для проверки фазы 5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class AnalogChannel:
    name: str
    unit: str
    values: np.ndarray
    phase: str = ""
    circuit: str = ""


@dataclass(frozen=True)
class DigitalChannel:
    name: str
    values: np.ndarray
    normal_state: int = 0
    phase: str = ""
    circuit: str = ""


@dataclass(frozen=True)
class ExportRecord:
    station_name: str
    recorder_id: str
    sample_rate_hz: float
    network_frequency_hz: float
    start_datetime: datetime
    trigger_datetime: datetime
    analog: tuple[AnalogChannel, ...]
    digital: tuple[DigitalChannel, ...]
    cfg_encoding: str = "ascii"


def write_comtrade_ascii(record: ExportRecord, cfg_path: Path, dat_path: Path) -> None:
    """Атомарно записать воспроизводимую пару CFG/DAT с окончаниями CRLF."""

    cfg_path, dat_path = Path(cfg_path), Path(dat_path)
    if cfg_path.stem != dat_path.stem or cfg_path.parent != dat_path.parent:
        raise ValueError("CFG и DAT должны лежать рядом и иметь одинаковое базовое имя")
    n_samples = _validate(record)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex
    cfg_tmp = cfg_path.with_name(f".{cfg_path.name}.{suffix}.tmp")
    dat_tmp = dat_path.with_name(f".{dat_path.name}.{suffix}.tmp")
    try:
        _write_cfg(record, n_samples, cfg_tmp)
        _write_dat(record, n_samples, dat_tmp)
        dat_tmp.replace(dat_path)
        cfg_tmp.replace(cfg_path)
    except BaseException:
        cfg_tmp.unlink(missing_ok=True)
        dat_tmp.unlink(missing_ok=True)
        raise


def _validate(record: ExportRecord) -> int:
    if not math.isfinite(record.sample_rate_hz) or record.sample_rate_hz <= 0:
        raise ValueError("Частота дискретизации должна быть конечной и положительной")
    if record.trigger_datetime < record.start_datetime:
        raise ValueError("trigger_datetime не может быть раньше start_datetime")
    channels = (*record.analog, *record.digital)
    if not channels:
        raise ValueError("Нужен хотя бы один канал")
    lengths = {np.asarray(channel.values).size for channel in channels}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("Все каналы должны иметь одинаковую ненулевую длину")
    if record.cfg_encoding not in ("ascii", "cp1251", "utf-8"):
        raise ValueError("Неподдерживаемая кодировка CFG")
    names = [_clean(channel.name, record.cfg_encoding) for channel in channels]
    if len(names) != len(set(names)):
        raise ValueError("Имена каналов после очистки должны быть уникальны")
    for channel in record.analog:
        values = np.asarray(channel.values)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Аналоговый канал {channel.name!r} содержит NaN/Inf или имеет неверную форму")
    for channel in record.digital:
        values = np.asarray(channel.values)
        if values.ndim != 1 or not np.isin(values, (0, 1)).all():
            raise ValueError(f"Дискретный канал {channel.name!r} должен содержать только 0/1")
        if channel.normal_state not in (0, 1):
            raise ValueError("normal_state должен быть 0 или 1")
    return next(iter(lengths))


def _write_cfg(record: ExportRecord, n_samples: int, path: Path) -> None:
    clean = lambda value: _clean(value, record.cfg_encoding)
    analog_count, digital_count = len(record.analog), len(record.digital)
    lines = [
        f"{clean(record.station_name)},{clean(record.recorder_id)},1999",
        f"{analog_count + digital_count},{analog_count}A,{digital_count}D",
    ]
    for index, channel in enumerate(record.analog, start=1):
        values = np.asarray(channel.values, dtype=np.float64)
        minimum, maximum = float(values.min()), float(values.max())
        lines.append(
            f"{index},{clean(channel.name)},{clean(channel.phase)},"
            f"{clean(channel.circuit)},{clean(channel.unit)},1,0,0,"
            f"{minimum:.17g},{maximum:.17g},1,1,S"
        )
    for index, channel in enumerate(record.digital, start=1):
        lines.append(
            f"{index},{clean(channel.name)},{clean(channel.phase)},"
            f"{clean(channel.circuit)},{channel.normal_state}"
        )
    lines.extend((
        f"{record.network_frequency_hz:.12g}",
        "1",
        f"{record.sample_rate_hz:.12g},{n_samples}",
        f"{_format_date(record.start_datetime)},{_format_time(record.start_datetime)}",
        f"{_format_date(record.trigger_datetime)},{_format_time(record.trigger_datetime)}",
        "ASCII",
        "1",
    ))
    _write_crlf(path, lines, record.cfg_encoding)


def _write_dat(record: ExportRecord, n_samples: int, path: Path) -> None:
    analog = [np.asarray(channel.values, dtype=np.float64) for channel in record.analog]
    digital = [np.asarray(channel.values, dtype=np.uint8) for channel in record.digital]
    with path.open("w", encoding="ascii", newline="") as stream:
        for index in range(n_samples):
            timestamp_us = round(index * 1_000_000.0 / record.sample_rate_hz)
            fields = [str(index + 1), str(timestamp_us)]
            fields.extend(f"{values[index]:.17g}" for values in analog)
            fields.extend(str(int(values[index])) for values in digital)
            stream.write(",".join(fields) + "\r\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_crlf(path: Path, lines: list[str], encoding: str = "ascii") -> None:
    with path.open("w", encoding=encoding, newline="") as stream:
        stream.write("\r\n".join(lines) + "\r\n")
        stream.flush()
        os.fsync(stream.fileno())


def _clean(value: object, encoding: str = "ascii") -> str:
    text = str(value).encode(encoding, "replace").decode(encoding)
    return text.replace(",", "_").replace("\r", "_").replace("\n", "_").strip()


def _format_date(value: datetime) -> str:
    return value.strftime("%d/%m/%Y")


def _format_time(value: datetime) -> str:
    return value.strftime("%H:%M:%S.%f")
