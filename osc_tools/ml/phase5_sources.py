"""Адаптеры реальных источников Phase 5 без зависимости от PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .phase5_contracts import CHANNEL_ORDER, ChannelProvenance


OPEN_EE_CHANNEL_PRIORITY: dict[str, tuple[str, ...]] = {
    "IA": ("IA",), "IB": ("IB",), "IC": ("IC",), "IN": ("IN",),
    "UA": ("UA BB", "UA CL"), "UB": ("UB BB", "UB CL"), "UC": ("UC BB", "UC CL"),
    "UN": ("UN BB", "UN CL"),
}
LINE_VOLTAGE_COLUMNS: tuple[str, ...] = ("UAB BB", "UBC BB", "UCA BB")


@dataclass(frozen=True)
class AdaptedOpenEERecord:
    """Одна осциллограмма в контракте Phase 5."""

    signals: np.ndarray
    provenance: np.ndarray
    voltage_basis: str
    source_columns: tuple[str | None, ...]


def adapt_open_ee_rows(rows: Sequence[Mapping[str, str]]) -> AdaptedOpenEERecord:
    """Привести строки одной Open_EE осциллограммы к (T, 8) float32.

    Линейные напряжения помещаются в voltage-slots только с ``basis='line'``;
    downstream feature builder обязан проверить basis до расчёта фазных ветвей.
    """

    if not rows:
        raise ValueError("Нельзя адаптировать пустую осциллограмму")
    columns = set(rows[0])
    phase_available = all(any(name in columns for name in OPEN_EE_CHANNEL_PRIORITY[key]) for key in ("UA", "UB", "UC"))
    line_available = all(name in columns for name in LINE_VOLTAGE_COLUMNS)
    voltage_basis = "phase" if phase_available else "line" if line_available else "missing"
    source_columns: list[str | None] = []
    values = np.full((len(rows), len(CHANNEL_ORDER)), np.nan, dtype=np.float32)
    provenance = np.full(len(CHANNEL_ORDER), int(ChannelProvenance.MISSING), dtype=np.uint8)

    for index, logical_name in enumerate(CHANNEL_ORDER):
        candidates = OPEN_EE_CHANNEL_PRIORITY[logical_name]
        source = next((name for name in candidates if name in columns), None)
        if logical_name in {"UA", "UB", "UC"} and not phase_available:
            source = LINE_VOLTAGE_COLUMNS[index - 4] if line_available else None
        source_columns.append(source)
        if source is None:
            continue
        channel = np.asarray([_as_float(row.get(source)) for row in rows], dtype=np.float32)
        values[:, index] = channel
        provenance[index] = int(ChannelProvenance.MEASURED)
    return AdaptedOpenEERecord(values, provenance, voltage_basis, tuple(source_columns))


def _as_float(value: str | None) -> float:
    """Сохранить пустое/некорректное значение как физически отсутствующее NaN."""

    if value is None or not value.strip():
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")
