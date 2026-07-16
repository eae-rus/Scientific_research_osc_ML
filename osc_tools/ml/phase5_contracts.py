"""Базовые контракты данных и времени для Phase 5.

Модуль не зависит от PyTorch и не меняет legacy pipeline Phase 4. Здесь
сосредоточены правила, которые должны одинаково применяться всеми adapters,
feature builders и task datasets Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Literal


CHANNEL_ORDER: tuple[str, ...] = (
    "IA",
    "IB",
    "IC",
    "IN",
    "UA",
    "UB",
    "UC",
    "UN",
)


class ChannelProvenance(IntEnum):
    """Происхождение физического канала в унифицированном представлении."""

    MISSING = 0
    MEASURED = 1
    DERIVED = 2


def round_samples(value: float) -> int:
    """Детерминированно округлить положительное число отсчётов вверх от .5.

    Встроенный ``round`` использует банковское округление, из-за чего шаги с
    половиной отсчёта могут чередовать направление округления между группами.
    Для временного контракта Phase 5 фиксируем правило ``floor(x + 0.5)``.
    """

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Число отсчётов должно быть положительным: {value!r}")
    return max(1, math.floor(value + 0.5))


def samples_per_period(sampling_rate_hz: float, network_frequency_hz: float) -> int:
    """Рассчитать целое число отсчётов на период сети."""

    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("Частота дискретизации должна быть положительной")
    if not math.isfinite(network_frequency_hz) or network_frequency_hz <= 0:
        raise ValueError("Частота сети должна быть положительной")
    return round_samples(sampling_rate_hz / network_frequency_hz)


def period_fraction_stride(spp: int, denominator: int = 8) -> int:
    """Получить шаг в отсчётах для заданной доли периода."""

    if spp <= 0:
        raise ValueError("SPP должен быть положительным")
    if denominator <= 0:
        raise ValueError("Знаменатель доли периода должен быть положительным")
    return round_samples(spp / denominator)


def periods_to_samples(periods: float, spp: int) -> int:
    """Перевести длительность в периодах в целое число отсчётов."""

    if spp <= 0:
        raise ValueError("SPP должен быть положительным")
    return round_samples(periods * spp)


def available_harmonics(spp: int, requested: int = 9) -> tuple[int, ...]:
    """Вернуть гармоники, доступные согласно пределу Найквиста.

    Гармоника с номером ``SPP / 2`` допускается для чётного SPP и соответствует
    частоте Найквиста. Feature builder позднее обязан сохранить mask для всех
    запрошенных, но недоступных гармоник.
    """

    if spp <= 0:
        raise ValueError("SPP должен быть положительным")
    if requested <= 0:
        raise ValueError("Число гармоник должно быть положительным")
    maximum = min(requested, spp // 2)
    return tuple(range(1, maximum + 1))


def snapshot_indices(first: int, last: int, count: int) -> tuple[int, ...]:
    """Построить равномерные snapshot-индексы с включёнными границами."""

    if first < 0 or last < first:
        raise ValueError("Некорректные границы snapshot-последовательности")
    if count < 2:
        raise ValueError("Snapshot-режим требует минимум две точки")
    if last - first + 1 < count:
        raise ValueError("Недостаточно уникальных позиций для snapshot-режима")

    span = last - first
    indices = tuple(
        first + math.floor((span * idx) / (count - 1) + 0.5)
        for idx in range(count)
    )
    if len(set(indices)) != count:
        raise RuntimeError("Округление создало повторяющиеся snapshot-индексы")
    return indices


TemporalMode = Literal["sequence_1_8", "snapshot_2", "snapshot_5"]


def spectral_positions(
    n_samples: int,
    spp: int,
    mode: TemporalMode,
    history_periods: float = 10.0,
    stride_fraction: int = 8,
) -> tuple[int, ...]:
    """Выбрать causal позиции спектральных токенов после предыстории.

    Позиция обозначает последний (включённый) отсчёт FFT-окна. Поэтому
    ``history_periods`` должны быть не меньше самого длинного low-period окна.
    Snapshot-режимы всегда включают первую и последнюю допустимые позиции.
    """

    if n_samples <= 0 or spp <= 0:
        raise ValueError("Размер сигнала и SPP должны быть положительными")
    first = periods_to_samples(history_periods, spp) - 1
    last = n_samples - 1
    if first > last:
        raise ValueError("Сигнал короче требуемой спектральной предыстории")
    if mode == "snapshot_2":
        return snapshot_indices(first, last, 2)
    if mode == "snapshot_5":
        return snapshot_indices(first, last, 5)
    if mode == "sequence_1_8":
        stride = period_fraction_stride(spp, stride_fraction)
        return tuple(range(first, last + 1, stride))
    raise ValueError(f"Неизвестный temporal mode: {mode!r}")


@dataclass(frozen=True)
class TimebaseContract:
    """Сериализуемые параметры временной сетки одной группы данных."""

    sampling_rate_hz: float
    network_frequency_hz: float
    spp: int
    window_periods: float
    window_samples: int
    stride_fraction: int
    stride_samples: int

    @classmethod
    def create(
        cls,
        sampling_rate_hz: float,
        network_frequency_hz: float,
        window_periods: float = 10.0,
        stride_fraction: int = 8,
    ) -> "TimebaseContract":
        """Создать контракт, используя единые правила округления."""

        spp = samples_per_period(sampling_rate_hz, network_frequency_hz)
        return cls(
            sampling_rate_hz=sampling_rate_hz,
            network_frequency_hz=network_frequency_hz,
            spp=spp,
            window_periods=window_periods,
            window_samples=periods_to_samples(window_periods, spp),
            stride_fraction=stride_fraction,
            stride_samples=period_fraction_stride(spp, stride_fraction),
        )

    @property
    def actual_stride_periods(self) -> float:
        """Фактический шаг после округления, в периодах сети."""

        return self.stride_samples / self.spp

    def to_metadata(self) -> dict[str, float | int]:
        """Вернуть JSON-совместимые metadata для окна/checkpoint."""

        return {
            "sampling_rate_hz": self.sampling_rate_hz,
            "network_frequency_hz": self.network_frequency_hz,
            "spp": self.spp,
            "window_periods": self.window_periods,
            "window_samples": self.window_samples,
            "stride_fraction": self.stride_fraction,
            "stride_samples": self.stride_samples,
            "actual_stride_periods": self.actual_stride_periods,
        }
