"""Versioned spectral feature contracts Phase 5, independent of legacy Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from osc_tools.features.phasor import (
    calculate_symmetrical_components,
    calculate_symmetrical_components_from_line,
)
from .phase5_contracts import CHANNEL_ORDER, ChannelProvenance, available_harmonics


FeatureVersion = Literal["A", "B"]
SYMMETRIC_NAMES = ("I1", "I2", "I0", "U1", "U2", "U0")


@dataclass(frozen=True)
class SpectralFeatureConfig:
    """Сериализуемый feature contract v2 без неявных legacy defaults."""

    version: FeatureVersion
    standard_harmonics: int = 9
    low_periods: tuple[int, ...] = (2, 4, 6, 10)

    @property
    def contract_name(self) -> str:
        return f"feature_contract_v2_{self.version.lower()}"


@dataclass(frozen=True)
class FeatureSchema:
    """Порядок внешних ``magnitude, angle`` признаков и их физические группы."""

    contract_name: str
    names: tuple[str, ...]
    groups: tuple[str, ...]


class SpectralFeatureBuilder:
    """Строитель Phase 5 A/B с NaN propagation и harmonics availability mask."""

    def __init__(self, config: SpectralFeatureConfig) -> None:
        self.config = config
        self.schema = self._schema()

    def _schema(self) -> FeatureSchema:
        names: list[str] = []
        groups: list[str] = []
        harmonics = [f"h{h}" for h in range(1, self.config.standard_harmonics + 1)] + [f"lp{p}" for p in self.config.low_periods]
        bases = CHANNEL_ORDER if self.config.version == "A" else SYMMETRIC_NAMES
        for harmonic in harmonics:
            for base in bases:
                names.extend((f"{base}_{harmonic}_magnitude", f"{base}_{harmonic}_angle"))
                groups.extend((base, base))
            if self.config.version == "A" and harmonic == "h1":
                for base in SYMMETRIC_NAMES:
                    names.extend((f"{base}_{harmonic}_magnitude", f"{base}_{harmonic}_angle"))
                    groups.extend((base, base))
        return FeatureSchema(self.config.contract_name, tuple(names), tuple(groups))

    def build(self, raw: np.ndarray, spp: int, positions: Sequence[int] | None = None,
              voltage_basis: str = "phase", channel_provenance: Sequence[int] | None = None,
              ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Вернуть ``features, missing_mask, metadata`` для raw `(T, 8)` окна.

        Недоступные по Найквисту и физически отсутствующие компоненты остаются
        NaN; mask имеет ту же форму, что и внешний mag+angle contract.
        """
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 8:
            raise ValueError(f"Ожидался raw (T, 8), получен {raw.shape}")
        if channel_provenance is None:
            channel_provenance_array = np.where(
                np.isfinite(raw).any(axis=0),
                int(ChannelProvenance.MEASURED),
                int(ChannelProvenance.MISSING),
            ).astype(np.uint8)
        else:
            channel_provenance_array = np.asarray(channel_provenance, dtype=np.uint8)
            if channel_provenance_array.shape != (8,):
                raise ValueError("channel_provenance должен иметь форму (8,)")
        positions_array = np.asarray((spp - 1,) if positions is None else positions, dtype=np.int64)
        if np.any(positions_array < 0):
            raise ValueError("Позиции FFT не могут быть отрицательными")
        standard = self._phasors(raw, positions_array, spp, self.config.standard_harmonics)
        lows = [self._phasors(raw, positions_array, spp * period, 1)[:, :, 0] for period in self.config.low_periods]
        available = set(available_harmonics(spp, self.config.standard_harmonics))
        blocks: list[np.ndarray] = []
        for harmonic in range(self.config.standard_harmonics):
            phase = standard[:, :, harmonic]
            if harmonic + 1 not in available:
                phase[:] = np.nan + 1j * np.nan
            blocks.append(self._block(
                phase, voltage_basis, include_phase=self.config.version == "A",
                include_symmetric=self.config.version == "B" or harmonic == 0,
            ))
        for phase in lows:
            blocks.append(self._block(phase, voltage_basis, include_phase=self.config.version == "A",
                                      include_symmetric=self.config.version == "B"))
        features = np.concatenate(blocks, axis=1).astype(np.float32)
        missing_mask = ~np.isfinite(features)
        feature_provenance = self._feature_provenance(
            channel_provenance_array, voltage_basis, available
        )
        return features, missing_mask, {
            "feature_contract": self.schema.contract_name, "feature_names": self.schema.names,
            "available_harmonics": sorted(available), "spp": spp, "voltage_basis": voltage_basis,
            "positions": positions_array.tolist(), "position_semantics": "window_end_inclusive",
            "feature_provenance": feature_provenance.tolist(),
        }

    def _feature_provenance(
        self,
        channels: np.ndarray,
        voltage_basis: str,
        available: set[int],
    ) -> np.ndarray:
        """Спроецировать provenance физических каналов на feature schema."""

        current_ok = np.all(channels[:3] != int(ChannelProvenance.MISSING))
        voltage_ok = np.all(channels[4:7] != int(ChannelProvenance.MISSING))
        result: list[int] = []
        for name in self.schema.names:
            base, harmonic, _component = name.rsplit("_", 2)
            if harmonic.startswith("h") and int(harmonic[1:]) not in available:
                result.append(int(ChannelProvenance.MISSING))
            elif base in CHANNEL_ORDER:
                result.append(int(channels[CHANNEL_ORDER.index(base)]))
            elif base.startswith("I"):
                result.append(int(ChannelProvenance.DERIVED if current_ok else ChannelProvenance.MISSING))
            elif base == "U0" and voltage_basis == "line":
                result.append(int(ChannelProvenance.MISSING))
            else:
                result.append(int(ChannelProvenance.DERIVED if voltage_ok else ChannelProvenance.MISSING))
        return np.asarray(result, dtype=np.uint8)

    @staticmethod
    def _phasors(raw: np.ndarray, positions: np.ndarray, window: int, count: int) -> np.ndarray:
        out = np.full((len(positions), 8, count), np.nan + 1j * np.nan, dtype=np.complex64)
        for row, end in enumerate(positions):
            start = int(end) - window + 1
            if start < 0 or end >= raw.shape[0]:
                continue
            segment = raw[start:int(end) + 1]
            spectrum = np.fft.fft(segment, axis=0) / window
            max_h = min(count, window // 2)
            out[row, :, :max_h] = (2.0 * spectrum[1:max_h + 1]).T
        return out

    def _block(self, phase: np.ndarray, voltage_basis: str, include_phase: bool,
               include_symmetric: bool) -> np.ndarray:
        symmetric = self._symmetric(phase, voltage_basis)
        complex_values = np.concatenate((phase, symmetric), axis=1) if include_phase and include_symmetric else phase if include_phase else symmetric
        voltage_reference = phase[:, 4]
        voltage_is_valid = np.isfinite(voltage_reference.real) & np.isfinite(voltage_reference.imag)
        reference = np.where(voltage_is_valid, voltage_reference, phase[:, 0])
        angle = (np.angle(complex_values) - np.angle(reference)[:, None]) % (2 * np.pi)
        polar = np.empty((phase.shape[0], complex_values.shape[1] * 2), dtype=np.float32)
        polar[:, 0::2] = np.abs(complex_values)
        polar[:, 1::2] = angle
        return polar

    @staticmethod
    def _symmetric(phase: np.ndarray, voltage_basis: str) -> np.ndarray:
        i1, i2, i0 = calculate_symmetrical_components(phase[:, 0], phase[:, 1], phase[:, 2])
        if voltage_basis == "phase":
            u1, u2, u0 = calculate_symmetrical_components(phase[:, 4], phase[:, 5], phase[:, 6])
        elif voltage_basis == "line":
            u1, u2 = calculate_symmetrical_components_from_line(
                phase[:, 4], phase[:, 5], phase[:, 6]
            )
            u0 = np.full(len(phase), np.nan + 1j * np.nan, dtype=np.complex64)
        else:
            u1 = u2 = u0 = np.full(len(phase), np.nan + 1j * np.nan, dtype=np.complex64)
        return np.stack((i1, i2, i0, u1, u2, u0), axis=1)
