"""Versioned spectral feature contracts Phase 5, independent of legacy Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from osc_tools.features.phasor import calculate_symmetrical_components
from .phase5_contracts import CHANNEL_ORDER, available_harmonics


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
              voltage_basis: str = "phase") -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Вернуть ``features, missing_mask, metadata`` для raw `(T, 8)` окна.

        Недоступные по Найквисту и физически отсутствующие компоненты остаются
        NaN; mask имеет ту же форму, что и внешний mag+angle contract.
        """
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 8:
            raise ValueError(f"Ожидался raw (T, 8), получен {raw.shape}")
        positions_array = np.asarray((0,) if positions is None else positions, dtype=np.int64)
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
        return features, ~np.isfinite(features), {
            "feature_contract": self.schema.contract_name, "feature_names": self.schema.names,
            "available_harmonics": sorted(available), "spp": spp, "voltage_basis": voltage_basis,
        }

    @staticmethod
    def _phasors(raw: np.ndarray, positions: np.ndarray, window: int, count: int) -> np.ndarray:
        out = np.full((len(positions), 8, count), np.nan + 1j * np.nan, dtype=np.complex64)
        for row, start in enumerate(positions):
            if start + window > raw.shape[0]:
                continue
            segment = raw[start:start + window]
            spectrum = np.fft.fft(segment, axis=0) / window
            max_h = min(count, window // 2)
            out[row, :, :max_h] = (2.0 * spectrum[1:max_h + 1]).T
        return out

    def _block(self, phase: np.ndarray, voltage_basis: str, include_phase: bool,
               include_symmetric: bool) -> np.ndarray:
        symmetric = self._symmetric(phase, voltage_basis)
        complex_values = np.concatenate((phase, symmetric), axis=1) if include_phase and include_symmetric else phase if include_phase else symmetric
        reference = phase[:, 4] if voltage_basis == "phase" else phase[:, 0]
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
        else:
            u1 = u2 = u0 = np.full(len(phase), np.nan + 1j * np.nan, dtype=np.complex64)
        return np.stack((i1, i2, i0, u1, u2, u0), axis=1)
