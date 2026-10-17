"""Адаптеры реальных источников Phase 5 без зависимости от PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from abc import ABC, abstractmethod
import json
from pathlib import Path
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


class DatasetSource(ABC):
    """Общий read-only API реального источника Phase 5."""

    name: str
    channel_order = CHANNEL_ORDER

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_metadata(self, idx: int) -> dict[str, object]: ...

    @abstractmethod
    def load_signal(self, idx: int) -> np.ndarray: ...

    def get_provenance(self, idx: int) -> np.ndarray:
        """Вернуть provenance восьми каналов; default годится для measured-only sources."""

        signal = self.load_signal(idx)
        present = np.isfinite(signal).any(axis=1)
        return np.where(
            present,
            int(ChannelProvenance.MEASURED),
            int(ChannelProvenance.MISSING),
        ).astype(np.uint8)


class OpenEEShardedSource(DatasetSource):
    """Lazy reader flat Open_EE shards с ограниченным LRU-кэшем открытых ZIP."""

    name = "open_ee"

    def __init__(self, manifest_path: Path, max_cached_shards: int = 2) -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "open_ee_sharded":
            raise ValueError("Manifest не является open_ee_sharded")
        self.entries: list[dict[str, object]] = manifest["records"]
        self.root = self.manifest_path.parent.parent
        self.max_cached_shards = max_cached_shards
        self._cache: OrderedDict[Path, object] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self) -> int:
        return len(self.entries)

    def get_metadata(self, idx: int) -> dict[str, object]:
        return dict(self.entries[idx])

    def _shard(self, relative_path: str) -> object:
        path = self.root / relative_path
        cached = self._cache.pop(path, None)
        if cached is not None:
            self.cache_hits += 1
            self._cache[path] = cached
            return cached
        self.cache_misses += 1
        loaded = np.load(path, allow_pickle=False)
        self._cache[path] = loaded
        while len(self._cache) > self.max_cached_shards:
            _, stale = self._cache.popitem(last=False)
            stale.close()
        return loaded

    def load_signal(self, idx: int) -> np.ndarray:
        entry = self.entries[idx]
        shard = self._shard(str(entry["shard_path"]))
        offsets = shard["offsets"]
        local_index = int(entry["local_index"])
        signal = shard["signals"][int(offsets[local_index]):int(offsets[local_index + 1])]
        return np.asarray(signal, dtype=np.float32).T

    def get_provenance(self, idx: int) -> np.ndarray:
        entry = self.entries[idx]
        shard = self._shard(str(entry["shard_path"]))
        return np.asarray(shard["provenance"][int(entry["local_index"])], dtype=np.uint8)

    def close(self) -> None:
        for shard in self._cache.values():
            shard.close()
        self._cache.clear()

    def __getstate__(self) -> dict[str, object]:
        """Не передавать открытые NPZ/ZIP-дескрипторы DataLoader workers.

        На Windows multiprocessing использует ``spawn`` и сериализует dataset.
        Объекты ``numpy.lib.npyio.NpzFile`` внутри LRU-кэша содержат
        ``BufferedReader`` и не поддерживают pickle. Каждый worker безопасно
        создаст собственный ленивый кэш при первом обращении к shard.
        """

        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["cache_hits"] = 0
        state["cache_misses"] = 0
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._cache = OrderedDict()


class FrenchRTESource(DatasetSource):
    """Lazy mmap reader French/RTE; без тока per-unit источник не считается нормированным."""

    name = "french_rte"

    def __init__(self, prepared_path: Path, current_nominal_a: float | None = 300.0) -> None:
        self.prepared_path = Path(prepared_path)
        if self.prepared_path.suffix != ".npy" or not self.prepared_path.exists():
            raise FileNotFoundError("French training source требует существующий mmap-доступный .npy")
        self.data = np.load(self.prepared_path, mmap_mode="r", allow_pickle=False)
        if self.data.ndim != 3 or self.data.shape[1] != 6:
            raise ValueError(f"Ожидалась French форма (N, 6, T), получена {self.data.shape}")
        self.current_nominal_a = current_nominal_a
        self.current_reserve = 20.0
        self.voltage_nominal_v = 90000.0
        self.voltage_reserve = 3.0

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def get_metadata(self, idx: int) -> dict[str, object]:
        return {"source": self.name, "record_id": idx, "f_network": 50, "f_adc": 6400, "spp": 128,
                "normalization_profile": "per_unit" if self.current_nominal_a else "physical_units",
                "channels_available": ["IA", "IB", "IC", "UA", "UB", "UC"]}

    def load_signal(self, idx: int) -> np.ndarray:
        raw = np.asarray(self.data[idx], dtype=np.float32)
        out = np.full((8, raw.shape[1]), np.nan, dtype=np.float32)
        out[4:7] = raw[:3] * 18.310
        out[:3] = raw[3:] * 4.314
        if self.current_nominal_a is not None:
            out[4:7] /= self.voltage_nominal_v * self.voltage_reserve
            out[:3] /= self.current_nominal_a * self.current_reserve
        return out

    def get_provenance(self, idx: int) -> np.ndarray:
        return np.asarray([
            ChannelProvenance.MEASURED,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MISSING,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MEASURED,
            ChannelProvenance.MISSING,
        ], dtype=np.uint8)

    def __getstate__(self) -> dict[str, object]:
        """Передать worker только путь, не сериализовать содержимое memmap."""

        state = self.__dict__.copy()
        state["data"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.data = np.load(self.prepared_path, mmap_mode="r", allow_pickle=False)


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
