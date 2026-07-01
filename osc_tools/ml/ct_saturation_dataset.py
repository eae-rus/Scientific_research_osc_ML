"""Ленивый датасет для архива моделирования насыщения ТТ объёмом 352 ГБ.

Входом модели служат только измеренные вторичные токи ``I2_CT1``. Идеальные
первичные токи ``I1_CT1`` намеренно исключены: они являются физическим эталоном,
которого нет в эксплуатационных осциллограммах COMTRADE.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from osc_tools.ml.augmented_dataset import compute_spectral_from_raw


FILE_RE = re.compile(
    r"^A0_INT_SAT_CT1_regime_1_(?P<tau>0\.015|0\.4)_(?P<record>\d+)_exp1\.mat$",
    re.IGNORECASE,
)
DEFAULT_SUB_PERIODS = (2, 4, 6, 10)
SECONDARY_NOMINAL_A = 5.0
CURRENT_RESERVE = 20.0


@dataclass(frozen=True)
class CTSaturationFile:
    path: Path
    record_id: int
    source_tau_s: float

    @property
    def split_group(self) -> str:
        return f"tau={self.source_tau_s:g}:record={self.record_id}"


def scan_ct_saturation_files(root: str | Path) -> list[CTSaturationFile]:
    """Просканировать только имена, не открывая содержимое MAT-файлов."""
    root = Path(root)
    result: list[CTSaturationFile] = []
    for path in root.glob("A0_INT_SAT_CT1_regime_1_*_exp1.mat"):
        match = FILE_RE.match(path.name)
        if match:
            result.append(CTSaturationFile(
                path=path,
                record_id=int(match.group("record")),
                source_tau_s=float(match.group("tau")),
            ))
    return sorted(result, key=lambda x: (x.source_tau_s, x.record_id))


def deterministic_split(
    files: Iterable[CTSaturationFile], val_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[CTSaturationFile], list[CTSaturationFile]]:
    """Стабильное групповое разбиение, не зависящее от порядка файлов."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction должен находиться между 0 и 1")
    train, val = [], []
    threshold = int(val_fraction * 10_000)
    for item in files:
        key = f"{seed}:{item.split_group}".encode("utf-8")
        bucket = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % 10_000
        (val if bucket < threshold else train).append(item)
    return train, val


def _field(obj: object, name: str) -> object:
    if isinstance(obj, dict):
        return obj[name]
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        value = obj[name]
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
        return value
    raise KeyError(f"В объекте MAT отсутствует поле {name!r}")


def load_ct_saturation_mat(
    path: str | Path,
    loadmat_fn: Callable[..., dict] | None = None,
) -> dict[str, np.ndarray | float]:
    """Загрузить одну подготовленную MAT-запись в обычные массивы NumPy."""
    if loadmat_fn is None:
        try:
            from scipy.io import loadmat as loadmat_fn
        except ImportError as exc:  # Зависит от локального окружения исследователя.
            raise ImportError("Для чтения архива ТТ требуется scipy.io.loadmat") from exc
    data = loadmat_fn(str(path), squeeze_me=True, struct_as_record=False)
    if {"secondary_a", "labels", "time_s"}.issubset(data):
        secondary = np.asarray(data["secondary_a"], dtype=np.float32)
        time = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
        labels = np.asarray(data["labels"], dtype=np.float32)
    else:
        obj = data.get("I2_CT1")
        if type(obj).__name__ == "MatlabOpaque":
            raise ValueError(
                "I2_CT1 сохранён как непрозрачный MATLAB MCOS timeseries. "
                "Запустите единый сценарий обучения: он предварительно распакует данные."
            )
        secondary = np.asarray(_field(obj, "Data"), dtype=np.float32)
        time = np.asarray(_field(obj, "Time"), dtype=np.float64).reshape(-1)
        labels = np.column_stack([
            np.asarray(data[f"flag_sat_phs{phase}"], dtype=bool).reshape(-1)
            for phase in "ABC"
        ]).astype(np.float32)
    if secondary.ndim != 2 or secondary.shape[1] != 3:
        raise ValueError(f"I2_CT1.Data должен иметь форму (T, 3), получено {secondary.shape}")
    if len(time) != len(secondary) or len(time) < 2:
        raise ValueError("I2_CT1.Time не согласован с I2_CT1.Data")
    if labels.shape != secondary.shape:
        raise ValueError(f"Флаги насыщения должны иметь форму {secondary.shape}, получено {labels.shape}")
    return {
        "secondary_a": secondary,
        "labels": labels.astype(np.float32),
        "time_s": time,
        "fs_hz": float(1.0 / np.median(np.diff(time))),
    }


def current_feature_count(num_harmonics: int = 9, sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS) -> int:
    """Три фазы с polar-признаками плюс polar-признаки I1/I2/I0."""
    return 3 * (num_harmonics + len(sub_periods)) * 2 + 3 * 2


def compute_current_spectral_features(
    currents: np.ndarray,
    *,
    samples_per_period: int,
    stride: int,
    warmup: int,
    num_harmonics: int = 9,
    sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS,
) -> np.ndarray:
    """Применить проверенный FFT-контур Фазы 4 и оставить токовые признаки."""
    raw8 = np.full((len(currents), 8), np.nan, dtype=np.float32)
    raw8[:, :3] = currents
    all_features = compute_spectral_from_raw(
        raw8,
        num_harmonics=num_harmonics,
        sub_periods=list(sub_periods),
        include_symmetric=True,
        stride=stride,
        warmup=warmup,
        fft_window=samples_per_period,
        samples_per_period=samples_per_period,
    )
    phase_width = (num_harmonics + len(sub_periods)) * 2
    phase_currents = all_features[:, :3 * phase_width]
    symmetric_currents = all_features[:, 8 * phase_width:8 * phase_width + 6]
    return np.concatenate([phase_currents, symmetric_currents], axis=1).astype(np.float32)


class _LRU:
    def __init__(self, maxsize: int):
        self.maxsize = max(0, int(maxsize))
        self.data: OrderedDict[str, dict] = OrderedDict()

    def get(self, path: Path, loader: Callable[[Path], dict]) -> dict:
        key = str(path)
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        value = loader(path)
        if self.maxsize:
            self.data[key] = value
            while len(self.data) > self.maxsize:
                self.data.popitem(last=False)
        return value


class CTSaturationLazyDataset(Dataset):
    """Одно лениво выбранное окно из 10 периодов на выбранный MAT-файл.

    Конечная длина эпохи задаётся через ``RandomSampler`` с возвращением.
    Выбор окон возле события не позволяет короткому насыщению потеряться
    среди равномерно выбранных нормальных участков.
    """

    def __init__(
        self,
        files: Sequence[CTSaturationFile],
        *,
        num_periods: int = 10,
        stride_fraction: int = 8,
        num_harmonics: int = 9,
        sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS,
        event_window_probability: float = 0.8,
        nominal_secondary_a: float = SECONDARY_NOMINAL_A,
        reserve_factor: float = CURRENT_RESERVE,
        phase_permutation: bool = False,
        cache_size: int = 2,
        seed: int = 42,
        loader: Callable[[Path], dict] = load_ct_saturation_mat,
    ) -> None:
        self.files = list(files)
        self.num_periods = num_periods
        self.stride_fraction = stride_fraction
        self.num_harmonics = num_harmonics
        self.sub_periods = tuple(sub_periods)
        self.event_window_probability = event_window_probability
        self.divisor = float(nominal_secondary_a * reserve_factor)
        self.phase_permutation = phase_permutation
        self.seed = seed
        self.loader = loader
        self._cache = _LRU(cache_size)
        if self.divisor <= 0:
            raise ValueError("Делитель нормализации должен быть положительным")

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, idx: int) -> np.random.Generator:
        worker = torch.utils.data.get_worker_info()
        worker_seed = worker.seed if worker is not None else torch.initial_seed()
        return np.random.default_rng((worker_seed + self.seed + idx) % (2**63 - 1))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.files[idx]
        record = self._cache.get(item.path, self.loader)
        currents = np.asarray(record["secondary_a"], dtype=np.float32)
        labels = np.asarray(record["labels"], dtype=np.float32)
        fs = float(record["fs_hz"])
        spp = round(fs / 50.0)
        stride = max(1, spp // self.stride_fraction)
        window = self.num_periods * spp
        context = max(self.sub_periods) * spp
        rng = self._rng(idx)

        positive = np.flatnonzero(labels.any(axis=1))
        if len(positive) and rng.random() < self.event_window_probability:
            centre = int(positive[0])
            start = centre - window // 2
        else:
            start = int(rng.integers(0, max(1, len(currents) - window + 1)))
        start = int(np.clip(start, 0, max(0, len(currents) - window)))
        left = max(0, start - context)
        raw = currents[left:start + window] / self.divisor
        missing_context = context - (start - left)
        if missing_context:
            raw = np.pad(raw, ((missing_context, 0), (0, 0)), mode="edge")
        y_raw = labels[start:start + window].copy()

        if self.phase_permutation:
            shift = int(rng.integers(0, 3))
            raw = np.roll(raw, shift, axis=1)
            y_raw = np.roll(y_raw, shift, axis=1)

        features = compute_current_spectral_features(
            raw,
            samples_per_period=spp,
            stride=stride,
            warmup=context + spp,
            num_harmonics=self.num_harmonics,
            sub_periods=self.sub_periods,
        )
        n_zones = (self.num_periods - 1) * self.stride_fraction
        features = features[:n_zones]
        targets = np.zeros((n_zones, 3), dtype=np.float32)
        for z in range(n_zones):
            lo = min(spp + z * stride, len(y_raw) - 1)
            hi = min(lo + stride, len(y_raw))
            targets[z] = y_raw[lo:hi].max(axis=0)
        return torch.from_numpy(features.T.copy()), torch.from_numpy(targets)
