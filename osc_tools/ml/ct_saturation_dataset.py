"""Ленивый датасет для архива моделирования насыщения ТТ объёмом 352 ГБ.

Входом спектральной Physical KAN-модели служат измеренные вторичные токи
``I2_CT1`` и вторичные напряжения ``V2_VT1``. Идеальные первичные токи
``I1_CT1`` намеренно исключены: они являются физическим эталоном, которого нет
в эксплуатационных осциллограммах COMTRADE.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from osc_tools.features.phasor import calculate_symmetrical_components
from osc_tools.features.polar import calculate_polar_features


FILE_RE = re.compile(
    r"^A0_INT_SAT_CT1_regime_1_(?P<tau>0\.015|0\.4)_(?P<record>\d+)_exp1\.mat$",
    re.IGNORECASE,
)
DEFAULT_SUB_PERIODS = (2, 4, 6, 10)
SECONDARY_NOMINAL_A = 5.0
CURRENT_RESERVE = 20.0
SECONDARY_NOMINAL_V = 100.0
VOLTAGE_RESERVE = 3.0


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
) -> dict[str, np.ndarray | float | None]:
    """Загрузить одну подготовленную MAT-запись в обычные массивы NumPy."""
    if loadmat_fn is None:
        try:
            from scipy.io import loadmat as loadmat_fn
        except ImportError as exc:  # Зависит от локального окружения исследователя.
            raise ImportError("Для чтения архива ТТ требуется scipy.io.loadmat") from exc
    data = loadmat_fn(str(path), squeeze_me=True, struct_as_record=False)
    voltage: np.ndarray | None
    if {"secondary_a", "labels", "time_s"}.issubset(data):
        secondary = np.asarray(data["secondary_a"], dtype=np.float32)
        voltage = (
            np.asarray(data["voltage_v"], dtype=np.float32)
            if "voltage_v" in data else None
        )
        time = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
        labels = np.asarray(data["labels"], dtype=np.float32)
        current_error = (
            np.asarray(data["current_error_a"], dtype=np.float32)
            if "current_error_a" in data else None
        )
    else:
        obj = data.get("I2_CT1")
        if type(obj).__name__ == "MatlabOpaque":
            raise ValueError(
                "I2_CT1 сохранён как непрозрачный MATLAB MCOS timeseries. "
                "Запустите единый сценарий обучения: он предварительно распакует данные."
            )
        secondary = np.asarray(_field(obj, "Data"), dtype=np.float32)
        voltage_obj = data.get("V2_VT1")
        voltage = (
            np.asarray(_field(voltage_obj, "Data"), dtype=np.float32)
            if voltage_obj is not None and type(voltage_obj).__name__ != "MatlabOpaque"
            else None
        )
        time = np.asarray(_field(obj, "Time"), dtype=np.float64).reshape(-1)
        labels = np.column_stack([
            np.asarray(data[f"flag_sat_phs{phase}"], dtype=bool).reshape(-1)
            for phase in "ABC"
        ]).astype(np.float32)
        current_error = None
    if secondary.ndim != 2 or secondary.shape[1] != 3:
        raise ValueError(f"I2_CT1.Data должен иметь форму (T, 3), получено {secondary.shape}")
    if len(time) != len(secondary) or len(time) < 2:
        raise ValueError("I2_CT1.Time не согласован с I2_CT1.Data")
    if labels.shape != secondary.shape:
        raise ValueError(f"Флаги насыщения должны иметь форму {secondary.shape}, получено {labels.shape}")
    if voltage is not None and voltage.shape != secondary.shape:
        raise ValueError(f"V2_VT1.Data должен иметь форму {secondary.shape}, получено {voltage.shape}")
    if current_error is not None and current_error.shape != secondary.shape:
        raise ValueError(
            f"current_error_a должен иметь форму {secondary.shape}, получено {current_error.shape}"
        )
    return {
        "secondary_a": secondary,
        "voltage_v": voltage,
        "current_error_a": current_error,
        "labels": labels.astype(np.float32),
        "time_s": time,
        "fs_hz": float(1.0 / np.median(np.diff(time))),
    }


def current_feature_count(num_harmonics: int = 9, sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS) -> int:
    """Три фазы с polar-признаками плюс polar-признаки I1/I2/I0."""
    return 3 * (num_harmonics + len(sub_periods)) * 2 + 3 * 2


def ct_spectral_feature_count(
    num_harmonics: int = 9,
    sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS,
    include_voltage: bool = True,
) -> int:
    """Число полярных признаков CT: токи, напряжения и h1-симм. составляющие."""
    current_count = current_feature_count(num_harmonics, sub_periods)
    return current_count * 2 if include_voltage else current_count


def compute_current_spectral_features(
    currents: np.ndarray,
    *,
    samples_per_period: int,
    stride: int,
    warmup: int,
    num_harmonics: int = 9,
    sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS,
) -> np.ndarray:
    """Рассчитать токовую часть причинного FFT-контура ТТ."""
    return compute_ct_spectral_features(
        currents,
        None,
        samples_per_period=samples_per_period,
        stride=stride,
        warmup=warmup,
        num_harmonics=num_harmonics,
        sub_periods=sub_periods,
        include_voltage=False,
    )


def compute_ct_spectral_features(
    currents: np.ndarray,
    voltages: np.ndarray | None,
    *,
    samples_per_period: int,
    stride: int,
    warmup: int,
    num_harmonics: int = 9,
    sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS,
    include_voltage: bool = True,
) -> np.ndarray:
    """Рассчитать причинные FFT-признаки в точках с шагом ``stride``.

    ``warmup`` задаёт начало исследуемого окна. Каждый фазор считается по
    отсчётам, предшествующим точке прогноза. Это не даёт утечки будущего и
    позволяет получить 64 валидных токена из 40 мс при шаге 1/32 периода.
    """
    if include_voltage and voltages is None:
        raise ValueError(
            "Для spectral-режима с напряжением отсутствует voltage_v. "
            "Повторно запустите подготовку плоских MAT-файлов."
        )
    signals = currents if not include_voltage else np.concatenate([currents, voltages], axis=1)
    endpoints = np.arange(warmup + stride, len(signals) + 1, stride, dtype=np.int64)
    total_harmonics = num_harmonics + len(sub_periods)
    complex_all = np.zeros(
        (len(endpoints), signals.shape[1], total_harmonics), dtype=np.complex64,
    )
    for row, endpoint in enumerate(endpoints):
        period_window = signals[endpoint - samples_per_period:endpoint]
        spectrum = np.fft.fft(period_window, axis=0) / len(period_window) * 2.0
        complex_all[row, :, :num_harmonics] = spectrum[1:num_harmonics + 1].T
        for low_index, periods in enumerate(sub_periods):
            length = int(periods * samples_per_period)
            low_window = signals[endpoint - length:endpoint]
            low_spectrum = np.fft.fft(low_window, axis=0) / len(low_window) * 2.0
            complex_all[row, :, num_harmonics + low_index] = low_spectrum[1]

    reference = complex_all[:, 3, 0] if include_voltage else complex_all[:, 0, 0]
    phase_polar = calculate_polar_features(
        complex_all.reshape(len(endpoints), -1), reference,
    )
    phase_width = (num_harmonics + len(sub_periods)) * 2
    current_phase = phase_polar[:, :3 * phase_width]
    i1, i2, i0 = calculate_symmetrical_components(
        complex_all[:, 0, 0], complex_all[:, 1, 0], complex_all[:, 2, 0],
    )
    current_symmetric = calculate_polar_features(
        np.stack([i1, i2, i0], axis=1), reference,
    )
    if not include_voltage:
        return np.concatenate([current_phase, current_symmetric], axis=1).astype(np.float32)
    voltage_phase = phase_polar[:, 3 * phase_width:6 * phase_width]
    u1, u2, u0 = calculate_symmetrical_components(
        complex_all[:, 3, 0], complex_all[:, 4, 0], complex_all[:, 5, 0],
    )
    voltage_symmetric = calculate_polar_features(
        np.stack([u1, u2, u0], axis=1), reference,
    )
    return np.concatenate(
        [current_phase, current_symmetric, voltage_phase, voltage_symmetric], axis=1,
    ).astype(np.float32)


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
    """Одно лениво выбранное окно насыщения ТТ на выбранный MAT-файл.

    Конечная длина эпохи задаётся через ``RandomSampler`` с возвращением.
    Выбор окон возле события не позволяет короткому насыщению потеряться
    среди равномерно выбранных нормальных участков.
    """

    def __init__(
        self,
        files: Sequence[CTSaturationFile],
        *,
        num_periods: int = 2,
        stride_fraction: int = 32,
        num_harmonics: int = 9,
        sub_periods: Sequence[int] = DEFAULT_SUB_PERIODS,
        event_window_probability: float = 0.8,
        nominal_secondary_a: float = SECONDARY_NOMINAL_A,
        reserve_factor: float = CURRENT_RESERVE,
        nominal_secondary_v: float = SECONDARY_NOMINAL_V,
        voltage_reserve_factor: float = VOLTAGE_RESERVE,
        input_mode: str = "spectral",
        include_voltage: bool = True,
        raw_target_spp: int = 32,
        gain_jitter_range: tuple[float, float] | None = None,
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
        if input_mode not in {"spectral", "raw"}:
            raise ValueError("input_mode должен быть 'spectral' или 'raw'")
        self.input_mode = input_mode
        self.include_voltage = include_voltage
        self.raw_target_spp = raw_target_spp
        self.gain_jitter_range = gain_jitter_range
        self.current_divisor = float(nominal_secondary_a * reserve_factor)
        self.voltage_divisor = float(nominal_secondary_v * voltage_reserve_factor)
        self.phase_permutation = phase_permutation
        self.seed = seed
        self.loader = loader
        self._cache = _LRU(cache_size)
        if self.current_divisor <= 0 or self.voltage_divisor <= 0:
            raise ValueError("Делители нормализации должны быть положительными")
        if gain_jitter_range is not None:
            low, high = gain_jitter_range
            if not 0 < low <= high:
                raise ValueError("gain_jitter_range должен быть положительным и упорядоченным")

    @property
    def num_input_channels(self) -> int:
        """Число каналов, возвращаемых датасетом."""
        if self.input_mode == "raw":
            return 6 if self.include_voltage else 3
        return ct_spectral_feature_count(
            self.num_harmonics, self.sub_periods, self.include_voltage,
        )

    @property
    def sequence_length(self) -> int:
        """Длина последовательности модели в выбранном режиме."""
        if self.input_mode == "raw":
            return self.num_periods * self.raw_target_spp
        return self.num_periods * self.stride_fraction

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
        voltages = (
            np.asarray(record["voltage_v"], dtype=np.float32)
            if record.get("voltage_v") is not None else None
        )
        labels = np.asarray(record["labels"], dtype=np.float32)
        fs = float(record["fs_hz"])
        spp = round(fs / 50.0)
        window = self.num_periods * spp
        context = max(self.sub_periods) * spp
        rng = self._rng(idx)
        gain = 1.0
        if self.gain_jitter_range is not None:
            gain = float(rng.uniform(*self.gain_jitter_range))

        positive = np.flatnonzero(labels.any(axis=1))
        if len(positive) and rng.random() < self.event_window_probability:
            centre = int(positive[0])
            start = centre - window // 2
        else:
            start = int(rng.integers(0, max(1, len(currents) - window + 1)))
        start = int(np.clip(start, 0, max(0, len(currents) - window)))
        current_window = currents[start:start + window] / self.current_divisor * gain
        voltage_window = (
            voltages[start:start + window] / self.voltage_divisor * gain
            if voltages is not None else None
        )
        y_raw = labels[start:start + window].copy()

        if self.phase_permutation:
            shift = int(rng.integers(0, 3))
            current_window = np.roll(current_window, shift, axis=1)
            if voltage_window is not None:
                voltage_window = np.roll(voltage_window, shift, axis=1)
            y_raw = np.roll(y_raw, shift, axis=1)

        if self.input_mode == "raw":
            return self._format_raw(current_window, voltage_window, y_raw, spp)

        left = max(0, start - context)
        current_context = currents[left:start + window] / self.current_divisor * gain
        voltage_context = (
            voltages[left:start + window] / self.voltage_divisor * gain
            if voltages is not None else None
        )
        missing_context = context - (start - left)
        if missing_context:
            current_context = np.pad(current_context, ((missing_context, 0), (0, 0)), mode="edge")
            if voltage_context is not None:
                voltage_context = np.pad(voltage_context, ((missing_context, 0), (0, 0)), mode="edge")
        if self.phase_permutation:
            current_context = np.roll(current_context, shift, axis=1)
            if voltage_context is not None:
                voltage_context = np.roll(voltage_context, shift, axis=1)

        stride = max(1, round(spp / self.stride_fraction))
        features = compute_ct_spectral_features(
            current_context,
            voltage_context,
            samples_per_period=spp,
            stride=stride,
            warmup=context,
            num_harmonics=self.num_harmonics,
            sub_periods=self.sub_periods,
        )
        n_zones = self.sequence_length
        features = features[:n_zones]
        targets = np.zeros((n_zones, 3), dtype=np.float32)
        for z in range(n_zones):
            lo = min(z * stride, len(y_raw) - 1)
            hi = min(lo + stride, len(y_raw))
            targets[z] = y_raw[lo:hi].max(axis=0)
        return torch.from_numpy(features.T.copy()), torch.from_numpy(targets)

    def _format_raw(
        self,
        currents: np.ndarray,
        voltages: np.ndarray | None,
        labels: np.ndarray,
        source_spp: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Антиалиасинговое приведение raw-сигналов к 32 отсчётам на период."""
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise ImportError("Для raw-режима требуется scipy.signal.resample_poly") from exc
        if self.include_voltage and voltages is None:
            raise ValueError("Для raw-режима с напряжением отсутствует voltage_v")
        signals = currents if not self.include_voltage else np.concatenate([currents, voltages], axis=1)
        common = math.gcd(int(source_spp), int(self.raw_target_spp))
        resampled = resample_poly(
            signals,
            up=self.raw_target_spp // common,
            down=source_spp // common,
            axis=0,
        ).astype(np.float32)
        expected = self.sequence_length
        resampled = resampled[:expected]
        if len(resampled) < expected:
            resampled = np.pad(resampled, ((0, expected - len(resampled)), (0, 0)), mode="edge")

        targets = np.zeros((expected, 3), dtype=np.float32)
        edges = np.linspace(0, len(labels), expected + 1).round().astype(int)
        for index in range(expected):
            lo, hi = edges[index], max(edges[index] + 1, edges[index + 1])
            targets[index] = labels[lo:min(hi, len(labels))].max(axis=0)
        return torch.from_numpy(resampled.T.copy()), torch.from_numpy(targets)
