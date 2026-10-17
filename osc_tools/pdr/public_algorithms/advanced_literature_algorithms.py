"""Открытые исследовательские РНМ по современным публикациям и РЭ.

Субцикловые органы используют исходные мгновенные отсчёты. Интервал около
1 мс переводится в целое число отсчётов отдельно для каждой частоты АЦП; поэтому
число точек между выборками не зашито в алгоритм. Решение появляется причинно
после поступления последней необходимой выборки.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from osc_tools.pdr.base import PDRAlgorithm, PDRDirection, PDRInputData, PDROutput
from osc_tools.pdr.pdr_signal_utils import (
    compute_negative_sequence,
    compute_positive_sequence,
    derive_unified_currents,
    derive_unified_voltages,
    scale_thresholds_for_profile,
)


def _wrap_angle_deg(angle_deg: float) -> float:
    value = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if value <= -180.0 else value


def _raw_context(input_data: PDRInputData) -> tuple[np.ndarray, int, float, float] | None:
    signal = input_data.raw_signals
    end_idx = input_data.end_sample_index
    sampling_rate = input_data.sampling_rate_hz
    network_frequency = input_data.network_frequency_hz
    if (
        signal is None
        or end_idx is None
        or sampling_rate is None
        or network_frequency is None
        or np.asarray(signal).ndim != 2
        or np.asarray(signal).shape[0] != 8
        or sampling_rate <= 0.0
        or network_frequency <= 0.0
    ):
        return None
    return np.asarray(signal), int(end_idx), float(sampling_rate), float(network_frequency)


def _sample_lag(sampling_rate_hz: float, interval_ms: float) -> int:
    return max(1, int(math.floor(sampling_rate_hz * interval_ms / 1000.0 + 0.5)))


def _complete_three(values: np.ndarray) -> np.ndarray:
    """Векторно восстановить третью фазу/линию из любых двух."""

    completed = np.asarray(values, dtype=np.complex128).copy()
    finite = np.isfinite(completed)
    exactly_two = finite.sum(axis=0) == 2
    for row in range(3):
        missing = exactly_two & ~finite[row]
        if np.any(missing):
            completed[row, missing] = -np.nansum(completed[:, missing], axis=0)
    return completed


def _vectorized_sequences(
    signals: np.ndarray,
    sample_indices: np.ndarray,
    estimator: Callable[[np.ndarray], np.ndarray],
    voltage_basis: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Подготовить U1/I1 для всех точек одной записью NumPy."""

    n_points = sample_indices.shape[1]
    u1 = np.full(n_points, np.nan + 1j * np.nan, dtype=np.complex128)
    i1 = np.full(n_points, np.nan + 1j * np.nan, dtype=np.complex128)
    valid_columns = np.all(
        (sample_indices >= 0) & (sample_indices < signals.shape[1]), axis=0
    )
    if not np.any(valid_columns):
        return u1, i1

    indices = sample_indices[:, valid_columns]
    channels = np.asarray((0, 1, 2, 4, 5, 6), dtype=np.int64)
    samples = np.take(np.asarray(signals)[channels], indices, axis=1)
    values = np.asarray(estimator(samples), dtype=np.complex128)
    values[~np.isfinite(samples).all(axis=1)] = np.nan + 1j * np.nan

    currents = _complete_three(values[:3])
    voltage_values = _complete_three(values[3:])
    if voltage_basis == "line":
        u_ab, u_bc, u_ca = voltage_values
        voltages = np.vstack((
            (u_ab - u_ca) / 3.0,
            (u_bc - u_ab) / 3.0,
            (u_ca - u_bc) / 3.0,
        ))
    else:
        voltages = voltage_values

    a = np.exp(1j * 2.0 * np.pi / 3.0)
    valid_i = np.isfinite(currents).all(axis=0)
    valid_u = np.isfinite(voltages).all(axis=0)
    i_values = (currents[0] + a * currents[1] + a ** 2 * currents[2]) / 3.0
    u_values = (voltages[0] + a * voltages[1] + a ** 2 * voltages[2]) / 3.0
    target_positions = np.flatnonzero(valid_columns)
    i1[target_positions[valid_i]] = i_values[valid_i]
    u1[target_positions[valid_u]] = u_values[valid_u]
    return u1, i1


def _prepared_position(algorithm: PDRAlgorithm, end_idx: int) -> Optional[int]:
    first = getattr(algorithm, "_prepared_first", None)
    step = getattr(algorithm, "_prepared_step", None)
    count = getattr(algorithm, "_prepared_count", 0)
    if first is None or step is None or step <= 0 or end_idx < first:
        return None
    offset = end_idx - first
    if offset % step:
        return None
    position = offset // step
    return position if position < count else None


def _complex_channels(
    input_data: PDRInputData,
    estimator: Callable[[np.ndarray], complex],
    sample_indices: list[int],
) -> tuple[dict[str, complex], dict[str, complex]] | None:
    context = _raw_context(input_data)
    if context is None:
        return None
    signal, _, _, _ = context
    if min(sample_indices) < 0 or max(sample_indices) >= signal.shape[1]:
        return None

    values: list[Optional[complex]] = []
    for channel_index in (0, 1, 2, 4, 5, 6):
        samples = np.asarray(signal[channel_index, sample_indices], dtype=np.float64)
        values.append(estimator(samples) if np.isfinite(samples).all() else None)

    currents_raw = {
        phase: value for phase, value in zip(("A", "B", "C"), values[:3])
        if value is not None
    }
    currents = derive_unified_currents(currents_raw)

    voltage_keys = (
        ("AB", "BC", "CA")
        if input_data.voltage_basis == "line"
        else ("A", "B", "C")
    )
    voltages_raw = {
        key: value for key, value in zip(voltage_keys, values[3:])
        if value is not None
    }
    voltages = derive_unified_voltages(voltages_raw)
    if set(currents) != {"A", "B", "C"} or voltages.mode == "invalid":
        return None
    return voltages.u_phase, currents


def _threshold_output(reason: str, *, unlabeled: bool, margin: float = 0.0) -> PDROutput:
    direction = PDRDirection.UNLABELED if unlabeled else PDRDirection.REVERSE
    return PDROutput(
        direction=direction,
        is_tripped=False,
        margin=margin,
        confidence=0.0 if unlabeled else 1.0,
        diagnostics={"reason": reason},
    )


class Sivokobylenko2PtPDRAlgorithm(PDRAlgorithm):
    """Двухвыборочный субцикловый угловой РНМ прямой последовательности.

    Реализована причинная версия формул Сивокобыленко и Деркачева (2016):
    вектор относится к первой выборке, а решение выдаётся после второй, то есть
    с задержкой около ``sample_interval_ms``.
    """

    algorithm_id = "pdr_sivokobylenko_2pt"
    name = "Sivokobylenko Two-Sample Positive-Sequence PDR"
    is_public = True
    tunable_parameters = {
        "sample_interval_ms": 1.0,
        "mta_deg": 60.0,
        "sector_width_deg": 180.0,
        "u_min_pu": 0.05,
        "i_min_pu": 0.05,
        "scale_profile": "physical_pu",
    }

    def prepare_record(
        self,
        signals: np.ndarray,
        end_indices: list[int],
        *,
        sampling_rate_hz: float,
        network_frequency_hz: float,
        voltage_basis: str,
    ) -> None:
        ends = np.asarray(end_indices, dtype=np.int64)
        self._prepared_first = int(ends[0]) if ends.size else None
        self._prepared_step = int(ends[1] - ends[0]) if ends.size > 1 else 1
        self._prepared_count = int(ends.size)
        lag = _sample_lag(sampling_rate_hz, float(self.params["sample_interval_ms"]))
        theta = 2.0 * math.pi * network_frequency_hz * lag / sampling_rate_hz
        sin_theta = math.sin(theta)
        self._prepared_lag = lag
        self._prepared_interval_ms = 1000.0 * lag / sampling_rate_hz
        if not ends.size or abs(sin_theta) < 1e-8:
            self._prepared_u1 = np.full(ends.size, np.nan + 1j * np.nan)
            self._prepared_i1 = self._prepared_u1.copy()
            return

        indices = np.vstack((ends - lag, ends))

        def estimator(samples: np.ndarray) -> np.ndarray:
            return samples[:, 0] + 1j * (
                (samples[:, 0] * math.cos(theta) - samples[:, 1]) / sin_theta
            )

        self._prepared_u1, self._prepared_i1 = _vectorized_sequences(
            signals, indices, estimator, voltage_basis
        )

    def compute(self, input_data: PDRInputData) -> PDROutput:
        context = _raw_context(input_data)
        if context is None:
            return _threshold_output("raw_samples_unavailable", unlabeled=True)
        _, end_idx, sampling_rate, frequency = context
        lag = _sample_lag(sampling_rate, float(self.params["sample_interval_ms"]))
        theta = 2.0 * math.pi * frequency * lag / sampling_rate
        sin_theta = math.sin(theta)
        if end_idx < lag or abs(sin_theta) < 1e-8:
            return _threshold_output("two_sample_window_unavailable", unlabeled=True)

        prepared_position = _prepared_position(self, end_idx)
        if prepared_position is not None:
            u1 = self._prepared_u1[prepared_position]
            i1 = self._prepared_i1[prepared_position]
            lag = self._prepared_lag
        else:
            def estimator(samples: np.ndarray) -> complex:
                first, second = float(samples[0]), float(samples[1])
                beta = (first * math.cos(theta) - second) / sin_theta
                return complex(first, beta)

            phase_vectors = _complex_channels(
                input_data, estimator, [end_idx - lag, end_idx]
            )
            if phase_vectors is None:
                return _threshold_output("insufficient_raw_phase_channels", unlabeled=True)
            voltages, currents = phase_vectors
            u1 = compute_positive_sequence(voltages, is_voltage=True)
            i1 = compute_positive_sequence(currents, is_voltage=False)
        if u1 is None or i1 is None or not np.isfinite([u1, i1]).all():
            return _threshold_output("sequence_unavailable", unlabeled=True)

        u_min, i_min, _ = scale_thresholds_for_profile(
            float(self.params["u_min_pu"]),
            float(self.params["i_min_pu"]),
            0.0,
            str(self.params["scale_profile"]),
        )
        if abs(u1) < u_min:
            return _threshold_output("voltage_below_threshold", unlabeled=True, margin=abs(u1)-u_min)
        if abs(i1) < i_min:
            return _threshold_output("current_below_threshold", unlabeled=False, margin=abs(i1)-i_min)

        phi_deg = _wrap_angle_deg(math.degrees(np.angle(u1) - np.angle(i1)))
        delta_deg = _wrap_angle_deg(phi_deg - float(self.params["mta_deg"]))
        half_sector = float(self.params["sector_width_deg"]) / 2.0
        margin = half_sector - abs(delta_deg)
        direction = PDRDirection.FORWARD if margin >= 0.0 else PDRDirection.REVERSE
        return PDROutput(
            direction=direction,
            is_tripped=direction == PDRDirection.FORWARD,
            margin=margin,
            diagnostics={
                "phi_deg": phi_deg,
                "delta_phi_deg": delta_deg,
                "sample_lag": lag,
                "actual_interval_ms": 1000.0 * lag / sampling_rate,
            },
        )


class Sivokobylenko5PtPDRAlgorithm(PDRAlgorithm):
    """Пятиточечный субцикловый мощностной РНМ с памятью напряжения."""

    algorithm_id = "pdr_sivokobylenko_5pt"
    name = "Sivokobylenko Five-Sample Memory-Polarized PDR"
    is_public = True
    requires_history = True
    history_fallback_to_earliest = True
    tunable_parameters = {
        "sample_interval_ms": 1.0,
        "mta_deg": 60.0,
        "u_min_pu": 0.05,
        "i_min_pu": 0.05,
        "p_thresh_pu": 0.001,
        "history_periods": 10.0,
        "scale_profile": "physical_pu",
    }

    def prepare_record(
        self,
        signals: np.ndarray,
        end_indices: list[int],
        *,
        sampling_rate_hz: float,
        network_frequency_hz: float,
        voltage_basis: str,
    ) -> None:
        ends = np.asarray(end_indices, dtype=np.int64)
        self._prepared_first = int(ends[0]) if ends.size else None
        self._prepared_step = int(ends[1] - ends[0]) if ends.size > 1 else 1
        self._prepared_count = int(ends.size)
        lag = _sample_lag(sampling_rate_hz, float(self.params["sample_interval_ms"]))
        h = lag / sampling_rate_hz
        omega = 2.0 * math.pi * network_frequency_hz
        theta = omega * h
        gain = 1.0 + 2.0 * math.cos(theta) + 2.0 * math.cos(2.0 * theta)
        self._prepared_lag = lag
        self._prepared_gain = gain
        self._prepared_window_ms = 1000.0 * 4.0 * lag / sampling_rate_hz
        self._prepared_delay_ms = 1000.0 * 2.0 * lag / sampling_rate_hz
        if not ends.size or abs(gain) < 1e-8:
            self._prepared_i1 = np.full(ends.size, np.nan + 1j * np.nan)
            return

        indices = np.vstack((
            ends - 4 * lag,
            ends - 3 * lag,
            ends - 2 * lag,
            ends - lag,
            ends,
        ))

        def estimator(samples: np.ndarray) -> np.ndarray:
            center = np.sum(samples, axis=1) / gain
            derivative = (
                samples[:, 0] - 8.0 * samples[:, 1]
                + 8.0 * samples[:, 3] - samples[:, 4]
            ) / (12.0 * h)
            return center - 1j * derivative / omega

        _, self._prepared_i1 = _vectorized_sequences(
            signals, indices, estimator, voltage_basis
        )

    def compute(self, input_data: PDRInputData) -> PDROutput:
        context = _raw_context(input_data)
        if context is None:
            return _threshold_output("raw_samples_unavailable", unlabeled=True)
        _, end_idx, sampling_rate, frequency = context
        lag = _sample_lag(sampling_rate, float(self.params["sample_interval_ms"]))
        indices = [end_idx - 4 * lag, end_idx - 3 * lag, end_idx - 2 * lag, end_idx - lag, end_idx]
        if min(indices) < 0:
            return _threshold_output("five_sample_window_unavailable", unlabeled=True)
        h = lag / sampling_rate
        omega = 2.0 * math.pi * frequency
        theta = omega * h
        averaging_gain = 1.0 + 2.0 * math.cos(theta) + 2.0 * math.cos(2.0 * theta)
        if abs(averaging_gain) < 1e-8:
            return _threshold_output("invalid_averaging_gain", unlabeled=True)

        prepared_position = _prepared_position(self, end_idx)
        if prepared_position is not None:
            i1 = self._prepared_i1[prepared_position]
            lag = self._prepared_lag
            averaging_gain = self._prepared_gain
        else:
            def estimator(samples: np.ndarray) -> complex:
                center_value = float(np.sum(samples) / averaging_gain)
                derivative = float(
                    (samples[0] - 8.0 * samples[1] + 8.0 * samples[3] - samples[4])
                    / (12.0 * h)
                )
                return complex(center_value, -derivative / omega)

            phase_vectors = _complex_channels(input_data, estimator, indices)
            if phase_vectors is None:
                return _threshold_output("insufficient_raw_phase_channels", unlabeled=True)
            _, currents = phase_vectors
            i1 = compute_positive_sequence(currents, is_voltage=False)
        u1_mem = compute_positive_sequence(
            input_data.history_phasors_u or {}, is_voltage=True
        )
        if i1 is None or u1_mem is None or not np.isfinite([i1, u1_mem]).all():
            return _threshold_output("memory_or_current_sequence_unavailable", unlabeled=True)

        u_min, i_min, p_thresh = scale_thresholds_for_profile(
            float(self.params["u_min_pu"]),
            float(self.params["i_min_pu"]),
            float(self.params["p_thresh_pu"]),
            str(self.params["scale_profile"]),
        )
        if abs(u1_mem) < u_min:
            return _threshold_output("memory_voltage_below_threshold", unlabeled=True, margin=abs(u1_mem)-u_min)
        if abs(i1) < i_min:
            return _threshold_output("current_below_threshold", unlabeled=False, margin=abs(i1)-i_min)

        mta_rad = math.radians(float(self.params["mta_deg"]))
        i1_rotated = i1 * np.exp(1j * mta_rad)
        operating_power = float(np.real(u1_mem * np.conj(i1_rotated)))
        margin = operating_power - p_thresh
        direction = PDRDirection.FORWARD if margin >= 0.0 else PDRDirection.REVERSE
        return PDROutput(
            direction=direction,
            is_tripped=direction == PDRDirection.FORWARD,
            margin=margin,
            diagnostics={
                "operating_power": operating_power,
                "sample_lag": lag,
                "actual_window_ms": 1000.0 * 4.0 * lag / sampling_rate,
                "estimate_delay_ms": 1000.0 * 2.0 * lag / sampling_rate,
                "averaging_gain": averaging_gain,
            },
        )


class BMRZReactiveAssistedPDRAlgorithm(PDRAlgorithm):
    """Мощностной РНМ БМРЗ с реактивно-токовой ветвью при U2."""

    algorithm_id = "pdr_bmrz_q_assisted"
    name = "BMRZ Positive-Sequence Power PDR with Reactive-Current Assist"
    is_public = True
    is_stateful = True
    tunable_parameters = {
        "u_min_pu": 0.05,
        "i_min_pu": 0.05,
        "u2_start_pu": 0.08,
        "reactive_current_start_pu": 0.05,
        "p_thresh_pu": 0.001,
        "angle_hysteresis_deg": 8.0,
        "scale_profile": "physical_pu",
    }

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._last_direction: Optional[PDRDirection] = None

    def reset_state(self) -> None:
        self._last_direction = None

    def compute(self, input_data: PDRInputData) -> PDROutput:
        u1 = compute_positive_sequence(input_data.phasors_u, is_voltage=True)
        u2 = compute_negative_sequence(input_data.phasors_u, is_voltage=True)
        i1 = compute_positive_sequence(input_data.phasors_i, is_voltage=False)
        if u1 is None or u2 is None or i1 is None or not np.isfinite([u1, u2, i1]).all():
            return _threshold_output("sequence_unavailable", unlabeled=True)

        scale_profile = str(self.params["scale_profile"])
        u_min, i_min, p_thresh = scale_thresholds_for_profile(
            float(self.params["u_min_pu"]),
            float(self.params["i_min_pu"]),
            float(self.params["p_thresh_pu"]),
            scale_profile,
        )
        u2_start, reactive_start, _ = scale_thresholds_for_profile(
            float(self.params["u2_start_pu"]),
            float(self.params["reactive_current_start_pu"]),
            0.0,
            scale_profile,
        )
        if abs(u1) < u_min:
            return _threshold_output("voltage_below_threshold", unlabeled=True, margin=abs(u1)-u_min)
        if abs(i1) < i_min:
            self._last_direction = PDRDirection.REVERSE
            return _threshold_output("current_below_threshold", unlabeled=False, margin=abs(i1)-i_min)

        phi_deg = _wrap_angle_deg(math.degrees(np.angle(u1) - np.angle(i1)))
        reactive_current = float(abs(i1) * math.sin(math.radians(phi_deg)))
        operating_power = float(np.real(u1 * np.conj(i1)))
        reactive_assist = abs(u2) > u2_start and reactive_current > reactive_start

        if reactive_assist:
            direction = PDRDirection.REVERSE
            margin = -(reactive_current - reactive_start)
        else:
            half_hysteresis = float(self.params["angle_hysteresis_deg"]) / 2.0
            abs_phi = abs(phi_deg)
            if abs_phi <= 90.0 - half_hysteresis and operating_power >= p_thresh:
                direction = PDRDirection.FORWARD
            elif abs_phi >= 90.0 + half_hysteresis or operating_power <= -p_thresh:
                direction = PDRDirection.REVERSE
            else:
                direction = self._last_direction or (
                    PDRDirection.FORWARD if operating_power >= p_thresh else PDRDirection.REVERSE
                )
            margin = operating_power - p_thresh

        self._last_direction = direction
        return PDROutput(
            direction=direction,
            is_tripped=direction == PDRDirection.FORWARD,
            margin=margin,
            diagnostics={
                "u1_abs": abs(u1),
                "u2_abs": abs(u2),
                "i1_abs": abs(i1),
                "phi_deg": phi_deg,
                "operating_power": operating_power,
                "reactive_current": reactive_current,
                "reactive_assist": reactive_assist,
            },
        )


class BAVR072CrossPolarizedPDRAlgorithm(PDRAlgorithm):
    """Открытый memory/cross-polarized benchmark по принципу БАВР-072.

    Серийное устройство может использовать напряжение резервной секции.
    В общем контракте Phase 5 такого канала нет, поэтому здесь явно
    зафиксирована воспроизводимая адаптация U_pol = U1 + k_mem*U1_mem.
    Она нужна как открытый аналог для сравнения со скрытым адаптивным РНМ.
    """

    algorithm_id = "pdr_bavr072_crosspol"
    name = "BAVR-072 Inspired Memory Cross-Polarized PDR"
    is_public = True
    requires_history = True
    history_fallback_to_earliest = True
    tunable_parameters = {
        "u_min_pu": 0.05,
        "i_min_pu": 0.05,
        "memory_weight": 0.3,
        "mta_deg": 45.0,
        "p_thresh_pu": 0.0,
        "history_periods": 10.0,
        "scale_profile": "physical_pu",
    }

    def compute(self, input_data: PDRInputData) -> PDROutput:
        u1 = compute_positive_sequence(input_data.phasors_u, is_voltage=True)
        u1_memory = compute_positive_sequence(
            input_data.history_phasors_u or {}, is_voltage=True
        )
        i1 = compute_positive_sequence(input_data.phasors_i, is_voltage=False)
        if (
            u1 is None
            or u1_memory is None
            or i1 is None
            or not np.isfinite([u1, u1_memory, i1]).all()
        ):
            return _threshold_output("memory_or_sequence_unavailable", unlabeled=True)

        u_min, i_min, p_thresh = scale_thresholds_for_profile(
            float(self.params["u_min_pu"]),
            float(self.params["i_min_pu"]),
            float(self.params["p_thresh_pu"]),
            str(self.params["scale_profile"]),
        )
        u_polarizing = u1 + float(self.params["memory_weight"]) * u1_memory
        if abs(u_polarizing) < u_min:
            return _threshold_output(
                "polarizing_voltage_below_threshold",
                unlabeled=True,
                margin=abs(u_polarizing) - u_min,
            )
        if abs(i1) < i_min:
            return _threshold_output(
                "current_below_threshold", unlabeled=False, margin=abs(i1) - i_min
            )

        mta_rad = math.radians(float(self.params["mta_deg"]))
        operating_power = float(
            np.real(u_polarizing * np.conj(i1 * np.exp(1j * mta_rad)))
        )
        margin = operating_power - p_thresh
        direction = (
            PDRDirection.FORWARD
            if margin >= 0.0
            else PDRDirection.REVERSE
        )
        return PDROutput(
            direction=direction,
            is_tripped=direction == PDRDirection.FORWARD,
            margin=margin,
            diagnostics={
                "operating_power": operating_power,
                "u1_abs": abs(u1),
                "u1_memory_abs": abs(u1_memory),
                "u_polarizing_abs": abs(u_polarizing),
                "memory_weight": float(self.params["memory_weight"]),
                "mta_deg": float(self.params["mta_deg"]),
                "polarization_source": "current_U1_plus_memory_U1_proxy",
            },
        )
