"""Публичные открытые алгоритмы РНМ (PDR).

Содержит:
1. PhasePDRAlgorithm — базовый фазный алгоритм РНМ (FORWARD=1, REVERSE=0).
2. PositiveSequencePDRAlgorithm — базовый алгоритм РНМ прямой последовательности.
3. Заглушки для исследования и быстрой интеграции алгоритмов БАВР других производителей.
"""

from __future__ import annotations

import math
from typing import Dict, Any
import numpy as np

from .base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection


def _wrap_angle_deg(angle_deg: float) -> float:
    """Приведение угла в градусах к диапазону [-180, 180]."""
    res = (angle_deg + 180.0) % 360.0 - 180.0
    return res if res != -180.0 else 180.0


def _symmetrical_components_positive(
    phasor_a: complex, phasor_b: complex, phasor_c: complex
) -> complex:
    """Вычисление прямой последовательности h1: X_1 = 1/3 * (X_A + a*X_B + a^2*X_C)."""
    a = complex(-0.5, math.sqrt(3) / 2.0)
    a_sq = complex(-0.5, -math.sqrt(3) / 2.0)
    return (phasor_a + a * phasor_b + a_sq * phasor_c) / 3.0


class PhasePDRAlgorithm(PDRAlgorithm):
    """Базовый фазный алгоритм РНМ (публичная версия).

    Выполняет пофазный анализ угла сдвига между напряжением и током:
    phi_k = arg(U_k) - arg(I_k).

    Результаты:
    - FORWARD (1): Прямое направление мощности.
    - REVERSE (0): Обратное направление / Блокировка / Малый сигнал.

    Уставки в относительно-именуемых единицах (per-unit):
    - phi_mch_deg: Угол максимальной чувствительности (по умолчанию 45.0 град, ток отстает от U)
    - u_min_pu: Порог по модулю напряжения (по умолчанию 0.05 о.е.)
    - i_min_pu: Порог по модулю тока (по умолчанию 0.02 о.е.)
    - sector_width_deg: Ширина зоны срабатывания (по умолчанию 180.0 град, т.е. +/- 90 от phi_mch)
    """

    algorithm_id = "phase_pdr_basic"
    name = "Basic Phase PDR Algorithm"
    is_public = True
    tunable_parameters = {
        "phi_mch_deg": 45.0,
        "u_min_pu": 0.05,
        "i_min_pu": 0.02,
        "sector_width_deg": 180.0,
    }

    def compute(self, input_data: PDRInputData) -> PDROutput:
        phi_mch = float(self.params["phi_mch_deg"])
        u_min = float(self.params["u_min_pu"])
        i_min = float(self.params["i_min_pu"])
        half_sector = float(self.params["sector_width_deg"]) / 2.0

        phase_results: Dict[str, Dict[str, Any]] = {}
        forward_count = 0
        margins: list[float] = []

        for phase in ("A", "B", "C"):
            u_ph = input_data.phasors_u.get(phase)
            i_ph = input_data.phasors_i.get(phase)

            if u_ph is None or i_ph is None or not (np.isfinite(u_ph) and np.isfinite(i_ph)):
                phase_results[phase] = {"direction": PDRDirection.REVERSE, "reason": "missing_signal"}
                continue

            u_abs = abs(u_ph)
            i_abs = abs(i_ph)

            if u_abs < u_min or i_abs < i_min:
                phase_results[phase] = {
                    "direction": PDRDirection.REVERSE,
                    "reason": "below_threshold",
                    "u_abs": u_abs,
                    "i_abs": i_abs,
                }
                continue

            # Фазовые углы в градусах
            u_angle = math.degrees(math.atan2(u_ph.imag, u_ph.real))
            i_angle = math.degrees(math.atan2(i_ph.imag, i_ph.real))
            # Угол между напряжением и током: phi = arg(U) - arg(I)
            phi = _wrap_angle_deg(u_angle - i_angle)
            delta_phi = _wrap_angle_deg(phi - phi_mch)

            # Запас срабатывания: P_dir = |U|*|I|*cos(delta_phi)
            margin_ph = u_abs * i_abs * math.cos(math.radians(delta_phi))
            margins.append(margin_ph)

            if abs(delta_phi) <= half_sector:
                ph_dir = PDRDirection.FORWARD
                forward_count += 1
            else:
                ph_dir = PDRDirection.REVERSE

            phase_results[phase] = {
                "direction": ph_dir,
                "phi_deg": phi,
                "delta_phi_deg": delta_phi,
                "margin": margin_ph,
                "u_abs": u_abs,
                "i_abs": i_abs,
            }

        # Если хотя бы в одной фазе зафиксировано прямое направление
        final_dir = PDRDirection.FORWARD if forward_count > 0 else PDRDirection.REVERSE
        total_margin = float(np.mean(margins)) if margins else 0.0
        is_tripped = (final_dir == PDRDirection.FORWARD)

        return PDROutput(
            direction=final_dir,
            is_tripped=is_tripped,
            margin=total_margin,
            confidence=1.0 if margins else 0.0,
            diagnostics={"phases": phase_results, "forward_count": forward_count},
        )


class PositiveSequencePDRAlgorithm(PDRAlgorithm):
    """Базовый алгоритм РНМ прямой последовательности (публичная версия).

    Результаты:
    - FORWARD (1): Прямое направление.
    - REVERSE (0): Обратное направление.

    Уставки в относительно-именуемых единицах (per-unit):
    - phi_mch_deg: Угол максимальной чувствительности (по умолчанию 45.0 град)
    - u1_min_pu: Порог по напряжению прямой последовательности (по умолчанию 0.05 о.е.)
    - i1_min_pu: Порог по току прямой последовательности (по умолчанию 0.02 о.е.)
    - sector_width_deg: Ширина зоны срабатывания (по умолчанию 180.0 град)
    """

    algorithm_id = "pos_seq_pdr_basic"
    name = "Basic Positive Sequence PDR Algorithm"
    is_public = True
    tunable_parameters = {
        "phi_mch_deg": 45.0,
        "u1_min_pu": 0.05,
        "i1_min_pu": 0.02,
        "sector_width_deg": 180.0,
    }

    def compute(self, input_data: PDRInputData) -> PDROutput:
        phi_mch = float(self.params["phi_mch_deg"])
        u1_min = float(self.params["u1_min_pu"])
        i1_min = float(self.params["i1_min_pu"])
        half_sector = float(self.params["sector_width_deg"]) / 2.0

        ua = input_data.phasors_u.get("A")
        ub = input_data.phasors_u.get("B")
        uc = input_data.phasors_u.get("C")
        ia = input_data.phasors_i.get("A")
        ib = input_data.phasors_i.get("B")
        ic = input_data.phasors_i.get("C")

        valid_u = ua is not None and ub is not None and uc is not None and np.isfinite([ua, ub, uc]).all()
        valid_i = ia is not None and ib is not None and ic is not None and np.isfinite([ia, ib, ic]).all()

        if not (valid_u and valid_i):
            return PDROutput(
                direction=PDRDirection.REVERSE,
                is_tripped=False,
                margin=0.0,
                confidence=0.0,
                diagnostics={"reason": "missing_three_phase_signals"},
            )

        u1 = _symmetrical_components_positive(ua, ub, uc)
        i1 = _symmetrical_components_positive(ia, ib, ic)

        u1_abs = abs(u1)
        i1_abs = abs(i1)

        if u1_abs < u1_min or i1_abs < i1_min:
            return PDROutput(
                direction=PDRDirection.REVERSE,
                is_tripped=False,
                margin=0.0,
                confidence=0.0,
                diagnostics={"reason": "below_threshold", "u1_abs": u1_abs, "i1_abs": i1_abs},
            )

        u1_angle = math.degrees(math.atan2(u1.imag, u1.real))
        i1_angle = math.degrees(math.atan2(i1.imag, i1.real))
        phi1 = _wrap_angle_deg(u1_angle - i1_angle)
        delta_phi = _wrap_angle_deg(phi1 - phi_mch)

        margin = u1_abs * i1_abs * math.cos(math.radians(delta_phi))

        if abs(delta_phi) <= half_sector:
            direction = PDRDirection.FORWARD
        else:
            direction = PDRDirection.REVERSE

        return PDROutput(
            direction=direction,
            is_tripped=(direction == PDRDirection.FORWARD),
            margin=margin,
            confidence=1.0,
            diagnostics={
                "u1_abs": u1_abs,
                "i1_abs": i1_abs,
                "phi1_deg": phi1,
                "delta_phi_deg": delta_phi,
            },
        )


class ManufacturerPowerPDRStub(PDRAlgorithm):
    """Публичная заглушка-напоминание для алгоритмов РНМ по мощности других производителей БАВР."""

    algorithm_id = "bavr_manufacturer_power_stub"
    name = "BAVR Manufacturer Power PDR (Stub / Research Task)"
    is_public = True
    tunable_parameters = {"power_threshold_pu": 0.01}

    def compute(self, input_data: PDRInputData) -> PDROutput:
        return PDROutput(
            direction=PDRDirection.REVERSE,
            is_tripped=False,
            margin=0.0,
            confidence=0.0,
            diagnostics={"status": "stub_pending_research_implementation"},
        )


class ManufacturerCurrentPDRStub(PDRAlgorithm):
    """Публичная заглушка-напоминание для алгоритмов РНМ с токовыми адаптивностями других производителей."""

    algorithm_id = "bavr_manufacturer_current_stub"
    name = "BAVR Manufacturer Current PDR (Stub / Research Task)"
    is_public = True
    tunable_parameters = {"i_threshold_pu": 0.02}

    def compute(self, input_data: PDRInputData) -> PDROutput:
        return PDROutput(
            direction=PDRDirection.REVERSE,
            is_tripped=False,
            margin=0.0,
            confidence=0.0,
            diagnostics={"status": "stub_pending_research_implementation"},
        )
