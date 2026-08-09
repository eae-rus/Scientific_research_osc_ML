"""Базовый пофазный угловой алгоритм РНМ (PhasePDRAlgorithm).

Реализует пофазный анализ угла сдвига между напряжением и током.
Поддерживает поляризацию:
- polarization_mode = 'quadrature_90' (по умолчанию): Ia <-> Ubc, Ib <-> Uca, Ic <-> Uab.
- polarization_mode = 'direct': Ia <-> Ua, Ib <-> Ub, Ic <-> Uc.

Уставка угла максимальной чувствительности (УМЧ / RCA): phi_mch_deg = 45.0° (ток отстает от Ua на 45°).
"""

from __future__ import annotations

import math
from typing import Dict, Any
import numpy as np

from osc_tools.pdr.base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection
from osc_tools.pdr.pdr_signal_utils import (
    derive_unified_currents,
    derive_unified_voltages,
    get_memory_voltage,
    scale_thresholds_for_profile,
)


def _wrap_angle_deg(angle_deg: float) -> float:
    """Детерминированное приведение угла в градусах к диапазону (-180, 180]."""
    res = (angle_deg + 180.0) % 360.0 - 180.0
    if res <= -180.0:
        return 180.0
    return res


class PhasePDRAlgorithm(PDRAlgorithm):
    """Публичный базовый пофазный угловой алгоритм РНМ."""

    algorithm_id = "phase_pdr_basic"
    name = "Basic Phase Angular PDR Algorithm"
    is_public = True
    tunable_parameters = {
        "phi_mch_deg": 45.0,
        "u_min_pu": 0.05,
        "i_min_pu": 0.05,
        "sector_width_deg": 180.0,
        "polarization_mode": "quadrature_90",
        "scale_profile": "physical_pu",
    }

    def compute(self, input_data: PDRInputData) -> PDROutput:
        phi_mch = float(self.params["phi_mch_deg"])
        raw_u_min = float(self.params["u_min_pu"])
        raw_i_min = float(self.params["i_min_pu"])
        half_sector = float(self.params["sector_width_deg"]) / 2.0
        pol_mode = str(self.params.get("polarization_mode", "quadrature_90"))
        scale_prof = str(self.params.get("scale_profile", "physical_pu"))

        u_min, i_min, _ = scale_thresholds_for_profile(raw_u_min, raw_i_min, 0.0, scale_prof)

        uv = derive_unified_voltages(input_data.phasors_u)
        if uv.mode == "invalid":
            return PDROutput(
                direction=PDRDirection.UNLABELED,
                is_tripped=False,
                margin=-half_sector,
                diagnostics={"reason": "missing_voltage_signals"},
            )
        currents = derive_unified_currents(input_data.phasors_i)
        if set(currents) != {"A", "B", "C"}:
            return PDROutput(
                direction=PDRDirection.UNLABELED,
                is_tripped=False,
                margin=-half_sector,
                confidence=0.0,
                diagnostics={"reason": "missing_current_signals"},
            )

        phase_pairs = {
            "A": ("A", "BC" if pol_mode == "quadrature_90" else "A"),
            "B": ("B", "CA" if pol_mode == "quadrature_90" else "B"),
            "C": ("C", "AB" if pol_mode == "quadrature_90" else "C"),
        }

        # Для 90° квадратурной схемы фазор Ubc отстает на 90° относительно Ua, поэтому угол МЧ = phi_mch - 90°
        target_mch = (phi_mch - 90.0) if pol_mode == "quadrature_90" else phi_mch

        phase_results: Dict[str, Dict[str, Any]] = {}
        forward_count = 0
        margins: list[float] = []

        for phase, (i_ch, u_ch) in phase_pairs.items():
            i_ph = currents[i_ch]
            u_ph_raw = uv.u_line.get(u_ch) if pol_mode == "quadrature_90" else uv.u_phase.get(u_ch)

            if i_ph is None or not np.isfinite(i_ph):
                phase_results[phase] = {"direction": PDRDirection.REVERSE, "reason": "missing_current"}
                continue

            i_abs = abs(i_ph)
            if i_abs < i_min:
                phase_results[phase] = {"direction": PDRDirection.REVERSE, "reason": "current_below_threshold"}
                margins.append(i_abs - i_min)
                continue

            u_ph = get_memory_voltage(
                current_u=u_ph_raw,
                history_phasors_u=input_data.history_phasors_u,
                key=u_ch,
                u_min_thresh=u_min,
            )

            if u_ph is None or not np.isfinite(u_ph) or abs(u_ph) < u_min:
                phase_results[phase] = {"direction": PDRDirection.REVERSE, "reason": "voltage_below_threshold"}
                u_abs = abs(u_ph) if u_ph is not None and np.isfinite(u_ph) else 0.0
                margins.append(u_abs - u_min)
                continue

            u_angle = math.degrees(math.atan2(u_ph.imag, u_ph.real))
            i_angle = math.degrees(math.atan2(i_ph.imag, i_ph.real))

            phi = _wrap_angle_deg(u_angle - i_angle)
            delta_phi = _wrap_angle_deg(phi - target_mch)

            margin_ph = half_sector - abs(delta_phi)
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
                "u_abs": abs(u_ph),
                "i_abs": i_abs,
            }

        valid_count = len(margins)
        if valid_count == 3 and forward_count == 3:
            overall_direction = PDRDirection.FORWARD
            is_tripped = True
            overall_margin = float(np.min(margins))
        else:
            overall_direction = PDRDirection.REVERSE
            is_tripped = False
            overall_margin = float(np.min(margins)) if margins else -half_sector

        return PDROutput(
            direction=overall_direction,
            is_tripped=is_tripped,
            margin=overall_margin,
            diagnostics={
                "forward_count": forward_count,
                "valid_count": valid_count,
                "polarization_mode": pol_mode,
                "voltage_mode": uv.mode,
                "phase_results": phase_results,
            },
        )
