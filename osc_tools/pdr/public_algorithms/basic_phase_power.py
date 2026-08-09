"""Базовый пофазный алгоритм РНМ по активной мощности (PhasePowerPDRAlgorithm).

Реализует пофазный расчёт электромагнитного мощностного момента по 90-градусной квадратурной схеме
для всех 3 фаз с объединением по условию «И» (AND):
- Фаза A: Top_A = |U_BC| * |I_A| * cos(arg I_A - arg U_BC - phi_mch_90)
- Фаза B: Top_B = |U_CA| * |I_B| * cos(arg I_B - arg U_CA - phi_mch_90)
- Фаза C: Top_C = |U_AB| * |I_C| * cos(arg I_C - arg U_AB - phi_mch_90)

Уставка по углу: phi_mch_deg = 45.0° (ток отстает от Ua на 45°, уставка для 90° схемы phi_mch_90 = phi_mch - 90° = -45.0°).

Совпадение с техникой:
- БМРЗ-БАВР-01 / БМРЗ-БАВР-56 (НТЦ «Механотроника»).
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


class PhasePowerPDRAlgorithm(PDRAlgorithm):
    """Публичный базовый пофазный алгоритм РНМ по активной мощности (90° схема)."""

    algorithm_id = "phase_power_pdr_basic"
    name = "Basic Phase Power PDR Algorithm"
    is_public = True
    tunable_parameters = {
        "phi_mch_deg": 45.0,
        "i_min_pu": 0.05,
        "u_min_pu": 0.05,
        "scale_profile": "physical_pu",
    }

    def compute(self, input_data: PDRInputData) -> PDROutput:
        phi_mch = float(self.params["phi_mch_deg"])
        # Для 90-градусной схемы линейное напряжение Ubc опережает Ua на 90°, поэтому физический угол чувствительности: -45°
        phi_mch_90_rad = math.radians(phi_mch - 90.0)

        raw_i_min = float(self.params["i_min_pu"])
        raw_u_min = float(self.params["u_min_pu"])
        scale_prof = str(self.params.get("scale_profile", "physical_pu"))

        u_min, i_min, _ = scale_thresholds_for_profile(raw_u_min, raw_i_min, 0.0, scale_prof)

        uv = derive_unified_voltages(input_data.phasors_u)
        if uv.mode == "invalid":
            return PDROutput(
                direction=PDRDirection.UNLABELED,
                is_tripped=False,
                margin=0.0,
                diagnostics={"reason": "missing_voltage_signals"},
            )
        currents = derive_unified_currents(input_data.phasors_i)
        if set(currents) != {"A", "B", "C"}:
            return PDROutput(
                direction=PDRDirection.UNLABELED,
                is_tripped=False,
                margin=0.0,
                confidence=0.0,
                diagnostics={"reason": "missing_current_signals"},
            )

        phase_quads = {
            "A": ("A", "B", "C", "BC"),
            "B": ("B", "C", "A", "CA"),
            "C": ("C", "A", "B", "AB"),
        }

        phase_results: Dict[str, Dict[str, Any]] = {}
        forward_count = 0
        margins: list[float] = []

        for phase, (i_ch, u_ch1, u_ch2, u_ln_ch) in phase_quads.items():
            i_ph = currents[i_ch]

            if i_ph is None or not np.isfinite(i_ph) or abs(i_ph) < i_min:
                phase_results[phase] = {"direction": PDRDirection.REVERSE, "reason": "current_below_threshold"}
                i_abs = abs(i_ph) if i_ph is not None and np.isfinite(i_ph) else 0.0
                margins.append(i_abs - i_min)
                continue

            u1_raw = uv.u_phase.get(u_ch1)
            u2_raw = uv.u_phase.get(u_ch2)

            u1_mem = get_memory_voltage(u1_raw, input_data.history_phasors_u, key=u_ch1, u_min_thresh=u_min)
            u2_mem = get_memory_voltage(u2_raw, input_data.history_phasors_u, key=u_ch2, u_min_thresh=u_min)

            if u1_mem is None or u2_mem is None or not (np.isfinite(u1_mem) and np.isfinite(u2_mem)):
                phase_results[phase] = {"direction": PDRDirection.REVERSE, "reason": "voltage_missing"}
                continue

            u_ln = u1_mem - u2_mem
            u_ln_abs = abs(u_ln)
            i_abs = abs(i_ph)

            if u_ln_abs < u_min:
                phase_results[phase] = {"direction": PDRDirection.REVERSE, "reason": "voltage_below_threshold"}
                margins.append(u_ln_abs - u_min)
                continue

            phi_diff = math.atan2(i_ph.imag, i_ph.real) - math.atan2(u_ln.imag, u_ln.real)
            t_op = u_ln_abs * i_abs * math.cos(phi_diff - phi_mch_90_rad)

            # Единый контракт margin: положительное значение внутри зоны
            # FORWARD, отрицательное — снаружи.
            margin_ph = t_op
            margins.append(margin_ph)

            if t_op > 0.0:
                ph_dir = PDRDirection.FORWARD
                forward_count += 1
            else:
                ph_dir = PDRDirection.REVERSE

            phase_results[phase] = {
                "direction": ph_dir,
                "t_op": t_op,
                "margin": margin_ph,
                "u_ln_abs": u_ln_abs,
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
            overall_margin = float(np.min(margins)) if margins else 0.0

        return PDROutput(
            direction=overall_direction,
            is_tripped=is_tripped,
            margin=overall_margin,
            diagnostics={
                "forward_count": forward_count,
                "valid_count": valid_count,
                "voltage_mode": uv.mode,
                "phase_results": phase_results,
            },
        )
