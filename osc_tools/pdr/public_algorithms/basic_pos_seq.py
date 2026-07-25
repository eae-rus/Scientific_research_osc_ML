"""Базовый алгоритм РНМ прямой последовательности (PositiveSequencePDRAlgorithm).

Реализует секторный анализ угла сдвига между сигналами прямой последовательности U1 и I1.

Уставка угла максимальной чувствительности (УМЧ / RCA): phi_mch_deg = 45.0° (ток отстает от U).
"""

from __future__ import annotations

import math
from typing import Dict, Any
import numpy as np

from osc_tools.pdr.base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection
from osc_tools.pdr.voltage_utils import compute_positive_sequence, get_memory_voltage, scale_thresholds_for_profile


def _wrap_angle_deg(angle_deg: float) -> float:
    """Детерминированное приведение угла в градусах к диапазону (-180, 180]."""
    res = (angle_deg + 180.0) % 360.0 - 180.0
    if res <= -180.0:
        return 180.0
    return res


class PositiveSequencePDRAlgorithm(PDRAlgorithm):
    """Публичный базовый алгоритм РНМ прямой последовательности."""

    algorithm_id = "pos_seq_pdr_basic"
    name = "Basic Positive Sequence Angular PDR Algorithm"
    is_public = True
    tunable_parameters = {
        "phi_mch_deg": 45.0,
        "u_min_pu": 0.05,
        "i_min_pu": 0.05,
        "sector_width_deg": 180.0,
        "scale_profile": "physical_pu",
    }

    def compute(self, input_data: PDRInputData) -> PDROutput:
        phi_mch = float(self.params["phi_mch_deg"])
        raw_u1_min = float(self.params["u_min_pu"])
        raw_i1_min = float(self.params["i_min_pu"])
        half_sector = float(self.params["sector_width_deg"]) / 2.0
        scale_prof = str(self.params.get("scale_profile", "physical_pu"))

        u1_min, i1_min, _ = scale_thresholds_for_profile(raw_u1_min, raw_i1_min, 0.0, scale_prof)

        i1 = compute_positive_sequence(input_data.phasors_i, is_voltage=False)

        if i1 is None or not np.isfinite(i1):
            return PDROutput(
                direction=PDRDirection.REVERSE,
                is_tripped=False,
                margin=-half_sector,
                diagnostics={"reason": "missing_current_sequence"},
            )

        i1_abs = abs(i1)
        if i1_abs < i1_min:
            return PDROutput(
                direction=PDRDirection.REVERSE,
                is_tripped=False,
                margin=-half_sector,
                diagnostics={"reason": "current_below_threshold", "i1_abs": i1_abs},
            )

        u1_raw = compute_positive_sequence(input_data.phasors_u, is_voltage=True)

        u1 = get_memory_voltage(
            current_u=u1_raw,
            history_phasors_u=input_data.history_phasors_u,
            key="1",
            u_min_thresh=u1_min,
        )

        if u1 is None or not np.isfinite(u1) or abs(u1) < 1e-5:
            return PDROutput(
                direction=PDRDirection.REVERSE,
                is_tripped=False,
                margin=-half_sector,
                diagnostics={"reason": "voltage_below_threshold"},
            )

        u1_abs = abs(u1)
        u1_angle = math.degrees(math.atan2(u1.imag, u1.real))
        i1_angle = math.degrees(math.atan2(i1.imag, i1.real))

        phi1 = _wrap_angle_deg(u1_angle - i1_angle)
        delta_phi1 = _wrap_angle_deg(phi1 - phi_mch)

        margin = half_sector - abs(delta_phi1)

        if abs(delta_phi1) <= half_sector:
            direction = PDRDirection.FORWARD
            is_tripped = True
        else:
            direction = PDRDirection.REVERSE
            is_tripped = False

        return PDROutput(
            direction=direction,
            is_tripped=is_tripped,
            margin=margin,
            diagnostics={
                "phi1_deg": phi1,
                "delta_phi1_deg": delta_phi1,
                "u1_abs": u1_abs,
                "i1_abs": i1_abs,
            },
        )
