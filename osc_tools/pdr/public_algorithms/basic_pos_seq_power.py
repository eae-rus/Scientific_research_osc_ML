"""Базовый алгоритм РНМ по мощностному моменту прямой последовательности (PositiveSequencePowerPDRAlgorithm).

Реализует электромагнитный момент / мощность прямой последовательности (Torque Equation):
T_op = Re[ V1_pol * (I1 * exp(-j * MTA))* ]

Уставка по углу: mta_deg = 45.0° (MTA / phi_mch, ток отстает от U).

Совпадение с техникой:
- SEL 32P (Schweitzer Engineering Laboratories);
- ЭКРА 217 БАВР (P1 < -P_thresh);
- ABB Relion (DOPPDPR 32R).
"""

from __future__ import annotations

import math
from typing import Dict, Any
import numpy as np

from osc_tools.pdr.base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection
from osc_tools.pdr.pdr_signal_utils import compute_positive_sequence, get_memory_voltage, scale_thresholds_for_profile


class PositiveSequencePowerPDRAlgorithm(PDRAlgorithm):
    """Публичный базовый алгоритм РНМ по мощностному моменту прямой последовательности."""

    algorithm_id = "pos_seq_power_pdr_basic"
    name = "Basic Positive Sequence Power PDR Algorithm"
    is_public = True
    tunable_parameters = {
        "mta_deg": 45.0,
        "u_min_pu": 0.05,
        "i_min_pu": 0.05,
        "p_thresh_pu": 0.0866,
        "scale_profile": "physical_pu",
    }

    def compute(self, input_data: PDRInputData) -> PDROutput:
        mta_rad = math.radians(float(self.params["mta_deg"]))
        raw_u_min = float(self.params["u_min_pu"])
        raw_i_min = float(self.params["i_min_pu"])
        raw_p_thresh = float(self.params["p_thresh_pu"])
        scale_prof = str(self.params.get("scale_profile", "physical_pu"))

        u_min, i_min, p_thresh = scale_thresholds_for_profile(raw_u_min, raw_i_min, raw_p_thresh, scale_prof)

        i1 = compute_positive_sequence(input_data.phasors_i, is_voltage=False)

        if i1 is None or not np.isfinite(i1):
            return PDROutput(
                direction=PDRDirection.UNLABELED,
                is_tripped=False,
                margin=0.0,
                diagnostics={"reason": "missing_current_sequence"},
            )

        u1_raw = compute_positive_sequence(input_data.phasors_u, is_voltage=True)

        u1 = get_memory_voltage(
            current_u=u1_raw,
            history_phasors_u=input_data.history_phasors_u,
            key="1",
            u_min_thresh=u_min,
        )

        if u1 is None or not np.isfinite(u1) or abs(u1) < 1e-5:
            return PDROutput(
                direction=PDRDirection.REVERSE,
                is_tripped=False,
                margin=0.0,
                diagnostics={"reason": "voltage_below_threshold"},
            )

        u1_abs = abs(u1)
        i1_abs = abs(i1)

        # Момент / Мощность: T_op = Re[ V1 * (I1 * exp(-j * MTA))* ]
        i1_rot = i1 * np.exp(-1j * mta_rad)
        t_op = float(np.real(u1 * np.conj(i1_rot)))

        if t_op >= -p_thresh:
            direction = PDRDirection.FORWARD
            is_tripped = True
        else:
            direction = PDRDirection.REVERSE
            is_tripped = False

        margin = abs(t_op - (-p_thresh))

        return PDROutput(
            direction=direction,
            is_tripped=is_tripped,
            margin=margin,
            diagnostics={
                "t_op": t_op,
                "p_thresh": p_thresh,
                "u1_abs": u1_abs,
                "i1_abs": i1_abs,
                "mta_deg": self.params["mta_deg"],
            },
        )
