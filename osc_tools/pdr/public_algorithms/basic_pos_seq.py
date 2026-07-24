"""Базовый алгоритм РНМ прямой последовательности (PositiveSequencePDRAlgorithm).

Оценивает направление мощности по симметричным составляющим U1, I1 первой гармоники.
"""

from __future__ import annotations

import math
from typing import Dict, Any
import numpy as np

from osc_tools.pdr.base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection


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

        margin = half_sector - abs(delta_phi)

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
