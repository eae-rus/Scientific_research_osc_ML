"""Базовый пофазный алгоритм РНМ (PhasePDRAlgorithm).

Выполняет пофазный анализ угла сдвига между напряжением и током с применением логики 'И' (AND).
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


class PhasePDRAlgorithm(PDRAlgorithm):
    """Базовый фазный алгоритм РНМ (публичная версия).

    Выполняет пофазный анализ угла сдвига между напряжением и током:
    phi_k = arg(U_k) - arg(I_k).

    Результаты:
    - FORWARD (1): Прямое направление мощности (при наличии прямого направления во ВСЕХ фазах).
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

            u_angle = math.degrees(math.atan2(u_ph.imag, u_ph.real))
            i_angle = math.degrees(math.atan2(i_ph.imag, i_ph.real))
            phi = _wrap_angle_deg(u_angle - i_angle)
            delta_phi = _wrap_angle_deg(phi - phi_mch)

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
                "u_abs": u_abs,
                "i_abs": i_abs,
            }

        valid_phase_count = len(margins)
        # Логика "И": все имеющиеся/измеренные фазы должны зафиксировать прямое направление
        final_dir = (
            PDRDirection.FORWARD
            if (valid_phase_count > 0 and forward_count == valid_phase_count)
            else PDRDirection.REVERSE
        )
        total_margin = float(np.mean(margins)) if margins else 0.0
        is_tripped = (final_dir == PDRDirection.FORWARD)

        return PDROutput(
            direction=final_dir,
            is_tripped=is_tripped,
            margin=total_margin,
            confidence=1.0 if margins else 0.0,
            diagnostics={
                "phases": phase_results,
                "forward_count": forward_count,
                "valid_phase_count": valid_phase_count,
            },
        )
