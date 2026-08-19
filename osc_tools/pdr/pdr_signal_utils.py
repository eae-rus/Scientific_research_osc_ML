"""Утилиты работы с фазорами напряжений и токов, восстановлением линейных/фазных цепей и памятью предыстории.

Обеспечивают:
- Восстановление 3-го линейного напряжения для ВСЕХ комбинаций пары (AB/BC, BC/CA, CA/AB).
- Расчёт эквивалентных фазных напряжений из 3 линейных (Ua = (Uab - Uca)/3).
- Восстановление любого 3-го недостающего фазного тока из двух любых (Ia + Ib + Ic = 0).
- Расчёт напряжений и токов прямой последовательности.
- Извлечение напряжения и тока предыстории U_mem, I_mem.
- Автоматический масштаб уставок под внутренний контракт датасета (деление на 20/3/60).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, NamedTuple
import numpy as np

_A_OPERATOR = np.exp(1j * 2.0 * np.pi / 3.0)
_A2_OPERATOR = np.exp(1j * 4.0 * np.pi / 3.0)


def power_threshold_for_internal_phasors(
    current_threshold: float,
    voltage_reserve: float = 3.0,
) -> float:
    """Согласовать мощностной порог с токовым во внутренних FFT-координатах.

    Мгновенное фазное напряжение нормируется на
    ``voltage_reserve * U_line_nom``, а одночастотный DFT возвращает пиковую
    амплитуду. Поэтому номинальный фазный фазор напряжения имеет модуль
    ``sqrt(2) / (voltage_reserve * sqrt(3))``.
    """

    if current_threshold < 0.0:
        raise ValueError("Токовая уставка не может быть отрицательной")
    if voltage_reserve <= 0.0:
        raise ValueError("Коэффициент запаса напряжения должен быть положительным")
    return current_threshold * math.sqrt(2.0) / (voltage_reserve * math.sqrt(3.0))


def _first_present(mapping: Dict[str, complex], *keys: str) -> Optional[complex]:
    """Вернуть первое присутствующее значение, не считая корректный ``0j`` отсутствующим."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


class UnifiedVoltagePhasors(NamedTuple):
    """Единая структура фазных и линейных напряжений."""

    u_phase: Dict[str, complex]  # {"A": ..., "B": ..., "C": ...}
    u_line: Dict[str, complex]   # {"AB": ..., "BC": ..., "CA": ...}
    mode: str                    # 'phase', 'line' или 'invalid'


def has_voltage_above_threshold(
    voltages: UnifiedVoltagePhasors,
    threshold: float,
) -> bool:
    """Проверить, что хотя бы одно фазное U пригодно для определения направления."""
    if voltages.mode == "invalid":
        return False
    return any(
        value is not None and np.isfinite(value) and abs(value) >= threshold
        for value in voltages.u_phase.values()
    )


def derive_unified_voltages(
    phasors_u: Dict[str, complex],
) -> UnifiedVoltagePhasors:
    """Универсальное восстановление фазных и линейных напряжений.

    Поддерживает:
    - 3 фазных напряжения (UA, UB, UC) -> строятся UAB, UBC, UCA.
    - ЛЮБАЯ пара из 2 линейных напряжений:
      * UAB и UBC -> UCA = -(UAB + UBC)
      * UBC и UCA -> UAB = -(UBC + UCA)
      * UCA и UAB -> UBC = -(UAB + UCA)
      После чего восстанавливаются эквивалентные фазные:
      UA = (UAB - UCA)/3, UB = (UBC - UAB)/3, UC = (UCA - UBC)/3.
    """
    empty_res = UnifiedVoltagePhasors({}, {}, "invalid")
    if not phasors_u:
        return empty_res

    ua = _first_present(phasors_u, "A", "UA")
    ub = _first_present(phasors_u, "B", "UB")
    uc = _first_present(phasors_u, "C", "UC")

    if ua is not None and ub is not None and uc is not None:
        if np.isfinite([ua, ub, uc]).all():
            u_ph = {"A": ua, "B": ub, "C": uc}
            u_ln = {"AB": ua - ub, "BC": ub - uc, "CA": uc - ua}
            return UnifiedVoltagePhasors(u_ph, u_ln, "phase")

    # При наличии любых двух фазных напряжений восстанавливаем третье из
    # UA + UB + UC = 0. Нулевая последовательность при этом недоступна, но для
    # направленных органов получается согласованная трёхфазная система.
    phase_values = {"A": ua, "B": ub, "C": uc}
    valid_phase = {
        key: value
        for key, value in phase_values.items()
        if value is not None and np.isfinite(value)
    }
    if len(valid_phase) == 2:
        missing_phase = next(key for key in ("A", "B", "C") if key not in valid_phase)
        completed_phase = dict(valid_phase)
        completed_phase[missing_phase] = -sum(valid_phase.values())
        ua, ub, uc = completed_phase["A"], completed_phase["B"], completed_phase["C"]
        u_ph = {"A": ua, "B": ub, "C": uc}
        u_ln = {"AB": ua - ub, "BC": ub - uc, "CA": uc - ua}
        return UnifiedVoltagePhasors(u_ph, u_ln, "phase_derived")

    u_ab = _first_present(phasors_u, "AB", "UAB")
    u_bc = _first_present(phasors_u, "BC", "UBC")
    u_ca = _first_present(phasors_u, "CA", "UCA")

    if u_ab is not None and u_bc is not None and np.isfinite([u_ab, u_bc]).all():
        u_ca = -(u_ab + u_bc)
    elif u_bc is not None and u_ca is not None and np.isfinite([u_bc, u_ca]).all():
        u_ab = -(u_bc + u_ca)
    elif u_ca is not None and u_ab is not None and np.isfinite([u_ca, u_ab]).all():
        u_bc = -(u_ab + u_ca)
    else:
        return empty_res

    if u_ab is not None and u_bc is not None and u_ca is not None and np.isfinite([u_ab, u_bc, u_ca]).all():
        u_ln = {"AB": u_ab, "BC": u_bc, "CA": u_ca}
        ua_eq = (u_ab - u_ca) / 3.0
        ub_eq = (u_bc - u_ab) / 3.0
        uc_eq = (u_ca - u_bc) / 3.0
        u_ph = {"A": ua_eq, "B": ub_eq, "C": uc_eq}
        return UnifiedVoltagePhasors(u_ph, u_ln, "line")

    return empty_res


def derive_unified_currents(
    phasors_i: Dict[str, complex],
) -> Dict[str, complex]:
    """Универсальное восстановление любого 3-го фазного тока из 2-х любых (Ia + Ib + Ic = 0).

    Returns:
        Словарь фазных токов {"A": ..., "B": ..., "C": ...}
    """
    if not phasors_i:
        return {}

    ia = _first_present(phasors_i, "A", "IA")
    ib = _first_present(phasors_i, "B", "IB")
    ic = _first_present(phasors_i, "C", "IC")

    ia_ok = ia is not None and np.isfinite(ia)
    ib_ok = ib is not None and np.isfinite(ib)
    ic_ok = ic is not None and np.isfinite(ic)

    if ia_ok and ib_ok and ic_ok:
        return {"A": ia, "B": ib, "C": ic}
    elif ia_ok and ib_ok:
        return {"A": ia, "B": ib, "C": -(ia + ib)}
    elif ib_ok and ic_ok:
        return {"A": -(ib + ic), "B": ib, "C": ic}
    elif ia_ok and ic_ok:
        return {"A": ia, "B": -(ia + ic), "C": ic}

    res = {}
    if ia_ok: res["A"] = ia
    if ib_ok: res["B"] = ib
    if ic_ok: res["C"] = ic
    return res


def compute_positive_sequence(
    phasors: Dict[str, complex],
    is_voltage: bool = False,
) -> Optional[complex]:
    """Вычисление фазора прямой последовательности X1."""
    if not phasors:
        return None

    if is_voltage:
        uv = derive_unified_voltages(phasors)
        if uv.mode != "invalid":
            ua, ub, uc = uv.u_phase["A"], uv.u_phase["B"], uv.u_phase["C"]
            return (ua + ub * _A_OPERATOR + uc * _A2_OPERATOR) / 3.0
        return None
    else:
        currents = derive_unified_currents(phasors)
        ia, ib, ic = currents.get("A"), currents.get("B"), currents.get("C")
        if ia is not None and ib is not None and ic is not None:
            if np.isfinite([ia, ib, ic]).all():
                return (ia + ib * _A_OPERATOR + ic * _A2_OPERATOR) / 3.0
        return None


def get_memory_voltage(
    current_u: Optional[complex],
    history_phasors_u: Optional[Dict[str, complex]],
    key: str = "1",
    u_min_thresh: float = 0.05,
) -> Optional[complex]:
    """Получение напряжения с поддержкой памяти предыстории (U_mem)."""
    if current_u is not None and abs(current_u) >= u_min_thresh:
        return current_u

    if history_phasors_u is not None:
        if key == "1":
            mem_u = compute_positive_sequence(history_phasors_u, is_voltage=True)
            if mem_u is not None and np.isfinite(mem_u) and abs(mem_u) > 1e-4:
                return mem_u
        else:
            uv_hist = derive_unified_voltages(history_phasors_u)
            mem_u = _first_present(uv_hist.u_phase, key)
            if mem_u is None:
                mem_u = _first_present(uv_hist.u_line, key)
            if mem_u is not None and np.isfinite(mem_u) and abs(mem_u) > 1e-4:
                return mem_u

    return current_u


def scale_thresholds_for_profile(
    u_min: float,
    i_min: float,
    p_thresh: float,
    scale_profile: str = "physical_pu",
    current_reserve: float = 20.0,
    voltage_reserve: float = 3.0,
) -> Tuple[float, float, float]:
    """Автоматический пересчёт уставок в зависимости от профиля входных сигналов."""
    if scale_profile == "dataset_peak_phasor":
        # Исходные параметры заданы в физических RMS p.u., тогда как Phase 5
        # хранит мгновенные токи как I/(20*I_nom), напряжения как
        # U_phase/(3*U_line_nom), а одночастотный DFT возвращает peak-фазор.
        i_eff = i_min * math.sqrt(2.0) / current_reserve
        u_eff = u_min * math.sqrt(2.0) / (voltage_reserve * math.sqrt(3.0))
        p_eff = p_thresh * 2.0 / (
            current_reserve * voltage_reserve * math.sqrt(3.0)
        )
        return u_eff, i_eff, p_eff
    if scale_profile == "dataset_internal":
        i_eff = i_min / current_reserve
        u_eff = u_min / voltage_reserve
        p_eff = p_thresh / (current_reserve * voltage_reserve)
        return u_eff, i_eff, p_eff
    return u_min, i_min, p_thresh
