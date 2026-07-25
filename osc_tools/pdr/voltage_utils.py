"""Утилиты работы с фазорами напряжений, восстановлением любых линейных цепей, восстановлением токов и памятью предыстории.

Обеспечивают:
- Восстановление 3-го линейного напряжения для ВСЕХ комбинаций пары (AB/BC, BC/CA, CA/AB).
- Расчёт эквивалентных фазных напряжений из 3 линейных (Ua = (Uab - Uca)/3).
- Восстановление любого 3-го недостающего фазного тока из двух любых (Ia + Ib + Ic = 0).
- Расчёт напряжений и токов прямой последовательности.
- Извлечение напряжения и тока предыстории U_mem, I_mem.
- Автоматический масштаб уставок под внутренний контракт датасета (деление на 20/3/60).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, NamedTuple
import numpy as np

_A_OPERATOR = np.exp(1j * 2.0 * np.pi / 3.0)
_A2_OPERATOR = np.exp(1j * 4.0 * np.pi / 3.0)


class UnifiedVoltagePhasors(NamedTuple):
    """Единая структура фазных и линейных напряжений."""

    u_phase: Dict[str, complex]  # {"A": ..., "B": ..., "C": ...}
    u_line: Dict[str, complex]   # {"AB": ..., "BC": ..., "CA": ...}
    mode: str                    # 'phase', 'line' или 'invalid'


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

    # 1. Попытка извлечения 3 фазных напряжений
    ua = phasors_u.get("A") or phasors_u.get("UA")
    ub = phasors_u.get("B") or phasors_u.get("UB")
    uc = phasors_u.get("C") or phasors_u.get("UC")

    if ua is not None and ub is not None and uc is not None:
        if np.isfinite([ua, ub, uc]).all():
            u_ph = {"A": ua, "B": ub, "C": uc}
            u_ln = {"AB": ua - ub, "BC": ub - uc, "CA": uc - ua}
            return UnifiedVoltagePhasors(u_ph, u_ln, "phase")

    # 2. Попытка извлечения линейных напряжений (любая пара из 3-х!)
    u_ab = phasors_u.get("AB") or phasors_u.get("UAB")
    u_bc = phasors_u.get("BC") or phasors_u.get("UBC")
    u_ca = phasors_u.get("CA") or phasors_u.get("UCA")

    # Автоматическое восстановление 3-го линейного из любой пары
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
        # Эквивалентные фазные напряжения при нулевом потенциале нейтрали
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

    ia = phasors_i.get("A") or phasors_i.get("IA")
    ib = phasors_i.get("B") or phasors_i.get("IB")
    ic = phasors_i.get("C") or phasors_i.get("IC")

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
        if uv.mode == "phase":
            ua, ub, uc = uv.u_phase["A"], uv.u_phase["B"], uv.u_phase["C"]
            return (ua + ub * _A_OPERATOR + uc * _A2_OPERATOR) / 3.0
        elif uv.mode == "line":
            u_ab, u_bc = uv.u_line["AB"], uv.u_line["BC"]
            return (u_ab + u_bc * _A_OPERATOR) / 3.0
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
            mem_u = uv_hist.u_phase.get(key) or uv_hist.u_line.get(key)
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
    """Автоматический пересчёт уставок в зависимости от профиля входных сигналов.

    - scale_profile = 'physical_pu': сигналы в чистых физических о.е.
      Уставки не изменяются (0.05, 0.05, 0.0866).
    - scale_profile = 'dataset_internal': внутренний контракт датасета (ток делен на 20, напряжение на 3).
      Уставки автоматически делятся:
      I_min_eff = I_min / 20.0 (0.0025 о.е.)
      U_min_eff = U_min / 3.0 (0.01667 о.е.)
      P_thresh_eff = P_thresh / (20.0 * 3.0) = P_thresh / 60.0 (0.001443 о.е.)
    """
    if scale_profile == "dataset_internal":
        i_eff = i_min / current_reserve
        u_eff = u_min / voltage_reserve
        p_eff = p_thresh / (current_reserve * voltage_reserve)
        return u_eff, i_eff, p_eff
    return u_min, i_min, p_thresh
