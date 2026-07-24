"""Модуль анализа полноты сигналов и их восстановления для РНМ.

Реализует правила оценки доступности каналов напряжений и токов, а также
векторный расчёт отсутствующего тока фазы B: I_B = -(I_A + I_C).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from osc_tools.ml.phase5_contracts import CHANNEL_ORDER, ChannelProvenance


@dataclass(frozen=True)
class PDRSignalAuditResult:
    """Результат оценки пригодности осциллограммы для расчёта РНМ."""

    can_run_phase_pdr: bool
    can_run_pos_seq_pdr: bool
    missing_channels: List[str] = field(default_factory=list)
    derived_channels: List[str] = field(default_factory=list)
    measured_channels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def derive_missing_currents(
    signals: np.ndarray,
    provenance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Векторный расчёт отсутствующего тока фазы B из токов фаз A и C.

    Если ток фазы A (индекс 0) и фазы C (индекс 2) измерены (MEASURED),
    а ток фазы B (индекс 1) отсутствует (MISSING) или содержит NaN,
    он вычисляется как: I_B = -(I_A + I_C), а provenance для IB меняется на DERIVED.

    Args:
        signals: Массив сигналов формы (8, T)
        provenance: Массив provenance формы (8,)

    Returns:
        Кортеж (модифицированные signals, модифицированный provenance)
    """
    signals = np.copy(signals)
    provenance = np.copy(provenance)

    idx_ia = CHANNEL_ORDER.index("IA")
    idx_ib = CHANNEL_ORDER.index("IB")
    idx_ic = CHANNEL_ORDER.index("IC")

    ia_ok = (provenance[idx_ia] != ChannelProvenance.MISSING) and np.isfinite(signals[idx_ia]).any()
    ic_ok = (provenance[idx_ic] != ChannelProvenance.MISSING) and np.isfinite(signals[idx_ic]).any()
    ib_missing = (provenance[idx_ib] == ChannelProvenance.MISSING) or not np.isfinite(signals[idx_ib]).all()

    if ia_ok and ic_ok and ib_missing:
        # Расчёт I_B = -(I_A + I_C)
        signals[idx_ib] = -(signals[idx_ia] + signals[idx_ic])
        provenance[idx_ib] = int(ChannelProvenance.DERIVED)

    return signals, provenance


def check_pdr_signal_sufficiency(
    provenance: np.ndarray,
    voltage_basis: str = "phase",
) -> PDRSignalAuditResult:
    """Аудит полноты сигналов для применения алгоритмов РНМ.

    Args:
        provenance: Массив provenance для 8 каналов
        voltage_basis: Базис напряжений ('phase' или 'line')

    Returns:
        PDRSignalAuditResult с флагами пригодности и примечаниями
    """
    missing_ch: List[str] = []
    derived_ch: List[str] = []
    measured_ch: List[str] = []
    notes: List[str] = []

    for i, ch_name in enumerate(CHANNEL_ORDER):
        prov = provenance[i]
        if prov == ChannelProvenance.MISSING:
            missing_ch.append(ch_name)
        elif prov == ChannelProvenance.DERIVED:
            derived_ch.append(ch_name)
        else:
            measured_ch.append(ch_name)

    # Проверка напряжений: нужно ли фазное напряжение UA, UB, UC
    # В Phase 5 принята приоритетная работа по фазным напряжениям СШ (UA BB, UB BB, UC BB)
    has_u_phase = all(
        provenance[CHANNEL_ORDER.index(ch)] != ChannelProvenance.MISSING
        for ch in ("UA", "UB", "UC")
    )
    
    # Проверка токов: нужны все три фазных тока (измеренные или рассчитанные)
    has_i_three_phase = all(
        provenance[CHANNEL_ORDER.index(ch)] != ChannelProvenance.MISSING
        for ch in ("IA", "IB", "IC")
    )

    if voltage_basis != "phase" and not has_u_phase:
        notes.append(f"Напряжения имеют базис {voltage_basis!r}, фазные напряжения отсутствуют.")

    can_phase = has_u_phase and any(
        provenance[CHANNEL_ORDER.index(ch)] != ChannelProvenance.MISSING
        for ch in ("IA", "IB", "IC")
    )

    can_pos_seq = has_u_phase and has_i_three_phase

    if "IB" in derived_ch:
        notes.append("Ток фазы B (IB) рассчитан векторно из IA и IC.")

    return PDRSignalAuditResult(
        can_run_phase_pdr=can_phase,
        can_run_pos_seq_pdr=can_pos_seq,
        missing_channels=missing_ch,
        derived_channels=derived_ch,
        measured_channels=measured_ch,
        notes=notes,
    )
