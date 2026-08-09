"""Модуль анализа полноты сигналов и их восстановления для РНМ.

Реализует правила оценки доступности каналов напряжений и токов, а также
векторное восстановление любого 3-го недостающего тока (Ia + Ib + Ic = 0).
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
    """Векторный расчёт любого 3-го недостающего фазного тока из двух любых (Ia + Ib + Ic = 0).

    Если любые два фазных тока из (IA, IB, IC) присутствуют, а 3-й отсутствует
    или содержит NaN, он восстанавливается векторно, а его provenance меняется на DERIVED.

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
    ib_ok = (provenance[idx_ib] != ChannelProvenance.MISSING) and np.isfinite(signals[idx_ib]).any()
    ic_ok = (provenance[idx_ic] != ChannelProvenance.MISSING) and np.isfinite(signals[idx_ic]).any()

    ia_missing = (provenance[idx_ia] == ChannelProvenance.MISSING) or not np.isfinite(signals[idx_ia]).all()
    ib_missing = (provenance[idx_ib] == ChannelProvenance.MISSING) or not np.isfinite(signals[idx_ib]).all()
    ic_missing = (provenance[idx_ic] == ChannelProvenance.MISSING) or not np.isfinite(signals[idx_ic]).all()

    if ib_ok and ic_ok and ia_missing:
        valid = np.isfinite(signals[idx_ib]) & np.isfinite(signals[idx_ic])
        signals[idx_ia, valid] = -(signals[idx_ib, valid] + signals[idx_ic, valid])
        provenance[idx_ia] = int(ChannelProvenance.DERIVED)
    elif ia_ok and ic_ok and ib_missing:
        valid = np.isfinite(signals[idx_ia]) & np.isfinite(signals[idx_ic])
        signals[idx_ib, valid] = -(signals[idx_ia, valid] + signals[idx_ic, valid])
        provenance[idx_ib] = int(ChannelProvenance.DERIVED)
    elif ia_ok and ib_ok and ic_missing:
        valid = np.isfinite(signals[idx_ia]) & np.isfinite(signals[idx_ib])
        signals[idx_ic, valid] = -(signals[idx_ia, valid] + signals[idx_ib, valid])
        provenance[idx_ic] = int(ChannelProvenance.DERIVED)

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

    # В унифицированных slots достаточно любых двух напряжений: для phase
    # восстанавливается третья фаза, для line — третье линейное напряжение.
    valid_u_count = sum(
        1 for ch in ("UA", "UB", "UC")
        if provenance[CHANNEL_ORDER.index(ch)] != ChannelProvenance.MISSING
    )
    has_u_sufficient = valid_u_count >= 2

    # Проверка токов: доступно ли хотя бы 2 фазных тока для восстановления 3-го
    valid_i_count = sum(
        1 for ch in ("IA", "IB", "IC")
        if provenance[CHANNEL_ORDER.index(ch)] != ChannelProvenance.MISSING
    )
    has_i_sufficient = valid_i_count >= 2

    if voltage_basis != "phase" and valid_u_count < 3:
        notes.append(f"Напряжения имеют базис {voltage_basis!r}, выполнена генерация эквивалентных фазных напряжений.")

    can_phase = has_u_sufficient and has_i_sufficient
    can_pos_seq = has_u_sufficient and has_i_sufficient

    for ch in ("IA", "IB", "IC"):
        if ch in derived_ch:
            notes.append(f"Ток фазы {ch[1]} ({ch}) рассчитан векторно из двух других фаз.")

    return PDRSignalAuditResult(
        can_run_phase_pdr=can_phase,
        can_run_pos_seq_pdr=can_pos_seq,
        missing_channels=missing_ch,
        derived_channels=derived_ch,
        measured_channels=measured_ch,
        notes=notes,
    )
