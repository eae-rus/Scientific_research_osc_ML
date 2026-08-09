"""Unit-тесты для публичных алгоритмов РНМ pdr/public_algorithms.py."""

import math
import pytest
import numpy as np

from osc_tools.pdr.base import PDRInputData, PDRDirection
from osc_tools.pdr.public_algorithms import (
    PhasePDRAlgorithm,
    PositiveSequencePDRAlgorithm,
    PositiveSequencePowerPDRAlgorithm,
)
from osc_tools.pdr.pdr_signal_utils import (
    compute_positive_sequence,
    derive_unified_voltages,
)
from osc_tools.pdr.placeholder import PlaceholderPDRAlgorithm


def test_phase_pdr_forward():
    """Тест фазного РНМ на прямое направление (ток отстает от напряжения на 45 град)."""
    alg = PhasePDRAlgorithm(phi_mch_deg=45.0, u_min_pu=0.05, i_min_pu=0.02)

    u_ph = complex(1.0, 0.0)
    i_ph = complex(math.cos(math.radians(-45)), math.sin(math.radians(-45)))

    inp = PDRInputData(
        phasors_u={"A": u_ph, "B": u_ph * complex(-0.5, -math.sqrt(3)/2), "C": u_ph * complex(-0.5, math.sqrt(3)/2)},
        phasors_i={"A": i_ph, "B": i_ph * complex(-0.5, -math.sqrt(3)/2), "C": i_ph * complex(-0.5, math.sqrt(3)/2)},
    )

    out = alg.compute(inp)
    assert out.direction == PDRDirection.FORWARD
    assert out.direction == 1
    assert out.is_tripped is True
    assert out.margin > 0.9


def test_phase_pdr_reverse():
    """Тест фазного РНМ на обратное направление (ток опережает напряжение на 135 град)."""
    alg = PhasePDRAlgorithm(phi_mch_deg=45.0)

    u_ph = complex(1.0, 0.0)
    i_ph = complex(math.cos(math.radians(135)), math.sin(math.radians(135)))

    inp = PDRInputData(
        phasors_u={"A": u_ph, "B": u_ph, "C": u_ph},
        phasors_i={"A": i_ph, "B": i_ph, "C": i_ph},
    )

    out = alg.compute(inp)
    assert out.direction == PDRDirection.REVERSE
    assert out.direction == 0
    assert out.is_tripped is False


def test_phase_pdr_below_threshold():
    """Тест фазного РНМ на занижение при малом токе."""
    alg = PhasePDRAlgorithm(i_min_pu=0.05)

    inp = PDRInputData(
        phasors_u={"A": complex(1.0, 0.0), "B": complex(1.0, 0.0), "C": complex(1.0, 0.0)},
        phasors_i={"A": complex(0.01, 0.0), "B": complex(0.01, 0.0), "C": complex(0.01, 0.0)},
    )

    out = alg.compute(inp)
    assert out.direction == PDRDirection.REVERSE
    assert out.direction == 0
    assert out.is_tripped is False


def test_pos_seq_pdr_forward():
    """Тест РНМ прямой последовательности на прямое направление."""
    alg = PositiveSequencePDRAlgorithm(phi_mch_deg=45.0)

    a = complex(-0.5, math.sqrt(3) / 2)
    a_sq = complex(-0.5, -math.sqrt(3) / 2)

    ua = complex(1.0, 0.0)
    ub = ua * a_sq
    uc = ua * a

    i_mag = complex(math.cos(math.radians(-45)), math.sin(math.radians(-45)))
    ia = i_mag
    ib = ia * a_sq
    ic = ia * a

    inp = PDRInputData(
        phasors_u={"A": ua, "B": ub, "C": uc},
        phasors_i={"A": ia, "B": ib, "C": ic},
    )

    out = alg.compute(inp)
    assert out.direction == PDRDirection.FORWARD
    assert out.direction == 1
    assert out.is_tripped is True


def test_placeholder_fallback():
    """Тест заглушки для закрытых алгоритмов."""
    stub = PlaceholderPDRAlgorithm(target_algorithm_id="secret_mir_pdr")
    out = stub.compute(PDRInputData(phasors_u={}, phasors_i={}))

    assert out.direction == PDRDirection.REVERSE
    assert out.direction == 0
    assert out.confidence == 0.0
    assert out.diagnostics["is_placeholder_fallback"] is True


def test_positive_sequence_is_equal_for_phase_and_line_voltages():
    """U1 должна быть инвариантна к фазному/линейному представлению входа."""
    a = np.exp(1j * 2 * np.pi / 3)
    phase = {"A": 1.0 + 0j, "B": a ** 2, "C": a}
    line = derive_unified_voltages(phase).u_line

    u1_phase = compute_positive_sequence(phase, is_voltage=True)
    u1_line = compute_positive_sequence(line, is_voltage=True)

    assert u1_phase == pytest.approx(u1_line, abs=1e-12)


def test_phase_pdr_requires_two_currents_and_restores_third():
    """Одна фаза недостаточна, две позволяют восстановить третью."""
    a = np.exp(1j * 2 * np.pi / 3)
    voltages = {"A": 1.0 + 0j, "B": a ** 2, "C": a}
    ia = np.exp(-1j * np.pi / 4)
    currents = {"A": ia, "C": ia * a}
    algorithm = PhasePDRAlgorithm(polarization_mode="direct")

    insufficient = algorithm.compute(PDRInputData(phasors_u=voltages, phasors_i={"A": ia}))
    restored = algorithm.compute(PDRInputData(phasors_u=voltages, phasors_i=currents))

    assert insufficient.direction == PDRDirection.UNLABELED
    assert restored.direction == PDRDirection.FORWARD
    assert restored.diagnostics["valid_count"] == 3


def test_positive_sequence_power_honours_minimum_current():
    """Публичный мощностной орган не действует ниже i_min_pu."""
    a = np.exp(1j * 2 * np.pi / 3)
    voltages = {"A": 1.0 + 0j, "B": a ** 2, "C": a}
    currents = {"A": 0.001 + 0j, "B": 0.001 * a ** 2, "C": 0.001 * a}
    out = PositiveSequencePowerPDRAlgorithm(i_min_pu=0.05).compute(
        PDRInputData(phasors_u=voltages, phasors_i=currents)
    )

    assert out.direction == PDRDirection.REVERSE
    assert out.diagnostics["reason"] == "current_below_threshold"
