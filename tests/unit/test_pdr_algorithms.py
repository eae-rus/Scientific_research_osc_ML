"""Unit-тесты для публичных алгоритмов РНМ pdr/public_algorithms.py."""

import math
import pytest
import numpy as np

from osc_tools.pdr.base import PDRInputData, PDRDirection
from osc_tools.pdr.public_algorithms import PhasePDRAlgorithm, PositiveSequencePDRAlgorithm
from osc_tools.pdr.placeholder import PlaceholderPDRAlgorithm


def test_phase_pdr_forward():
    """Тест фазного РНМ на прямое направление (ток отстает от напряжения на 45 град)."""
    alg = PhasePDRAlgorithm(phi_mch_deg=45.0, u_min_pu=0.05, i_min_pu=0.02)

    # U: 1.0 < 0 deg; I: 1.0 < -45 deg -> phi = arg(U) - arg(I) = 45 deg = phi_mch
    u_ph = complex(1.0, 0.0)
    i_ph = complex(math.cos(math.radians(-45)), math.sin(math.radians(-45)))

    inp = PDRInputData(
        phasors_u={"A": u_ph, "B": u_ph * complex(-0.5, -math.sqrt(3)/2), "C": u_ph * complex(-0.5, math.sqrt(3)/2)},
        phasors_i={"A": i_ph, "B": i_ph * complex(-0.5, -math.sqrt(3)/2), "C": i_ph * complex(-0.5, math.sqrt(3)/2)},
    )

    out = alg.compute(inp)
    assert out.direction == PDRDirection.FORWARD
    assert out.is_tripped is True
    assert out.margin > 0.9


def test_phase_pdr_reverse():
    """Тест фазного РНМ на обратное направление (ток опережает напряжение на 135 град)."""
    alg = PhasePDRAlgorithm(phi_mch_deg=45.0)

    # U: 1.0 < 0 deg; I: 1.0 < 135 deg -> phi = arg(U) - arg(I) = -135 deg -> delta_phi = -180 deg
    u_ph = complex(1.0, 0.0)
    i_ph = complex(math.cos(math.radians(135)), math.sin(math.radians(135)))

    inp = PDRInputData(
        phasors_u={"A": u_ph, "B": u_ph, "C": u_ph},
        phasors_i={"A": i_ph, "B": i_ph, "C": i_ph},
    )

    out = alg.compute(inp)
    assert out.direction == PDRDirection.REVERSE
    assert out.is_tripped is False


def test_phase_pdr_below_threshold():
    """Тест фазного РНМ на блокировку при малом токе."""
    alg = PhasePDRAlgorithm(i_min_pu=0.05)

    inp = PDRInputData(
        phasors_u={"A": complex(1.0, 0.0), "B": complex(1.0, 0.0), "C": complex(1.0, 0.0)},
        phasors_i={"A": complex(0.01, 0.0), "B": complex(0.01, 0.0), "C": complex(0.01, 0.0)},
    )

    out = alg.compute(inp)
    assert out.direction == PDRDirection.BLOCK
    assert out.is_tripped is False


def test_pos_seq_pdr_forward():
    """Тест РНМ прямой последовательности на прямое направление."""
    alg = PositiveSequencePDRAlgorithm(phi_mch_deg=45.0)

    a = complex(-0.5, math.sqrt(3) / 2)
    a_sq = complex(-0.5, -math.sqrt(3) / 2)

    # Симметричная 3-фазная система напряжений 1.0 < 0
    ua = complex(1.0, 0.0)
    ub = ua * a_sq
    uc = ua * a

    # Ток прямой последовательности с углом -45 град
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
    assert out.is_tripped is True


def test_placeholder_fallback():
    """Тест заглушки для закрытых алгоритмов."""
    stub = PlaceholderPDRAlgorithm(target_algorithm_id="secret_mir_pdr")
    out = stub.compute(PDRInputData(phasors_u={}, phasors_i={}))

    assert out.direction == PDRDirection.BLOCK
    assert out.confidence == 0.0
    assert out.diagnostics["is_placeholder_fallback"] is True
