"""Unit-тесты для публичных алгоритмов РНМ pdr/public_algorithms.py."""

import math
import pytest
import numpy as np

from osc_tools.pdr.base import PDRInputData, PDRDirection
from osc_tools.pdr.public_algorithms import (
    PhasePDRAlgorithm,
    PhasePowerPDRAlgorithm,
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


@pytest.mark.parametrize(
    ("current_angle_deg", "expected_direction"),
    [(-45.0, PDRDirection.FORWARD), (135.0, PDRDirection.REVERSE)],
)
def test_power_algorithms_match_on_balanced_phase_system(
    current_angle_deg: float,
    expected_direction: PDRDirection,
) -> None:
    """Пофазный и U1/I1 моменты имеют общий знак, масштаб и уставку."""

    a = np.exp(1j * 2 * np.pi / 3)
    voltages = {"A": 1.0 + 0j, "B": a ** 2, "C": a}
    ia = 0.2 * np.exp(1j * np.deg2rad(current_angle_deg))
    currents = {"A": ia, "B": ia * a ** 2, "C": ia * a}
    input_data = PDRInputData(phasors_u=voltages, phasors_i=currents)

    phase = PhasePowerPDRAlgorithm().compute(input_data)
    sequence = PositiveSequencePowerPDRAlgorithm().compute(input_data)

    assert phase.direction == expected_direction
    assert sequence.direction == expected_direction
    assert phase.margin == pytest.approx(sequence.margin, abs=1e-12)
    for phase_result in phase.diagnostics["phase_results"].values():
        assert phase_result["t_op"] == pytest.approx(sequence.diagnostics["t_op"], abs=1e-12)


def test_power_algorithms_use_same_nonzero_threshold() -> None:
    """Небольшая обратная мощность остаётся внутри общей зоны p_thresh."""

    a = np.exp(1j * 2 * np.pi / 3)
    voltages = {"A": 1.0 + 0j, "B": a ** 2, "C": a}
    # Обратный максимальный момент -0.05 меньше по модулю стандартной уставки 0.0866.
    ia = 0.05 * np.exp(1j * np.deg2rad(135.0))
    currents = {"A": ia, "B": ia * a ** 2, "C": ia * a}
    input_data = PDRInputData(phasors_u=voltages, phasors_i=currents)

    phase = PhasePowerPDRAlgorithm(i_min_pu=0.01).compute(input_data)
    sequence = PositiveSequencePowerPDRAlgorithm(i_min_pu=0.01).compute(input_data)

    assert phase.direction == PDRDirection.FORWARD
    assert sequence.direction == PDRDirection.FORWARD
    assert phase.margin == pytest.approx(0.0366, abs=1e-12)
    assert phase.margin == pytest.approx(sequence.margin, abs=1e-12)
