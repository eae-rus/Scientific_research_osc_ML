"""Unit-тесты для реестра алгоритмов РНМ pdr/registry.py."""

import pytest

from osc_tools.pdr.registry import PDRRegistry, get_pdr_algorithm
from osc_tools.pdr.public_algorithms import PhasePDRAlgorithm, PositiveSequencePDRAlgorithm
from osc_tools.pdr.placeholder import PlaceholderPDRAlgorithm


def test_list_algorithms():
    """Проверка наличия базовых алгоритмов в реестре."""
    algs = PDRRegistry.list_algorithms()
    assert "phase_pdr_basic" in algs
    assert "pos_seq_pdr_basic" in algs
    assert "bavr_manufacturer_power_stub" in algs


def test_get_pdr_algorithm_valid():
    """Получение экземпляра зарегистрированного алгоритма."""
    alg = get_pdr_algorithm("phase_pdr_basic", phi_mch_deg=60.0)
    assert isinstance(alg, PhasePDRAlgorithm)
    assert alg.params["phi_mch_deg"] == 60.0


def test_get_pdr_algorithm_fallback():
    """Запрос неизвестного алгоритма должен возвращать fallback."""
    alg = get_pdr_algorithm("unknown_algorithm_123", fallback_id="pos_seq_pdr_basic")
    assert isinstance(alg, PositiveSequencePDRAlgorithm)


def test_get_pdr_algorithm_placeholder_fallback():
    """Запрос закрытого алгоритма при отсутствии реализации вернёт PlaceholderPDRAlgorithm."""
    alg = get_pdr_algorithm("unknown_algorithm_123", fallback_id="private_pdr_placeholder")
    assert isinstance(alg, PlaceholderPDRAlgorithm)
