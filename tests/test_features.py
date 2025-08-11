import numpy as np
import pytest
from osc_tools.features.pdr_calculator import calculate_symmetrical_components

def test_calculate_symmetrical_components_balanced_system():
    """
    Тест для сбалансированной трехфазной системы.
    В сбалансированной системе должны присутствовать только компоненты прямой последовательности.
    Обратная и нулевая последовательности должны быть равны нулю.
    """
    # Создаем сбалансированную систему:
    # Фаза A: амплитуда 1, угол 0 градусов
    # Фаза B: амплитуда 1, угол -120 градусов
    # Фаза C: амплитуда 1, угол +120 градусов
    phasor_a = np.array([1 + 0j])
    phasor_b = np.array([np.exp(1j * -2 * np.pi / 3)])
    phasor_c = np.array([np.exp(1j * 2 * np.pi / 3)])

    # Вычисляем симметричные составляющие
    phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)

    # ПРОВЕРКА:
    # 1. Прямая последовательность (phasor_1) должна быть равна фазе A.
    assert np.allclose(phasor_1, phasor_a)

    # 2. Обратная последовательность (phasor_2) должна быть близка к нулю.
    assert np.allclose(phasor_2, np.zeros_like(phasor_2))

    # 3. Нулевая последовательность (phasor_0) должна быть близка к нулю.
    assert np.allclose(phasor_0, np.zeros_like(phasor_0))

def test_calculate_symmetrical_components_zero_sequence():
    """
    Тест для системы, где все три фазы идентичны.
    В такой системе должны быть только компоненты нулевой последовательности.
    """
    # Создаем систему, где все три фазы одинаковы
    phasor_a = np.array([1 + 0.5j, 2 - 1j])
    phasor_b = np.array([1 + 0.5j, 2 - 1j])
    phasor_c = np.array([1 + 0.5j, 2 - 1j])

    # Вычисляем симметричные составляющие
    phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(phasor_a, phasor_b, phasor_c)

    # ПРОВЕРКА:
    # 1. Нулевая последовательность (phasor_0) должна быть равна исходным фазорам.
    assert np.allclose(phasor_0, phasor_a)

    # 2. Прямая последовательность (phasor_1) должна быть близка к нулю.
    assert np.allclose(phasor_1, np.zeros_like(phasor_1))

    # 3. Обратная последовательность (phasor_2) должна быть близка к нулю.
    assert np.allclose(phasor_2, np.zeros_like(phasor_2))
