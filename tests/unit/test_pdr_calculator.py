"""
Unit-тесты для osc_tools.features.pdr_calculator.

Тестирует функции расчёта симметричных составляющих и другие вычисления.
"""

import numpy as np
import pytest
import os
import sys

# Ensure project root is available for imports when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from osc_tools.features.pdr_calculator import calculate_symmetrical_components


@pytest.mark.unit
class TestSymmetricalComponents:
    """Тесты для функции расчёта симметричных составляющих."""
    
    def test_balanced_system(self, balanced_three_phase):
        """
        Тест для сбалансированной трёхфазной системы.
        
        В сбалансированной системе:
        - Прямая последовательность (positive) == основной сигнал
        - Обратная последовательность (negative) ≈ 0
        - Нулевая последовательность (zero) ≈ 0
        """
        phasor_a = balanced_three_phase["phasor_a"]
        phasor_b = balanced_three_phase["phasor_b"]
        phasor_c = balanced_three_phase["phasor_c"]
        
        # Вычисляем симметричные составляющие
        phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(
            phasor_a, phasor_b, phasor_c
        )
        
        # Проверки:
        # 1. Прямая последовательность ≈ фаза A
        assert np.allclose(phasor_1, phasor_a), \
            f"Прямая последовательность должна быть близка к фазе A. " \
            f"Получено: {phasor_1}, ожидалось: {phasor_a}"
        
        # 2. Обратная последовательность ≈ 0
        assert np.allclose(phasor_2, np.zeros_like(phasor_2), atol=1e-10), \
            f"Обратная последовательность должна быть нулевой. " \
            f"Получено: {phasor_2}"
        
        # 3. Нулевая последовательность ≈ 0
        assert np.allclose(phasor_0, np.zeros_like(phasor_0), atol=1e-10), \
            f"Нулевая последовательность должна быть нулевой. " \
            f"Получено: {phasor_0}"
    
    def test_zero_sequence_only(self, zero_sequence_three_phase):
        """
        Тест для системы только с нулевой последовательностью.
        
        Когда все три фазы идентичны:
        - Прямая последовательность ≈ 0
        - Обратная последовательность ≈ 0
        - Нулевая последовательность == исходный фазор
        """
        phasor_a = zero_sequence_three_phase["phasor_a"]
        phasor_b = zero_sequence_three_phase["phasor_b"]
        phasor_c = zero_sequence_three_phase["phasor_c"]
        
        phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(
            phasor_a, phasor_b, phasor_c
        )
        
        # Проверки:
        # 1. Нулевая последовательность ≈ исходный фазор
        assert np.allclose(phasor_0, phasor_a), \
            f"Нулевая последовательность должна равняться исходному фазору. " \
            f"Получено: {phasor_0}, ожидалось: {phasor_a}"
        
        # 2. Прямая и обратная ≈ 0
        assert np.allclose(phasor_1, np.zeros_like(phasor_1), atol=1e-10)
        assert np.allclose(phasor_2, np.zeros_like(phasor_2), atol=1e-10)
    
    def test_negative_sequence_only(self, negative_sequence_three_phase):
        """
        Тест для системы только с обратной последовательностью.
        
        Обратная последовательность имеет обратный порядок фаз.
        """
        phasor_a = negative_sequence_three_phase["phasor_a"]
        phasor_b = negative_sequence_three_phase["phasor_b"]
        phasor_c = negative_sequence_three_phase["phasor_c"]
        
        phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(
            phasor_a, phasor_b, phasor_c
        )
        
        # В системе с только обратной последовательностью:
        # - Обратная последовательность ≠ 0
        # - Прямая и нулевая ≈ 0
        assert np.allclose(phasor_1, np.zeros_like(phasor_1), atol=1e-10)
        assert np.linalg.norm(phasor_2) > 1e-10, \
            "Обратная последовательность должна быть ненулевой"
        assert np.allclose(phasor_0, np.zeros_like(phasor_0), atol=1e-10)
    
    def test_multi_sample_input(self):
        """
        Тест для многоточечных входных данных.
        
        Функция должна работать с массивами, содержащими несколько отсчётов.
        """
        # Создаём 10 отсчётов сбалансированной системы
        n_samples = 10
        phase_shift = 2 * np.pi / 3
        
        phasor_a = np.exp(1j * np.random.rand(n_samples) * 2 * np.pi)
        phasor_b = phasor_a * np.exp(1j * -phase_shift)
        phasor_c = phasor_a * np.exp(1j * phase_shift)
        
        phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(
            phasor_a, phasor_b, phasor_c
        )
        
        # Проверяем, что формы совпадают
        assert phasor_1.shape == phasor_a.shape
        assert phasor_2.shape == phasor_a.shape
        assert phasor_0.shape == phasor_a.shape
        
        # И что прямая последовательность близка к фазе A (для сбалансированной)
        assert np.allclose(phasor_1, phasor_a)
    
    def test_output_is_complex(self):
        """Проверяет, что результаты — комплексные массивы."""
        phasor_a = np.array([1.0 + 0.0j])
        phasor_b = np.array([np.exp(1j * -2 * np.pi / 3)])
        phasor_c = np.array([np.exp(1j * 2 * np.pi / 3)])
        
        phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(
            phasor_a, phasor_b, phasor_c
        )
        
        assert np.iscomplexobj(phasor_1)
        assert np.iscomplexobj(phasor_2)
        assert np.iscomplexobj(phasor_0)
    
    def test_decomposition_completeness(self, balanced_three_phase):
        """
        Тест: сумма симметричных составляющих должна восстанавливать исходный сигнал.
        
        Формула: Ia = I1a + I2a + I0a
        где I1a, I2a, I0a — фазоры прямой, обратной, нулевой последовательностей.
        """
        phasor_a = balanced_three_phase["phasor_a"]
        phasor_b = balanced_three_phase["phasor_b"]
        phasor_c = balanced_three_phase["phasor_c"]
        
        phasor_1, phasor_2, phasor_0 = calculate_symmetrical_components(
            phasor_a, phasor_b, phasor_c
        )
        
        # Преобразование Фортескье: I1, I2, I0 -> Ia, Ib, Ic
        # Обратная трансформация:
        a_operator = np.exp(2j * np.pi / 3)  # e^(j*120°)
        
        # Ia = I1 + I2 + I0
        ia_reconstructed = phasor_1 + phasor_2 + phasor_0
        
        assert np.allclose(ia_reconstructed, phasor_a, atol=1e-10), \
            f"Реконструированная фаза A не совпадает с исходной. " \
            f"Получено: {ia_reconstructed}, ожидалось: {phasor_a}"
