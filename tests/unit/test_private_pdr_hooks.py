"""Тестовый хук для опциональной проверки закрытых алгоритмов РНМ.

Для внешних пользователей (в публичном репозитории, где папка private/ отсутствует)
тесты скрытых алгоритмов автоматически пропускаются (pytest.skip).
Для исследователя с локальным папкой private/ тесты подгружаются и выполняются.
"""

from pathlib import Path
import pytest


def test_private_pdr_algorithms_if_available():
    """Безопасный запуск закрытых тестов РНМ при наличии локальной папки private/."""
    private_dir = Path(__file__).resolve().parents[2] / "private" / "pdr_algorithms"
    private_file = private_dir / "adaptive_pdr.py"

    if not private_file.exists():
        pytest.skip(
            "Закрытые алгоритмы РНМ отсутствуют (папка private/ не включена в Git). "
            "Публичный пайплайн использует фолбэк PlaceholderPDRAlgorithm."
        )

    # Динамический запуск тестов из private/
    try:
        from private.pdr_algorithms.test_adaptive_pdr import (
            test_adaptive_current_lockout_conditions,
            test_adaptive_pdr_positive_sequence_default,
            test_adaptive_pdr_starts_with_current_phasor_as_earliest_history,
            test_adaptive_pdr_vector_memory_and_missing_current,
        )
        test_adaptive_pdr_positive_sequence_default()
        test_adaptive_pdr_vector_memory_and_missing_current()
        test_adaptive_current_lockout_conditions()
        test_adaptive_pdr_starts_with_current_phasor_as_earliest_history()
    except Exception as exc:
        pytest.fail(f"Ошибка при выполнении закрытых тестов РНМ: {exc}")
