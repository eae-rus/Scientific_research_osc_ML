"""Заглушка для закрытых алгоритмов РНМ (PDR).

Применяется в публичном коде по умолчанию, когда закрытый модуль из каталога `private/`
отсутствует или недоступен. Предотвращает сбои и фиксирует факт фолбэка.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from .base import PDRAlgorithm, PDRInputData, PDROutput, PDRDirection

logger = logging.getLogger(__name__)


class PlaceholderPDRAlgorithm(PDRAlgorithm):
    """Публичная заглушка для закрытых проприетарных алгоритмов РНМ.

    Если закрытые алгоритмы из `private/pdr_algorithms/` не загружены,
    эта заглушка перехватывает вызов, выдаёт предупреждение в лог
    и возвращает блокировку со 100% прозрачностью.
    """

    algorithm_id = "private_pdr_placeholder"
    name = "Proprietary PDR Algorithm (Placeholder / Stub)"
    is_public = True
    tunable_parameters = {}

    def __init__(self, target_algorithm_id: str = "proprietary_pdr", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.target_algorithm_id = target_algorithm_id
        logger.warning(
            f"ВНИМАНИЕ: Запрошен закрытый алгоритм РНМ '{target_algorithm_id}', "
            f"но доступен только публичный плагин-заглушка {self.name}. "
            f"Результаты будут помечены как BLOCK (неразмеченный фолбэк)."
        )

    def compute(self, input_data: PDRInputData) -> PDROutput:
        return PDROutput(
            direction=PDRDirection.BLOCK,
            is_tripped=False,
            margin=0.0,
            confidence=0.0,
            diagnostics={
                "is_placeholder_fallback": True,
                "target_algorithm_id": self.target_algorithm_id,
                "warning": "Closed source algorithm is missing from private/ directory.",
            },
        )
