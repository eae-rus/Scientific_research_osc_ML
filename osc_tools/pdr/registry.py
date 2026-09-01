"""Реестр алгоритмов РНМ (PDRRegistry).

Управляет поиском, регистрацией базовых и литературных открытых алгоритмов РНМ,
а также закрытых (private) адаптивных плагинов.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Type, Optional, List, Any

from .base import PDRAlgorithm
from .public_algorithms import (
    PhasePDRAlgorithm,
    PositiveSequencePDRAlgorithm,
    PhasePowerPDRAlgorithm,
    PositiveSequencePowerPDRAlgorithm,
    ManufacturerPowerPDRStub,
    ManufacturerCurrentPDRStub,
    Sivokobylenko2PtPDRAlgorithm,
    Sivokobylenko5PtPDRAlgorithm,
    BMRZReactiveAssistedPDRAlgorithm,
    BAVR072CrossPolarizedPDRAlgorithm,
)
from .placeholder import PlaceholderPDRAlgorithm

logger = logging.getLogger(__name__)

# Карта удобных алиасов для вызова алгоритмов
ALGORITHM_ALIASES: Dict[str, str] = {
    "adaptive_pdr": "adaptive_pdr_mir",
    "mir_adaptive": "adaptive_pdr_mir",
    "bavr_adaptive": "adaptive_pdr_mir",
    "phase_pdr": "phase_pdr_basic",
    "pos_seq_pdr": "pos_seq_pdr_basic",
    "phase_power_pdr": "phase_power_pdr_basic",
    "pos_seq_power_pdr": "pos_seq_power_pdr_basic",
    "sivokobylenko_2pt": "pdr_sivokobylenko_2pt",
    "sivokobylenko_5pt": "pdr_sivokobylenko_5pt",
    "bmrz_q_assisted": "pdr_bmrz_q_assisted",
    "bavr072_crosspol": "pdr_bavr072_crosspol",
}


class PDRRegistry:
    """Реестр всех известных алгоритмов РНМ."""

    _registry: Dict[str, Type[PDRAlgorithm]] = {}
    _initialized_private: bool = False

    @classmethod
    def register(cls, algorithm_cls: Type[PDRAlgorithm]) -> Type[PDRAlgorithm]:
        """Зарегистрировать алгоритм в реестре."""
        alg_id = algorithm_cls.algorithm_id
        if not alg_id:
            raise ValueError(f"Класс {algorithm_cls.__name__} не имеет algorithm_id")
        cls._registry[alg_id] = algorithm_cls
        return algorithm_cls

    @classmethod
    def list_algorithms(cls, include_private: bool = True) -> List[str]:
        """Получить список всех доступных идентификаторов алгоритмов."""
        cls._try_load_private_plugins()
        if include_private:
            return list(cls._registry.keys())
        return [
            alg_id
            for alg_id, alg_cls in cls._registry.items()
            if getattr(alg_cls, "is_public", True)
        ]

    @classmethod
    def _try_load_private_plugins(cls) -> None:
        """Попытка динамической подгрузки приватных алгоритмов из private/pdr_algorithms/."""
        if cls._initialized_private:
            return
        cls._initialized_private = True
        try:
            importlib.import_module("private.pdr_algorithms")
            logger.info("Закрытый адаптивный орган РНМ из private/pdr_algorithms успешно подключен.")
        except ImportError as exc:
            logger.info("Закрытые PDR-плагины недоступны: %s", exc)

    @classmethod
    def get_class(
        cls,
        algorithm_id: str,
        fallback_id: Optional[str] = "pos_seq_pdr_basic",
    ) -> Type[PDRAlgorithm]:
        """Получить класс алгоритма по его ID или алиасу с поддержкой fallback."""
        cls._try_load_private_plugins()

        # Разрешение алиасов
        resolved_id = ALGORITHM_ALIASES.get(algorithm_id, algorithm_id)

        if resolved_id in cls._registry:
            return cls._registry[resolved_id]

        resolved_fallback = ALGORITHM_ALIASES.get(fallback_id, fallback_id) if fallback_id else None

        logger.warning(
            f"Алгоритм РНМ '{algorithm_id}' (запрошенный как '{resolved_id}') не найден в реестре. "
            f"Применяется fallback на '{resolved_fallback}'."
        )

        if resolved_fallback and resolved_fallback in cls._registry:
            return cls._registry[resolved_fallback]

        return PlaceholderPDRAlgorithm


# Регистрация открытых физических алгоритмов и заглушек
PDRRegistry.register(PhasePDRAlgorithm)
PDRRegistry.register(PositiveSequencePDRAlgorithm)
PDRRegistry.register(PhasePowerPDRAlgorithm)
PDRRegistry.register(PositiveSequencePowerPDRAlgorithm)
PDRRegistry.register(Sivokobylenko2PtPDRAlgorithm)
PDRRegistry.register(Sivokobylenko5PtPDRAlgorithm)
PDRRegistry.register(BMRZReactiveAssistedPDRAlgorithm)
PDRRegistry.register(BAVR072CrossPolarizedPDRAlgorithm)
PDRRegistry.register(ManufacturerPowerPDRStub)
PDRRegistry.register(ManufacturerCurrentPDRStub)
PDRRegistry.register(PlaceholderPDRAlgorithm)


def get_pdr_algorithm(
    algorithm_id: str = "adaptive_pdr_mir",
    fallback_id: Optional[str] = "pos_seq_pdr_basic",
    **kwargs: Any,
) -> PDRAlgorithm:
    """Вспомогательная функция для создания экземпляра алгоритма из реестра.

    По умолчанию вызовет закрытый адаптивный орган 'adaptive_pdr_mir' со всеми
    включенными адаптивностями и стандартными уставками БАВР.
    """
    PDRRegistry._try_load_private_plugins()
    requested_resolved = ALGORITHM_ALIASES.get(algorithm_id, algorithm_id)
    requested_available = requested_resolved in PDRRegistry._registry
    alg_cls = PDRRegistry.get_class(algorithm_id, fallback_id=fallback_id)
    algorithm = alg_cls(**kwargs)
    algorithm.requested_algorithm_id = algorithm_id
    algorithm.resolved_algorithm_id = alg_cls.algorithm_id
    algorithm.fallback_applied = not requested_available
    algorithm.fallback_algorithm_id = alg_cls.algorithm_id if not requested_available else None
    return algorithm
