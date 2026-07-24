"""Реестр алгоритмов РНМ (PDRRegistry).

Управляет поиском, регистрацией и динамической подгрузкой алгоритмов РНМ.
Поддерживает открытые и закрытые (private) плагины с безопасным fallback.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Type, Optional, List, Any

from .base import PDRAlgorithm
from .public_algorithms import (
    PhasePDRAlgorithm,
    PositiveSequencePDRAlgorithm,
    ManufacturerPowerPDRStub,
    ManufacturerCurrentPDRStub,
)
from .placeholder import PlaceholderPDRAlgorithm

logger = logging.getLogger(__name__)


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
            # Импорт модуля плагинов если он существует
            importlib.import_module("private.pdr_algorithms")
            logger.info("Закрытые алгоритмы РНМ из private/pdr_algorithms успешно подключены.")
        except ImportError:
            # Разрешённое отсутствие закрытых файлов в публичном репозитории
            pass

    @classmethod
    def get_class(
        cls,
        algorithm_id: str,
        fallback_id: Optional[str] = "pos_seq_pdr_basic",
    ) -> Type[PDRAlgorithm]:
        """Получить класс алгоритма по его ID с возможностью использования fallback."""
        cls._try_load_private_plugins()

        if algorithm_id in cls._registry:
            return cls._registry[algorithm_id]

        logger.warning(
            f"Алгоритм РНМ '{algorithm_id}' не найден в реестре. "
            f"Применяется fallback на '{fallback_id}'."
        )

        if fallback_id and fallback_id in cls._registry:
            return cls._registry[fallback_id]

        return PlaceholderPDRAlgorithm


# Регистрация встроенных публичных алгоритмов
PDRRegistry.register(PhasePDRAlgorithm)
PDRRegistry.register(PositiveSequencePDRAlgorithm)
PDRRegistry.register(ManufacturerPowerPDRStub)
PDRRegistry.register(ManufacturerCurrentPDRStub)
PDRRegistry.register(PlaceholderPDRAlgorithm)


def get_pdr_algorithm(
    algorithm_id: str,
    fallback_id: Optional[str] = "pos_seq_pdr_basic",
    **kwargs: Any,
) -> PDRAlgorithm:
    """Удобный фабричный метод получения экземпляра алгоритма РНМ."""
    cls = PDRRegistry.get_class(algorithm_id, fallback_id=fallback_id)
    if issubclass(cls, PlaceholderPDRAlgorithm):
        return cls(target_algorithm_id=algorithm_id, **kwargs)
    return cls(**kwargs)
