"""Пакет измерительных органов реле направления мощности (РНМ / PDR) для Phase 5.

Предоставляет единый API контрактов данных, модуль анализа полноты сигналов,
публичные алгоритмы РНМ, адаптивный реестр с плагинами, генератор псевдоразметки
и PyTorch Task Dataset.
"""

from .base import (
    PDRAlgorithm,
    PDRInputData,
    PDROutput,
    PDRDirection,
)
from .signal_analysis import (
    derive_missing_currents,
    check_pdr_signal_sufficiency,
    PDRSignalAuditResult,
)
from .public_algorithms import (
    PhasePDRAlgorithm,
    PositiveSequencePDRAlgorithm,
)
from .registry import (
    PDRRegistry,
    get_pdr_algorithm,
)

__all__ = [
    "PDRAlgorithm",
    "PDRInputData",
    "PDROutput",
    "PDRDirection",
    "derive_missing_currents",
    "check_pdr_signal_sufficiency",
    "PDRSignalAuditResult",
    "PhasePDRAlgorithm",
    "PositiveSequencePDRAlgorithm",
    "PDRRegistry",
    "get_pdr_algorithm",
]
