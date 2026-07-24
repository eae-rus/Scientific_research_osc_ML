"""Базовые контракты данных и интерфейс алгоритмов РНМ (PDR).

Этот модуль задаёт типы данных, перечисления и абстрактный базовый класс `PDRAlgorithm`.
Все публичные и закрытые измерительные органы РНМ должны наследоваться от `PDRAlgorithm`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, Optional, Sequence
import numpy as np

from osc_tools.ml.phase5_contracts import ChannelProvenance


class PDRDirection(IntEnum):
    """Направление мощности / решение органа РНМ.
    
    - UNLABELED (-999): Неразмеченная область (прогрев 10 периодов, недостаточная предыстория).
      Исключает путаницу со значениями срабатывания.
    - REVERSE (0): Обратное направление (КЗ за спиной / блокировка срабатывания БАВР).
    - FORWARD (1): Прямое направление (КЗ в зоне действия / разрешение БАВР).
    """

    UNLABELED = -999
    REVERSE = 0
    FORWARD = 1


@dataclass(frozen=True)
class PDRInputData:
    """Входные данные для расчёта органа РНМ в конкретный момент времени."""

    # Комплексные фазоры первой гармоники напряжений {"A", "B", "C", "N"}
    phasors_u: Dict[str, complex]
    # Комплексные фазоры первой гармоники токов {"A", "B", "C", "N"}
    phasors_i: Dict[str, complex]
    # Доступные фазоры предыстории (t - 200 мс) для адаптивных органов
    history_phasors_u: Optional[Dict[str, complex]] = None
    history_phasors_i: Optional[Dict[str, complex]] = None
    # Provenance каналов (8 элементов согласно CHANNEL_ORDER)
    provenance: Optional[np.ndarray] = None
    # Базис напряжений ('phase' или 'line')
    voltage_basis: str = "phase"
    # Временная метка окна в секундах
    timestamp_sec: float = 0.0


@dataclass(frozen=True)
class PDROutput:
    """Выходные данные и диагностическая информация работы органа РНМ."""

    # Решение по направлению (FORWARD = 1, REVERSE = 0, UNLABELED = -999)
    direction: PDRDirection
    # Факт пуска/срабатывания (True если FORWARD = 1)
    is_tripped: bool
    # Непрерывная величина запаса срабатывания (margin / мощность / проекция)
    margin: float
    # Уверенность решения (от 0.0 до 1.0)
    confidence: float = 1.0
    # Подробная диагностика для отладки и статистики
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "direction": int(self.direction),
            "is_tripped": self.is_tripped,
            "margin": float(self.margin),
            "confidence": float(self.confidence),
            "diagnostics": self.diagnostics,
        }


class PDRAlgorithm(ABC):
    """Абстрактный базовый класс для всех измерительных органов РНМ."""

    algorithm_id: str = "base_pdr"
    name: str = "Base PDR Algorithm"
    is_public: bool = True
    tunable_parameters: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        """Инициализация алгоритма с возможностью переопределения уставок."""
        self.params = dict(self.tunable_parameters)
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    @abstractmethod
    def compute(self, input_data: PDRInputData) -> PDROutput:
        """Расчёт решения органа РНМ для одной точки/окна."""
        pass

    def vectorized_compute(self, inputs: Sequence[PDRInputData]) -> list[PDROutput]:
        """Векторизованный или последовательный расчёт для списка входов."""
        return [self.compute(inp) for inp in inputs]
