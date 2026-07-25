"""Базовые контракты данных и интерфейс алгоритмов РНМ (PDR).

Этот модуль задаёт типы данных, перечисления и абстрактный базовый класс `PDRAlgorithm`.
Все измерительные органы РНМ (публичные и приватные) наследуются от `PDRAlgorithm`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, Optional, Sequence
import numpy as np


class PDRDirection(IntEnum):
    """Направление мощности / решение органа РНМ.

    - UNLABELED (-999): Неразмеченная или невалидная область (зона начального прогрева БПФ,
      отсутствие входных данных или недостаточная длина осциллограммы).
    - REVERSE (0): Обратное направление (мощность в сеть / P < 0 / выбег двигателей /
      КЗ во внешней сети выше ввода) => РАЗРЕШЕНИЕ БАВР (is_tripped = False).
    - FORWARD (1): Прямое направление (мощность в нагрузку / P > 0 / нормальный режим /
      КЗ на шинах подстанции) => БЛОКИРОВКА БАВР (is_tripped = True).
    """

    UNLABELED = -999
    REVERSE = 0
    FORWARD = 1


@dataclass(frozen=True)
class PDRInputData:
    """Входные данные для расчёта органа РНМ в конкретный момент времени."""

    # Комплексные фазоры 1-й гармоники напряжений (фазные {"A", "B", "C"} и/или линейные {"AB", "BC", "CA"})
    phasors_u: Dict[str, complex]
    # Комплексные фазоры 1-й гармоники токов {"A", "B", "C"}
    phasors_i: Dict[str, complex]
    # Доступные фазоры предыстории для органов с памятью по напряжению/току
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

    # Решение по направлению (FORWARD = 1 / БЛОКИРОВКА, REVERSE = 0 / РАЗРЕШЕНИЕ, UNLABELED = -999)
    direction: PDRDirection
    # Блокировка БАВР (True при FORWARD = 1; False при REVERSE = 0)
    is_tripped: bool
    # Непрерывная величина запаса срабатывания (margin / мощность / проекция)
    margin: float
    # Уверенность решения (от 0.0 до 1.0)
    confidence: float = 1.0
    # Служебная диагностика для отладки и единичной статистики (не сохраняется на диск при генерации датасетов)
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
