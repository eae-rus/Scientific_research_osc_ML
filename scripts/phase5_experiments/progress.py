"""Минимальный progress reporter без внешних зависимостей."""

from __future__ import annotations

import sys
import time


class ProgressReporter:
    """Показывать детерминированный прогресс длительного локального сценария."""

    def __init__(self, label: str, total: int) -> None:
        if total <= 0:
            raise ValueError("Общий объём progress должен быть положительным")
        self.label = label
        self.total = total
        self._last_percent = -1
        self._started = time.monotonic()

    def update(self, completed: int) -> None:
        """Обновить индикатор, не печатая лишние строки."""

        bounded = min(max(completed, 0), self.total)
        percent = int(100 * bounded / self.total)
        if percent == self._last_percent and bounded != self.total:
            return
        elapsed = max(time.monotonic() - self._started, 1e-9)
        rate = bounded / elapsed
        remaining = (self.total - bounded) / rate if rate else 0.0
        print(
            f"\r{self.label}: {percent:3d}% ({bounded:,}/{self.total:,}; ETA {remaining:.0f} c)",
            end="", file=sys.stderr, flush=True,
        )
        self._last_percent = percent

    def finish(self) -> None:
        """Завершить строку индикатора."""

        self.update(self.total)
        print(file=sys.stderr, flush=True)
