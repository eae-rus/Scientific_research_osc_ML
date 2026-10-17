"""Минимальный progress reporter без внешних зависимостей."""

from __future__ import annotations

import sys
import time


class ProgressReporter:
    """Показывать детерминированный прогресс длительного локального сценария."""

    def __init__(
        self,
        label: str,
        total: int,
        unit: str = "элем.",
        initial_completed: int = 0,
    ) -> None:
        if total <= 0:
            raise ValueError("Общий объём progress должен быть положительным")
        self.label = label
        self.total = total
        self.unit = unit
        self._initial_completed = min(max(initial_completed, 0), total)
        self._last_percent = -1
        self._started = time.monotonic()

    def update(self, completed: int) -> None:
        """Обновить индикатор, не печатая лишние строки."""

        bounded = min(max(completed, 0), self.total)
        percent = int(100 * bounded / self.total)
        if percent == self._last_percent:
            return
        elapsed = max(time.monotonic() - self._started, 1e-9)
        completed_this_run = max(0, bounded - self._initial_completed)
        rate = completed_this_run / elapsed
        remaining = (self.total - bounded) / rate if rate else 0.0
        print(
            f"\r{self.label}: {percent:3d}% ({bounded:,}/{self.total:,}; "
            f"{rate:.2f} {self.unit}/с; прошло {elapsed:.0f} с; ETA {remaining:.0f} с)",
            end="", file=sys.stderr, flush=True,
        )
        self._last_percent = percent

    def finish(self) -> None:
        """Завершить строку индикатора."""

        self.update(self.total)
        print(file=sys.stderr, flush=True)

    def snapshot(self, completed: int) -> dict[str, float | int | str]:
        """Вернуть машинно-читаемые rate/ETA для progress.json."""

        bounded = min(max(completed, 0), self.total)
        elapsed = max(time.monotonic() - self._started, 1e-9)
        completed_this_run = max(0, bounded - self._initial_completed)
        rate = completed_this_run / elapsed
        return {
            "label": self.label,
            "completed": bounded,
            "total": self.total,
            "percent": 100.0 * bounded / self.total,
            "elapsed_seconds": elapsed,
            "records_per_second": rate,
            "eta_seconds": (self.total - bounded) / rate if rate else 0.0,
        }
