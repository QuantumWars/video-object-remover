"""Tiny per-stage timing so optimizations can be measured, not guessed."""
from __future__ import annotations
import time
from contextlib import contextmanager


class Timer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._order: list[str] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        if name not in self.stages:
            self._order.append(name)
            self.stages[name] = 0.0
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] += time.perf_counter() - start

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def summary(self) -> str:
        total = self.total()
        lines = ["[timing]"]
        for name in self._order:
            secs = self.stages[name]
            pct = 100 * secs / total if total else 0
            lines.append(f"  {name:<12} {secs:7.1f}s  {pct:4.0f}%")
        lines.append(f"  {'TOTAL':<12} {total:7.1f}s")
        return "\n".join(lines)
