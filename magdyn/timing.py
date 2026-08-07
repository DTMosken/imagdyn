"""Lightweight step timers for compute jobs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60.0)
    if m < 60:
        return f"{int(m)}m {s:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m}m {s:04.1f}s"


class StepTimer:
    """Track wall-clock time for named steps; report avg / total / ETA."""

    def __init__(self, label: str = "", *, total_steps: int | None = None) -> None:
        self.label = label
        self.total_steps = total_steps
        self.t0 = time.perf_counter()
        self.steps: list[tuple[str, float]] = []
        self._step_t0: float | None = None
        self._step_name: str | None = None

    def begin(self, name: str = "") -> None:
        self.end()
        self._step_t0 = time.perf_counter()
        self._step_name = name

    def end(self) -> float | None:
        if self._step_t0 is None:
            return None
        dt = time.perf_counter() - self._step_t0
        self.steps.append((self._step_name or f"step{len(self.steps)+1}", dt))
        self._step_t0 = None
        self._step_name = None
        return dt

    def tick(self, name: str = "") -> float | None:
        """Finish previous step (if any) and start ``name``. Return previous dt."""
        dt = self.end()
        self.begin(name)
        return dt

    @contextmanager
    def step(self, name: str = "") -> Iterator[None]:
        self.begin(name)
        try:
            yield
        finally:
            self.end()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def step_count(self) -> int:
        n = len(self.steps)
        if self._step_t0 is not None:
            n += 1
        return n

    @property
    def mean_step(self) -> float:
        if not self.steps:
            return 0.0
        return sum(d for _, d in self.steps) / len(self.steps)

    def eta(self) -> float | None:
        if self.total_steps is None or not self.steps:
            return None
        done = len(self.steps)
        remaining = max(0, self.total_steps - done)
        return remaining * self.mean_step

    def progress_line(self, *, last_name: str | None = None, last_dt: float | None = None) -> str:
        done = len(self.steps)
        parts: list[str] = []
        if last_name is not None and last_dt is not None:
            parts.append(f"{last_name} {format_duration(last_dt)}")
        if self.total_steps is not None:
            parts.append(f"{done}/{self.total_steps}")
        else:
            parts.append(f"n={done}")
        parts.append(f"avg {format_duration(self.mean_step)}")
        parts.append(f"elapsed {format_duration(self.elapsed)}")
        eta = self.eta()
        if eta is not None:
            parts.append(f"ETA {format_duration(eta)}")
        return "  ".join(parts)

    def summary(self, *, finalize: bool = True) -> str:
        if finalize:
            self.end()
        total = self.elapsed
        n = len(self.steps)
        avg = self.mean_step
        label = f"{self.label}: " if self.label else ""
        lines = [
            f"{label}total {format_duration(total)}",
            f"  steps {n}  avg/step {format_duration(avg)}",
        ]
        if n and n <= 24:
            for name, dt in self.steps:
                lines.append(f"    {name}: {format_duration(dt)}")
        elif n > 24:
            slow = sorted(self.steps, key=lambda x: x[1], reverse=True)[:5]
            lines.append("  slowest:")
            for name, dt in slow:
                lines.append(f"    {name}: {format_duration(dt)}")
        return "\n".join(lines)

    def as_meta(self) -> dict[str, Any]:
        self.end()
        return {
            "label": self.label,
            "total_s": round(self.elapsed, 3),
            "n_steps": len(self.steps),
            "avg_step_s": round(self.mean_step, 3),
            "steps": [{"name": n, "s": round(d, 3)} for n, d in self.steps],
        }
