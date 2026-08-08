"""Tests for StepTimer and duration formatting."""

from __future__ import annotations

import time

import pytest

from imagdyn.timing import StepTimer, format_duration


def test_format_duration() -> None:
    assert format_duration(-1) == "0.00s"
    assert format_duration(1.5) == "1.50s"
    assert format_duration(65) == "1m 05.0s"
    assert "h" in format_duration(3700)


def test_step_timer_context() -> None:
    t = StepTimer("job", total_steps=2)
    with t.step("a"):
        time.sleep(0.01)
    with t.step("b"):
        time.sleep(0.01)
    assert len(t.steps) == 2
    assert t.mean_step > 0
    assert t.eta() == pytest.approx(0.0, abs=1e-3)
    meta = t.as_meta()
    assert meta["label"] == "job"
    assert meta["n_steps"] == 2
    assert "total" in t.summary()
