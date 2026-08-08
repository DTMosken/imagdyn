"""Unit tests for temperature helpers (no full map pipeline)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from imagdyn.temperature import (
    solar_declination_deg,
    soft_step,
    soft_tropics_weight,
    temperature_to_gray,
)


def test_solar_declination_equinox_near_zero() -> None:
    # DOY 81 ≈ March equinox → declination ≈ 0
    d = float(solar_declination_deg(81.0))
    assert abs(d) < 0.5


def test_solar_declination_solstice_near_obliquity() -> None:
    # DOY ~172 ≈ June solstice
    d = float(solar_declination_deg(172.0))
    assert d == pytest.approx(23.5, abs=1.0)


def test_temperature_to_gray_endpoints() -> None:
    t = np.array([-60.0, 45.0, 0.0], dtype=np.float32)
    g = temperature_to_gray(t, -60.0, 45.0)
    assert g[0] == 0
    assert g[1] == 255
    assert 0 < g[2] < 255


def test_soft_tropics_and_step() -> None:
    lat = torch.tensor([0.0, 23.5, 60.0])
    w = soft_tropics_weight(lat.abs(), 23.5)
    assert float(w[0]) > float(w[1]) > float(w[2])
    s = soft_step(torch.tensor([-2.0, 0.0, 2.0]), center=0.0, scale=0.5)
    assert float(s[0]) < 0.5 < float(s[2])
