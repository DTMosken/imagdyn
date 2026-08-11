"""Unit tests for temperature helpers (no full map pipeline)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from imagdyn.temperature import (
    small_inland_lake_mask,
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
    assert float(soft_step(torch.tensor([0.0]), 0.0, 1.0)) == pytest.approx(0.5, abs=0.05)


def test_small_inland_lake_mask() -> None:
    h, w = 64, 128
    land = np.ones((h, w), dtype=bool)
    # World ocean: bottom strip (largest water body)
    land[50:, :] = False
    # Small inland lake (4×5)
    land[10:14, 40:45] = False
    # Larger inland sea
    land[20:35, 80:110] = False
    # Coarse grid → large px area; threshold between small lake and inland sea size
    km_per_deg = (np.pi * 6371.0) / 180.0
    px_km2 = (180.0 / h) * km_per_deg * (360.0 / w) * km_per_deg
    small_area = 4 * 5 * px_km2
    mid_area = 15 * 30 * px_km2
    thresh = 0.5 * (small_area + mid_area)
    mask = small_inland_lake_mask(land, planet_radius_km=6371.0, max_area_km2=thresh)
    assert bool(mask[12, 42].item())
    assert not bool(mask[55, 10].item())  # ocean
    assert not bool(mask[25, 90].item())  # larger inland sea
    mask_tiny = small_inland_lake_mask(land, planet_radius_km=6371.0, max_area_km2=1.0)
    assert not bool(mask_tiny.any().item())
    s = soft_step(torch.tensor([-2.0, 0.0, 2.0]), center=0.0, scale=0.5)
    assert float(s[0]) < 0.5 < float(s[2])
