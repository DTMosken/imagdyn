"""Unit tests for temperature helpers (no full map pipeline)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from imagdyn.params import STEFAN_BOLTZMANN, TEMPERATURE, temperature_call_kwargs
from imagdyn.temperature import (
    SIGMA,
    current_flux_W_m2,
    depth_inertia,
    ice_albedo_weight,
    inland_lake_area_km2,
    lake_inertia_from_area_km2,
    latent_heat_capacity,
    latent_peak_J_m2_K,
    small_inland_lake_mask,
    solar_declination_deg,
    soft_step,
    synthesize_temperatures,
    temperature_to_gray,
    transport_flux_W_m2,
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


def test_soft_step_midpoint() -> None:
    assert float(soft_step(torch.tensor([0.0]), 0.0, 1.0)) == pytest.approx(0.5, abs=0.05)
    s = soft_step(torch.tensor([-2.0, 0.0, 2.0]), center=0.0, scale=0.5)
    assert float(s[0]) < 0.5 < float(s[2])


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


def test_inland_lake_area_uses_cos_lat() -> None:
    h, w = 180, 360
    land = np.ones((h, w), dtype=bool)
    land[170:, :] = False  # world ocean (south)
    land[85:95, 10:20] = False  # ~equator 10×10
    land[5:15, 10:20] = False  # high latitude 10×10
    area = inland_lake_area_km2(land, planet_radius_km=6371.0, max_area_km2=1.0e12)
    a_eq = float(area[90, 15].item())
    a_hi = float(area[10, 15].item())
    assert a_eq > 0.0 and a_hi > 0.0
    assert a_hi < 0.4 * a_eq


def test_lake_inertia_ramps_to_one() -> None:
    a = torch.tensor([10_000.0, 20_000.0, 35_000.0, 50_000.0, 80_000.0])
    i = lake_inertia_from_area_km2(a)
    assert float(i[0]) == pytest.approx(0.45)
    assert float(i[1]) == pytest.approx(0.45)
    assert float(i[2]) == pytest.approx(0.725)
    assert float(i[3]) == pytest.approx(1.0)
    assert float(i[4]) == pytest.approx(1.0)


def test_area_weighted_mean_downweights_poles() -> None:
    from imagdyn.temperature import area_weighted_mean, latitude_grid

    h, w = 180, 360
    lat = latitude_grid(h, torch.device("cpu"))
    field = torch.zeros(h, w)
    field[0] = 100.0
    mu = float(area_weighted_mean(field, lat).item())
    pixel_mean = float(field.mean().item())
    assert mu < pixel_mean
    assert mu < 1.0


def test_depth_inertia_endpoints() -> None:
    z = torch.tensor([0.0, 200.0, 1.0e6])
    i = depth_inertia(z, inertia_shallow=0.25, inertia_deep=1.0, mix_depth_m=200.0)
    assert float(i[0]) == pytest.approx(0.25)
    mid = 0.25 + 0.75 * (1.0 - math.exp(-1.0))
    assert float(i[1]) == pytest.approx(mid, rel=1e-5)
    assert float(i[2]) == pytest.approx(1.0, abs=1e-5)


def test_latent_heat_capacity_peaks_at_freeze() -> None:
    peak = latent_peak_J_m2_K(0.5, 0.8)
    t = torch.tensor([-10.0, 0.0, 10.0])
    c = latent_heat_capacity(t, freeze_c=0.0, delta_c=0.8, peak=peak)
    assert float(c[1]) == pytest.approx(peak, rel=1e-5)
    assert float(c[1]) > float(c[0]) > 0.0
    assert float(c[0]) == pytest.approx(float(c[2]), rel=1e-5)
    assert float(c[0]) < 0.05 * peak


def test_ice_albedo_freeze_points_split() -> None:
    t = torch.tensor([0.0, -1.8])
    land_w = ice_albedo_weight(t, freeze_c=0.0, soft_c=1.5)
    ocean_w = ice_albedo_weight(t, freeze_c=-1.8, soft_c=1.5)
    assert float(land_w[0]) == pytest.approx(0.5, abs=0.02)
    assert float(ocean_w[1]) == pytest.approx(0.5, abs=0.02)
    assert float(land_w[0]) > float(ocean_w[0])  # 0°C is freezing on land, not ocean


def test_current_flux_linearized_olr() -> None:
    t_k = torch.tensor([280.0])
    dT = torch.tensor([3.0])
    g = 1.35
    q = current_flux_W_m2(t_k, dT, g)
    expect = (4.0 * SIGMA * (280.0**3) / g) * 3.0
    assert float(q[0]) == pytest.approx(expect, rel=1e-5)
    assert STEFAN_BOLTZMANN == SIGMA


def test_transport_flux_relaxes_to_global_mean() -> None:
    from imagdyn.temperature import area_weighted_mean, latitude_grid

    h, w = 180, 360
    lat = latitude_grid(h, torch.device("cpu"))
    t = torch.zeros(h, w)
    t[h // 2] = 10.0
    t[0] = -20.0
    lam = 3.8
    q = transport_flux_W_m2(t, lat, lam)
    t_bar = float(area_weighted_mean(t, lat).item())
    assert float(q[h // 2, 0]) == pytest.approx(lam * (t_bar - 10.0), rel=1e-5)
    assert float(q[0, 0]) == pytest.approx(lam * (t_bar - (-20.0)), rel=1e-5)
    assert float(q[h // 2, 0]) < 0.0
    assert float(q[0, 0]) > 0.0
    assert float(area_weighted_mean(q, lat).item()) == pytest.approx(0.0, abs=1e-5)
    assert float(transport_flux_W_m2(t, lat, 0.0).abs().max()) == 0.0


def test_spinup_returns_last_twelve_months() -> None:
    h, w = 16, 32
    elev = np.full((h, w), 0.5, dtype=np.float32)
    land = np.zeros((h, w), dtype=bool)
    land[6:10, 8:16] = True
    elev[land] = 0.55
    monthly, annual, meta = synthesize_temperatures(
        elev,
        land,
        device=torch.device("cpu"),
        **temperature_call_kwargs(
            spinup_years=2,
            currents=False,
            maritime_diffuse_passes=1,
            aa_blend_px=1,
            transport_lambda=0.0,
        ),
    )
    assert monthly.shape == (12, h, w)
    assert annual.shape == (h, w)
    assert meta["spinup_years"] == 2
    assert meta["month_steps"] == 24
    assert meta["transport_lambda"] == 0.0
    assert np.isfinite(monthly).all()
