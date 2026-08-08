"""Tests for ocean current coastline filter."""

from __future__ import annotations

import torch

from imagdyn.currents import OceanCurrentFilter


def _continent_strip() -> torch.Tensor:
    """Land in columns 4..7 of a 10×12 grid (east face at col 7, west at col 4)."""
    land = torch.zeros((10, 12), dtype=torch.bool)
    land[:, 4:8] = True
    return land


def test_detect_east_west_coasts() -> None:
    land = _continent_strip()
    east, west = OceanCurrentFilter.detect_coast_edges(land)
    # Ocean just east of continent (col 8) has land to the west → east coast
    assert bool(east[:, 8].any())
    assert not bool(west[:, 8].any())
    # Ocean just west of continent (col 3) has land to the east → west coast
    assert bool(west[:, 3].any())
    assert not bool(east[:, 3].any())


def test_latitude_weight_peaks_near_30() -> None:
    filt = OceanCurrentFilter(peak_lat_deg=30.0, lat_sigma_deg=12.0)
    lat = torch.tensor([0.0, 30.0, 60.0, 90.0])
    w = filt.latitude_weight(lat)
    assert float(w[1]) == max(float(x) for x in w)
    assert float(w[0]) < float(w[1])
    assert float(w[3]) < float(w[1])


def test_compute_warm_east_cold_west() -> None:
    land = _continent_strip()
    # Put mid-lat rows around index where lat≈30 if H maps 90→-90
    h, w = land.shape
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    filt = OceanCurrentFilter(
        warm_delta_C=4.0,
        cold_delta_C=-4.0,
        reach_km=200.0,
        diffuse_passes=2,
        land_bleed=0.0,
        planet_radius_km=6371.0,
    )
    corr = filt.compute(land, lat)
    assert corr.humidity is None  # placeholder
    dT = corr.temperature_C
    # East ocean column should be warmer than west ocean column on average
    assert float(dT[:, 8].mean()) > float(dT[:, 3].mean())


def test_apply_adds_delta() -> None:
    land = _continent_strip()
    lat = torch.zeros(land.shape[0])
    lat[:] = 30.0
    filt = OceanCurrentFilter(warm_delta_C=2.0, cold_delta_C=-2.0, diffuse_passes=1, reach_km=100.0)
    base = torch.zeros_like(land, dtype=torch.float32)
    out, hum = filt.apply(base, land=land, lat_deg=lat)
    assert hum is None
    assert not torch.allclose(out, base)
