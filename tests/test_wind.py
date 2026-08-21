"""Tests for wind / pressure helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from imagdyn.wind import (
    WindField,
    enforce_area_mean_zero,
    omega_from_spin,
    read_dot_speed_rgb_png,
    read_uvp_rgb_png,
    thermal_to_pressure_anomaly,
    write_dot_speed_rgb_png,
    write_uvp_rgb_png,
)


def test_omega_from_spin_24h() -> None:
    w = omega_from_spin(24.0 * 3600.0)
    assert w == pytest.approx(2 * np.pi / (24 * 3600), rel=1e-6)


def test_thermal_to_pressure_hot_is_low() -> None:
    t = torch.tensor([[5.0, -5.0]])
    p = thermal_to_pressure_anomaly(t, k_hpa_per_c=1.0)
    assert float(p[0, 0]) < float(p[0, 1])


def test_thermal_latitude_weight_weaker_at_equator() -> None:
    from imagdyn.wind import thermal_latitude_weight

    lat = torch.tensor([-45.0, 0.0, 45.0])
    w = thermal_latitude_weight(lat, sigma_deg=18.0, equator_frac=0.2)
    assert float(w[1]) == pytest.approx(0.2, abs=1e-5)
    assert float(w[0]) > float(w[1])
    assert float(w[2]) > float(w[1])
    assert float(w[0]) == pytest.approx(float(w[2]), abs=1e-5)


def test_mean_zero_after_expand() -> None:
    h, w = 36, 72
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    field = torch.randn(h, w)
    out = enforce_area_mean_zero(field, lat)
    wt = torch.cos(torch.deg2rad(lat)).clamp(0, 1)[:, None].expand(h, w)
    mean = float((out * wt).sum() / wt.sum())
    assert abs(mean) < 1e-4


def test_windfield_outputs_and_files(tmp_path: Path) -> None:
    h, w = 64, 128
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = 10.0 + 20.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))
    T = T.astype(np.float32)
    land = np.zeros((h, w), dtype=bool)
    land[:, 40:60] = True
    elev = np.zeros((h, w), dtype=np.float32)
    elev[land] = 500.0
    elev[20:30, 45:55] = 2000.0

    wf = WindField(t_spin_s=24 * 3600)
    res = wf.compute(T, land, elev, device=torch.device("cpu"))
    assert res.u.shape == (h, w)
    assert res.terrain_dot.shape == (h, w)
    assert "thermal_equator_lat_deg" in res.meta
    assert abs(res.meta["thermal_equator_lat_deg"]) < 20.0
    belts = res.meta["amplitudes_hpa"]
    p0 = float(belts["p0"])
    assert belts["design"]["equator"] < p0
    assert belts["design"]["polar_n"] > p0
    assert belts["design"]["subtropical_n"] > p0
    assert belts["design"]["subpolar_n"] < p0

    u = res.u.numpy()
    v = res.v.numpy()
    p = res.pressure.numpy()
    uv_path = tmp_path / "uv.png"
    scale = write_uvp_rgb_png(
        uv_path,
        u,
        v,
        p,
        u_range=(float(u.min()), float(u.max() + 1e-6)),
        v_range=(float(v.min()), float(v.max() + 1e-6)),
        p_range=(float(p.min()), float(p.max() + 1e-6)),
    )
    u2, v2, p2 = read_uvp_rgb_png(uv_path, scale["u"], scale["v"], scale["pressure"])
    assert u2.shape == u.shape
    du = (scale["u"][1] - scale["u"][0]) / 255
    dv = (scale["v"][1] - scale["v"][0]) / 255
    assert np.max(np.abs(u2 - u)) <= du * 1.1 + 1e-5
    assert np.max(np.abs(v2 - v)) <= dv * 1.1 + 1e-5

    dot_path = tmp_path / "dot.png"
    dscale = write_dot_speed_rgb_png(dot_path, res.terrain_dot.numpy(), res.speed.numpy())
    d2, s2 = read_dot_speed_rgb_png(dot_path, dscale["terrain_dot"], dscale["speed"])
    assert d2.shape == (h, w)
    assert s2.shape == (h, w)


def test_write_wind_products_color_uv(tmp_path: Path) -> None:
    from imagdyn.wind import write_wind_products

    h, w = 32, 64
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 10.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    land = np.zeros((h, w), dtype=bool)
    elev = np.zeros((h, w), dtype=np.float32)
    res = WindField().compute(T, land, elev, device=torch.device("cpu"))
    meta = write_wind_products(
        res,
        tmp_path,
        label="test",
        uv_name="uv.png",
        dot_name="dot.png",
    )
    assert (tmp_path / "uv.png").is_file()
    assert (tmp_path / "dot.png").is_file()
    assert not (tmp_path / "p.png").exists()
    assert not (tmp_path / "u.npy").exists()
    assert not list(tmp_path.glob("*.f32"))
    assert "uv_scale" in meta
    assert "terrain_dot_scale" in meta
    assert "gray_scale_pressure_hpa" in meta
    assert "gray_scale_speed" in meta
    assert "gray_scale_terrain_dot" not in meta


def test_cli_parser() -> None:
    from imagdyn.cli import build_parser

    p = build_parser()
    args = p.parse_args(["wind", "--", "--annual-only", "--cpu"])
    assert args.command == "wind"


def test_coriolis_deflects_meridional_force() -> None:
    h, w = 180, 360
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 20.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    res = WindField(friction_ocean=5e-7, friction_land=5e-7).compute(
        T, np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=np.float32), device=torch.device("cpu")
    )
    sth_n = float(res.meta["belt_centers_design_deg"]["subtropical_high"][0])
    sample_lat = sth_n + 8.0  # poleward flank → midlatitude westerlies
    row = int(np.argmin(np.abs(lat - sample_lat)))
    u_mean = float(res.u[row].mean())
    v_mean = float(res.v[row].mean())
    assert abs(u_mean) > 0.15
    assert abs(u_mean) > 0.12 * abs(v_mean) + 0.05


def test_subtropical_high_near_30deg() -> None:
    h, w = 180, 360
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 20.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    res = WindField(
        drag_kappa0=1.2e-6,
        drag_kappa_lat=8.0,
        hadley_v_m_s=1.5,
        thermal_wind_scale=8882.0,
        belt_lon_sectors=12,
    ).compute(
        T, np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=np.float32), device=torch.device("cpu")
    )
    design = res.meta["belt_centers_design_deg"]["subtropical_high"]
    teq_d = res.meta["belt_centers_design_deg"]["thermal_equator"]
    # Rayleigh-drag AMC + thermal-wind u_crit → Earth-like ~28–32° when teq≈0
    assert abs(teq_d) < 3.0
    assert 26.0 <= design[0] <= 34.0
    assert -34.0 <= design[1] <= -26.0
    assert res.meta["belt_centers_design_deg"]["polar_high"] == [89.0, -89.0]
    assert int(res.meta["belt_centers_design_deg"].get("n_lon_sectors", 0)) >= 1
    assert "upper_amc" in res.meta
    assert res.meta["upper_amc"]["method"].startswith("rayleigh_amc")
    assert 20.0 <= res.meta["upper_amc"]["u_crit_n_m_s"] <= 70.0
    assert res.meta["belt_width_deg"]["profile"] == "cosine_halfwave_between_centers"
    assert res.meta["belt_width_deg"]["blend"] == pytest.approx(1.5, abs=0.01)


def test_nh_high_has_easterlies_on_equatorward_flank() -> None:
    h, w = 120, 240
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (18.0 + 18.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    res = WindField(friction_ocean=5e-7, friction_land=5e-7).compute(
        T, np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=np.float32), device=torch.device("cpu")
    )
    sth_n = float(res.meta["subtropical_high_lat_deg"][0])
    # Equatorward flank of the high (into the trades)
    sample_lat = sth_n - 8.0
    row = int(np.argmin(np.abs(lat - sample_lat)))
    assert float(res.u[row].mean()) < -0.05


def test_quadratic_drag_equator_balance() -> None:
    """At f=0, quadratic drag ⇒ |V| = sqrt(|F|/c_d)."""
    cd = 2.0e-6
    force = 1.0e-3  # m/s²
    s_expect = (abs(force) / cd) ** 0.5
    f2 = 0.0
    F2 = force * force
    disc = f2 * f2 + 4.0 * cd * cd * F2
    speed2 = (2.0 * F2) / (f2 + disc**0.5)
    assert abs(speed2**0.5 - s_expect) < 1e-9
    h, width = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 20.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, width))).astype(
        np.float32
    )
    res = WindField().compute(
        T,
        np.zeros((h, width), dtype=bool),
        np.zeros((h, width), dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert "quadratic" in res.meta.get("friction_kind", "")


def test_belt_peak_matches_meta_latitude() -> None:
    h, w = 180, 360
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 20.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    res = WindField().compute(
        T, np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=np.float32), device=torch.device("cpu")
    )
    p_z = res.pressure.mean(dim=1).numpy()
    sth_n = float(res.meta["subtropical_high_lat_deg"][0])
    row = int(np.argmin(np.abs(lat - sth_n)))
    # Local neighborhood maximum near the reported center
    lo, hi = max(0, row - 8), min(h, row + 9)
    assert int(np.argmax(p_z[lo:hi]) + lo) == row


def test_near_equator_not_pure_meridional() -> None:
    h, w = 180, 360
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 20.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    res = WindField().compute(
        T, np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=np.float32), device=torch.device("cpu")
    )
    row = int(np.argmin(np.abs(lat - 5.0)))
    u_mean = float(res.u[row].mean())
    v_mean = float(res.v[row].mean())
    assert abs(u_mean) > 0.2 * abs(v_mean)


def test_belt_amplitudes_capped() -> None:
    h, w = 64, 128
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (10.0 + 40.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    res = WindField().compute(
        T, np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=np.float32), device=torch.device("cpu")
    )
    amps = res.meta["amplitudes_hpa"]
    assert amps["a"] <= 7.0 + 1e-6
    assert amps["b_north"] <= 7.0 + 1e-6
    assert amps["design"]["subtropical_n"] > amps["design"]["equator"]
    assert abs(amps["design"]["subtropical_n"] - amps["design"]["equator"]) < 25.0


def test_speed_scale_in_meta(tmp_path: Path) -> None:
    from imagdyn.wind import write_wind_products

    h, w = 32, 64
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 10.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    res = WindField().compute(
        T, np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=np.float32), device=torch.device("cpu")
    )
    meta = write_wind_products(
        res,
        tmp_path,
        label="t",
        uv_name="uv.png",
        dot_name="dot.png",
    )
    s_max = meta["gray_scale_speed"][1]
    assert s_max < 200.0


def test_grad_lonlat_polar_ns_zero() -> None:
    from imagdyn.wind import grad_lonlat

    h, w = 48, 96
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    # Strong N–S ramp so interior dfdy ≠ 0
    field = torch.arange(h, dtype=torch.float32)[:, None].expand(h, w).contiguous()
    _dfdx, dfdy = grad_lonlat(field, dy_km=10.0, dx_eq_km=10.0, lat_deg=lat)
    assert float(dfdy[0].abs().max()) == 0.0
    assert float(dfdy[-1].abs().max()) == 0.0
    assert float(dfdy[h // 2].abs().mean()) > 0.0


def test_temperature_to_sea_level() -> None:
    from imagdyn.wind import temperature_to_sea_level

    T = torch.tensor([[2.0, 15.0]])
    elev = torch.tensor([[2000.0, 0.0]])
    T_sl = temperature_to_sea_level(T, elev, lapse_k_per_km=6.5)
    assert float(T_sl[0, 0]) == pytest.approx(15.0, abs=1e-5)
    assert float(T_sl[0, 1]) == pytest.approx(15.0, abs=1e-5)


def test_mountain_slp_not_from_surface_cold() -> None:
    """Cold mountaintop surface T must not invent a local thermal high after SLP."""
    h, w = 48, 96
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 10.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    land = np.zeros((h, w), dtype=bool)
    elev = np.zeros((h, w), dtype=np.float32)
    # Plateau with surface cooling = 6.5 K/km * 2 km
    land[16:32, 30:50] = True
    elev[land] = 2000.0
    T[land] -= 13.0
    res = WindField(lapse_k_per_km=6.5, wind_smooth_sigma_px=0.0).compute(
        T, land, elev, device=torch.device("cpu")
    )
    # Plateau SLP thermal contribution ≈ neighbors; local p close to same-latitude ocean mean
    row = 24
    p_plat = float(res.pressure[row, 40].item())
    p_ocean = float(res.pressure[row, 10].item())
    assert abs(p_plat - p_ocean) < 3.0


def test_smooth_wind_uv_mixes_neighbors() -> None:
    from imagdyn.wind import smooth_wind_uv

    h, w = 32, 64
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    u = torch.zeros(h, w)
    v = torch.zeros(h, w)
    u[16, 32] = 10.0
    us, vs = smooth_wind_uv(u, v, sigma_px=1.5, lat_deg=lat)
    assert float(us[16, 33].abs()) > 0.1
    assert float(us[16, 32]) < 10.0


def test_land_damps_planetary_belt() -> None:
    """Over land, belt anomaly is weakened; thermal can dominate."""
    h, w = 90, 180
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (15.0 + 18.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    land = np.zeros((h, w), dtype=bool)
    land[:, 40:90] = True
    elev = np.zeros((h, w), dtype=np.float32)
    elev[land] = 200.0
    # Uniform T in lon → thermal≈0; land vs ocean pressure contrast from belt damp only
    res = WindField(land_belt_frac=0.2, pressure_smooth_sigma_px=0.0).compute(
        T, land, elev, device=torch.device("cpu")
    )
    sth = float(res.meta["belt_centers_design_deg"]["subtropical_high"][0])
    row = int(np.argmin(np.abs(lat - sth)))
    p_ocean = float(res.pressure[row, 10].item())
    p_land = float(res.pressure[row, 60].item())
    # Ocean keeps full subtropical high; land belt damped toward p0
    assert p_ocean > p_land + 0.3


def test_land_block_stronger_on_highs() -> None:
    from imagdyn.wind import land_belt_block_scale

    land = torch.ones(2, 2)
    high = torch.full((2, 2), 4.0)
    low = torch.full((2, 2), -4.0)
    s_hi = land_belt_block_scale(high, land, land_belt_frac=0.1, block_half_hpa=1.5)
    s_lo = land_belt_block_scale(low, land, land_belt_frac=0.1, block_half_hpa=1.5)
    assert float(s_hi.mean()) < float(s_lo.mean())
    assert float(s_hi.mean()) < 0.25
    assert float(s_lo.mean()) > 0.7


def test_pressure_diffuse_wraps_longitude() -> None:
    from imagdyn.wind import diffuse_pressure_2d

    h, w = 48, 96
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    p = torch.zeros(h, w)
    p[:, 0] = 10.0
    p[:, -1] = 10.0
    out = diffuse_pressure_2d(p, lat, sigma_px=2.0, polar_cut_lat=89.0, polar_fade_lat=87.0)
    mid = h // 2
    assert float(out[mid, 1]) > 0.5
    assert float(out[mid, w - 2]) > 0.5
    # Gradual E–W damp: fine grid must not hard-jump across the cut
    h2, w2 = 360, 180
    lat2 = 90.0 - (torch.arange(h2, dtype=torch.float32) + 0.5) * (180.0 / h2)
    lon = torch.linspace(0.0, 2.0 * math.pi, w2)
    p2 = torch.sin(lon)[None, :].expand(h2, w2).contiguous()
    out2 = diffuse_pressure_2d(p2, lat2, sigma_px=1.0, polar_cut_lat=89.0, polar_fade_lat=87.0)
    i_cut = int(torch.argmin((lat2.abs() - 89.0).abs()).item())
    j = w2 // 4
    dp = float((out2[i_cut, j] - out2[i_cut + 1, j]).abs().item())
    assert dp < 0.25
    assert float(out2[0].std()) < 1e-5


def test_prepare_surface_polar_and_water_zero() -> None:
    from imagdyn.wind import prepare_surface_for_wind

    h, w = 180, 60
    land = np.ones((h, w), dtype=bool)
    elev = np.full((h, w), 500.0, dtype=np.float32)
    elev[:5] = 2000.0
    land_o, elev_o, elev_t, soft = prepare_surface_for_wind(
        land, elev, polar_ocean_lat=89.0, polar_fade_lat=87.0, coast_sigma_px=2.0
    )
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    polar = np.abs(lat) >= 89.0
    assert not land_o[polar].any()
    assert float(np.abs(elev_o[polar]).max()) == 0.0
    assert float(np.abs(elev_t[polar]).max()) == 0.0
    # Ocean cells (forced polar) contribute 0 height before soft
    mid = h // 2
    assert elev_t[mid].mean() > 100.0


def test_water_zero_in_terrain_dot_field() -> None:
    from imagdyn.wind import prepare_surface_for_wind

    h, w = 64, 128
    land = np.zeros((h, w), dtype=bool)
    land[:, 40:80] = True
    elev = np.full((h, w), -2000.0, dtype=np.float32)
    elev[land] = 31.0
    elev[20:40, 50:70] = 1200.0
    _lo, _eo, elev_t, soft = prepare_surface_for_wind(land, elev, coast_sigma_px=3.0)
    # Far-ocean soft≈0 → terrain≈0 (no bathymetry cliff into ∇h)
    assert float(elev_t[:, :10].mean()) < 5.0
    assert float(elev_t[25:35, 55:65].mean()) > 200.0
    assert 0.0 <= float(soft.min()) <= float(soft.max()) <= 1.0


def test_wind_stats_waterworld(tmp_path: Path) -> None:
    from imagdyn.wind import build_wind_stats, write_wind_stats

    h, w = 64, 128
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    T = (12.0 + 18.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))).astype(np.float32)
    land = np.zeros((h, w), dtype=bool)
    land[20:40, 30:50] = True
    elev = np.zeros((h, w), dtype=np.float32)
    elev[land] = 400.0
    wf = WindField()
    res = wf.compute(T, land, elev, device=torch.device("cpu"))
    stats = build_wind_stats(res, land, temperature_C=T, wind=wf, device=torch.device("cpu"))
    assert "pressure_hpa" in stats
    assert "waterworld_1d" in stats
    ww = stats["waterworld_1d"]
    assert "cases" in ww
    assert "synthesize_temperatures" in (ww.get("note") or "")
    for key in ("tropic_cancer", "equator", "tropic_capricorn"):
        c = ww["cases"][key]
        assert len(c["pressure_hpa"]) == len(c["speed_m_s"]) == len(c["to_direction_deg"])
        assert len(c["temperature_C"]) == len(c["pressure_hpa"])
        assert "lat_deg" not in c
        assert "u_east_m_s" not in c and "v_south_m_s" not in c
        assert "month_name" in c
        # Aquaplanet SST keeps poles cold (not TOA polar-day peak)
        assert abs(c["thermal_equator_lat_deg"]) < 15.0
        peak_i = int(np.argmax(c["temperature_C"]))
        # Reconstruct sample latitudes from grid + lat_step
        step = max(1, int(round(float(ww["lat_step_deg"]) / (180.0 / h))))
        lats = 90.0 - (np.arange(0, h, step) + 0.5) * (180.0 / h)
        assert abs(float(lats[peak_i])) < 20.0
    assert "u_east_m_s" not in stats and "v_south_m_s" not in stats
    teq_n = ww["cases"]["tropic_cancer"]["thermal_equator_lat_deg"]
    teq_0 = ww["cases"]["equator"]["thermal_equator_lat_deg"]
    teq_s = ww["cases"]["tropic_capricorn"]["thermal_equator_lat_deg"]
    # Ocean heat capacity lags insolation, so teq does not track δ monotonically.
    assert abs(teq_n) < 15.0 and abs(teq_0) < 15.0 and abs(teq_s) < 15.0
    # Seasonal ocean asymmetry: NH warmer in N-tropic month than S-tropic month
    def _t_at(case: dict, target_lat: float) -> float:
        step = max(1, int(round(float(ww["lat_step_deg"]) / (180.0 / h))))
        lats = 90.0 - (np.arange(0, h, step) + 0.5) * (180.0 / h)
        ts = np.asarray(case["temperature_C"], dtype=np.float64)
        n = min(len(lats), len(ts))
        return float(ts[int(np.argmin(np.abs(lats[:n] - target_lat)))])

    assert _t_at(ww["cases"]["tropic_cancer"], 45.0) > _t_at(
        ww["cases"]["tropic_capricorn"], 45.0
    )
    assert _t_at(ww["cases"]["tropic_capricorn"], -45.0) > _t_at(
        ww["cases"]["tropic_cancer"], -45.0
    )
    path = write_wind_stats(stats, tmp_path)
    assert path.is_file()
    assert path.name == "wind_stats.json"


def test_aquaplanet_t_peaks_near_equator_not_pole() -> None:
    from imagdyn.wind import synthesize_aquaplanet_temperatures, _aquaplanet_month_indices

    h = 180
    monthly, meta = synthesize_aquaplanet_temperatures(
        h, device=torch.device("cpu"), sample_width=64, obliquity_deg=23.5,
        spinup_years=1,
    )
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)
    decls = np.asarray(meta["declination_deg"], dtype=np.float64)
    mi = _aquaplanet_month_indices(decls)["tropic_cancer"]
    t = monthly[mi].mean(axis=1)
    peak_lat = float(lat[int(np.argmax(t))])
    assert abs(peak_lat) < 15.0
    assert float(t[0]) < float(t[h // 2])  # north pole colder than equator


def test_amc_profile_at_30deg() -> None:
    from imagdyn.wind import omega_from_spin, upper_amc_u_profile

    lat = torch.tensor([0.0, 15.0, 30.0, 45.0])
    omega = omega_from_spin(24.0 * 3600.0)
    R = 6371e3
    u = upper_amc_u_profile(lat, 0.0, omega=omega, radius_m=R)
    assert float(u[0]) == pytest.approx(0.0, abs=1e-3)
    assert float(u[2]) == pytest.approx(134.0, abs=5.0)
    assert float(u[3]) > float(u[2]) > float(u[1])


def test_drag_amc_subtropical_near_30() -> None:
    from imagdyn.wind import find_subtropical_high_from_amc, omega_from_spin

    h = 360
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    T = 15.0 + 20.0 * torch.cos(torch.deg2rad(lat))
    omega = omega_from_spin(24.0 * 3600.0)
    sth_n, sth_s, meta = find_subtropical_high_from_amc(
        lat,
        0.0,
        T,
        omega=omega,
        radius_m=6371e3,
    )
    assert 28.0 <= sth_n <= 32.0
    assert -32.0 <= sth_s <= -28.0
    assert abs(meta["u_at_phi0_n_m_s"]) < 1.0
    assert 20.0 <= meta["u_crit_n_m_s"] <= 70.0


def test_drag_amc_itcz_drift_shifts_subtropical_highs() -> None:
    from imagdyn.wind import find_subtropical_high_from_amc, omega_from_spin

    h = 360
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    T = 15.0 + 20.0 * torch.cos(torch.deg2rad(lat))
    omega = omega_from_spin(24.0 * 3600.0)
    kwargs = dict(omega=omega, radius_m=6371e3)
    n0, s0, _ = find_subtropical_high_from_amc(lat, 0.0, T, **kwargs)
    n1, s1, _ = find_subtropical_high_from_amc(lat, 10.0, T, **kwargs)
    assert n1 > n0
    assert s1 < s0
    assert abs(n1 - 10.0) < abs(n1)


def test_stronger_meridional_dt_raises_u_crit_and_sth() -> None:
    from imagdyn.wind import find_subtropical_high_from_amc, omega_from_spin

    h = 360
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    omega = omega_from_spin(24.0 * 3600.0)
    T_weak = 20.0 + 10.0 * torch.cos(torch.deg2rad(lat))
    T_strong = 10.0 + 35.0 * torch.cos(torch.deg2rad(lat))
    kwargs = dict(omega=omega, radius_m=6371e3)
    n_w, _, m_w = find_subtropical_high_from_amc(lat, 0.0, T_weak, **kwargs)
    n_s, _, m_s = find_subtropical_high_from_amc(lat, 0.0, T_strong, **kwargs)
    assert m_s["u_crit_n_m_s"] > m_w["u_crit_n_m_s"]
    assert n_s >= n_w


def test_subpolar_from_temp_gradient_kink() -> None:
    from imagdyn.wind import find_subpolar_low_from_temp_gradient

    h = 360
    lat = 90.0 - (torch.arange(h, dtype=torch.float32) + 0.5) * (180.0 / h)
    # Localized front: steepest |dT/dφ| near 55°N
    T = 5.0 + 15.0 * torch.tanh((55.0 - lat) / 2.5)
    spl = find_subpolar_low_from_temp_gradient(lat, T, sth=30.0, toward_pole_sign=1.0)
    assert 50.0 <= spl <= 60.0


def test_smooth_lonlat_metric_wider_ew_at_high_lat() -> None:
    from imagdyn.temperature import latitude_grid, smooth_lonlat_metric

    h, w = 180, 360
    lat = latitude_grid(h, torch.device("cpu"))
    field = torch.zeros(h, w)
    i_eq = int(torch.argmin(lat.abs()).item())
    i_hi = int(torch.argmin((lat.abs() - 75.0).abs()).item())
    field[i_eq, 180] = 1.0
    field[i_hi, 180] = 1.0
    out = smooth_lonlat_metric(field, lat, 2.0)
    assert float(out[i_hi, 184]) > float(out[i_eq, 184])
