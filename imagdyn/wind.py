#!/usr/bin/env python3
"""
Wind / pressure field synthesizer.

Pipeline
--------
1. Sea-levelize surface temperature (undo elevation lapse) so pressure is SLP.
2. Planetary belts: 36 lon sectors; each builds cosine-segment ``build_belt_anomaly_1d``
   from sector-mean T; composite via periodic longitude interpolation. Meta: mean latitudes.

3. Local thermal pressure from SLP temperature anomaly.
4. Mild 2D diffusion of total SLP (lon wrap; polar safe truncate) + ∇ via convolution.
5. Coriolis + quadratic surface drag (−c_d |V| V); UV convolution for neighborhood wind direction.
6. Terrain block / divert / leeside; keep ``u·∇h`` as float32.

Outputs under graphs/wind/: two RGB PNGs per period —
  (1) U→R, V→G, pressure→B (8-bit); (2) terrain_dot→R,G (16-bit), speed→B (8-bit);
plus wind_meta.json / wind_stats.json.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import paths
from .io_gray import save_png_atomic, write_text_atomic
from .params import ENCODE, PLANET, TEMPERATURE, WIND, temperature_call_kwargs, wind_field_kwargs
from .temperature import (
    METRIC_COS_EPS,
    area_weighted_mean,
    area_weighted_mean_np,
    area_weights,
    conv1d_lat_replicate,
    conv1d_lon_grouped,
    get_device,
    latitude_grid,
    load_grayscale,
    lon_sigma_px,
    release_torch_memory,
    smooth_lonlat_metric,
)


def json_safe(obj: Any) -> Any:
    """Convert tensors / numpy scalars to JSON-serializable Python types."""
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return float(obj.item())
        return [json_safe(x) for x in obj.detach().cpu().tolist()]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def thermal_to_pressure_anomaly(
    t_anom: torch.Tensor,
    *,
    k_hpa_per_c: float = WIND.k_hpa_per_c,
) -> torch.Tensor:
    """Hot → low pressure anomaly (hPa-ish)."""
    return -k_hpa_per_c * t_anom


def thermal_latitude_weight(
    lat: torch.Tensor,
    *,
    sigma_deg: float = WIND.thermal_lat_sigma_deg,
    equator_frac: float = WIND.thermal_equator_frac,
) -> torch.Tensor:
    """
    Latitude Gaussian damper for local thermal → pressure.

    Near the equator the conversion is weakest (``equator_frac``);
    weight → 1 as |φ| ≫ ``sigma_deg``.
    """
    sig = max(float(sigma_deg), 0.5)
    eq = float(np.clip(equator_frac, 0.0, 1.0))
    g = torch.exp(-0.5 * (lat / sig) ** 2)
    return eq + (1.0 - eq) * (1.0 - g)


def temperature_to_sea_level(
    temperature_C: torch.Tensor,
    elev_asl_m: torch.Tensor,
    *,
    lapse_k_per_km: float = PLANET.lapse_k_per_km,
) -> torch.Tensor:
    """
    Undo elevation lapse so thermal → pressure maps to sea-level pressure (SLP).

    Surface maps are colder on mountains; adding ``lapse * z/1000`` recovers the
    temperature used to drive horizontal SLP gradients (not station pressure).
    Ocean / z≤0 cells are unchanged.
    """
    z_km = elev_asl_m.clamp(min=0.0) / 1000.0
    return temperature_C + float(lapse_k_per_km) * z_km


def omega_from_spin(t_spin_s: float) -> float:
    """Angular velocity (rad/s): one revolution in ``t_spin_s``."""
    return float(2.0 * np.pi / max(t_spin_s, 1e-6))


def planet_radius_km_from_grid(
    width: int,
    *,
    planet_radius_km: float | None = None,
) -> float:
    """
    Equirectangular: equator circumference spans ``width`` pixels.
    If ``planet_radius_km`` is given, use it; else default Earth-like 6371.
    (Width still defines Ω via spin period; R sets km/px for terrain gradients.)
    """
    if planet_radius_km is not None and planet_radius_km > 0:
        return float(planet_radius_km)
    return PLANET.radius_km


def km_per_px(height: int, width: int, radius_km: float) -> tuple[float, float]:
    km_per_deg = (np.pi * radius_km) / 180.0
    dy = (180.0 / height) * km_per_deg
    dx_eq = (360.0 / width) * km_per_deg
    return float(dy), float(dx_eq)


def _pad_lonlat(field: torch.Tensor, py: int, px: int) -> torch.Tensor:
    """Replicate latitude edges; wrap longitude."""
    if px > 0:
        field = torch.cat([field[:, -px:], field, field[:, :px]], dim=1)
    if py > 0:
        field = torch.cat(
            [field[:1].expand(py, -1), field, field[-1:].expand(py, -1)],
            dim=0,
        )
    return field


def _gauss_kernels_2d(
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separable Gaussian and its derivative (1D), for outer-product 2D kernels."""
    sig = max(float(sigma), 0.35)
    r = max(1, int(math.ceil(3.0 * sig)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sig) ** 2)
    g = g / g.sum().clamp_min(1e-12)
    dg = (-x / (sig * sig)) * g
    # Normalize so a unit ramp (slope 1 / px) maps to ≈1 (mom is typically negative)
    mom = (x * dg).sum()
    if float(mom.abs()) < 1e-12:
        dg = torch.zeros_like(dg)
        if r >= 1:
            dg[r - 1] = -0.5
            dg[r + 1] = 0.5
    else:
        dg = dg / mom
    return g, dg


def _gauss_and_deriv_rows(
    sigma_row: torch.Tensor,
    r: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row 1D Gaussian and derivative kernels, shape (H, K)."""
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    sig = sigma_row.clamp(min=0.35)[:, None]
    g = torch.exp(-0.5 * (x[None, :] / sig) ** 2)
    g = g / g.sum(dim=1, keepdim=True).clamp_min(1e-12)
    dg = (-x[None, :] / (sig * sig)) * g
    mom = (x[None, :] * dg).sum(dim=1, keepdim=True)
    fallback = torch.zeros_like(dg)
    if r >= 1:
        fallback[:, r - 1] = -0.5
        fallback[:, r + 1] = 0.5
    dg = torch.where(mom.abs() < 1e-12, fallback, dg / mom)
    return g, dg


def conv_grad_lonlat(
    field: torch.Tensor,
    *,
    dy_km: float,
    dx_eq_km: float,
    lat_deg: torch.Tensor,
    sigma_px: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Neighborhood-aware (∂/∂x_east, ∂/∂y_south) via separable Gaussian derivatives.

    EW kernel width is ``σ_eq / max(cos φ, ε)``; NS uses ``σ_eq``.
    After the pixel derivative, east–west distance is ``dx_eq * cos(φ)``.
    Lon wraps; lat replicates. Polar rows: NS gradient forced to 0.
    """
    h, w = field.shape
    lat1 = lat_deg.reshape(-1)[:h].to(device=field.device, dtype=field.dtype)
    sig_eq = max(float(sigma_px), 0.35)

    g_ns, dg_ns = _gauss_kernels_2d(sig_eq, device=field.device, dtype=field.dtype)
    sig_row = lon_sigma_px(lat1, sig_eq, eps=METRIC_COS_EPS)
    r_ew = max(1, min(max(1, w // 6), max(1, int(math.ceil(3.0 * float(sig_row.max().item()))))))
    g_ew, dg_ew = _gauss_and_deriv_rows(
        sig_row, r_ew, device=field.device, dtype=field.dtype
    )

    dfdx_px = conv1d_lat_replicate(
        conv1d_lon_grouped(field, dg_ew[:, None, :]), g_ns
    )
    dfdy_px = conv1d_lon_grouped(
        conv1d_lat_replicate(field, dg_ns), g_ew[:, None, :]
    )

    cos_lat = torch.cos(torch.deg2rad(lat1)).clamp(0.05, 1.0)[:, None]
    dx = dx_eq_km * cos_lat
    dfdx = dfdx_px / dx
    dfdy = dfdy_px / dy_km
    dfdy[0].zero_()
    dfdy[-1].zero_()
    return dfdx, dfdy


def grad_lonlat(
    field: torch.Tensor,
    *,
    dy_km: float,
    dx_eq_km: float,
    lat_deg: torch.Tensor,
    sigma_px: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (∂/∂x_east, ∂/∂y_south) with lon wrap; lat replicate. Units /km.

    Uses Gaussian-derivative convolution so each cell sees a local neighborhood
    (not a two-point stencil). Polar NS gradient is forced to 0.
    """
    return conv_grad_lonlat(
        field,
        dy_km=dy_km,
        dx_eq_km=dx_eq_km,
        lat_deg=lat_deg,
        sigma_px=sigma_px,
    )


def smooth_wind_uv(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    sigma_px: float = 2.0,
    lat_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian-convolve wind components (vector average → neighborhood direction).

    EW blur radius grows as ``1/cos(φ)``. Lon wraps; lat replicates.
    Speed is not renormalized — magnitude softens slightly.
    """
    if sigma_px <= 0:
        return u, v
    return (
        smooth_field_lonlat(u, sigma_px=sigma_px, lat_deg=lat_deg),
        smooth_field_lonlat(v, sigma_px=sigma_px, lat_deg=lat_deg),
    )


def smooth_field_lonlat(
    field: torch.Tensor,
    *,
    sigma_px: float,
    lat_deg: torch.Tensor,
) -> torch.Tensor:
    """Metric Gaussian diffusion: NS ``σ_eq``, EW ``σ_eq / cos(φ)``; lon wrap, lat replicate."""
    return smooth_lonlat_metric(field, lat_deg, sigma_px)


def diffuse_pressure_2d(
    pressure: torch.Tensor,
    lat_deg: torch.Tensor,
    *,
    sigma_px: float = 4.0,
    polar_cut_lat: float = 89.0,
    polar_fade_lat: float = 87.0,
) -> torch.Tensor:
    """
    Mild spatial diffusion of the total pressure field.

    - Lon edges are cyclic (map left↔right).
    - Lat uses replicate pad (poles do not wrap to each other).
    - Between ``polar_fade_lat`` and ``polar_cut_lat``, E–W anomalies are
      linearly damped toward the zonal mean (fully zonal at/above the cut)
      so the 89° ring has no hard pressure jump.
    """
    if sigma_px <= 0:
        return pressure
    out = smooth_field_lonlat(pressure, sigma_px=sigma_px, lat_deg=lat_deg)
    abs_lat = lat_deg.abs()
    span = max(float(polar_cut_lat - polar_fade_lat), 1e-3)
    # 0 equatorward of fade → 1 at/above cut
    w_zon = ((abs_lat - float(polar_fade_lat)) / span).clamp(0.0, 1.0)
    zm = out.mean(dim=1, keepdim=True)
    out = zm + (1.0 - w_zon[:, None]) * (out - zm)
    return out


def enforce_area_mean_zero(field: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    mean = area_weighted_mean(field, lat_deg)
    return field - mean


def gaussian_1d(lat: torch.Tensor, center: float, sigma: float) -> torch.Tensor:
    z = (lat - center) / max(sigma, 0.35)
    return torch.exp(-0.5 * z * z)


def smooth_lat_profile(values: torch.Tensor, sigma_deg: float, lat: torch.Tensor) -> torch.Tensor:
    """1D Gaussian smooth along latitude (replicate edges). sigma in degrees."""
    if sigma_deg <= 0 or values.numel() < 3:
        return values
    dlat = float((lat[1] - lat[0]).abs().clamp_min(1e-6).item())
    sig_px = max(float(sigma_deg) / dlat, 0.35)
    r = max(1, int(math.ceil(3.0 * sig_px)))
    x = torch.arange(-r, r + 1, device=values.device, dtype=values.dtype)
    g = torch.exp(-0.5 * (x / sig_px) ** 2)
    g = g / g.sum().clamp_min(1e-12)
    pad = torch.cat([values[:1].expand(r), values, values[-1:].expand(r)])
    return F.conv1d(pad[None, None], g[None, None])[0, 0]


def find_extremum_latitude(
    profile: torch.Tensor,
    lat: torch.Tensor,
    target: float,
    *,
    find_max: bool,
    window_deg: float = 12.0,
) -> float:
    """Latitude of local max/min of a 1D profile nearest ``target``."""
    lo, hi = target - window_deg, target + window_deg
    mask = (lat >= lo) & (lat <= hi)
    if not bool(mask.any()):
        return float(target)
    idx = torch.where(mask)[0]
    seg = profile[idx]
    j = int(torch.argmax(seg).item()) if find_max else int(torch.argmin(seg).item())
    return float(lat[idx[j]].item())


def land_belt_block_scale(
    belt_anom: torch.Tensor,
    land_s: torch.Tensor,
    *,
    land_belt_frac: float = 0.1,
    block_half_hpa: float = 1.5,
) -> torch.Tensor:
    """
    Dynamic land damping of planetary belt anomalies.

    Smooth gate on pressure anomaly: highs (positive) → strong land block;
    lows / near-zero → weak block so thermal still dominates less aggressively.

    ``scale = 1 - land · σ(anom/w) · (1 - land_belt_frac)``
    At full gate, land retains only ``land_belt_frac`` of the belt anomaly.
    """
    w = max(float(block_half_hpa), 1e-3)
    # σ(anom/w): →0 for strong lows, →1 for strong highs
    gate = torch.sigmoid(belt_anom - 0.5 / w)
    damp = gate * (1.0 - float(land_belt_frac))
    return 1.0 - land_s * damp


def build_belt_anomaly_1d(
    lat: torch.Tensor,
    *,
    teq_lat: float,
    sth_n: float,
    sth_s: float,
    spl_n: float,
    spl_s: float,
    pol_n: float,
    pol_s: float,
    a: float,
    b_n: float,
    b_s: float,
    c_n: float,
    c_s: float,
    blend_deg: float = 1.5,
) -> torch.Tensor:
    """
    Planetary pressure-belt anomaly via cosine segments between belt centers.

    At each design latitude the anomaly equals the belt peak; between adjacent
    centers it follows a half-period cosine
    ``½(Aᵢ+Aᵢ₊₁) + ½(Aᵢ−Aᵢ₊₁) cos(π t)``, ``t∈[0,1]``. Optional light
    latitude smooth (``blend_deg``) softens kink noise without Gaussian lobes.
    """
    nodes_lat = torch.tensor(
        [
            float(pol_n),
            float(spl_n),
            float(sth_n),
            float(teq_lat),
            float(sth_s),
            float(spl_s),
            float(pol_s),
        ],
        device=lat.device,
        dtype=lat.dtype,
    )
    nodes_amp = torch.tensor(
        [
            float(b_n),
            float(-(b_n + c_n)),
            float(a + c_n),
            float(-a),
            float(a + c_s),
            float(-(b_s + c_s)),
            float(b_s),
        ],
        device=lat.device,
        dtype=lat.dtype,
    )
    order = torch.argsort(nodes_lat)
    c = nodes_lat[order]
    amp = nodes_amp[order]
    # Merge nearly coincident nodes (keep amplitude of the last in the cluster)
    keep: list[int] = [0]
    for i in range(1, int(c.numel())):
        if float((c[i] - c[keep[-1]]).item()) < 0.25:
            keep[-1] = i
        else:
            keep.append(i)
    idx_k = torch.tensor(keep, device=lat.device, dtype=torch.long)
    c = c[idx_k]
    amp = amp[idx_k]
    n = int(c.numel())
    if n == 1:
        anom = amp[0].expand_as(lat).clone()
        return smooth_lat_profile(anom, blend_deg, lat)

    x = lat.reshape(-1)
    # Right edge of segment containing x (ascending centers)
    hi = torch.searchsorted(c.contiguous(), x.contiguous()).clamp(1, n - 1)
    lo = hi - 1
    c0, c1 = c[lo], c[hi]
    a0, a1 = amp[lo], amp[hi]
    span = (c1 - c0).clamp_min(1e-6)
    t = ((x - c0) / span).clamp(0.0, 1.0)
    anom = 0.5 * (a0 + a1) + 0.5 * (a0 - a1) * torch.cos(math.pi * t)
    anom = torch.where(x <= c[0], amp[0].expand_as(x), anom)
    anom = torch.where(x >= c[-1], amp[-1].expand_as(x), anom)
    anom = anom.reshape_as(lat)
    return smooth_lat_profile(anom, blend_deg, lat)


def upper_amc_u_profile(
    lat: torch.Tensor,
    phi0_deg: float,
    *,
    omega: float,
    radius_m: float,
    cos_floor: float = 0.05,
) -> torch.Tensor:
    """
    Ideal angular-momentum–conserving upper-level eastward wind (m/s), no drag.

    ``u = Ω R ( cos²(φ₀)/cos(φ) − cos(φ) )``.
    """
    phi = torch.deg2rad(lat)
    phi0 = math.radians(float(phi0_deg))
    cos_lat = torch.cos(phi).clamp(min=cos_floor)
    cos0 = max(math.cos(phi0), cos_floor)
    u = float(omega) * float(radius_m) * ((cos0 * cos0) / cos_lat - cos_lat)
    return u.clamp(min=0.0)


def rayleigh_kappa(phi_rad: float, *, kappa0: float, kappa_lat: float) -> float:
    """Latitude-dependent Rayleigh drag (1/s); larger in mid/high latitudes."""
    s = math.sin(float(phi_rad))
    return float(kappa0) * (1.0 + float(kappa_lat) * s * s)


def thermal_wind_u_crit_m_s(
    lat: torch.Tensor,
    T_bar: torch.Tensor,
    phi0_deg: float,
    *,
    toward_pole_sign: float,
    omega: float,
    radius_m: float,
    tropopause_h_m: float = 10_000.0,
    t_ref_k: float = 250.0,
    f_ref_lat_deg: float = 30.0,
    baro_outer_lat_abs: float = 50.0,
    thermal_wind_scale: float = 8882.0,
    u_crit_min: float = 20.0,
    u_crit_max: float = 70.0,
    smooth_deg: float = 1.5,
) -> tuple[float, dict[str, float]]:
    """
    Baroclinic breakdown threshold from thermal wind on zonal-mean T.

    ``u_crit = scale · (g H) / (f_ref T_ref) · max|∂T̄/∂y|`` in the φ₀→~50° band,
    clamped to ``[u_crit_min, u_crit_max]``. ``thermal_wind_scale`` maps typical
    surface |dT/dy| onto jet-core speeds (~30–40 m/s at equinox-like contrast).
    """
    T_s = smooth_lat_profile(T_bar, smooth_deg, lat)
    dlat = (lat[1:] - lat[:-1]).abs().clamp(min=1e-6)
    dT_dphi = (T_s[1:] - T_s[:-1]).abs() / dlat  # K / deg
    lat_m = 0.5 * (lat[1:] + lat[:-1])
    phi0 = float(phi0_deg)
    outer = float(baro_outer_lat_abs)
    if toward_pole_sign > 0:
        mask = (lat_m >= phi0) & (lat_m <= outer)
    else:
        mask = (lat_m <= phi0) & (lat_m >= -outer)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        dT_dphi_max = 0.0
    else:
        dT_dphi_max = float(dT_dphi[idx].max().item())
    g = 9.80665
    f_ref = abs(2.0 * float(omega) * math.sin(math.radians(float(f_ref_lat_deg))))
    dT_dy = dT_dphi_max * (math.pi / 180.0) / max(float(radius_m), 1.0)
    u_raw = (g * float(tropopause_h_m)) / max(f_ref * float(t_ref_k), 1e-12) * dT_dy
    u_crit = float(np.clip(float(thermal_wind_scale) * u_raw, u_crit_min, u_crit_max))
    return u_crit, {
        "dT_dphi_max_K_per_deg": dT_dphi_max,
        "dT_dy_K_per_m": dT_dy,
        "u_raw_m_s": float(u_raw),
        "u_crit_m_s": u_crit,
        "thermal_wind_scale": float(thermal_wind_scale),
        "tropopause_h_m": float(tropopause_h_m),
        "f_ref_s": float(f_ref),
    }


def integrate_amc_rayleigh_u(
    lat: torch.Tensor,
    phi0_deg: float,
    *,
    toward_pole_sign: float,
    omega: float,
    radius_m: float,
    kappa0: float,
    kappa_lat: float,
    hadley_v_m_s: float,
    max_lat_abs: float = 50.0,
    cos_floor: float = 0.05,
) -> torch.Tensor:
    """
    Path-integrate absolute angular momentum with Rayleigh drag from φ₀ toward a pole.

    ``M(φ₀)=Ω R² cos²φ₀`` (u=0); ``dM/dφ = −κ(φ)(M − Ω R² cos²φ)·R / v_merid``;
    ``u = M/(R cosφ) − Ω R cosφ``. Stateful walk — not a memoryless map on lat.
    """
    n = int(lat.numel())
    u_out = torch.zeros(n, device=lat.device, dtype=torch.float64)
    phi0 = float(phi0_deg)
    sign = 1.0 if toward_pole_sign > 0 else -1.0
    if sign > 0:
        idx = torch.where((lat >= phi0) & (lat <= float(max_lat_abs)))[0]
        order = idx[torch.argsort(lat[idx])]
    else:
        idx = torch.where((lat <= phi0) & (lat >= -float(max_lat_abs)))[0]
        order = idx[torch.argsort(lat[idx], descending=True)]
    if order.numel() == 0:
        return u_out.to(dtype=lat.dtype)

    R = float(radius_m)
    om = float(omega)
    v_m = max(float(hadley_v_m_s), 1e-6)
    phi0_r = math.radians(phi0)
    M = om * R * R * (math.cos(phi0_r) ** 2)
    prev_phi = phi0_r
    for step, i in enumerate(order.tolist()):
        phi = math.radians(float(lat[i].item()))
        dphi = abs(phi - prev_phi)
        if step == 0 and dphi < 1e-12:
            u_out[i] = 0.0
            prev_phi = phi
            continue
        if dphi < 1e-12:
            cosphi = max(math.cos(phi), cos_floor)
            u_out[i] = max(M / (R * cosphi) - om * R * cosphi, 0.0)
            continue
        k = rayleigh_kappa(0.5 * (phi + prev_phi), kappa0=kappa0, kappa_lat=kappa_lat)
        m_eq = om * R * R * (math.cos(phi) ** 2)
        coef = k * R / v_m
        # Semi-implicit in M for stability on coarse latitude grids
        M = (M + dphi * coef * m_eq) / (1.0 + dphi * coef)
        cosphi = max(math.cos(phi), cos_floor)
        u_out[i] = max(M / (R * cosphi) - om * R * cosphi, 0.0)
        prev_phi = phi
    return u_out.to(dtype=lat.dtype)


def _first_crossing_or_peak_latitude(
    lat: torch.Tensor,
    values: torch.Tensor,
    *,
    threshold: float,
    toward_pole_sign: float,
    start_lat: float,
    max_lat_abs: float,
) -> tuple[float, bool]:
    """First poleward crossing of threshold; else latitude of path max. Returns (lat, hit)."""
    if toward_pole_sign > 0:
        mask = (lat > start_lat) & (lat <= max_lat_abs)
        idx = torch.where(mask)[0]
        order = idx[torch.argsort(lat[idx])] if idx.numel() else idx
    else:
        mask = (lat < start_lat) & (lat >= -max_lat_abs)
        idx = torch.where(mask)[0]
        order = idx[torch.argsort(lat[idx], descending=True)] if idx.numel() else idx
    if order.numel() == 0:
        fallback = float(np.clip(start_lat + toward_pole_sign * 30.0, -max_lat_abs, max_lat_abs))
        return fallback, False
    prev_i = None
    for i in order.tolist():
        v = float(values[i].item())
        if v >= threshold:
            if prev_i is None:
                return float(lat[i].item()), True
            v0 = float(values[prev_i].item())
            v1 = v
            if v1 <= v0:
                return float(lat[i].item()), True
            t = float(np.clip((threshold - v0) / max(v1 - v0, 1e-9), 0.0, 1.0))
            return float(lat[prev_i].item() + t * (lat[i].item() - lat[prev_i].item())), True
        prev_i = i
    j = int(torch.argmax(values[order]).item())
    return float(lat[order[j]].item()), False


def find_subtropical_high_from_amc(
    lat: torch.Tensor,
    phi0_deg: float,
    T_bar: torch.Tensor,
    *,
    omega: float,
    radius_m: float,
    kappa0: float = 1.2e-6,
    kappa_lat: float = 8.0,
    hadley_v_m_s: float = 1.5,
    tropopause_h_m: float = 10_000.0,
    thermal_wind_scale: float = 8882.0,
    u_crit_min: float = 20.0,
    u_crit_max: float = 70.0,
    max_lat_abs: float = 50.0,
) -> tuple[float, float, dict[str, Any]]:
    """
    Subtropical highs from Rayleigh-drag AMC path integrals + thermal-wind u_crit.

    Integrates separately NH/SH from the thermal equator φ₀; STH = first latitude
    where path ``u(φ)`` reaches that hemisphere's ``u_crit(T̄)``.
    """
    phi0 = float(phi0_deg)
    u_n = integrate_amc_rayleigh_u(
        lat,
        phi0,
        toward_pole_sign=1.0,
        omega=omega,
        radius_m=radius_m,
        kappa0=kappa0,
        kappa_lat=kappa_lat,
        hadley_v_m_s=hadley_v_m_s,
        max_lat_abs=max_lat_abs,
    )
    u_s = integrate_amc_rayleigh_u(
        lat,
        phi0,
        toward_pole_sign=-1.0,
        omega=omega,
        radius_m=radius_m,
        kappa0=kappa0,
        kappa_lat=kappa_lat,
        hadley_v_m_s=hadley_v_m_s,
        max_lat_abs=max_lat_abs,
    )
    u_crit_n, tw_n = thermal_wind_u_crit_m_s(
        lat,
        T_bar,
        phi0,
        toward_pole_sign=1.0,
        omega=omega,
        radius_m=radius_m,
        tropopause_h_m=tropopause_h_m,
        thermal_wind_scale=thermal_wind_scale,
        u_crit_min=u_crit_min,
        u_crit_max=u_crit_max,
    )
    u_crit_s, tw_s = thermal_wind_u_crit_m_s(
        lat,
        T_bar,
        phi0,
        toward_pole_sign=-1.0,
        omega=omega,
        radius_m=radius_m,
        tropopause_h_m=tropopause_h_m,
        thermal_wind_scale=thermal_wind_scale,
        u_crit_min=u_crit_min,
        u_crit_max=u_crit_max,
    )
    sth_n, hit_n = _first_crossing_or_peak_latitude(
        lat,
        u_n,
        threshold=u_crit_n,
        toward_pole_sign=1.0,
        start_lat=phi0,
        max_lat_abs=float(max_lat_abs),
    )
    sth_s, hit_s = _first_crossing_or_peak_latitude(
        lat,
        u_s,
        threshold=u_crit_s,
        toward_pole_sign=-1.0,
        start_lat=phi0,
        max_lat_abs=float(max_lat_abs),
    )
    sth_n = float(max(sth_n, phi0 + 5.0))
    sth_s = float(min(sth_s, phi0 - 5.0))

    # Boundary checks at φ₀
    i0 = int(torch.argmin((lat - phi0).abs()).item())
    u0_n = float(u_n[i0].item()) if float(lat[i0].item()) >= phi0 - 1e-6 else 0.0
    u0_s = float(u_s[i0].item()) if float(lat[i0].item()) <= phi0 + 1e-6 else 0.0
    meta = {
        "phi0_deg": phi0,
        "omega_rad_s": float(omega),
        "radius_m": float(radius_m),
        "drag_kappa0": float(kappa0),
        "drag_kappa_lat": float(kappa_lat),
        "hadley_v_m_s": float(hadley_v_m_s),
        "u_crit_n_m_s": u_crit_n,
        "u_crit_s_m_s": u_crit_s,
        "u_crit_hit_n": bool(hit_n),
        "u_crit_hit_s": bool(hit_s),
        "u_path_max_n_m_s": float(u_n.max().item()),
        "u_path_max_s_m_s": float(u_s.max().item()),
        "u_at_phi0_n_m_s": u0_n,
        "u_at_phi0_s_m_s": u0_s,
        "thermal_wind_n": tw_n,
        "thermal_wind_s": tw_s,
        "method": "rayleigh_amc_path_integral + thermal_wind_u_crit",
    }
    return sth_n, sth_s, meta


def find_subpolar_low_from_temp_gradient(
    lat: torch.Tensor,
    T_bar: torch.Tensor,
    sth: float,
    *,
    toward_pole_sign: float,
    smooth_deg: float = 1.5,
    min_offset_deg: float = 5.0,
    pole_limit_deg: float = 80.0,
) -> float:
    """
    Subpolar low = latitude of maximum |dT̄/dφ| poleward of the subtropical high.
    """
    T_s = smooth_lat_profile(T_bar, smooth_deg, lat)
    # dT/dφ: lat decreases with row index in equirectangular maps
    dlat = (lat[1:] - lat[:-1]).clamp(min=1e-6)
    dT = (T_s[1:] - T_s[:-1]) / dlat
    lat_m = 0.5 * (lat[1:] + lat[:-1])
    abs_g = dT.abs()
    if toward_pole_sign > 0:
        lo = sth + min_offset_deg
        hi = pole_limit_deg
        mask = (lat_m >= lo) & (lat_m <= hi)
    else:
        lo = -pole_limit_deg
        hi = sth - min_offset_deg
        mask = (lat_m >= lo) & (lat_m <= hi)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return float(np.clip(sth + toward_pole_sign * 25.0, -pole_limit_deg, pole_limit_deg))
    j = int(torch.argmax(abs_g[idx]).item())
    return float(lat_m[idx[j]].item())


def belt_placement_from_T1d(
    lat: torch.Tensor,
    T1d: torch.Tensor,
    *,
    omega: float,
    radius_m: float,
    k_hpa_per_c: float,
    belt_k_frac: float,
    belt_amp_max_hpa: float,
    secondary_frac: float,
    kappa0: float,
    kappa_lat: float,
    hadley_v_m_s: float,
    tropopause_h_m: float,
    thermal_wind_scale: float,
    u_crit_min: float,
    u_crit_max: float,
    amc_max_lat_abs: float,
    belt_blend_deg: float,
) -> tuple[dict[str, float], torch.Tensor, dict[str, Any]]:
    """
    One meridian: belt centers + amplitudes, 1D cosine-belt anomaly, and AMC meta.
    """
    teq_i = int(torch.argmax(T1d).item())
    teq_lat = float(lat[teq_i].item())
    T_ref = float(area_weighted_mean(T1d, lat).item())
    k_belt = float(k_hpa_per_c) * float(belt_k_frac)
    a = float(np.clip(k_belt * (float(T1d[teq_i].item()) - T_ref), 0.8, belt_amp_max_hpa))
    h = int(T1d.shape[0])
    ncap = max(1, h // 20)
    t_n = float(area_weighted_mean(T1d[:ncap], lat[:ncap]).item())
    t_s = float(area_weighted_mean(T1d[-ncap:], lat[-ncap:]).item())
    b_n = float(np.clip(k_belt * (T_ref - t_n), 0.8, belt_amp_max_hpa))
    b_s = float(np.clip(k_belt * (T_ref - t_s), 0.8, belt_amp_max_hpa))
    c_n = float(secondary_frac) * (a + b_n)
    c_s = float(secondary_frac) * (a + b_s)
    sth_n, sth_s, amc_meta = find_subtropical_high_from_amc(
        lat,
        teq_lat,
        T1d,
        omega=omega,
        radius_m=radius_m,
        kappa0=kappa0,
        kappa_lat=kappa_lat,
        hadley_v_m_s=hadley_v_m_s,
        tropopause_h_m=tropopause_h_m,
        thermal_wind_scale=thermal_wind_scale,
        u_crit_min=u_crit_min,
        u_crit_max=u_crit_max,
        max_lat_abs=amc_max_lat_abs,
    )
    spl_n = find_subpolar_low_from_temp_gradient(
        lat, T1d, sth_n, toward_pole_sign=1.0
    )
    spl_s = find_subpolar_low_from_temp_gradient(
        lat, T1d, sth_s, toward_pole_sign=-1.0
    )
    pol_n, pol_s = 89.0, -89.0
    anom = build_belt_anomaly_1d(
        lat,
        teq_lat=teq_lat,
        sth_n=sth_n,
        sth_s=sth_s,
        spl_n=spl_n,
        spl_s=spl_s,
        pol_n=pol_n,
        pol_s=pol_s,
        a=a,
        b_n=b_n,
        b_s=b_s,
        c_n=c_n,
        c_s=c_s,
        blend_deg=belt_blend_deg,
    )
    centers = {
        "thermal_equator": teq_lat,
        "subtropical_high_n": float(sth_n),
        "subtropical_high_s": float(sth_s),
        "subpolar_low_n": float(spl_n),
        "subpolar_low_s": float(spl_s),
        "polar_high_n": pol_n,
        "polar_high_s": pol_s,
        "a": a,
        "b_n": b_n,
        "b_s": b_s,
        "c_n": c_n,
        "c_s": c_s,
    }
    return centers, anom, amc_meta


def _interp_lon_periodic(anoms_h_n: np.ndarray, src_j: np.ndarray, width: int) -> np.ndarray:
    """Interpolate (H, N) samples at column indices src_j → (H, width) with lon wrap."""
    h, n = anoms_h_n.shape
    if n == 1:
        return np.repeat(anoms_h_n, width, axis=1).astype(np.float32)
    src = src_j.astype(np.float64)
    src_ext = np.concatenate([src - width, src, src + width])
    val_ext = np.concatenate([anoms_h_n, anoms_h_n, anoms_h_n], axis=1)
    x = np.arange(width, dtype=np.float64)
    out = np.empty((h, width), dtype=np.float32)
    for i in range(h):
        out[i] = np.interp(x, src_ext, val_ext[i]).astype(np.float32)
    return out


def _mean_nested_numbers(objs: list[Any]) -> Any:
    """Element-wise mean of parallel nested dict/list trees of numbers."""
    if not objs:
        return None
    sample = objs[0]
    if isinstance(sample, dict):
        keys = sample.keys()
        return {k: _mean_nested_numbers([o[k] for o in objs if k in o]) for k in keys}
    if isinstance(sample, (list, tuple)):
        n = len(sample)
        return [_mean_nested_numbers([o[i] for o in objs]) for i in range(n)]
    if isinstance(sample, bool):
        return bool(sum(1 for o in objs if o) * 2 >= len(objs))
    if isinstance(sample, (int, float, np.floating, np.integer)):
        return float(np.mean([float(o) for o in objs]))
    return sample


def composite_belts_from_meridians(
    T: torch.Tensor,
    lat: torch.Tensor,
    *,
    omega: float,
    radius_m: float,
    n_sectors: int = 36,
    k_hpa_per_c: float,
    belt_k_frac: float,
    belt_amp_max_hpa: float,
    secondary_frac: float,
    kappa0: float,
    kappa_lat: float,
    hadley_v_m_s: float,
    tropopause_h_m: float,
    thermal_wind_scale: float,
    u_crit_min: float,
    u_crit_max: float,
    amc_max_lat_abs: float,
    belt_blend_deg: float,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """
    Split longitude into ``n_sectors`` equal bands. Each band: sector-mean T →
    cosine ``build_belt_anomaly_1d`` at the sector midpoint; composite with
    periodic longitude interpolation. Returns belt field, mean centers, mean AMC meta.
    """
    h, w = T.shape
    n = max(1, min(int(n_sectors), w))
    edges = np.linspace(0, w, n + 1)
    anoms: list[np.ndarray] = []
    mid_j: list[float] = []
    rows: list[dict[str, float]] = []
    amc_rows: list[dict[str, Any]] = []
    for s in range(n):
        j0 = int(round(edges[s]))
        j1 = int(round(edges[s + 1]))
        if j1 <= j0:
            j1 = min(j0 + 1, w)
        T1d = T[:, j0:j1].mean(dim=1)
        centers, anom, amc_meta = belt_placement_from_T1d(
            lat,
            T1d,
            omega=omega,
            radius_m=radius_m,
            k_hpa_per_c=k_hpa_per_c,
            belt_k_frac=belt_k_frac,
            belt_amp_max_hpa=belt_amp_max_hpa,
            secondary_frac=secondary_frac,
            kappa0=kappa0,
            kappa_lat=kappa_lat,
            hadley_v_m_s=hadley_v_m_s,
            tropopause_h_m=tropopause_h_m,
            thermal_wind_scale=thermal_wind_scale,
            u_crit_min=u_crit_min,
            u_crit_max=u_crit_max,
            amc_max_lat_abs=amc_max_lat_abs,
            belt_blend_deg=belt_blend_deg,
        )
        rows.append(centers)
        amc_rows.append(amc_meta)
        anoms.append(anom.detach().cpu().numpy().astype(np.float32))
        mid_j.append(0.5 * (j0 + j1 - 1))
    stacked = np.stack(anoms, axis=1)
    src = np.asarray(mid_j, dtype=np.float64)
    order = np.argsort(src)
    belt_np = _interp_lon_periodic(stacked[:, order], src[order], w)
    belt = torch.from_numpy(belt_np).to(device=T.device, dtype=T.dtype)

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    mean_centers = {
        "thermal_equator": _mean("thermal_equator"),
        "subtropical_high": [_mean("subtropical_high_n"), _mean("subtropical_high_s")],
        "subpolar_low": [_mean("subpolar_low_n"), _mean("subpolar_low_s")],
        "polar_high": [_mean("polar_high_n"), _mean("polar_high_s")],
        "a": _mean("a"),
        "b_n": _mean("b_n"),
        "b_s": _mean("b_s"),
        "c_n": _mean("c_n"),
        "c_s": _mean("c_s"),
        "n_sectors": n,
    }
    return belt, mean_centers, _mean_nested_numbers(amc_rows)

@dataclass
class WindResult:
    pressure: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    speed: torch.Tensor
    terrain_dot: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


class WindField:
    """Synthesize pressure + wind from an absolute temperature field."""

    def __init__(
        self,
        *,
        p0_hpa: float = WIND.p0_hpa,
        k_hpa_per_c: float = WIND.k_hpa_per_c,
        # Planetary belt amplitudes stay modest so local thermal p can show.
        belt_k_frac: float = WIND.belt_k_frac,
        belt_amp_max_hpa: float = WIND.belt_amp_max_hpa,
        secondary_frac: float = WIND.secondary_frac,
        # Light latitude smooth after cosine belt segments (no Gaussian σ).
        belt_blend_deg: float = WIND.belt_blend_deg,
        # Rayleigh-drag AMC path integral → subtropical high (no α, no fixed 40 m/s).
        drag_kappa0: float = WIND.drag_kappa0,
        drag_kappa_lat: float = WIND.drag_kappa_lat,
        hadley_v_m_s: float = WIND.hadley_v_m_s,
        tropopause_h_m: float = WIND.tropopause_h_m,
        thermal_wind_scale: float = WIND.thermal_wind_scale,
        u_crit_min_m_s: float = WIND.u_crit_min_m_s,
        u_crit_max_m_s: float = WIND.u_crit_max_m_s,
        amc_max_lat_abs: float = WIND.amc_max_lat_abs,
        belt_lon_sectors: int = WIND.belt_lon_sectors,
        t_spin_s: float = PLANET.t_spin_hours * 3600.0,
        planet_radius_km: float = PLANET.radius_km,
        drag: float = WIND.drag,
        # Quadratic surface drag coeff c_d (1/m): F_drag = −c_d |V| V.
        # Defaults ≈ old linear μ / 10 m/s so mid-lat speeds stay similar.
        friction_ocean: float = WIND.friction_ocean,
        friction_land: float = WIND.friction_land,
        air_density: float = WIND.air_density,
        speed_cap: float = WIND.speed_cap,
        # Terrain: ∇h in m/km — require real slopes; mild speed tweaks only.
        upslope_slow: float = WIND.upslope_slow,
        # Lee accelerates downslopes.
        downslope_boost: float = WIND.downslope_boost,
        block_speed: float = WIND.block_speed,
        divert_frac: float = WIND.divert_frac,
        slope_block_m_per_km: float = WIND.slope_block_m_per_km,
        force_scale: float = WIND.force_scale,
        polar_ocean_lat: float = WIND.polar_ocean_lat,
        polar_fade_lat: float = WIND.polar_fade_lat,
        coast_sigma_px: float = WIND.coast_sigma_px,
        # Match temperature.py default: undo land lapse before SLP pressure
        lapse_k_per_km: float = PLANET.lapse_k_per_km,
        # Convolution widths (px) for ∇p and neighborhood wind direction
        grad_sigma_px: float = WIND.grad_sigma_px,
        wind_smooth_sigma_px: float = WIND.wind_smooth_sigma_px,
        # Land keeps this fraction of belt anomaly at full (high-pressure) block.
        land_belt_frac: float = WIND.land_belt_frac,
        # Sigmoid half-width (hPa): gate = σ(belt_anom / block_half_hpa).
        land_block_half_hpa: float = WIND.land_block_half_hpa,
        # Local thermal→p weaker near equator (Gaussian notch).
        thermal_lat_sigma_deg: float = WIND.thermal_lat_sigma_deg,
        thermal_equator_frac: float = WIND.thermal_equator_frac,
        # Mild 2D diffusion of total SLP after belts + thermal.
        pressure_smooth_sigma_px: float = WIND.pressure_smooth_sigma_px,
    ) -> None:
        self.p0_hpa = float(p0_hpa)
        self.k_hpa_per_c = float(k_hpa_per_c)
        self.belt_k_frac = float(belt_k_frac)
        self.belt_amp_max_hpa = float(belt_amp_max_hpa)
        self.secondary_frac = float(secondary_frac)
        self.belt_blend_deg = float(belt_blend_deg)
        self.drag_kappa0 = float(drag_kappa0)
        self.drag_kappa_lat = float(drag_kappa_lat)
        self.hadley_v_m_s = float(hadley_v_m_s)
        self.tropopause_h_m = float(tropopause_h_m)
        self.thermal_wind_scale = float(thermal_wind_scale)
        self.u_crit_min_m_s = float(u_crit_min_m_s)
        self.u_crit_max_m_s = float(u_crit_max_m_s)
        self.amc_max_lat_abs = float(amc_max_lat_abs)
        self.belt_lon_sectors = int(belt_lon_sectors)
        self.t_spin_s = float(t_spin_s)
        self.planet_radius_km = float(planet_radius_km)
        self.drag = float(drag)
        self.friction_ocean = float(friction_ocean)
        self.friction_land = float(friction_land)
        self.air_density = float(air_density)
        self.speed_cap = float(speed_cap)
        self.upslope_slow = float(upslope_slow)
        self.downslope_boost = float(downslope_boost)
        self.block_speed = float(block_speed)
        self.divert_frac = float(divert_frac)
        self.slope_block_m_per_km = float(slope_block_m_per_km)
        self.force_scale = float(force_scale)
        self.polar_ocean_lat = float(polar_ocean_lat)
        self.polar_fade_lat = float(polar_fade_lat)
        self.coast_sigma_px = float(coast_sigma_px)
        self.lapse_k_per_km = float(lapse_k_per_km)
        self.grad_sigma_px = float(grad_sigma_px)
        self.wind_smooth_sigma_px = float(wind_smooth_sigma_px)
        self.land_belt_frac = float(land_belt_frac)
        self.land_block_half_hpa = float(land_block_half_hpa)
        self.thermal_lat_sigma_deg = float(thermal_lat_sigma_deg)
        self.thermal_equator_frac = float(thermal_equator_frac)
        self.pressure_smooth_sigma_px = float(pressure_smooth_sigma_px)

    def compute(
        self,
        temperature_C: torch.Tensor | np.ndarray,
        land: torch.Tensor | np.ndarray,
        elev_m: torch.Tensor | np.ndarray,
        *,
        device: torch.device | None = None,
        prepare_surface: bool = True,
    ) -> WindResult:
        if device is None:
            device = (
                temperature_C.device
                if isinstance(temperature_C, torch.Tensor)
                else torch.device("cpu")
            )
        T_sfc = torch.as_tensor(temperature_C, device=device, dtype=torch.float32)
        if isinstance(land, torch.Tensor):
            land_np = land.detach().cpu().numpy().astype(bool)
        else:
            land_np = np.asarray(land, dtype=bool)
        if isinstance(elev_m, torch.Tensor):
            elev_np = elev_m.detach().cpu().numpy().astype(np.float32)
        else:
            elev_np = np.asarray(elev_m, dtype=np.float32)

        if prepare_surface:
            land_np, elev_out, elev_terr, land_soft = prepare_surface_for_wind(
                land_np,
                elev_np,
                polar_ocean_lat=self.polar_ocean_lat,
                polar_fade_lat=self.polar_fade_lat,
                coast_sigma_px=self.coast_sigma_px,
            )
        else:
            elev_out = elev_np
            elev_terr = np.where(land_np, np.maximum(elev_np, 0.0), 0.0).astype(np.float32)
            land_soft = land_np.astype(np.float32)

        land_s = torch.as_tensor(land_soft, device=device, dtype=torch.float32)
        elev = torch.as_tensor(elev_terr, device=device, dtype=torch.float32)
        # ASL height for lapse undo (ocean / bathymetry → 0; matches temperature lapse)
        elev_asl = torch.as_tensor(
            np.where(land_np, np.maximum(elev_out, 0.0), 0.0).astype(np.float32),
            device=device,
            dtype=torch.float32,
        )
        # SLP driver: sea-levelize T before belts / thermal pressure anomalies
        T = temperature_to_sea_level(
            T_sfc, elev_asl, lapse_k_per_km=self.lapse_k_per_km
        )
        h, w = T.shape
        lat = latitude_grid(h, device)
        omega = omega_from_spin(self.t_spin_s)
        R = planet_radius_km_from_grid(w, planet_radius_km=self.planet_radius_km)
        dy_km, dx_eq = km_per_px(h, w, R)

        # --- 1. planetary belts: 36 lon sectors, periodic lon composite ---
        T_bar = T.mean(dim=1)
        radius_m = R * 1000.0
        belt_anom, belt_mean, amc_meta = composite_belts_from_meridians(
            T,
            lat,
            omega=omega,
            radius_m=radius_m,
            n_sectors=self.belt_lon_sectors,
            k_hpa_per_c=self.k_hpa_per_c,
            belt_k_frac=self.belt_k_frac,
            belt_amp_max_hpa=self.belt_amp_max_hpa,
            secondary_frac=self.secondary_frac,
            kappa0=self.drag_kappa0,
            kappa_lat=self.drag_kappa_lat,
            hadley_v_m_s=self.hadley_v_m_s,
            tropopause_h_m=self.tropopause_h_m,
            thermal_wind_scale=self.thermal_wind_scale,
            u_crit_min=self.u_crit_min_m_s,
            u_crit_max=self.u_crit_max_m_s,
            amc_max_lat_abs=self.amc_max_lat_abs,
            belt_blend_deg=self.belt_blend_deg,
        )
        teq_lat = float(belt_mean["thermal_equator"])
        sth_n = float(belt_mean["subtropical_high"][0])
        sth_s = float(belt_mean["subtropical_high"][1])
        spl_n = float(belt_mean["subpolar_low"][0])
        spl_s = float(belt_mean["subpolar_low"][1])
        pol_n = float(belt_mean["polar_high"][0])
        pol_s = float(belt_mean["polar_high"][1])
        a = float(belt_mean["a"])
        b_n = float(belt_mean["b_n"])
        b_s = float(belt_mean["b_s"])
        c_n = float(belt_mean["c_n"])
        c_s = float(belt_mean["c_s"])

        # Land damps planetary belts (stronger on positive anomalies / highs).
        belt_scale = land_belt_block_scale(
            belt_anom,
            land_s,
            land_belt_frac=self.land_belt_frac,
            block_half_hpa=self.land_block_half_hpa,
        )
        belt_anom = belt_anom * belt_scale
        belt_anom = enforce_area_mean_zero(belt_anom, lat)
        p_base = belt_anom + self.p0_hpa
        # Zonal-mean belt profile after mean-zero (ocean-weighted reference)
        p_zonal_base = p_base.mean(dim=1)
        # --- 2. local thermal (latitude-damped near equator) ---
        T_anom = T - T_bar[:, None]
        p_therm = thermal_to_pressure_anomaly(T_anom, k_hpa_per_c=self.k_hpa_per_c)
        w_lat = thermal_latitude_weight(
            lat,
            sigma_deg=self.thermal_lat_sigma_deg,
            equator_frac=self.thermal_equator_frac,
        )
        p_therm = p_therm * w_lat[:, None]
        p_therm = enforce_area_mean_zero(p_therm, lat)

        # --- 3. unify + mild 2D diffusion, then gradient ---
        p = p_base + p_therm
        p = diffuse_pressure_2d(
            p,
            lat,
            sigma_px=self.pressure_smooth_sigma_px,
            polar_cut_lat=self.polar_ocean_lat,
            polar_fade_lat=self.polar_fade_lat,
        )
        # grad_lonlat: (∂/∂x_east, ∂/∂y_south). Physics frame: +X east, +Y north.
        dpx_e, dpy_s = grad_lonlat(
            p,
            dy_km=dy_km,
            dx_eq_km=dx_eq,
            lat_deg=lat,
            sigma_px=self.grad_sigma_px,
        )
        dpx = dpx_e
        dpy = -dpy_s  # flip Y → north-positive
        # hPa/km → Pa/m (*0.1), then acceleration −∇p/ρ
        rho = max(self.air_density, 1e-3)
        Fx = -(dpx * 0.1 / rho) * self.force_scale
        Fy = -(dpy * 0.1 / rho) * self.force_scale

        # --- 4. Coriolis + quadratic surface drag (u east, v north) ---
        # c_d |V| u − f v = Fx ;  f u + c_d |V| v = Fy
        # ⇒ s² = |V|² solves c_d² s⁴ + f² s² − |F|² = 0 (stable rationalized root).
        phi = torch.deg2rad(lat)[:, None]
        f_real = 2.0 * omega * torch.sin(phi)
        f_inertia = 2.0 * omega * math.sin(math.radians(5.0))
        f = torch.where(f_real.abs() < f_inertia, torch.sign(phi) * f_inertia, f_real)
        cd = self.friction_ocean * (1.0 - land_s) + self.friction_land * land_s
        f2 = f * f
        F2 = Fx * Fx + Fy * Fy
        disc = f2 * f2 + (4.0 * cd * cd * F2)
        # speed² = 2|F|² / (f² + √(f⁴ + 4 c_d² |F|²))  — limits: geo / pure quadratic
        speed2 = (2.0 * F2) / (f2 + torch.sqrt(disc.clamp_min(0.0))).clamp_min(1e-14)
        speed_bal = torch.sqrt(speed2.clamp_min(0.0))
        mu = (cd * speed_bal).clamp_min(1e-14)
        eq_damp = 8.0e-5 * torch.exp(-0.5 * (lat / 3.0)**2)[:, None] 
        mu = mu + eq_damp
        denom = (mu * mu + f2).clamp_min(1e-14)
        u = (mu * Fx + f * Fy) / denom
        v = (mu * Fy - f * Fx) / denom

        # Soft speed cap (outliers from unresolved cells)
        spd0 = torch.sqrt(u * u + v * v).clamp_min(1e-6)
        if self.speed_cap > 0:
            scale = (self.speed_cap / spd0).clamp(max=1.0)
            u = u * scale
            v = v * scale

        # --- 5. terrain (elev_terrain: water=0, soft coast; cliffs inland stay) ---
        dhdx_e, dhdy_s = grad_lonlat(
            elev,
            dy_km=dy_km,
            dx_eq_km=dx_eq,
            lat_deg=lat,
            sigma_px=self.grad_sigma_px,
        )
        dhdx = dhdx_e
        dhdy = -dhdy_s  # north-positive slope
        slope = torch.sqrt(dhdx * dhdx + dhdy * dhdy).clamp_min(1e-6)
        nx, ny = dhdx / slope, dhdy / slope
        speed0 = torch.sqrt(u * u + v * v).clamp_min(1e-6)
        terrain_dot_phys = u * dhdx + v * dhdy
        up_cos = (terrain_dot_phys / (speed0 * slope)).clamp(-1.0, 1.0)
        up_w = up_cos.clamp(min=0.0)
        slope_w = (slope / (slope + self.slope_block_m_per_km)).clamp(0.0, 1.0)
        slow = 1.0 - self.upslope_slow * up_w * slope_w
        u = u * slow
        v = v * slow
        blocked = (up_w > 0.35) & (speed0 > self.block_speed) & (slope > self.slope_block_m_per_km)
        tx, ty = -ny, nx
        align = torch.where((u * tx + v * ty) < 0, -torch.ones_like(u), torch.ones_like(u))
        tux, tuy = tx * align, ty * align
        spd = torch.sqrt(u * u + v * v)
        div_w = self.divert_frac * slope_w
        u = torch.where(blocked, u * (1.0 - div_w) + tux * spd * div_w, u)
        v = torch.where(blocked, v * (1.0 - div_w) + tuy * spd * div_w, v)
        down_w = (-up_cos).clamp(min=0.0)
        # Mild lee boost; slope_w already softens weak coastal ramps
        boost = 1.0 + self.downslope_boost * down_w * slope_w
        u = u * boost
        v = v * boost

        # Re-cap: terrain must not create super-critical jets (esp. false coast cliffs)
        spd_t = torch.sqrt(u * u + v * v).clamp_min(1e-6)
        max_spd = speed0 * (1.0 + self.downslope_boost * 1.35)
        if self.speed_cap > 0:
            max_spd = torch.minimum(max_spd, torch.full_like(max_spd, self.speed_cap))
        tscale = (max_spd / spd_t).clamp(max=1.0)
        u = u * tscale
        v = v * tscale

        # Neighborhood wind direction: convolve UV (vector average of nearby cells)
        u, v = smooth_wind_uv(u, v, sigma_px=self.wind_smooth_sigma_px, lat_deg=lat)

        # Image frame: +V south (row direction). Flip V after physics.
        v = -v
        terrain_dot = u * dhdx_e + v * dhdy_s
        speed = torch.sqrt(u * u + v * v)

        # Measured belt centers from zonal-mean SLP (matches what the map shows)
        p_zonal = p.mean(dim=1)
        teq_obs = find_extremum_latitude(p_zonal, lat, teq_lat, find_max=False, window_deg=15.0)
        sth_n_obs = find_extremum_latitude(p_zonal, lat, sth_n, find_max=True, window_deg=12.0)
        sth_s_obs = find_extremum_latitude(p_zonal, lat, sth_s, find_max=True, window_deg=12.0)
        spl_n_obs = find_extremum_latitude(p_zonal, lat, spl_n, find_max=False, window_deg=12.0)
        spl_s_obs = find_extremum_latitude(p_zonal, lat, spl_s, find_max=False, window_deg=12.0)
        pol_n_obs = find_extremum_latitude(p_zonal, lat, pol_n, find_max=True, window_deg=12.0)
        pol_s_obs = find_extremum_latitude(p_zonal, lat, pol_s, find_max=True, window_deg=12.0)

        def _p_at(lat0: float) -> float:
            i = int(torch.argmin((lat - lat0).abs()).item())
            return float(p_zonal[i].item())

        meta = {
            # Observed centers from zonal-mean SLP (map)
            "thermal_equator_lat_deg": teq_obs,
            "subtropical_high_lat_deg": [sth_n_obs, sth_s_obs],
            "subpolar_low_lat_deg": [spl_n_obs, spl_s_obs],
            "polar_high_lat_deg": [pol_n_obs, pol_s_obs],
            # Design = mean over meridian samples only
            "belt_centers_design_deg": {
                "thermal_equator": teq_lat,
                "subtropical_high": [sth_n, sth_s],
                "subpolar_low": [spl_n, spl_s],
                "polar_high": [pol_n, pol_s],
                "n_lon_sectors": int(belt_mean["n_sectors"]),
            },
            "belt_width_deg": {
                "note": "cosine segments between belt centers; light lat blend only",
                "blend": self.belt_blend_deg,
                "profile": "cosine_halfwave_between_centers",
            },
            "belt_center_method": {
                "note": (
                    f"{int(belt_mean['n_sectors'])} lon sectors: each cosine "
                    "build_belt_anomaly_1d from sector-mean T (Rayleigh-drag AMC + "
                    "thermal-wind u_crit; subpolar max|dT/dφ|; polar ±89°); "
                    "periodic lon interpolate; meta = mean latitudes"
                ),
                "subtropical_high": (
                    "mean of sector AMC path integrals (Rayleigh drag + thermal-wind u_crit)"
                ),
                "subpolar_low": "mean of sector max |dT_bar/dphi| poleward of subtropical high",
                "polar_high": "fixed ±89°",
            },
            "upper_amc": amc_meta,            "amplitudes_hpa": {
                "note": "design = mean sample peaks; equator/subtropical/… = zonal-mean total SLP at observed centers",
                "a": a,
                "b_north": b_n,
                "b_south": b_s,
                "c_north": c_n,
                "c_south": c_s,
                "p0": self.p0_hpa,
                "design": {
                    "equator": self.p0_hpa - a,
                    "subtropical_n": self.p0_hpa + a + c_n,
                    "subtropical_s": self.p0_hpa + a + c_s,
                    "subpolar_n": self.p0_hpa - b_n - c_n,
                    "subpolar_s": self.p0_hpa - b_s - c_s,
                    "polar_n": self.p0_hpa + b_n,
                    "polar_s": self.p0_hpa + b_s,
                },
                "equator": _p_at(teq_obs),
                "subtropical": _p_at(sth_n_obs),
                "subtropical_s": _p_at(sth_s_obs),
                "subpolar": _p_at(spl_n_obs),
                "subpolar_s": _p_at(spl_s_obs),
                "polar": _p_at(pol_n_obs),
                "polar_s": _p_at(pol_s_obs),
                "zonal_base_at_design": {
                    "equator": float(p_zonal_base[int(torch.argmin((lat - teq_lat).abs()).item())].item()),
                    "subtropical_n": float(
                        p_zonal_base[int(torch.argmin((lat - sth_n).abs()).item())].item()
                    ),
                    "subtropical_s": float(
                        p_zonal_base[int(torch.argmin((lat - sth_s).abs()).item())].item()
                    ),
                },
            },
            "t_spin_s": self.t_spin_s,
            "omega_rad_s": omega,
            "planet_radius_km": R,
            "drag": self.drag,
            "friction_kind": "quadratic (−c_d |V| V); friction_* are c_d in 1/m",
            "friction_ocean": self.friction_ocean,
            "friction_land": self.friction_land,
            "grid": {"height": int(h), "width": int(w)},
            "axes": {
                "physics": "+X east, +Y north",
                "stored_uv": "+U east, +V south (image row direction)",
            },
            "pressure_mean_hpa": float(area_weighted_mean(p, lat).item()),
            "speed_mean": float(area_weighted_mean(speed, lat).item()),
            "speed_max": float(speed.max().item()),
            "terrain_dot_mean": float(area_weighted_mean(terrain_dot, lat).item()),
            "pressure_kind": "sea_level (from sea-levelized temperature)",
            "lapse_k_per_km": self.lapse_k_per_km,
            "grad_sigma_px": self.grad_sigma_px,
            "wind_smooth_sigma_px": self.wind_smooth_sigma_px,
            "land_belt_frac": self.land_belt_frac,
            "land_block_half_hpa": self.land_block_half_hpa,
            "land_block": "gate=sigmoid(belt_anom/half); highs blocked more than lows",
            "thermal_lat_sigma_deg": self.thermal_lat_sigma_deg,
            "thermal_equator_frac": self.thermal_equator_frac,
            "pressure_smooth_sigma_px": self.pressure_smooth_sigma_px,
            "surface_prep": {
                "polar_ocean_lat": self.polar_ocean_lat,
                "polar_fade_lat": self.polar_fade_lat,
                "coast_sigma_px": self.coast_sigma_px,
                "elev_terrain": "water=0; soft land·height coast; polar fade→0",
            },
        }
        return WindResult(
            pressure=p,
            u=u,
            v=v,
            speed=speed,
            terrain_dot=terrain_dot,
            meta=meta,
        )


def value_to_gray(v: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    g = (v - vmin) / max(vmax - vmin, 1e-6) * 255.0
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def save_gray_png(path: Path, gray: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_png_atomic(path, Image.fromarray(gray, mode="L"))


def _save_png_atomic(path: Path, img: Image.Image, *, attempts: int = 8) -> None:
    save_png_atomic(path, img, attempts=attempts)


def _finite_range(arr: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(a, lo_pct))
    vmax = float(np.percentile(a, hi_pct))
    if vmax - vmin < 1e-6:
        vmin = float(a.min())
        vmax = float(a.max() + 1e-6)
    return vmin, vmax


def encode_u8(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    t = (values.astype(np.float64) - vmin) / max(vmax - vmin, 1e-12)
    return np.clip(np.rint(t * 255.0), 0, 255).astype(np.uint8)


def decode_u8(gray: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (vmin + (gray.astype(np.float32) / 255.0) * (vmax - vmin)).astype(np.float32)


def encode_u16_pair(values: np.ndarray, vmin: float, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Map floats → 16-bit, split into high/low uint8 planes (≃65536 levels)."""
    t = (values.astype(np.float64) - vmin) / max(vmax - vmin, 1e-12)
    q = np.clip(np.rint(t * 65535.0), 0, 65535).astype(np.uint16)
    return (q >> 8).astype(np.uint8), (q & np.uint16(255)).astype(np.uint8)


def decode_u16_pair(hi: np.ndarray, lo: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    q = (hi.astype(np.uint16) << 8) | lo.astype(np.uint16)
    return (vmin + (q.astype(np.float32) / 65535.0) * (vmax - vmin)).astype(np.float32)


def write_uvp_rgb_png(
    path: Path,
    u: np.ndarray,
    v: np.ndarray,
    pressure: np.ndarray,
    *,
    u_range: tuple[float, float] | None = None,
    v_range: tuple[float, float] | None = None,
    p_range: tuple[float, float] | None = None,
) -> dict[str, list[float]]:
    """RGB8: U→R, V→G, pressure→B."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u_min, u_max = u_range or _finite_range(u, 0.5, 99.5)
    v_min, v_max = v_range or _finite_range(v, 0.5, 99.5)
    p_min, p_max = p_range or _finite_range(pressure, 1.0, 99.0)
    rgb = np.stack(
        [
            encode_u8(u, u_min, u_max),
            encode_u8(v, v_min, v_max),
            encode_u8(pressure, p_min, p_max),
        ],
        axis=-1,
    )
    _save_png_atomic(path, Image.fromarray(rgb, mode="RGB"))
    return {"u": [u_min, u_max], "v": [v_min, v_max], "pressure": [p_min, p_max]}


def read_uvp_rgb_png(
    path: Path,
    u_range: list[float] | tuple[float, float],
    v_range: list[float] | tuple[float, float],
    p_range: list[float] | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(path).convert("RGB"))
    u = decode_u8(img[..., 0], float(u_range[0]), float(u_range[1]))
    v = decode_u8(img[..., 1], float(v_range[0]), float(v_range[1]))
    p = decode_u8(img[..., 2], float(p_range[0]), float(p_range[1]))
    return u, v, p


def write_dot_speed_rgb_png(
    path: Path,
    terrain_dot: np.ndarray,
    speed: np.ndarray,
    *,
    dot_range: tuple[float, float] | None = None,
    speed_range: tuple[float, float] | None = None,
) -> dict[str, list[float]]:
    """RGB: terrain_dot→R,G (16-bit), wind speed→B (8-bit)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d_min, d_max = dot_range or _finite_range(terrain_dot, 0.5, 99.5)
    s_min, s_max = speed_range or (0.0, float(_finite_range(speed, 0.0, 98.0)[1]))
    if s_max < 1e-3:
        s_max = 1e-3
    r, g = encode_u16_pair(terrain_dot, d_min, d_max)
    b = encode_u8(speed, s_min, s_max)
    rgb = np.stack([r, g, b], axis=-1)
    _save_png_atomic(path, Image.fromarray(rgb, mode="RGB"))
    return {"terrain_dot": [d_min, d_max], "speed": [s_min, s_max]}


def read_dot_speed_rgb_png(
    path: Path,
    dot_range: list[float] | tuple[float, float],
    speed_range: list[float] | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(path).convert("RGB"))
    dot = decode_u16_pair(img[..., 0], img[..., 1], float(dot_range[0]), float(dot_range[1]))
    spd = decode_u8(img[..., 2], float(speed_range[0]), float(speed_range[1]))
    return dot, spd


# Back-compat: scalar R,G only (prefer write_dot_speed_rgb_png for wind products)
def write_scalar_rgb16_png(
    path: Path,
    field: np.ndarray,
    *,
    value_range: tuple[float, float] | None = None,
) -> list[float]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = value_range or _finite_range(field, 0.5, 99.5)
    r, g = encode_u16_pair(field, vmin, vmax)
    b = np.zeros_like(r, dtype=np.uint8)
    _save_png_atomic(path, Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB"))
    return [vmin, vmax]


def read_scalar_rgb16_png(path: Path, value_range: list[float] | tuple[float, float]) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    return decode_u16_pair(img[..., 0], img[..., 1], float(value_range[0]), float(value_range[1]))


def elev01_to_meters(
    elev01: np.ndarray, land: np.ndarray, max_elev_m: float = PLANET.max_elev_m
) -> np.ndarray:
    out = np.zeros_like(elev01, dtype=np.float32)
    out[land] = np.clip((elev01[land] - 0.5) / 0.5 * max_elev_m, 0.0, max_elev_m)
    out[~land] = np.clip((0.5 - elev01[~land]) / 0.5, 0.0, 1.0) * (-max_elev_m)
    return out


def prepare_surface_for_wind(
    land: np.ndarray,
    elev_m: np.ndarray,
    *,
    polar_ocean_lat: float = WIND.polar_ocean_lat,
    polar_fade_lat: float = WIND.polar_fade_lat,
    coast_sigma_px: float = WIND.coast_sigma_px,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Polar + coast prep for the wind engine.

    Returns
    -------
    land_out : bool
        Land mask with |lat| ≥ polar_ocean_lat forced to ocean.
    elev_out : float32
        Elevation with polar fade to sea level (0) and polar cap zeroed.
    elev_terrain : float32
        Terrain height for ∇h / terrain_dot: water = 0, soft land·height coast.
        Inland coastal cliffs remain where soft≈1 and land elevation rises.
    land_soft : float32
        Soft land fraction [0,1] for friction blending across coasts.
    """
    from scipy.ndimage import gaussian_filter1d

    h = land.shape[0]
    lat = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * (180.0 / h)
    abs_lat = np.abs(lat)[:, None]
    land_out = np.asarray(land, dtype=bool).copy()
    elev_out = np.asarray(elev_m, dtype=np.float32).copy()

    # Smooth elev → 0 approaching the polar ocean ring, then force ocean
    span = max(float(polar_ocean_lat - polar_fade_lat), 1e-3)
    fade = np.clip((abs_lat - polar_fade_lat) / span, 0.0, 1.0).astype(np.float32)
    elev_out *= 1.0 - fade
    polar = abs_lat >= polar_ocean_lat
    land_out = np.where(polar, False, land_out)
    elev_out = np.where(polar, 0.0, elev_out).astype(np.float32)

    # Water = 0 for terrain; soft land weight so 31 m mask edge is not a cliff
    land_h = np.where(land_out, np.maximum(elev_out, 0.0), 0.0).astype(np.float32)
    soft = land_out.astype(np.float32)
    sig = max(float(coast_sigma_px), 0.5)
    cos = np.clip(np.abs(np.cos(np.deg2rad(lat))), 0.015, 1.0)
    sig_lon = sig / cos
    for i in range(h):
        soft[i] = gaussian_filter1d(soft[i], sigma=float(sig_lon[i]), mode="wrap")
    soft = gaussian_filter1d(soft, sigma=sig * 0.6, axis=0, mode="nearest")
    soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
    elev_terrain = (soft * land_h).astype(np.float32)
    return land_out, elev_out, elev_terrain, soft


def _percentile_stats(
    arr: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a)
    a = a[finite]
    if a.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
    if weights is None:
        mean = float(a.mean())
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)[finite]
        sw = float(w.sum())
        mean = float((a * w).sum() / sw) if sw > 0.0 else 0.0
    return {
        "mean": mean,
        "min": float(a.min()),
        "max": float(a.max()),
        "p5": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def _aquaplanet_month_indices(declinations: np.ndarray) -> dict[str, int]:
    """Map waterworld cases → month index (0–11) by solar declination."""
    d = np.asarray(declinations, dtype=np.float64).reshape(-1)
    if d.size != 12:
        raise ValueError(f"expected 12 monthly declinations, got {d.size}")
    return {
        "tropic_cancer": int(np.argmax(d)),
        "equator": int(np.argmin(np.abs(d))),
        "tropic_capricorn": int(np.argmin(d)),
    }


def synthesize_aquaplanet_temperatures(
    height: int,
    *,
    device: torch.device,
    sample_width: int = 64,
    obliquity_deg: float = PLANET.obliquity_deg,
    s0: float = PLANET.s0,
    greenhouse_factor: float = TEMPERATURE.greenhouse_factor,
    planet_radius_km: float = PLANET.radius_km,
    spinup_years: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Same temperature pipeline as the full map, on an aquaplanet strip:
    all ocean, elev01 = 0.5 (0 m ASL). Currents off (no coasts).

    ``spinup_years`` defaults to 2 so wind-stats debug stays cheap; the
    mapped ``temperature`` CLI uses ``TEMPERATURE.spinup_years``.

    Returns monthly (12,H,W) °C and synthesize meta.
    """
    from .temperature import synthesize_temperatures

    tw = max(int(sample_width), 64)  # wide enough for WindField pressure smooth kernel
    elev01 = np.full((int(height), tw), 0.5, dtype=np.float32)
    land = np.zeros((int(height), tw), dtype=bool)
    monthly, _annual, meta = synthesize_temperatures(
        elev01,
        land,
        device=device,
        **temperature_call_kwargs(
            s0=float(s0),
            obliquity_deg=float(obliquity_deg),
            greenhouse_factor=float(greenhouse_factor),
            planet_radius_km=float(planet_radius_km),
            currents=False,
            spinup_years=int(spinup_years),
        ),
    )
    return monthly, meta


def _waterworld_profile_from_month(
    t2: np.ndarray,
    wind: WindField,
    *,
    device: torch.device,
    lat_step_deg: float,
    solar_declination_deg: float,
    case_id: str,
    case_label: str,
    month_index: int,
    month_name: str,
) -> dict[str, Any]:
    """Run WindField.compute on a 2D aquaplanet T strip (same path as full maps)."""
    h, w = t2.shape
    land = np.zeros((h, w), dtype=bool)
    elev = np.zeros((h, w), dtype=np.float32)  # global 0 m ASL
    res = wind.compute(t2, land, elev, device=device, prepare_surface=True)
    t_bar = np.asarray(t2, dtype=np.float64).mean(axis=1)
    p = res.pressure.mean(dim=1).detach().cpu().numpy().astype(np.float64)
    u = res.u.mean(dim=1).detach().cpu().numpy().astype(np.float64)
    v = res.v.mean(dim=1).detach().cpu().numpy().astype(np.float64)
    speed = np.sqrt(u * u + v * v)
    to_dir = np.degrees(np.arctan2(u, -v)) % 360.0
    lat = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * (180.0 / h)
    step = max(1, int(round(lat_step_deg / (180.0 / h))))
    idx = np.arange(0, h, step)
    design = res.meta.get("belt_centers_design_deg") or {}
    return {
        "id": case_id,
        "label": case_label,
        "month_index": int(month_index),
        "month_name": month_name,
        "solar_declination_deg": float(solar_declination_deg),
        "thermal_equator_lat_deg": float(
            design.get("thermal_equator", res.meta.get("thermal_equator_lat_deg", 0.0))
        ),
        "subtropical_high_lat_deg": design.get(
            "subtropical_high", res.meta.get("subtropical_high_lat_deg")
        ),
        "subpolar_low_lat_deg": design.get(
            "subpolar_low", res.meta.get("subpolar_low_lat_deg")
        ),
        "temperature_C": [float(x) for x in t_bar[idx]],
        "pressure_hpa": [float(x) for x in p[idx]],
        "speed_m_s": [float(x) for x in speed[idx]],
        "to_direction_deg": [float(x) for x in to_dir[idx]],
        "summary": {
            "temperature_C": _percentile_stats(t_bar),
            "pressure_hpa": _percentile_stats(p),
            "speed_m_s": _percentile_stats(speed),
        },
    }


def waterworld_1d_profile(
    wind: WindField,
    *,
    device: torch.device,
    height: int,
    sample_width: int = 64,
    lat_step_deg: float = 1.0,
    obliquity_deg: float = PLANET.obliquity_deg,
    s0: float = PLANET.s0,
    greenhouse_factor: float = TEMPERATURE.greenhouse_factor,
    planet_radius_km: float = PLANET.radius_km,
) -> dict[str, Any]:
    """
    Aquaplanet debug of the full wind pipeline: all ocean, elev = 0 m.

    Temperature uses the same ``synthesize_temperatures`` path as the mapped
    planet (defaults match ``temperature`` CLI; currents off). Three months are
    selected for sun over N tropic / equator / S tropic, then each is fed through
    ``WindField.compute`` exactly as full maps are.
    """
    from .temperature import MONTH_NAMES

    obl = float(obliquity_deg)
    tw = max(int(sample_width), 64)
    monthly, t_meta = synthesize_aquaplanet_temperatures(
        height,
        device=device,
        sample_width=tw,
        obliquity_deg=obl,
        s0=s0,
        greenhouse_factor=greenhouse_factor,
        planet_radius_km=planet_radius_km,
    )
    decls = np.asarray(t_meta.get("declination_deg"), dtype=np.float64)
    month_idx = _aquaplanet_month_indices(decls)
    labels = {
        "tropic_cancer": f"太阳直射北回归线 (δ≈+{obl:g}°)",
        "equator": "太阳直射赤道 (δ≈0°)",
        "tropic_capricorn": f"太阳直射南回归线 (δ≈−{obl:g}°)",
    }
    cases: dict[str, Any] = {}
    for case_id, mi in month_idx.items():
        cases[case_id] = _waterworld_profile_from_month(
            monthly[mi],
            wind,
            device=device,
            lat_step_deg=lat_step_deg,
            solar_declination_deg=float(decls[mi]),
            case_id=case_id,
            case_label=labels[case_id],
            month_index=mi,
            month_name=MONTH_NAMES[mi],
        )
        release_torch_memory()
    return {
        "note": (
            "aquaplanet debug of full pipeline: all ocean, elev=0 m; "
            "T from synthesize_temperatures (same defaults as temperature CLI, "
            "currents=off); wind via WindField.compute(prepare_surface=True); "
            "cases = months nearest δ=+obl / 0 / −obl"
        ),
        "obliquity_deg": obl,
        "s0_W_m2": float(s0),
        "greenhouse_factor": float(greenhouse_factor),
        "lat_step_deg": float(lat_step_deg),
        "sample_width": tw,
        "temperature_meta": {
            k: t_meta[k]
            for k in (
                "declination_deg",
                "t_annual_mean_C",
                "t_ocean_mean_C",
                "spinup_years",
                "heat_capacity_ocean",
            )
            if k in t_meta
        },
        "direction_convention": "to_direction_deg: 0=north, 90=east (matches viewer 去向); profiles omit lat_deg / u,v",
        "cases": cases,
    }


def build_wind_stats(
    result: WindResult,
    land: np.ndarray,
    *,
    temperature_C: np.ndarray,
    wind: WindField,
    device: torch.device,
) -> dict[str, Any]:
    """Annual wind/pressure stats plus all-water 1D profile."""
    p = result.pressure.detach().cpu().numpy().astype(np.float32)
    spd = result.speed.detach().cpu().numpy().astype(np.float32)
    dot = result.terrain_dot.detach().cpu().numpy().astype(np.float32)
    land_b = np.asarray(land, dtype=bool)
    if land_b.shape != p.shape:
        # Re-prep may have changed polar land; match by shape only
        land_b = np.zeros(p.shape, dtype=bool)
    ocean = ~land_b
    h, w = p.shape
    lat = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * (180.0 / h)
    wt = np.clip(np.cos(np.deg2rad(lat)), 0.0, 1.0)[:, None]
    wt2 = np.broadcast_to(wt, p.shape)
    return {
        "grid": {
            "height": h,
            "width": w,
            "land_pct": float(100.0 * area_weighted_mean_np(land_b.astype(np.float64), lat)),
        },
        "pressure_hpa": {
            "global": _percentile_stats(p, wt2),
            "land": _percentile_stats(p[land_b], wt2[land_b]) if land_b.any() else {},
            "ocean": _percentile_stats(p[ocean], wt2[ocean]) if ocean.any() else {},
        },
        "speed_m_s": {
            "global": _percentile_stats(spd, wt2),
            "land": _percentile_stats(spd[land_b], wt2[land_b]) if land_b.any() else {},
            "ocean": _percentile_stats(spd[ocean], wt2[ocean]) if ocean.any() else {},
        },
        "terrain_dot": _percentile_stats(dot, wt2),
        "meta_snapshot": {
            k: result.meta[k]
            for k in (
                "thermal_equator_lat_deg",
                "subtropical_high_lat_deg",
                "subpolar_low_lat_deg",
                "belt_centers_design_deg",
                "upper_amc",
                "amplitudes_hpa",
                "surface_prep",
                "axes",
            )
            if k in result.meta
        },
        "waterworld_1d": waterworld_1d_profile(
            wind, device=device, height=h
        ),
    }


def write_wind_stats(stats: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths.WIND_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / paths.WIND_STATS
    write_text_atomic(
        path, json.dumps(json_safe(stats), indent=2, ensure_ascii=False) + "\n"
    )
    return path


def compute_and_write_wind_stats(
    annual_T: np.ndarray,
    land: np.ndarray,
    elev_m: np.ndarray,
    *,
    device: torch.device | None = None,
    out_dir: Path | None = None,
    wind: WindField | None = None,
) -> Path:
    """
    Annual WindField.compute + build_wind_stats (incl. waterworld 1D) → wind_stats.json.

    Does not write wind PNG products — for summarize / stats-only paths.
    """
    wind = wind or WindField()
    if device is None:
        device = get_device(prefer_gpu=True)
    res = wind.compute(annual_T, land, elev_m, device=device)
    land_prep, _, _, _ = prepare_surface_for_wind(
        np.asarray(land, dtype=bool),
        np.asarray(elev_m, dtype=np.float32),
        polar_ocean_lat=wind.polar_ocean_lat,
        polar_fade_lat=wind.polar_fade_lat,
        coast_sigma_px=wind.coast_sigma_px,
    )
    stats = build_wind_stats(
        res,
        land_prep,
        temperature_C=np.asarray(annual_T, dtype=np.float32),
        wind=wind,
        device=device,
    )
    path = write_wind_stats(stats, out_dir)
    del res
    release_torch_memory()
    return path


def load_temperature_png(path: Path, t_min: float, t_max: float) -> np.ndarray:
    g = load_grayscale(path)
    if g.max() <= 1.5:
        g = g * 255.0
    return (t_min + (g / 255.0) * (t_max - t_min)).astype(np.float32)


def write_wind_products(
    result: WindResult,
    out_dir: Path,
    *,
    label: str,
    uv_name: str,
    dot_name: str,
    pressure_name: str | None = None,
) -> dict[str, Any]:
    """
    Write two RGB PNGs:
      uv_name  — U→R, V→G, pressure→B (8-bit)
      dot_name — terrain_dot→R,G (16-bit), speed→B (8-bit)
    ``pressure_name`` is ignored (kept for call-site compat).
    """
    del pressure_name  # pressure packed into UVP B channel
    out_dir.mkdir(parents=True, exist_ok=True)
    p = result.pressure.detach().cpu().numpy().astype(np.float32)
    u = result.u.detach().cpu().numpy().astype(np.float32)
    v = result.v.detach().cpu().numpy().astype(np.float32)
    spd = result.speed.detach().cpu().numpy().astype(np.float32)
    dot = result.terrain_dot.detach().cpu().numpy().astype(np.float32)

    p_min, p_max = _finite_range(p, 1.0, 99.0)
    s_min = 0.0
    s_max = float(np.percentile(spd[np.isfinite(spd)], 98)) if np.isfinite(spd).any() else 1.0
    if s_max < 1e-3:
        s_max = float(max(float(np.nanmax(spd)), 1e-3))

    uvp_scale = write_uvp_rgb_png(
        out_dir / uv_name,
        u,
        v,
        p,
        p_range=(p_min, p_max),
    )
    ds_scale = write_dot_speed_rgb_png(
        out_dir / dot_name,
        dot,
        spd,
        speed_range=(s_min, s_max),
    )

    return {
        "label": label,
        "files": {
            "uv": uv_name,
            "terrain_dot": dot_name,
            "pressure": uv_name,
        },
        "encoding": {
            "uv": "rgb8: U→R  V→G  pressure→B; +U east, +V south (image)",
            "terrain_dot": "rgb: terrain_dot→R,G (uint16)  speed→B (uint8)",
            "pressure": "packed in uv B channel (gray_scale_pressure_hpa)",
        },
        "uv_scale": {"u": uvp_scale["u"], "v": uvp_scale["v"]},
        "terrain_dot_scale": ds_scale["terrain_dot"],
        "gray_scale_pressure_hpa": uvp_scale["pressure"],
        "gray_scale_speed": ds_scale["speed"],
        "shape": list(p.shape),
        **result.meta,
    }


def synthesize_wind_maps(
    monthly_T: np.ndarray | None,
    annual_T: np.ndarray,
    land: np.ndarray,
    elev_m: np.ndarray,
    *,
    device: torch.device,
    wind: WindField | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute annual (+ monthly if provided) wind products; return meta dict."""
    from .timing import StepTimer, format_duration

    wind = wind or WindField()
    out_dir = out_dir or paths.WIND_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    job_timer = StepTimer("synthesize")

    print("  wind: annual…", flush=True)
    ann = wind.compute(annual_T, land, elev_m, device=device)
    products = [
        write_wind_products(
            ann,
            out_dir,
            label="annual",
            uv_name=paths.WIND_UV_ANNUAL,
            dot_name=paths.WIND_TERRAIN_DOT_ANNUAL,
        )
    ]
    land_prep, _, _, _ = prepare_surface_for_wind(
        land,
        elev_m,
        polar_ocean_lat=wind.polar_ocean_lat,
        polar_fade_lat=wind.polar_fade_lat,
        coast_sigma_px=wind.coast_sigma_px,
    )
    print("  wind: stats + waterworld 1D…", flush=True)
    stats = build_wind_stats(
        ann,
        land_prep,
        temperature_C=annual_T,
        wind=wind,
        device=device,
    )
    stats_path = write_wind_stats(stats, out_dir)
    print(f"  wrote {stats_path}")
    del ann
    release_torch_memory()

    month_meta: list[dict[str, Any]] = []
    month_timer = StepTimer("month", total_steps=12 if monthly_T is not None else None)
    if monthly_T is not None:
        assert monthly_T.shape[0] == 12
        for m, name in enumerate(paths.MONTH_NAMES_SHORT):
            month_timer.begin(name)
            res = wind.compute(monthly_T[m], land, elev_m, device=device)
            month_meta.append(
                write_wind_products(
                    res,
                    out_dir,
                    label=name,
                    uv_name=paths.MONTH_WIND_UV_NAMES[m],
                    dot_name=paths.MONTH_TERRAIN_DOT_NAMES[m],
                )
            )
            del res
            release_torch_memory()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = month_timer.end() or 0.0
            print(
                f"    month {m + 1}/12  "
                f"{month_timer.progress_line(last_name=name, last_dt=dt)}",
                flush=True,
            )

    print(
        f"  synthesize total {format_duration(job_timer.elapsed)}  "
        f"month steps {len(month_timer.steps)}  "
        f"avg/month {format_duration(month_timer.mean_step)}",
        flush=True,
    )

    meta = {
        "annual": products[0],
        "months": month_meta,
        "notes": {
            "uv": "RGB8: U→R, V→G, pressure→B; +U east, +V south (image)",
            "terrain_dot": "RGB: terrain_dot→R,G (16-bit), speed→B (8-bit)",
            "pressure": "packed in Wind-UV B channel; gray_scale_pressure_hpa",
            "speed": "packed in Terrain-Dot B channel; also derived from UV in viewer",
            "stats": paths.WIND_STATS,
            "polar": "E–W pressure anomaly damped 87→89° to zonal mean; |lat|≥89 forced ocean + elev→0",
            "slp": "temperature sea-levelized (undo land lapse) before pressure anomalies; products are SLP",
            "convolution": "∇p and wind UV use Gaussian convolution (neighborhood direction)",
        },
        "timing": {
            "synthesize_total_s": round(job_timer.elapsed, 3),
            "month_steps": len(month_timer.steps),
            "month_avg_s": round(month_timer.mean_step, 3),
            "month_total_s": round(sum(d for _, d in month_timer.steps), 3),
        },
    }
    meta_path = out_dir / paths.WIND_META
    write_text_atomic(meta_path, json.dumps(json_safe(meta), indent=2) + "\n")
    print(f"  wrote {meta_path}")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = paths.ROOT
    p = argparse.ArgumentParser(description="Generate wind / pressure fields from temperature maps.")
    p.add_argument("--temp-dir", type=Path, default=paths.TEMP_DIR)
    p.add_argument("--out-dir", type=Path, default=paths.WIND_DIR)
    p.add_argument("--elevation", type=Path, default=root / "graphs" / paths.FULL_ELEV)
    p.add_argument("--land-mask", type=Path, default=root / "graphs" / paths.LAND_MASK)
    p.add_argument("--max-elev-m", type=float, default=PLANET.max_elev_m)
    p.add_argument("--lapse", type=float, default=PLANET.lapse_k_per_km, help="K/km; undo land lapse for SLP")
    p.add_argument("--t-spin-hours", type=float, default=PLANET.t_spin_hours)
    p.add_argument("--radius-km", type=float, default=PLANET.radius_km)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--annual-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from .assets import ensure_derived_terrain, write_assets_json
    from .timing import StepTimer

    wall = StepTimer("wind")
    args = parse_args(argv)
    with wall.step("ensure"):
        ensure_derived_terrain(seed_template=True)
    device = get_device(prefer_gpu=not args.cpu)
    print(f"Wind device: {device}")

    with wall.step("load"):
        meta_path = args.temp_dir / paths.TEMP_META
        t_min, t_max = ENCODE.t_gray_min, ENCODE.t_gray_max
        if meta_path.is_file():
            tm = json.loads(meta_path.read_text(encoding="utf-8"))
            gs = tm.get("gray_scale_C")
            if isinstance(gs, list) and len(gs) == 2:
                t_min, t_max = float(gs[0]), float(gs[1])

        ann_path = args.temp_dir / paths.TEMP_ANNUAL
        if not ann_path.is_file():
            print(f"Missing {ann_path}; run temperature first.", flush=True)
            return 1
        annual = load_temperature_png(ann_path, t_min, t_max)

        monthly = None
        if not args.annual_only:
            months = []
            ok = True
            for name in paths.MONTH_TEMP_NAMES:
                fp = args.temp_dir / name
                if not fp.is_file():
                    ok = False
                    break
                months.append(load_temperature_png(fp, t_min, t_max))
            if ok:
                monthly = np.stack(months, axis=0)

        elev01 = load_grayscale(args.elevation)
        if args.land_mask.is_file():
            land = load_grayscale(args.land_mask) > 0.5
        else:
            land = elev01 > 0.5
        if elev01.shape != annual.shape:
            raise SystemExit(f"Shape mismatch elev {elev01.shape} vs T {annual.shape}")
        elev_m = elev01_to_meters(elev01, land, max_elev_m=args.max_elev_m)

    wind = WindField(
        **wind_field_kwargs(
            t_spin_s=args.t_spin_hours * 3600.0,
            planet_radius_km=args.radius_km,
            lapse_k_per_km=args.lapse,
        )
    )
    with wall.step("synthesize"):
        synthesize_wind_maps(
            monthly,
            annual,
            land,
            elev_m,
            device=device,
            wind=wind,
            out_dir=args.out_dir,
        )
    with wall.step("assets"):
        write_assets_json()
    release_torch_memory()
    print(f"Done. Wind outputs in {args.out_dir}")
    print(wall.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
