#!/usr/bin/env python3
"""
Generate 12 monthly temperature maps + annual mean from a grayscale
full-elevation terrain (0.5 = sea level).

Drivers
-------
- Top-of-atmosphere solar irradiance (default S0 = 1361 W/m^2)
- Axial tilt / tropics at +/-23.5 deg (smooth, no hard latitude bands)
- Elevation lapse rate on land
- Distance to *open* ocean + nearby open-ocean area (maritime moderation)
  Water with that month's temperature at or below 0 C does not provide
  maritime influence for that month

Uses PyTorch CUDA when available (conda env tf-gpu).

Outputs grayscale PNGs under graphs/temperature/ (brighter = warmer):
    gray = clip(round((T_C - T_MIN) / (T_MAX - T_MIN) * 255), 0, 255)
default T_MIN=-60 C, T_MAX=+45 C.

Usage::

    python generate_temperature.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from .timing import StepTimer, format_duration


SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W m^-2 K^-4

MID_MONTH_DOY = np.array(
    [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=np.float64
)
MONTH_NAMES = [
    "01 January",
    "02 February",
    "03 March",
    "04 April",
    "05 May",
    "06 June",
    "07 July",
    "08 August",
    "09 September",
    "10 October",
    "11 November",
    "12 December",
]


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_grayscale(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def latitude_grid(height: int, device: torch.device) -> torch.Tensor:
    """Equirectangular latitudes (deg): +90 at top row centers -> -90 at bottom."""
    return 90.0 - (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (
        180.0 / height
    )


def solar_declination_deg(doy: np.ndarray | float, obliquity_deg: float = 23.5) -> np.ndarray:
    doy = np.asarray(doy, dtype=np.float64)
    lam = np.deg2rad(360.0 * (doy - 81.0) / 365.25)
    eps = np.deg2rad(obliquity_deg)
    return np.rad2deg(np.arcsin(np.sin(eps) * np.sin(lam)))


def daily_mean_toa_insolation(
    lat_deg: torch.Tensor,
    decl_deg: float,
    s0: float = 1361.0,
) -> torch.Tensor:
    """Daily-mean TOA insolation (W/m^2), continuous through polar day/night."""
    phi = torch.deg2rad(lat_deg)
    delta = torch.deg2rad(
        torch.tensor(decl_deg, device=lat_deg.device, dtype=lat_deg.dtype)
    )
    # Clamp |phi| slightly below 90 to keep tan finite
    phi_safe = torch.clamp(phi, -np.pi / 2 + 1e-4, np.pi / 2 - 1e-4)
    x = -torch.tan(phi_safe) * torch.tan(delta)
    x_clamped = torch.clamp(x, -1.0, 1.0)
    h0 = torch.acos(x_clamped)
    q = (s0 / np.pi) * (
        h0 * torch.sin(phi_safe) * torch.sin(delta)
        + torch.cos(phi_safe) * torch.cos(delta) * torch.sin(h0)
    )
    return torch.clamp(q, min=0.0)


def soft_tropics_weight(abs_lat: torch.Tensor, obliquity_deg: float) -> torch.Tensor:
    """Smooth tropics weight in [0,1], peaking at equator, ~0 beyond tropics."""
    # Raised-cosine falloff to 0 at ~1.35 * obliquity (no hard edge at 23.5)
    x = torch.clamp(abs_lat / (obliquity_deg * 1.35), 0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(np.pi * x))


def soft_subtropical_dry(abs_lat: torch.Tensor, obliquity_deg: float) -> torch.Tensor:
    """Smooth subtropical dry peak near ~obliquity+8 deg."""
    center = obliquity_deg + 8.0
    width = 12.0
    return torch.exp(-0.5 * ((abs_lat - center) / width) ** 2)


def soft_step(x: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    """Smooth 0->1 step; larger scale = softer."""
    return torch.sigmoid((x - center) / max(scale, 1e-3))


def radiative_temperature_C(
    q: torch.Tensor,
    albedo: torch.Tensor,
    greenhouse_factor: float,
    q_floor: float = 25.0,
) -> torch.Tensor:
    absorbed = greenhouse_factor * (1.0 - albedo) * torch.clamp(q, min=q_floor)
    t_k = torch.pow(torch.clamp(absorbed, min=1.0) / SIGMA, 0.25)
    return t_k - 273.15


def apply_heat_transport(q: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0.0:
        return q
    return (1.0 - strength) * q + strength * q.mean()


def box_filter_wrap_lon(field: torch.Tensor, ry: int, rx: int) -> torch.Tensor:
    """Separable mean filter; longitude wraps, latitude uses replicate."""
    # field: (H, W)
    h, w = field.shape
    x = field[None, None]  # (1,1,H,W)
    # Pad lon wrap, lat replicate
    x = F.pad(x, (rx, rx, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, ry, ry), mode="replicate")
    ky = torch.ones((1, 1, 2 * ry + 1, 1), device=field.device, dtype=field.dtype) / (
        2 * ry + 1
    )
    kx = torch.ones((1, 1, 1, 2 * rx + 1), device=field.device, dtype=field.dtype) / (
        2 * rx + 1
    )
    x = F.conv2d(x, ky)
    x = F.conv2d(x, kx)
    return x[0, 0]


def open_water_distance_km(
    open_water_np: np.ndarray,
    lat_np: np.ndarray,
    planet_radius_km: float,
    max_dist_km: float = 2000.0,
) -> np.ndarray:
    """
    Distance (km) to nearest open-water pixel. Longitude wraps.
    open_water_np: bool (H,W)

    Pads only enough for max_dist_km (full half-width pad is too costly at 8k).
    """
    h, w = open_water_np.shape
    km_per_deg = (np.pi * planet_radius_km) / 180.0
    dy_km = (180.0 / h) * km_per_deg
    dx_eq = (360.0 / w) * km_per_deg

    if not open_water_np.any():
        return np.full((h, w), 1.0e6, dtype=np.float32)

    pad = min(w // 2, max(32, int(np.ceil(max_dist_km / max(dx_eq, 1e-6))) + 2))
    ow_pad = np.pad(open_water_np, ((0, 0), (pad, pad)), mode="wrap")
    dist_pad = ndimage.distance_transform_edt(~ow_pad, sampling=(dy_km, dx_eq))
    dist_km = dist_pad[:, pad : pad + w].astype(np.float32)
    cos_abs = np.abs(np.cos(np.deg2rad(lat_np))).astype(np.float32)
    dist_km *= np.sqrt(0.35 + 0.65 * cos_abs)[:, None]
    return np.minimum(dist_km, np.float32(max_dist_km * 1.5))


def kernel_radius_px(
    height: int,
    width: int,
    radius_km: float,
    planet_radius_km: float,
) -> tuple[int, int]:
    km_per_deg = (np.pi * planet_radius_km) / 180.0
    dy_km = (180.0 / height) * km_per_deg
    dx_eq = (360.0 / width) * km_per_deg
    ry = max(1, int(round(radius_km / dy_km)))
    rx = max(1, int(round(radius_km / dx_eq)))
    return ry, rx


def maritime_from_open_water(
    open_water: torch.Tensor,
    lat: torch.Tensor,
    *,
    planet_radius_km: float,
    neighbor_radius_km: float,
    maritime_e_fold_km: float,
    land: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Maritime weight from open (unfrozen) water only.
    Returns (maritime, dist_km, near_open).
    """
    h, w = open_water.shape
    device = open_water.device
    ry, rx = kernel_radius_px(h, w, neighbor_radius_km, planet_radius_km)

    # Nearby open-water fraction (GPU conv); soft weights allowed
    near = box_filter_wrap_lon(open_water, ry, rx).clamp(0.0, 1.0)

    # Distance on CPU via scipy EDT (binary open water)
    ow_bin = (open_water > 0.5).detach().cpu().numpy().astype(bool)
    lat_np = lat.detach().cpu().numpy()
    max_dist = max(2000.0, maritime_e_fold_km * 8.0, neighbor_radius_km * 4.0)
    dist_np = open_water_distance_km(
        ow_bin, lat_np, planet_radius_km, max_dist_km=max_dist
    )
    dist_km = torch.from_numpy(dist_np).to(device=device, dtype=torch.float32)

    dist_term = torch.exp(-dist_km / max(maritime_e_fold_km, 1.0))
    maritime = (0.55 * near + 0.45 * dist_term).clamp(0.0, 1.0)

    # Over open water itself: strong maritime; over frozen water / land: from proximity
    # Local open-water presence scales maritime on water cells (ice ~ land-like)
    maritime = torch.where(
        land,
        maritime,
        torch.maximum(maritime * open_water, open_water * 0.92),
    )
    return maritime, dist_km, near


def earthlike_sst_profile_C(lat_deg: torch.Tensor) -> torch.Tensor:
    """
    Approximate Earth zonal-mean annual SST (C) vs latitude.
    Equator ~27-28 C; polar seas near -1.8 C.
    """
    phi = torch.deg2rad(lat_deg)
    cos_lat = torch.cos(phi).clamp(0.0, 1.0)
    sst = 27.5 * torch.pow(cos_lat, 1.25) - 0.8
    sst = sst - 2.5 * soft_step(lat_deg.abs(), 55.0, 12.0)
    return sst.clamp(-1.8, 29.5)


def earthlike_sst_month_C(lat_deg: torch.Tensor, decl_deg: float) -> torch.Tensor:
    """Annual SST profile + modest midlatitude seasonal swing (weak inside polar circles)."""
    ann = earthlike_sst_profile_C(lat_deg)
    phi = torch.deg2rad(lat_deg)
    season = float(np.sin(np.deg2rad(decl_deg)))
    mid = torch.exp(-0.5 * ((lat_deg.abs() - 42.0) / 16.0) ** 2)
    # Gate out seasonal SST swing poleward of ~60° (polar oceans stay near annual)
    polar_gate = 1.0 - soft_step(lat_deg.abs(), 58.0, 8.0)
    d_t = 3.5 * season * torch.sin(phi) * mid * polar_gate
    return (ann + d_t).clamp(-1.8, 30.5)


def masked_local_mean(
    field: torch.Tensor,
    weight: torch.Tensor,
    ry: int,
    rx: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Neighborhood mean of field using soft mask weights (lon wraps)."""
    num = box_filter_wrap_lon(field * weight, ry, rx)
    den = box_filter_wrap_lon(weight, ry, rx).clamp_min(eps)
    return num / den


def blend_coastal_temperatures(
    t_c: torch.Tensor,
    land: torch.Tensor,
    open_water: torch.Tensor,
    dist_km: torch.Tensor,
    *,
    planet_radius_km: float,
    coast_blend_km: float,
    ocean_mix_km: float,
    land_pull: float,
    ocean_pull: float,
) -> torch.Tensor:
    """
    Light coastal continuity + ocean mixing.

    Allows a few C of land-sea contrast at the shore and a smooth inland
    fade. Avoids forcing coastal land onto SST (that over-cooled/warmed
    coasts and created an inland jump where a hard blend cut off).
    """
    h, w = t_c.shape
    ry_c, rx_c = kernel_radius_px(h, w, max(35.0, coast_blend_km), planet_radius_km)
    ry_o, rx_o = kernel_radius_px(h, w, ocean_mix_km, planet_radius_km)

    land_f = land.float()
    water = 1.0 - land_f
    ow = open_water.clamp(0.0, 1.0)

    local_ocean = masked_local_mean(t_c, ow.clamp_min(1e-4), ry_o, rx_o)
    local_any_water = masked_local_mean(t_c, water.clamp_min(1e-4), ry_c, rx_c)

    efold = max(coast_blend_km, 1.0)
    shore = torch.exp(-dist_km / efold)
    near_ow = box_filter_wrap_lon(ow, max(1, ry_c // 3), max(1, rx_c // 3)).clamp(0.0, 1.0)

    # Modest land -> nearby water pull (cap ~0.4 so coasts keep distinct climate)
    land_w = (land_f * shore * land_pull * (0.35 + 0.65 * near_ow)).clamp(0.0, 0.42)
    out = t_c * (1.0 - land_w) + local_any_water * land_w

    ocean_w = (water * ow * ocean_pull).clamp(0.0, 0.70)
    out = out * (1.0 - ocean_w) + local_ocean * ocean_w

    closed = water * (1.0 - ow)
    if float(closed.max().item()) > 0.0:
        local_closed = masked_local_mean(t_c, closed + 1e-3 * water, ry_o, rx_o)
        cw = (closed * 0.25 * shore.clamp_min(0.15)).clamp(0.0, 0.35)
        out = out * (1.0 - cw) + local_closed * cw

    return out


def synthesize_temperatures(
    elev01: np.ndarray,
    land_np: np.ndarray,
    *,
    device: torch.device,
    s0: float,
    obliquity_deg: float,
    max_elev_m: float,
    lapse_k_per_km: float,
    planet_radius_km: float,
    neighbor_radius_km: float,
    maritime_e_fold_km: float,
    greenhouse_factor: float,
    heat_transport: float,
    albedo_ocean: float,
    albedo_land: float,
    albedo_ice: float,
    ice_threshold_C: float,
    freeze_C: float,
    freeze_soft_C: float,
    ocean_inertia: float,
    land_inertia: float,
    continentality_amp: float,
    coast_blend_km: float = 100.0,
    ocean_mix_km: float = 450.0,
    coast_land_pull: float = 0.30,
    coast_ocean_pull: float = 0.45,
    ocean_sst_nudge: float = 0.55,
    maritime_iters: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns monthly (12,H,W) float32 C, annual (H,W) float32 C, meta dict.

    Open-ocean maritime influence is monthly: water with that month's
    temperature <= freeze_C does not count as open ocean for that month.
    """
    job_timer = StepTimer("synthesize")
    elev = torch.from_numpy(elev01).to(device=device, dtype=torch.float32)
    land = torch.from_numpy(land_np.astype(np.bool_)).to(device=device)
    h, w = elev.shape
    lat = latitude_grid(h, device)
    abs_lat = lat.abs()

    elev_m = torch.where(land, (elev - 0.5) / 0.5 * max_elev_m, torch.zeros_like(elev))
    elev_m = elev_m.clamp(0.0, max_elev_m)
    lapse_raw = lapse_k_per_km * (elev_m / 1000.0)

    ry_s, rx_s = kernel_radius_px(h, w, max(25.0, coast_blend_km * 0.4), planet_radius_km)
    land_soft = box_filter_wrap_lon(land.float(), ry_s, rx_s).clamp(0.0, 1.0)
    water_soft = 1.0 - land_soft

    declinations = solar_declination_deg(MID_MONTH_DOY, obliquity_deg)
    q_months = torch.empty((12, h, w), device=device, dtype=torch.float32)
    sst_targets = torch.empty((12, h, w), device=device, dtype=torch.float32)
    for m, dec in enumerate(declinations):
        q_row = daily_mean_toa_insolation(lat, float(dec), s0=s0)
        q_months[m] = q_row[:, None].expand(h, w)
        sst_row = earthlike_sst_month_C(lat, float(dec))
        sst_targets[m] = sst_row[:, None].expand(h, w)

    q_annual = q_months.mean(dim=0)
    q_ann_eff = apply_heat_transport(q_annual, heat_transport)

    tropics_w = soft_tropics_weight(abs_lat, obliquity_deg)[:, None].expand(h, w)
    dry_w = soft_subtropical_dry(abs_lat, obliquity_deg)[:, None].expand(h, w)
    water = (~land).float()

    open_water = water[None].expand(12, h, w).clone()
    monthly = torch.empty((12, h, w), device=device, dtype=torch.float32)
    dist_km_ref = torch.zeros((h, w), device=device, dtype=torch.float32)
    maritime_ref = torch.zeros((h, w), device=device, dtype=torch.float32)

    n_iters = max(1, maritime_iters)
    month_timer = StepTimer("month", total_steps=n_iters * 12)

    for it in range(n_iters):
        print(f"  maritime iter {it + 1}/{n_iters}…", flush=True)
        albedo = albedo_land * land_soft + albedo_ocean * water_soft
        t_rad0 = radiative_temperature_C(q_ann_eff, albedo, greenhouse_factor)
        ice_lat = soft_step(abs_lat, obliquity_deg + 28.0, 10.0)[:, None].expand(h, w)
        ice_t = soft_step(
            torch.full_like(t_rad0, ice_threshold_C) - t_rad0, 0.0, 4.0
        )
        ice_w = torch.clamp(
            ice_lat * ice_t * (0.85 * land_soft + 0.25 * water_soft), 0.0, 1.0
        )
        albedo = (albedo + ice_w * (albedo_ice - albedo)).clamp(0.0, 0.95)
        t_rad = radiative_temperature_C(q_ann_eff, albedo, greenhouse_factor)

        for m in range(12):
            step_name = f"i{it+1}m{m+1}"
            month_timer.begin(step_name)
            maritime, dist_km, _near = maritime_from_open_water(
                open_water[m],
                lat,
                planet_radius_km=planet_radius_km,
                neighbor_radius_km=neighbor_radius_km,
                maritime_e_fold_km=maritime_e_fold_km,
                land=land,
            )
            if m == 5:
                dist_km_ref = dist_km
                maritime_ref = maritime

            coast_prox = torch.exp(-dist_km / max(coast_blend_km * 0.8, 1.0))
            lapse = lapse_raw * (1.0 - 0.35 * coast_prox * land.float())

            inland = (1.0 - maritime) * land_soft
            # Tropics boost mainly on land; oceans follow SST climatology
            geo = (
                3.2 * tropics_w * land_soft
                + 0.6 * tropics_w * water_soft
                + 1.8 * dry_w * inland
                - 1.2 * inland * (abs_lat[:, None] / 90.0)
            )

            sens = (
                0.070 * (1.0 - maritime) * land_soft
                + 0.018 * maritime
                + 0.028 * land_soft * (1.0 - 0.65 * maritime)
            )
            sens = sens * (
                1.0 + continentality_amp * (1.0 - maritime) * land_soft * 0.30
            )
            sens = sens * (1.0 - 0.40 * water_soft * maritime)
            # Polar oceans: damp insolation-driven seasonal swing (ice / deep mixed layer)
            polar_w = soft_step(abs_lat, 58.0, 8.0)[:, None].expand(h, w)
            sens = sens * (1.0 - 0.75 * polar_w * water_soft)
            inertia = land_inertia + (ocean_inertia - land_inertia) * maritime
            inertia = torch.clamp(inertia + 0.08 * polar_w * water_soft, 0.0, 0.97)

            dq = q_months[m] - q_annual
            d_t = sens * dq * (1.0 - 0.55 * inertia)
            t_phys = t_rad + geo + d_t - lapse

            nudge = float(np.clip(ocean_sst_nudge, 0.0, 1.0))
            t_ocean = (1.0 - nudge) * t_phys + nudge * sst_targets[m]
            t_ocean = torch.maximum(t_ocean, torch.full_like(t_ocean, -1.8))
            monthly[m] = torch.where(land, t_phys, t_ocean)

            monthly[m] = blend_coastal_temperatures(
                monthly[m],
                land,
                open_water[m],
                dist_km,
                planet_radius_km=planet_radius_km,
                coast_blend_km=coast_blend_km,
                ocean_mix_km=ocean_mix_km,
                land_pull=coast_land_pull,
                ocean_pull=coast_ocean_pull,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = month_timer.end() or 0.0
            print(
                f"    month {m + 1}/12  "
                f"{month_timer.progress_line(last_name=step_name, last_dt=dt)}",
                flush=True,
            )

        open_water = water[None] * soft_step(monthly, freeze_C, freeze_soft_C)

        if device.type == "cuda":
            torch.cuda.synchronize()

    annual = monthly.mean(dim=0)
    open_frac_monthly = [float(open_water[m].mean().item()) for m in range(12)]

    lat2 = lat[:, None].expand(h, w)
    eq = (~land) & (lat2.abs() < 10.0)
    pol = (~land) & (lat2.abs() > 60.0)
    amp = (monthly[6] - monthly[0]).abs()
    timing = {
        "synthesize_total_s": round(job_timer.elapsed, 3),
        "month_steps": len(month_timer.steps),
        "month_avg_s": round(month_timer.mean_step, 3),
        "month_total_s": round(sum(d for _, d in month_timer.steps), 3),
    }
    print(
        f"  synthesize total {format_duration(job_timer.elapsed)}  "
        f"month steps {len(month_timer.steps)}  "
        f"avg/month {format_duration(month_timer.mean_step)}",
        flush=True,
    )
    meta = {
        "device": str(device),
        "declination_deg": declinations.tolist(),
        "q_annual_mean_W_m2": float(q_annual.mean().item()),
        "q_annual_max_W_m2": float(q_annual.max().item()),
        "t_annual_mean_C": float(annual.mean().item()),
        "t_annual_min_C": float(annual.min().item()),
        "t_annual_max_C": float(annual.max().item()),
        "t_land_mean_C": float(annual[land].mean().item()),
        "t_ocean_mean_C": float(annual[~land].mean().item()),
        "t_ocean_eq_mean_C": float(annual[eq].mean().item()) if bool(eq.any()) else None,
        "t_ocean_polar_mean_C": float(annual[pol].mean().item()) if bool(pol.any()) else None,
        "amp_jul_jan_land_mean_C": float(amp[land].mean().item()) if land.any() else 0.0,
        "amp_jul_jan_ocean_mean_C": float(amp[~land].mean().item()) if (~land).any() else 0.0,
        "open_water_frac_mean": float(open_water.mean().item()),
        "open_water_frac_monthly": open_frac_monthly,
        "elev_land_max_m": float(elev_m[land].max().item()) if land.any() else 0.0,
        "dist_land_mean_km": float(dist_km_ref[land].mean().item()) if land.any() else 0.0,
        "maritime_land_mean": float(maritime_ref[land].mean().item()) if land.any() else 0.0,
        "freeze_C": freeze_C,
        "freeze_rule": "monthly: water with T_month <= freeze_C is not open ocean",
        "coast_blend_km": coast_blend_km,
        "ocean_mix_km": ocean_mix_km,
        "ocean_sst_nudge": ocean_sst_nudge,
        "maritime_iters": maritime_iters,
        "timing": timing,
    }

    monthly_np = monthly.detach().cpu().numpy().astype(np.float32)
    annual_np = annual.detach().cpu().numpy().astype(np.float32)
    return monthly_np, annual_np, meta


def temperature_to_gray(t_c: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    g = (t_c - t_min) / (t_max - t_min) * 255.0
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def save_gray_png(path: Path, gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="L").save(path)


def parse_args() -> argparse.Namespace:
    from . import paths

    root = paths.ROOT
    p = argparse.ArgumentParser(description="Generate monthly / annual temperature maps (GPU).")
    p.add_argument(
        "--elevation",
        type=Path,
        default=root / "graphs" / "Terrain - Full Elevation.png",
    )
    p.add_argument(
        "--land-mask",
        type=Path,
        default=root / "graphs" / "Terrain - Land Mask.png",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "graphs" / "temperature",
    )
    p.add_argument("--s0", type=float, default=1361.0)
    p.add_argument("--obliquity", type=float, default=23.5)
    p.add_argument("--max-elev-m", type=float, default=8000.0)
    p.add_argument("--lapse", type=float, default=6.5)
    p.add_argument("--radius-km", type=float, default=6371.0)
    p.add_argument("--neighbor-radius-km", type=float, default=400.0)
    p.add_argument("--maritime-efold-km", type=float, default=250.0)
    p.add_argument("--greenhouse", type=float, default=1.46)
    p.add_argument("--heat-transport", type=float, default=0.58)
    p.add_argument("--albedo-ocean", type=float, default=0.11)
    p.add_argument("--albedo-land", type=float, default=0.18)
    p.add_argument("--albedo-ice", type=float, default=0.45)
    p.add_argument("--ice-threshold", type=float, default=-2.0)
    p.add_argument(
        "--freeze-c",
        type=float,
        default=0.0,
        help="Water with monthly T at/below this C does not provide maritime influence that month",
    )
    p.add_argument(
        "--freeze-soft-c",
        type=float,
        default=1.5,
        help="Softness (C) of open-water freeze transition",
    )
    p.add_argument("--maritime-iters", type=int, default=2)
    p.add_argument("--ocean-inertia", type=float, default=0.90)
    p.add_argument("--land-inertia", type=float, default=0.28)
    p.add_argument("--continentality-amp", type=float, default=1.2)
    p.add_argument(
        "--coast-blend-km",
        type=float,
        default=100.0,
        help="Land→SST coastal blend e-folding distance (km)",
    )
    p.add_argument(
        "--ocean-mix-km",
        type=float,
        default=450.0,
        help="Ocean horizontal mixing radius (km)",
    )
    p.add_argument("--coast-land-pull", type=float, default=0.30)
    p.add_argument("--coast-ocean-pull", type=float, default=0.45)
    p.add_argument(
        "--ocean-sst-nudge",
        type=float,
        default=0.55,
        help="Blend weight of Earth-like SST profile into ocean cells [0,1]",
    )
    p.add_argument("--t-gray-min", type=float, default=-60.0)
    p.add_argument("--t-gray-max", type=float, default=45.0)
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


def main() -> None:
    from .assets import ensure_derived_terrain, write_assets_json

    wall = StepTimer("temperature")
    args = parse_args()
    with wall.step("ensure"):
        ensure_derived_terrain(seed_template=True)
    device = get_device(prefer_gpu=not args.cpu)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    with wall.step("load"):
        elev = load_grayscale(args.elevation)
        if args.land_mask.is_file():
            land = load_grayscale(args.land_mask) > 0.5
        else:
            land = elev > 0.5

        if args.downsample > 1:
            s = args.downsample
            elev = elev[::s, ::s].copy()
            land = land[::s, ::s].copy()

    if elev.shape != land.shape:
        raise SystemExit(f"Shape mismatch: elev {elev.shape} vs land {land.shape}")

    print(f"Grid {elev.shape[1]}x{elev.shape[0]}  land={land.mean()*100:.1f}%")
    print(
        f"S0={args.s0} W/m^2  obliquity=+/-{args.obliquity} deg  "
        f"max_elev={args.max_elev_m} m  freeze<={args.freeze_c} C"
    )

    with wall.step("synthesize"):
        monthly, annual, meta = synthesize_temperatures(
            elev,
            land,
            device=device,
            s0=args.s0,
            obliquity_deg=args.obliquity,
            max_elev_m=args.max_elev_m,
            lapse_k_per_km=args.lapse,
            planet_radius_km=args.radius_km,
            neighbor_radius_km=args.neighbor_radius_km,
            maritime_e_fold_km=args.maritime_efold_km,
            greenhouse_factor=args.greenhouse,
            heat_transport=args.heat_transport,
            albedo_ocean=args.albedo_ocean,
            albedo_land=args.albedo_land,
            albedo_ice=args.albedo_ice,
            ice_threshold_C=args.ice_threshold,
            freeze_C=args.freeze_c,
            freeze_soft_C=args.freeze_soft_c,
            ocean_inertia=args.ocean_inertia,
            land_inertia=args.land_inertia,
            continentality_amp=args.continentality_amp,
            coast_blend_km=args.coast_blend_km,
            ocean_mix_km=args.ocean_mix_km,
            coast_land_pull=args.coast_land_pull,
            coast_ocean_pull=args.coast_ocean_pull,
            ocean_sst_nudge=args.ocean_sst_nudge,
            maritime_iters=args.maritime_iters,
        )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    with wall.step("write"):
        for m, name in enumerate(MONTH_NAMES):
            gray = temperature_to_gray(monthly[m], args.t_gray_min, args.t_gray_max)
            path = out / f"Temperature - {name}.png"
            save_gray_png(path, gray)
            print(
                f"  wrote {path.name}  "
                f"T=[{monthly[m].min():.1f}, {monthly[m].max():.1f}] C  "
                f"mean={monthly[m].mean():.1f}"
            )

        gray_ann = temperature_to_gray(annual, args.t_gray_min, args.t_gray_max)
        ann_path = out / "Temperature - Annual Mean.png"
        save_gray_png(ann_path, gray_ann)
        print(
            f"  wrote {ann_path.name}  "
            f"T=[{annual.min():.1f}, {annual.max():.1f}] C  mean={annual.mean():.1f}"
        )

        if args.save_npy:
            np.save(out / "temperature_monthly_C.npy", monthly)
            np.save(out / "temperature_annual_C.npy", annual)
            print("  wrote .npy float32 C arrays")

        meta_out = {
            **meta,
            "s0_W_m2": args.s0,
            "obliquity_deg": args.obliquity,
            "max_elev_m": args.max_elev_m,
            "min_elev_m": -abs(float(args.max_elev_m)),
            "lapse_K_per_km": args.lapse,
            "greenhouse_factor": args.greenhouse,
            "heat_transport": args.heat_transport,
            "gray_scale_C": [args.t_gray_min, args.t_gray_max],
            "gray_formula": "gray = clip(round((T_C - T_min)/(T_max - T_min)*255), 0, 255)",
            "months": MONTH_NAMES,
        }
        with open(out / "temperature_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)
            f.write("\n")
    meta_out["wall_timing"] = wall.as_meta()
    with open(out / "temperature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
        f.write("\n")
    print(f"Done. Outputs in {out}")
    print(
        f"Annual land/ocean mean: {meta['t_land_mean_C']:.1f} / "
        f"{meta['t_ocean_mean_C']:.1f} C  "
        f"open_water_mean={meta['open_water_frac_mean']*100:.1f}%"
    )
    print(
        f"Jul−Jan |amp| land/ocean: {meta['amp_jul_jan_land_mean_C']:.1f} / "
        f"{meta['amp_jul_jan_ocean_mean_C']:.1f} C"
    )
    if meta.get("t_ocean_eq_mean_C") is not None:
        print(
            f"Ocean eq(|lat|<10)/polar(|lat|>60): "
            f"{meta['t_ocean_eq_mean_C']:.1f} / {meta['t_ocean_polar_mean_C']:.1f} C"
        )
    fracs = meta["open_water_frac_monthly"]
    print(
        "  open_water by month (%): "
        + ", ".join(f"{100*f:.0f}" for f in fracs)
    )
    print(wall.summary())
    write_assets_json()


if __name__ == "__main__":
    main()
