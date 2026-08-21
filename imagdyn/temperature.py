#!/usr/bin/env python3
"""
Generate 12 monthly temperature maps + annual mean from a grayscale
full-elevation terrain (0.5 = sea level).

Drivers
-------
- Top-of-atmosphere solar irradiance (default S0 = 1361 W/m^2)
- Heat-capacity energy balance with greenhouse OLR = σT^4 / G
- Newtonian heat transport Q_transport = λ (T̄_global − T_local) in Q_abs
- Land / ocean / lake effective heat capacity (depth inertia + latent peak)
- GPU sea-level temperature diffusion (metric 1/cos φ kernels)
- Ocean currents as an energy flux in Q_abs (east-warm / west-cold)
- Elevation lapse applied after sea-level diffusion

Spin-up runs ``spinup_years`` virtual years; only the last 12 months are
returned and written.

Uses PyTorch CUDA when available (conda env tf-gpu).

Outputs grayscale PNGs under graphs/temperature/ (brighter = warmer):
    gray = clip(round((T_C - T_MIN) / (T_MAX - T_MIN) * 255), 0, 255)
default T_MIN=-60 C, T_MAX=+45 C.

Usage::

    python -m imagdyn temperature

Tunables default to ``imagdyn/params.py`` (CLI flags still override).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .io_gray import save_gray_png, write_text_atomic
from .params import (
    CURRENTS,
    ENCODE,
    METRIC_COS_EPS,
    PLANET,
    STEFAN_BOLTZMANN,
    TEMPERATURE,
    temperature_call_kwargs,
    wind_field_kwargs,
)
from .timing import StepTimer, format_duration


SIGMA = STEFAN_BOLTZMANN  # W m^-2 K^-4

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


def release_torch_memory() -> None:
    """Free Python refs' follow-on GPU cache right after a temperature job."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


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


def _lat_1d(lat_deg: torch.Tensor, height: int) -> torch.Tensor:
    lat = lat_deg[:, 0] if lat_deg.ndim >= 2 else lat_deg.reshape(-1)
    return lat[:height]


def cos_lat_weight(lat_deg: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    """``cos(φ)`` clamped to ``[eps, 1]``. ``lat_deg`` is 1-D (H,) or a column of (H, W)."""
    lat1 = lat_deg[:, 0] if lat_deg.ndim >= 2 else lat_deg.reshape(-1)
    return torch.cos(torch.deg2rad(lat1)).clamp(float(eps), 1.0)


def area_weights(lat_deg: torch.Tensor, width: int) -> torch.Tensor:
    """Spherical-cap cell weights ``dA ∝ cos(φ)`` as an (H, W) tensor."""
    w1 = cos_lat_weight(lat_deg, eps=0.0)
    return w1[:, None].expand(-1, int(width))


def area_weighted_mean(
    field: torch.Tensor,
    lat_deg: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Area-weighted mean with ``dA ∝ cos(φ)``.

    ``field`` is (H,), (H, W), or (..., H, W). ``mask`` broadcasts over spatial dims.
    """
    field = torch.as_tensor(field)
    if field.ndim == 1:
        lat1 = _lat_1d(lat_deg.to(device=field.device), field.shape[0])
        wt = cos_lat_weight(lat1, eps=0.0).to(dtype=field.dtype)
        if mask is not None:
            wt = wt * mask.to(device=field.device, dtype=field.dtype).reshape(-1)
        return (field * wt).sum() / wt.sum().clamp_min(1e-6)

    h, w = int(field.shape[-2]), int(field.shape[-1])
    lat1 = _lat_1d(lat_deg.to(device=field.device), h)
    wt = cos_lat_weight(lat1, eps=0.0).to(dtype=field.dtype)[:, None].expand(h, w)
    if mask is not None:
        m = mask.to(device=field.device, dtype=field.dtype)
        while m.ndim < field.ndim:
            m = m.unsqueeze(0)
        num = (field * wt * m).sum(dim=(-2, -1))
        den = (wt * m).sum(dim=(-2, -1)).clamp_min(1e-6)
        return num / den
    num = (field * wt).sum(dim=(-2, -1))
    den = wt.sum().clamp_min(1e-6)
    return num / den


def area_weights_np(lat_deg: np.ndarray, width: int | None = None) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    w1 = np.clip(np.cos(np.deg2rad(lat)), 0.0, 1.0)
    if width is None:
        return w1
    return np.broadcast_to(w1[:, None], (lat.size, int(width))).copy()


def area_weighted_mean_np(
    field: np.ndarray,
    lat_deg: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Numpy ``dA ∝ cos(φ)`` mean of a 1-D profile or 2-D map (optional bool mask)."""
    arr = np.asarray(field, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    wt1 = np.clip(np.cos(np.deg2rad(lat)), 0.0, 1.0)
    if arr.ndim == 1:
        wt = wt1[: arr.size]
        if mask is not None:
            wt = wt * np.asarray(mask, dtype=np.float64).reshape(-1)[: arr.size]
        ok = np.isfinite(arr)
        wt = np.where(ok, wt, 0.0)
        s = float(wt.sum())
        return float((arr * wt).sum() / s) if s > 0.0 else 0.0
    h, w = arr.shape[-2], arr.shape[-1]
    wt = np.broadcast_to(wt1[:h, None], (h, w))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != arr.shape[-2:]:
            m = np.broadcast_to(m, (h, w))
        sel = m & np.isfinite(arr if arr.ndim == 2 else arr.reshape(-1, h, w)[0])
        if arr.ndim != 2:
            raise ValueError("area_weighted_mean_np mask path expects a 2-D field")
        vals = arr[sel]
        wsel = wt[sel]
    else:
        if arr.ndim != 2:
            raise ValueError("area_weighted_mean_np expects a 1-D or 2-D field")
        ok = np.isfinite(arr)
        vals = arr[ok]
        wsel = wt[ok]
    s = float(np.sum(wsel))
    return float(np.sum(vals * wsel) / s) if s > 0.0 else 0.0


def conv1d_lon_grouped(field: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Per-row 1D conv along longitude (circular). ``kernels``: (H, 1, K), K odd."""
    h, _w = field.shape
    r = int(kernels.shape[-1]) // 2
    if r <= 0:
        return field
    padded = torch.cat([field[:, -r:], field, field[:, :r]], dim=1)
    return F.conv1d(padded[None], kernels, groups=h)[0]


def conv1d_lat_replicate(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Uniform 1D conv along latitude (replicate pad). ``kernel`` is (K,) or (1, 1, K)."""
    k = kernel.reshape(-1)
    r = int(k.numel()) // 2
    if r <= 0:
        return field
    pad = torch.cat([field[:1].expand(r, -1), field, field[-1:].expand(r, -1)], dim=0)
    out = F.conv1d(pad.T[:, None, :], k.reshape(1, 1, -1))
    return out[:, 0, :].T


def _ew_radius_cap(width: int, sigma_or_rx_max: float) -> int:
    return max(1, min(max(1, int(width) // 6), max(1, int(math.ceil(float(sigma_or_rx_max))))))


def lon_sigma_px(
    lat_deg: torch.Tensor,
    sigma_eq: float,
    *,
    eps: float = METRIC_COS_EPS,
) -> torch.Tensor:
    """Pixel EW Gaussian width ``σ_eq / max(cos φ, ε)`` per latitude row."""
    cos = torch.cos(torch.deg2rad(lat_deg)).abs().clamp(float(eps), 1.0)
    return float(sigma_eq) / cos


def smooth_lonlat_metric(
    field: torch.Tensor,
    lat_deg: torch.Tensor,
    sigma_eq_px: float,
    *,
    eps: float = METRIC_COS_EPS,
) -> torch.Tensor:
    """
    Separable metric Gaussian: NS uses ``σ_eq``; EW uses ``σ_eq / max(cos φ, ε)``.

    Longitude wraps; latitude replicates. Replaces a single isotropic 2D conv.
    """
    if sigma_eq_px <= 0:
        return field
    h, w = field.shape
    lat1 = _lat_1d(lat_deg.to(device=field.device, dtype=field.dtype), h)
    sig_eq = max(float(sigma_eq_px), 0.35)

    max_r_ns = max(1, h // 2)
    sig_ns = min(sig_eq, max_r_ns / 3.0)
    r_ns = min(max(1, int(math.ceil(3.0 * sig_ns))), max_r_ns)
    x_ns = torch.arange(-r_ns, r_ns + 1, device=field.device, dtype=field.dtype)
    g_ns = torch.exp(-0.5 * (x_ns / max(sig_ns, 0.35)) ** 2)
    g_ns = g_ns / g_ns.sum().clamp_min(1e-12)
    out = conv1d_lat_replicate(field, g_ns)

    sig_row = lon_sigma_px(lat1, sig_eq, eps=eps)
    r_ew = _ew_radius_cap(w, 3.0 * float(sig_row.max().item()))
    x = torch.arange(-r_ew, r_ew + 1, device=field.device, dtype=field.dtype)
    sig = sig_row.clamp(min=0.35)[:, None]
    kernels = torch.exp(-0.5 * (x[None, :] / sig) ** 2)
    kernels = kernels / kernels.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return conv1d_lon_grouped(out, kernels[:, None, :])


def box_filter_lonlat_metric(
    field: torch.Tensor,
    lat_deg: torch.Tensor,
    ry: int,
    rx_eq: int,
    *,
    eps: float = METRIC_COS_EPS,
) -> torch.Tensor:
    """Separable mean filter: NS radius ``ry``; EW radius ``rx_eq / max(cos φ, ε)`` per row."""
    h, w = field.shape
    ry_i = max(int(ry), 0)
    rx_i = max(int(rx_eq), 0)
    out = field
    if ry_i > 0:
        k = 2 * ry_i + 1
        kernel = torch.ones(k, device=field.device, dtype=field.dtype) / float(k)
        out = conv1d_lat_replicate(out, kernel)
    if rx_i <= 0:
        return out
    lat1 = _lat_1d(lat_deg.to(device=field.device, dtype=field.dtype), h)
    cos = torch.cos(torch.deg2rad(lat1)).abs().clamp(float(eps), 1.0)
    rx_row = (float(rx_i) / cos).round().clamp(min=1.0)
    r_ew = _ew_radius_cap(w, float(rx_row.max().item()))
    rx_row = rx_row.clamp(max=float(r_ew))
    x = torch.arange(-r_ew, r_ew + 1, device=field.device, dtype=field.dtype)
    kernels = (x.abs()[None, :] <= rx_row[:, None]).to(dtype=field.dtype)
    kernels = kernels / kernels.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return conv1d_lon_grouped(out, kernels[:, None, :])


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


def soft_step(x: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    """Smooth 0->1 step; larger scale = softer."""
    return torch.sigmoid((x - center) / max(scale, 1e-3))


MONTH_DT_S = 365.25 / 12.0 * 86400.0
WATER_DENSITY = 1000.0  # kg m^-3
LATENT_HEAT_FUSION = 3.34e5  # J kg^-1
T_K_FLOOR = 50.0


def depth_inertia(
    z_m: torch.Tensor,
    inertia_shallow: float = TEMPERATURE.inertia_shallow,
    inertia_deep: float = TEMPERATURE.inertia_deep,
    mix_depth_m: float = TEMPERATURE.mix_depth_m,
) -> torch.Tensor:
    """I(z) = I_shallow + (I_deep - I_shallow) * (1 - exp(-z / d0))."""
    d0 = max(float(mix_depth_m), 1e-6)
    z = z_m.clamp(min=0.0)
    i0 = float(inertia_shallow)
    i1 = float(inertia_deep)
    return i0 + (i1 - i0) * (1.0 - torch.exp(-z / d0))


def latent_peak_J_m2_K(
    latent_ice_m: float = TEMPERATURE.latent_ice_m,
    delta_c: float = TEMPERATURE.freeze_latent_delta_C,
) -> float:
    """C_peak = L ρ h / (δ √(2π)) for a Gaussian virtual heat capacity."""
    d = max(float(delta_c), 1e-6)
    h = max(float(latent_ice_m), 0.0)
    return LATENT_HEAT_FUSION * WATER_DENSITY * h / (d * math.sqrt(2.0 * math.pi))


def latent_heat_capacity(
    t_c: torch.Tensor,
    freeze_c: torch.Tensor | float,
    delta_c: float = TEMPERATURE.freeze_latent_delta_C,
    peak: float | None = None,
) -> torch.Tensor:
    """Gaussian virtual heat capacity peaked at the freeze point."""
    if peak is None:
        peak = latent_peak_J_m2_K(delta_c=delta_c)
    d = max(float(delta_c), 1e-6)
    x = (t_c - freeze_c) / d
    return float(peak) * torch.exp(-0.5 * x * x)


def ice_albedo_weight(
    t_c: torch.Tensor,
    freeze_c: torch.Tensor | float,
    soft_c: float = TEMPERATURE.ice_albedo_soft_C,
) -> torch.Tensor:
    """Sigmoid ice fraction: 0.5 at T = T_freeze, →1 as T drops."""
    s = max(float(soft_c), 1e-3)
    return torch.sigmoid((freeze_c - t_c) / s)


def current_flux_W_m2(
    t_k: torch.Tensor,
    dt_curr_c: torch.Tensor,
    greenhouse_factor: float,
) -> torch.Tensor:
    """Q_curr = (4 σ T^3 / G) ΔT so linearized equilibrium offset ≈ ΔT."""
    g = max(float(greenhouse_factor), 1e-6)
    t = t_k.clamp(min=T_K_FLOOR)
    return (4.0 * SIGMA * t.pow(3) / g) * dt_curr_c


def transport_flux_W_m2(
    t_c: torch.Tensor,
    lat_deg: torch.Tensor,
    transport_lambda: float,
) -> torch.Tensor:
    """Q_transport = λ (T̄_global − T_local); λ in W/m²/K. Area-weighted mean is energy-conserving."""
    lam = float(transport_lambda)
    if lam == 0.0:
        return torch.zeros_like(t_c)
    t_bar = area_weighted_mean(t_c, lat_deg)
    return lam * (t_bar - t_c)


def implicit_energy_step(
    t_c: torch.Tensor,
    q_abs: torch.Tensor,
    c_eff: torch.Tensor,
    dt_s: float = MONTH_DT_S,
    greenhouse_factor: float = TEMPERATURE.greenhouse_factor,
) -> torch.Tensor:
    """T_new = T + (Q_abs - σT^4/G) / (C/Δt + 4σT^3/G), T in °C on the grid."""
    g = max(float(greenhouse_factor), 1e-6)
    t_k = (t_c + 273.15).clamp(min=T_K_FLOOR)
    olr = SIGMA * t_k.pow(4) / g
    dolr = 4.0 * SIGMA * t_k.pow(3) / g
    c = c_eff.clamp(min=1.0)
    t_new_k = t_k + (q_abs - olr) / (c / float(dt_s) + dolr)
    return t_new_k - 273.15


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


def inland_lake_area_km2(
    land: torch.Tensor | np.ndarray,
    *,
    planet_radius_km: float = PLANET.radius_km,
    max_area_km2: float = TEMPERATURE.lake_max_area_km2,
) -> torch.Tensor:
    """
    Per-cell spherical area (km²) of inland lakes ≤ ``max_area_km2``; 0 elsewhere.

    Water connected components (8-neighbour, longitude wraps). The largest
    component is the world ocean. Cell area is ``dy · dx_eq · cos(φ)``.
    """
    from scipy import ndimage

    if isinstance(land, torch.Tensor):
        device = land.device
        land_np = land.detach().cpu().numpy().astype(bool)
    else:
        device = torch.device("cpu")
        land_np = np.asarray(land, dtype=bool)

    water = ~land_np
    h, w = water.shape
    z = torch.zeros((h, w), device=device, dtype=torch.float32)
    if not water.any():
        return z

    struct = ndimage.generate_binary_structure(2, 2)
    labeled, nlab = ndimage.label(water, structure=struct)
    if nlab == 0:
        return z

    # Merge labels that touch across the antimeridian (lon wrap).
    left = labeled[:, 0]
    right = labeled[:, -1]
    touch = (left > 0) & (right > 0) & (left != right)
    if np.any(touch):
        parent = np.arange(nlab + 1, dtype=np.int32)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        for a, b in zip(left[touch].tolist(), right[touch].tolist()):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        remap = np.zeros(nlab + 1, dtype=np.int32)
        for i in range(1, nlab + 1):
            remap[i] = find(i)
        uniq = {0: 0}
        next_id = 1
        for i in range(1, nlab + 1):
            r = int(remap[i])
            if r not in uniq:
                uniq[r] = next_id
                next_id += 1
            remap[i] = uniq[r]
        labeled = remap[labeled]
        nlab = next_id - 1

    lat_np = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * (180.0 / h)
    cos_row = np.clip(np.cos(np.deg2rad(lat_np)), 0.0, 1.0)
    cell_w = np.broadcast_to(cos_row[:, None], (h, w))
    area_w = np.bincount(
        labeled.ravel(), weights=cell_w.ravel(), minlength=nlab + 1
    )
    area_w[0] = 0.0
    if nlab == 0 or float(area_w.max()) == 0.0:
        return z

    ocean_id = int(np.argmax(area_w))
    km_per_deg = (np.pi * float(planet_radius_km)) / 180.0
    dy_km = (180.0 / h) * km_per_deg
    dx_eq = (360.0 / w) * km_per_deg
    area_km2 = area_w * float(dy_km * dx_eq)
    area_map = area_km2[labeled]
    keep = (labeled != ocean_id) & (area_map > 0.0) & (area_map <= float(max_area_km2))
    area_map = np.where(keep, area_map, 0.0).astype(np.float32)
    return torch.as_tensor(area_map, device=device, dtype=torch.float32)


def small_inland_lake_mask(
    land: torch.Tensor | np.ndarray,
    *,
    planet_radius_km: float = PLANET.radius_km,
    max_area_km2: float = TEMPERATURE.lake_max_area_km2,
) -> torch.Tensor:
    """True on inland-lake cells whose spherical area is ≤ ``max_area_km2``."""
    return inland_lake_area_km2(
        land, planet_radius_km=planet_radius_km, max_area_km2=max_area_km2
    ) > 0


def lake_inertia_from_area_km2(
    area_km2: torch.Tensor,
    *,
    small_km2: float = TEMPERATURE.lake_small_area_km2,
    large_km2: float = TEMPERATURE.lake_max_area_km2,
    inertia_small: float = TEMPERATURE.lake_inertia,
    inertia_large: float = 1.0,
) -> torch.Tensor:
    """Linear coefficient: ``inertia_small`` at ``small_km2``, ``inertia_large`` (≤1) at ``large_km2``."""
    lo = min(float(small_km2), float(large_km2))
    hi = max(float(small_km2), float(large_km2))
    span = max(hi - lo, 1e-6)
    t = ((area_km2 - lo) / span).clamp(0.0, 1.0)
    i0 = float(np.clip(inertia_small, 0.0, 1.0))
    i1 = float(np.clip(inertia_large, 0.0, 1.0))
    return i0 + t * (i1 - i0)


def diffuse_field(
    field: torch.Tensor,
    lat: torch.Tensor,
    *,
    planet_radius_km: float,
    length_km: float,
    passes: int = 6,
) -> torch.Tensor:
    """Repeated metric mean filters ≈ heat diffusion (no 0–1 clamp)."""
    h, w = field.shape
    x = field
    n = max(1, int(passes))
    r_km = max(float(length_km) * (3.0 / float(n)) ** 0.5, 20.0)

    for _ in range(n):
        ry, rx = kernel_radius_px(h, w, r_km, planet_radius_km)
        ry = min(max(ry, 1), max(3, h // 4))
        rx = min(max(rx, 1), max(3, w // 4))
        x = box_filter_lonlat_metric(x, lat, ry, rx)
    return x


def anti_alias_coast(
    t_c: torch.Tensor,
    land: torch.Tensor,
    lat: torch.Tensor,
    *,
    aa_blend_px: int,
) -> torch.Tensor:
    """1-ish pixel coast mix: edge = 4s(1-s) against a metric box blur."""
    px = max(int(aa_blend_px), 0)
    if px <= 0:
        return t_c
    soft = box_filter_lonlat_metric(land.float(), lat, px, px).clamp(0.0, 1.0)
    edge = (4.0 * soft * (1.0 - soft)).clamp(0.0, 1.0)
    blur = box_filter_lonlat_metric(t_c, lat, px, px)
    return t_c * (1.0 - edge) + blur * edge


def synthesize_temperatures(
    elev01: np.ndarray,
    land_np: np.ndarray,
    *,
    device: torch.device,
    s0: float = PLANET.s0,
    obliquity_deg: float = PLANET.obliquity_deg,
    max_elev_m: float = PLANET.max_elev_m,
    lapse_k_per_km: float = PLANET.lapse_k_per_km,
    planet_radius_km: float = PLANET.radius_km,
    maritime_e_fold_km: float = TEMPERATURE.maritime_e_fold_km,
    greenhouse_factor: float = TEMPERATURE.greenhouse_factor,
    transport_lambda: float = TEMPERATURE.transport_lambda,
    albedo_ocean: float = TEMPERATURE.albedo_ocean,
    albedo_land: float = TEMPERATURE.albedo_land,
    albedo_ice: float = TEMPERATURE.albedo_ice,
    maritime_diffuse_passes: int = TEMPERATURE.maritime_diffuse_passes,
    heat_capacity_land: float = TEMPERATURE.heat_capacity_land,
    heat_capacity_ocean: float = TEMPERATURE.heat_capacity_ocean,
    inertia_shallow: float = TEMPERATURE.inertia_shallow,
    inertia_deep: float = TEMPERATURE.inertia_deep,
    mix_depth_m: float = TEMPERATURE.mix_depth_m,
    lake_inertia: float = TEMPERATURE.lake_inertia,
    lake_small_area_km2: float = TEMPERATURE.lake_small_area_km2,
    lake_max_area_km2: float = TEMPERATURE.lake_max_area_km2,
    freeze_land_C: float = TEMPERATURE.freeze_land_C,
    freeze_ocean_C: float = TEMPERATURE.freeze_ocean_C,
    freeze_latent_delta_C: float = TEMPERATURE.freeze_latent_delta_C,
    latent_ice_m: float = TEMPERATURE.latent_ice_m,
    ice_albedo_soft_C: float = TEMPERATURE.ice_albedo_soft_C,
    spinup_years: int = TEMPERATURE.spinup_years,
    t_init_K: float = TEMPERATURE.t_init_K,
    aa_blend_px: int = TEMPERATURE.aa_blend_px,
    currents: bool = TEMPERATURE.currents,
    current_warm_delta_C: float = CURRENTS.warm_delta_C,
    current_cold_delta_C: float = CURRENTS.cold_delta_C,
    current_peak_lat_deg: float = CURRENTS.peak_lat_deg,
    current_lat_sigma_deg: float = CURRENTS.lat_sigma_deg,
    current_reach_km: float = CURRENTS.reach_km,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns last-year monthly (12,H,W) float32 °C, annual (H,W) float32 °C, meta.

    Spin-up runs ``spinup_years`` virtual years of month-steps. Only the last
    12 months are returned. Currents and Newtonian heat transport enter
    ``Q_abs`` as energy fluxes.
    """
    job_timer = StepTimer("synthesize")
    try:
        return _synthesize_temperatures_impl(
            elev01,
            land_np,
            device=device,
            s0=s0,
            obliquity_deg=obliquity_deg,
            max_elev_m=max_elev_m,
            lapse_k_per_km=lapse_k_per_km,
            planet_radius_km=planet_radius_km,
            maritime_e_fold_km=maritime_e_fold_km,
            greenhouse_factor=greenhouse_factor,
            transport_lambda=transport_lambda,
            albedo_ocean=albedo_ocean,
            albedo_land=albedo_land,
            albedo_ice=albedo_ice,
            maritime_diffuse_passes=maritime_diffuse_passes,
            heat_capacity_land=heat_capacity_land,
            heat_capacity_ocean=heat_capacity_ocean,
            inertia_shallow=inertia_shallow,
            inertia_deep=inertia_deep,
            mix_depth_m=mix_depth_m,
            lake_inertia=lake_inertia,
            lake_small_area_km2=lake_small_area_km2,
            lake_max_area_km2=lake_max_area_km2,
            freeze_land_C=freeze_land_C,
            freeze_ocean_C=freeze_ocean_C,
            freeze_latent_delta_C=freeze_latent_delta_C,
            latent_ice_m=latent_ice_m,
            ice_albedo_soft_C=ice_albedo_soft_C,
            spinup_years=spinup_years,
            t_init_K=t_init_K,
            aa_blend_px=aa_blend_px,
            currents=currents,
            current_warm_delta_C=current_warm_delta_C,
            current_cold_delta_C=current_cold_delta_C,
            current_peak_lat_deg=current_peak_lat_deg,
            current_lat_sigma_deg=current_lat_sigma_deg,
            current_reach_km=current_reach_km,
            job_timer=job_timer,
        )
    finally:
        release_torch_memory()



def _synthesize_temperatures_impl(
    elev01: np.ndarray,
    land_np: np.ndarray,
    *,
    device: torch.device,
    s0: float,
    obliquity_deg: float,
    max_elev_m: float,
    lapse_k_per_km: float,
    planet_radius_km: float,
    maritime_e_fold_km: float,
    greenhouse_factor: float,
    transport_lambda: float,
    albedo_ocean: float,
    albedo_land: float,
    albedo_ice: float,
    maritime_diffuse_passes: int,
    heat_capacity_land: float,
    heat_capacity_ocean: float,
    inertia_shallow: float,
    inertia_deep: float,
    mix_depth_m: float,
    lake_inertia: float,
    lake_small_area_km2: float,
    lake_max_area_km2: float,
    freeze_land_C: float,
    freeze_ocean_C: float,
    freeze_latent_delta_C: float,
    latent_ice_m: float,
    ice_albedo_soft_C: float,
    spinup_years: int,
    t_init_K: float,
    aa_blend_px: int,
    currents: bool,
    current_warm_delta_C: float,
    current_cold_delta_C: float,
    current_peak_lat_deg: float,
    current_lat_sigma_deg: float,
    current_reach_km: float,
    job_timer: StepTimer,
) -> tuple[np.ndarray, np.ndarray, dict]:
    elev = torch.from_numpy(elev01).to(device=device, dtype=torch.float32)
    land = torch.from_numpy(land_np.astype(np.bool_)).to(device=device)
    h, w = elev.shape
    lat = latitude_grid(h, device)

    elev_land = torch.where(
        land, (elev - 0.5) / 0.5 * max_elev_m, torch.zeros_like(elev)
    ).clamp(0.0, max_elev_m)
    depth_m = torch.where(
        ~land, (0.5 - elev) / 0.5 * max_elev_m, torch.zeros_like(elev)
    ).clamp(0.0, max_elev_m)
    lapse = float(lapse_k_per_km) * (elev_land / 1000.0)

    lake_area = inland_lake_area_km2(
        land,
        planet_radius_km=planet_radius_km,
        max_area_km2=lake_max_area_km2,
    )
    is_lake = lake_area > 0
    world_ocean = (~land) & (~is_lake)
    lake_coeff = lake_inertia_from_area_km2(
        lake_area,
        small_km2=lake_small_area_km2,
        large_km2=lake_max_area_km2,
        inertia_small=lake_inertia,
        inertia_large=1.0,
    )
    i_z = depth_inertia(
        depth_m,
        inertia_shallow=inertia_shallow,
        inertia_deep=inertia_deep,
        mix_depth_m=mix_depth_m,
    )
    c_water = float(heat_capacity_ocean) * i_z
    c_base = torch.where(land, torch.full_like(elev, float(heat_capacity_land)), c_water)
    c_base = torch.where(is_lake, lake_coeff * c_water, c_base)

    freeze_map = torch.where(
        land | is_lake,
        torch.full_like(elev, float(freeze_land_C)),
        torch.full_like(elev, float(freeze_ocean_C)),
    )
    c_peak = latent_peak_J_m2_K(latent_ice_m, freeze_latent_delta_C)
    bare_albedo = torch.where(
        land,
        torch.full_like(elev, float(albedo_land)),
        torch.full_like(elev, float(albedo_ocean)),
    )

    land_soft = box_filter_lonlat_metric(land.float(), lat, 2, 2).clamp(0.0, 1.0)

    declinations = solar_declination_deg(MID_MONTH_DOY, obliquity_deg)
    q_months = torch.empty((12, h, w), device=device, dtype=torch.float32)
    for m, dec in enumerate(declinations):
        q_row = daily_mean_toa_insolation(lat, float(dec), s0=s0)
        q_months[m] = q_row[:, None].expand(h, w)

    dt_curr = torch.zeros((h, w), device=device, dtype=torch.float32)
    currents_meta: dict | None = None
    if currents:
        from .currents import OceanCurrentFilter

        print("  ocean currents → Q_abs flux…", flush=True)
        curr_filt = OceanCurrentFilter(
            peak_lat_deg=current_peak_lat_deg,
            lat_sigma_deg=current_lat_sigma_deg,
            warm_delta_C=current_warm_delta_C,
            cold_delta_C=current_cold_delta_C,
            reach_km=current_reach_km,
            planet_radius_km=planet_radius_km,
            land_bleed=0.0,
            enabled=True,
        )
        curr_corr = curr_filt.compute(land, lat)
        dt_curr = curr_corr.temperature_C * world_ocean.float()
        currents_meta = dict(curr_corr.meta)
        del curr_corr, curr_filt
        print(
            f"    currents dT ocean "
            f"[{currents_meta.get('dT_min_C'):.2f}, {currents_meta.get('dT_max_C'):.2f}] C",
            flush=True,
        )

    n_years = max(1, int(spinup_years))
    n_steps = n_years * 12
    t_c = torch.full(
        (h, w), float(t_init_K) - 273.15, device=device, dtype=torch.float32
    )
    monthly = torch.empty((12, h, w), device=device, dtype=torch.float32)
    month_timer = StepTimer("month", total_steps=n_steps)

    print(f"  spin-up {n_years} year(s) ({n_steps} month_steps)…", flush=True)
    for step in range(n_steps):
        m = step % 12
        year = step // 12 + 1
        step_name = f"y{year}m{m + 1}"
        month_timer.begin(step_name)

        ice_w = ice_albedo_weight(t_c, freeze_map, ice_albedo_soft_C)
        albedo = (bare_albedo + ice_w * (float(albedo_ice) - bare_albedo)).clamp(0.0, 0.95)
        c_lat = latent_heat_capacity(
            t_c, freeze_map, freeze_latent_delta_C, peak=c_peak
        )
        c_eff = c_base + c_lat
        t_k = (t_c + 273.15).clamp(min=T_K_FLOOR)
        q_curr = current_flux_W_m2(t_k, dt_curr, greenhouse_factor)
        q_trans = transport_flux_W_m2(t_c, lat, transport_lambda)
        q_abs = (1.0 - albedo) * q_months[m] + q_curr + q_trans
        t_c = implicit_energy_step(
            t_c, q_abs, c_eff, MONTH_DT_S, greenhouse_factor
        )
        t_sl = t_c + lapse
        t_sl = diffuse_field(
            t_sl,
            lat,
            planet_radius_km=planet_radius_km,
            length_km=maritime_e_fold_km,
            passes=maritime_diffuse_passes,
        )
        t_c = t_sl - lapse
        if step >= n_steps - 12:
            monthly[m] = t_c
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = month_timer.end() or 0.0
        print(
            f"    year {year}/{n_years}  month {m + 1}/12  "
            f"{month_timer.progress_line(last_name=step_name, last_dt=dt)}",
            flush=True,
        )

    for m in range(12):
        monthly[m] = anti_alias_coast(
            monthly[m], land, lat, aa_blend_px=aa_blend_px
        )

    annual = monthly.mean(dim=0)
    ice_w_m = ice_albedo_weight(monthly, freeze_map[None], ice_albedo_soft_C)
    open_water = (~land).float()[None] * (1.0 - ice_w_m)
    open_frac_monthly = [
        float(area_weighted_mean(open_water[m], lat).item()) for m in range(12)
    ]

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
        "q_annual_mean_W_m2": float(area_weighted_mean(q_months.mean(dim=0), lat).item()),
        "q_annual_max_W_m2": float(q_months.mean(dim=0).max().item()),
        "t_annual_mean_C": float(area_weighted_mean(annual, lat).item()),
        "t_annual_min_C": float(annual.min().item()),
        "t_annual_max_C": float(annual.max().item()),
        "t_land_mean_C": float(area_weighted_mean(annual, lat, mask=land).item()),
        "t_ocean_mean_C": float(area_weighted_mean(annual, lat, mask=~land).item()),
        "t_ocean_eq_mean_C": (
            float(area_weighted_mean(annual, lat, mask=eq).item()) if bool(eq.any()) else None
        ),
        "t_ocean_polar_mean_C": (
            float(area_weighted_mean(annual, lat, mask=pol).item()) if bool(pol.any()) else None
        ),
        "amp_jul_jan_land_mean_C": (
            float(area_weighted_mean(amp, lat, mask=land).item()) if bool(land.any()) else 0.0
        ),
        "amp_jul_jan_ocean_mean_C": (
            float(area_weighted_mean(amp, lat, mask=~land).item()) if bool((~land).any()) else 0.0
        ),
        "open_water_frac_mean": float(area_weighted_mean(open_water.mean(dim=0), lat).item()),
        "open_water_frac_monthly": open_frac_monthly,
        "elev_land_max_m": float(elev_land[land].max().item()) if bool(land.any()) else 0.0,
        "freeze_land_C": freeze_land_C,
        "freeze_ocean_C": freeze_ocean_C,
        "freeze_latent_delta_C": freeze_latent_delta_C,
        "spinup_years": n_years,
        "month_steps": n_steps,
        "transport_lambda": float(transport_lambda),
        "maritime_e_fold_km": maritime_e_fold_km,
        "maritime_model": "gpu_diffusion_sea_level_T",
        "maritime_diffuse_passes": maritime_diffuse_passes,
        "heat_capacity_land": heat_capacity_land,
        "heat_capacity_ocean": heat_capacity_ocean,
        "inertia_shallow": inertia_shallow,
        "inertia_deep": inertia_deep,
        "mix_depth_m": mix_depth_m,
        "lake_inertia": lake_inertia,
        "lake_small_area_km2": lake_small_area_km2,
        "lake_max_area_km2": lake_max_area_km2,
        "lake_inertia_note": (
            f"spherical area dA∝cos(φ); coeff {float(np.clip(lake_inertia, 0.0, 1.0)):.2f}"
            f"@{float(lake_small_area_km2):.0f}km2 → 1.00@{float(lake_max_area_km2):.0f}km2"
        ),
        "small_inland_lake_pct": float(
            area_weighted_mean(is_lake.float(), lat).item() * 100.0
        ),
        "latent_ice_m": latent_ice_m,
        "ice_albedo_soft_C": ice_albedo_soft_C,
        "t_init_K": t_init_K,
        "aa_blend_px": aa_blend_px,
        "land_soft_mean": float(area_weighted_mean(land_soft, lat).item()),
        "ocean_currents": currents_meta,
        "timing": timing,
    }

    monthly_np = monthly.detach().cpu().numpy().astype(np.float32)
    annual_np = annual.detach().cpu().numpy().astype(np.float32)
    del (
        elev,
        land,
        lat,
        elev_land,
        depth_m,
        lapse,
        lake_area,
        is_lake,
        world_ocean,
        lake_coeff,
        i_z,
        c_water,
        c_base,
        freeze_map,
        bare_albedo,
        land_soft,
        q_months,
        dt_curr,
        t_c,
        monthly,
        annual,
        ice_w_m,
        open_water,
        lat2,
        eq,
        pol,
        amp,
    )
    release_torch_memory()
    return monthly_np, annual_np, meta


def temperature_to_gray(t_c: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    g = (t_c - t_min) / (t_max - t_min) * 255.0
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


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
    p.add_argument("--s0", type=float, default=PLANET.s0)
    p.add_argument("--obliquity", type=float, default=PLANET.obliquity_deg)
    p.add_argument("--max-elev-m", type=float, default=PLANET.max_elev_m)
    p.add_argument("--lapse", type=float, default=PLANET.lapse_k_per_km)
    p.add_argument("--radius-km", type=float, default=PLANET.radius_km)
    p.add_argument("--maritime-efold-km", type=float, default=TEMPERATURE.maritime_e_fold_km)
    p.add_argument(
        "--greenhouse",
        type=float,
        default=TEMPERATURE.greenhouse_factor,
        help="G in OLR = σT^4 / G (shortwave is not multiplied)",
    )
    p.add_argument(
        "--transport-lambda",
        type=float,
        default=TEMPERATURE.transport_lambda,
        help="λ in Q_transport = λ (T̄_global − T_local) (W/m²/K)",
    )
    p.add_argument("--albedo-ocean", type=float, default=TEMPERATURE.albedo_ocean)
    p.add_argument("--albedo-land", type=float, default=TEMPERATURE.albedo_land)
    p.add_argument("--albedo-ice", type=float, default=TEMPERATURE.albedo_ice)
    p.add_argument(
        "--maritime-diffuse-passes",
        type=int,
        default=TEMPERATURE.maritime_diffuse_passes,
        help="Repeated GPU mean-filter passes for sea-level T diffusion",
    )
    p.add_argument(
        "--heat-capacity-land",
        type=float,
        default=TEMPERATURE.heat_capacity_land,
        help="Land effective heat capacity (J/m²/K)",
    )
    p.add_argument(
        "--heat-capacity-ocean",
        type=float,
        default=TEMPERATURE.heat_capacity_ocean,
        help="Deep-ocean reference heat capacity (J/m²/K)",
    )
    p.add_argument("--inertia-shallow", type=float, default=TEMPERATURE.inertia_shallow)
    p.add_argument("--inertia-deep", type=float, default=TEMPERATURE.inertia_deep)
    p.add_argument("--mix-depth-m", type=float, default=TEMPERATURE.mix_depth_m)
    p.add_argument(
        "--lake-inertia",
        type=float,
        default=TEMPERATURE.lake_inertia,
        help="Inland-lake heat-capacity coefficient at --lake-small-area-km2 (ramps to 1 at --lake-max-area-km2)",
    )
    p.add_argument(
        "--lake-small-area-km2",
        type=float,
        default=TEMPERATURE.lake_small_area_km2,
        help="Inland-lake spherical area (km²) at which coefficient = --lake-inertia",
    )
    p.add_argument(
        "--lake-max-area-km2",
        type=float,
        default=TEMPERATURE.lake_max_area_km2,
        help="Max inland-lake spherical area (km²); coefficient → 1",
    )
    p.add_argument("--freeze-land-c", type=float, default=TEMPERATURE.freeze_land_C)
    p.add_argument("--freeze-ocean-c", type=float, default=TEMPERATURE.freeze_ocean_C)
    p.add_argument("--freeze-latent-delta-c", type=float, default=TEMPERATURE.freeze_latent_delta_C)
    p.add_argument("--latent-ice-m", type=float, default=TEMPERATURE.latent_ice_m)
    p.add_argument("--ice-albedo-soft-c", type=float, default=TEMPERATURE.ice_albedo_soft_C)
    p.add_argument("--spinup-years", type=int, default=TEMPERATURE.spinup_years)
    p.add_argument("--t-init-k", type=float, default=TEMPERATURE.t_init_K)
    p.add_argument("--aa-blend-px", type=int, default=TEMPERATURE.aa_blend_px)
    p.add_argument(
        "--currents",
        action=argparse.BooleanOptionalAction,
        default=TEMPERATURE.currents,
        help="Add east-warm / west-cold current ΔT as Q_abs flux (default: on)",
    )
    p.add_argument("--current-warm-c", type=float, default=CURRENTS.warm_delta_C, help="East-coast warm ΔT (°C)")
    p.add_argument("--current-cold-c", type=float, default=CURRENTS.cold_delta_C, help="West-coast cold ΔT (°C)")
    p.add_argument("--current-peak-lat", type=float, default=CURRENTS.peak_lat_deg, help="|lat| of max current contrast")
    p.add_argument("--current-lat-sigma", type=float, default=CURRENTS.lat_sigma_deg, help="Gaussian σ (° lat) for mid-lat weight")
    p.add_argument("--current-reach-km", type=float, default=CURRENTS.reach_km, help="Current footprint diffusion radius (km)")
    p.add_argument("--t-gray-min", type=float, default=ENCODE.t_gray_min)
    p.add_argument("--t-gray-max", type=float, default=ENCODE.t_gray_max)
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument(
        "--wind",
        action=argparse.BooleanOptionalAction,
        default=TEMPERATURE.sync_wind,
        help="After temperature, synthesize wind/pressure from T tensors (default: on)",
    )
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

    h_g = elev.shape[0]
    lat_g = 90.0 - (np.arange(h_g, dtype=np.float64) + 0.5) * (180.0 / h_g)
    print(
        f"Grid {elev.shape[1]}x{elev.shape[0]}  "
        f"land={100.0 * area_weighted_mean_np(land.astype(np.float64), lat_g):.1f}%"
    )
    print(
        f"S0={args.s0} W/m^2  obliquity=+/-{args.obliquity} deg  "
        f"max_elev={args.max_elev_m} m  spinup={args.spinup_years}y  "
        f"λ_transport={args.transport_lambda} W/m^2/K  "
        f"freeze land/ocean={args.freeze_land_c}/{args.freeze_ocean_c} C"
    )
    if args.currents:
        print(
            f"Ocean currents in Q_abs  east=+{args.current_warm_c:.1f}C  "
            f"west={args.current_cold_c:.1f}C  peak|lat|={args.current_peak_lat}°"
        )
    else:
        print("Ocean currents OFF")

    with wall.step("synthesize"):
        monthly, annual, meta = synthesize_temperatures(
            elev,
            land,
            device=device,
            **temperature_call_kwargs(
                s0=args.s0,
                obliquity_deg=args.obliquity,
                max_elev_m=args.max_elev_m,
                lapse_k_per_km=args.lapse,
                planet_radius_km=args.radius_km,
                maritime_e_fold_km=args.maritime_efold_km,
                greenhouse_factor=args.greenhouse,
                transport_lambda=args.transport_lambda,
                albedo_ocean=args.albedo_ocean,
                albedo_land=args.albedo_land,
                albedo_ice=args.albedo_ice,
                maritime_diffuse_passes=args.maritime_diffuse_passes,
                heat_capacity_land=args.heat_capacity_land,
                heat_capacity_ocean=args.heat_capacity_ocean,
                inertia_shallow=args.inertia_shallow,
                inertia_deep=args.inertia_deep,
                mix_depth_m=args.mix_depth_m,
                lake_inertia=args.lake_inertia,
                lake_small_area_km2=args.lake_small_area_km2,
                lake_max_area_km2=args.lake_max_area_km2,
                freeze_land_C=args.freeze_land_c,
                freeze_ocean_C=args.freeze_ocean_c,
                freeze_latent_delta_C=args.freeze_latent_delta_c,
                latent_ice_m=args.latent_ice_m,
                ice_albedo_soft_C=args.ice_albedo_soft_c,
                spinup_years=args.spinup_years,
                t_init_K=args.t_init_k,
                aa_blend_px=args.aa_blend_px,
                currents=bool(args.currents),
                current_warm_delta_C=args.current_warm_c,
                current_cold_delta_C=args.current_cold_c,
                current_peak_lat_deg=args.current_peak_lat,
                current_lat_sigma_deg=args.current_lat_sigma,
                current_reach_km=args.current_reach_km,
            ),
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
                f"mean={area_weighted_mean_np(monthly[m], lat_g):.1f}"
            )

        gray_ann = temperature_to_gray(annual, args.t_gray_min, args.t_gray_max)
        ann_path = out / "Temperature - Annual Mean.png"
        save_gray_png(ann_path, gray_ann)
        print(
            f"  wrote {ann_path.name}  "
            f"T=[{annual.min():.1f}, {annual.max():.1f}] C  "
            f"mean={area_weighted_mean_np(annual, lat_g):.1f}"
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
            "gray_scale_C": [args.t_gray_min, args.t_gray_max],
            "gray_formula": "gray = clip(round((T_C - T_min)/(T_max - T_min)*255), 0, 255)",
            "months": MONTH_NAMES,
        }
        write_text_atomic(
            out / "temperature_meta.json",
            json.dumps(meta_out, indent=2) + "\n",
        )

    if args.wind:
        from .wind import WindField, elev01_to_meters, synthesize_wind_maps

        with wall.step("wind"):
            print("Wind from temperature tensors…", flush=True)
            elev_m = elev01_to_meters(elev, land, max_elev_m=args.max_elev_m)
            synthesize_wind_maps(
                monthly,
                annual,
                land,
                elev_m,
                device=device,
                wind=WindField(
                    **wind_field_kwargs(
                        planet_radius_km=args.radius_km,
                        lapse_k_per_km=args.lapse,
                    )
                ),
            )

    # Input rasters no longer needed
    del elev, land
    release_torch_memory()

    meta_out["wall_timing"] = wall.as_meta()
    write_text_atomic(
        out / "temperature_meta.json",
        json.dumps(meta_out, indent=2) + "\n",
    )
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
    del monthly, annual, meta, meta_out
    release_torch_memory()


if __name__ == "__main__":
    main()
