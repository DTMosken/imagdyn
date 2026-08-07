#!/usr/bin/env python3
"""
Nonlinear remapping of terrain elevation rasters (baked into height maps).

- Land: control-point remap — expand plains relief, compress plateaus; avoid 8-bit lowland pile-up
- Ocean: store t' = t² so linear decode with 11000 m full-scale yields depth = 11000 · t²
- Then: light adaptive Gaussian smooth (stronger on high peaks); land/ocean filtered apart
- Land Mask unchanged; display code is not modified

Updates:
  Terrain - Full Elevation.png
  Terrain - Elevation Above Sea Level.png
  Terrain - Elevation Below Sea Level.png
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Intended ocean full-scale (m). Height map stores t' = t² so depth_m = OCEAN_MAX_M * t'.
OCEAN_MAX_M = 11000.0


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def land_plain_hills_peak(t: np.ndarray) -> np.ndarray:
    """
    Monotonic land remap via control points (no softstep shelves).

    - Low band (plains / low hills) ≈ identity so original lowland relief and
      occupancy are kept; avoids remapping mass onto the ~31 m gray step.
    - Above ~1000 m: compress so former plateaus settle into hills/upland.
    - Endpoints fixed: f(0)=0, f(1)=1.
    """
    t = np.clip(t, 0.0, 1.0)
    # tin → tout (fraction of 8000 m). Identity through ~960 m, then compress.
    xp = np.array(
        [0.00, 0.12, 0.18, 0.26, 0.36, 0.48, 0.62, 0.78, 1.00],
        dtype=np.float64,
    )
    fp = np.array(
        [0.00, 0.12, 0.162, 0.208, 0.262, 0.322, 0.410, 0.570, 1.00],
        dtype=np.float64,
    )
    out = np.interp(t, xp, fp).astype(np.float32)
    out = np.where(t <= 0.0, 0.0, out)
    out = np.where(t >= 1.0, 1.0, out)
    return np.clip(out, 0.0, 1.0)


def ocean_depth_t_sq(d: np.ndarray) -> np.ndarray:
    """Bake depth = OCEAN_MAX_M · t² into normalized gray depth t' = t²."""
    d = np.clip(d, 0.0, 1.0)
    return (d * d).astype(np.float32)


def gaussian_lon_wrap(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with wrap in longitude (axis=1) and reflect in latitude."""
    if sigma <= 0:
        return arr.astype(np.float32, copy=True)
    pad = max(1, int(math.ceil(4.0 * sigma)))
    x = np.pad(arr, ((0, 0), (pad, pad)), mode="wrap")
    x = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    y = gaussian_filter(x, sigma=sigma, mode="nearest")
    return y[pad:-pad, pad:-pad].astype(np.float32)


def masked_gaussian(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian within mask only (normalized), lon-wrap."""
    m = mask.astype(np.float32)
    v = np.where(mask, values, 0.0).astype(np.float32)
    vs = gaussian_lon_wrap(v, sigma)
    ms = gaussian_lon_wrap(m, sigma)
    return vs / np.maximum(ms, 1e-6)


def adaptive_elev_smooth(elev: np.ndarray, land: np.ndarray) -> np.ndarray:
    """
    Light global smooth; blend toward a wider kernel as land height rises.
    Land and ocean are filtered separately so the coastline stays sharp.
    """
    elev = elev.astype(np.float32)
    out = elev.copy()

    # --- land: mild everywhere, stronger on peaks ---
    sigma_mild, sigma_strong = 1.15, 5.0
    amount_lo, amount_hi = 0.22, 0.82
    land_t = np.clip((elev - 0.5) / 0.5, 0.0, 1.0)
    # ramp starts ~2800 m, strong by ~6000 m+
    peak_w = smoothstep(0.35, 0.78, land_t)
    amount = amount_lo + (amount_hi - amount_lo) * peak_w

    mild = masked_gaussian(elev, land, sigma_mild)
    strong = masked_gaussian(elev, land, sigma_strong)
    blurred = (1.0 - peak_w) * mild + peak_w * strong
    land_out = (1.0 - amount) * elev + amount * blurred
    # keep land at/above sea level
    out[land] = np.maximum(land_out[land], 0.5)

    # --- ocean: slight mild smooth only ---
    ocean = ~land
    mild_o = masked_gaussian(elev, ocean, 1.25)
    ocean_out = 0.78 * elev + 0.22 * mild_o
    out[ocean] = np.minimum(ocean_out[ocean], 0.5)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def mild_global_smooth(
    elev: np.ndarray,
    land: np.ndarray,
    *,
    sigma: float = 1.1,
    amount: float = 0.28,
) -> np.ndarray:
    """Uniform light Gaussian blend; land/ocean filtered apart."""
    elev = elev.astype(np.float32)
    out = elev.copy()
    ocean = ~land
    land_b = masked_gaussian(elev, land, sigma)
    ocean_b = masked_gaussian(elev, ocean, sigma)
    out[land] = np.maximum((1.0 - amount) * elev[land] + amount * land_b[land], 0.5)
    out[ocean] = np.minimum((1.0 - amount) * elev[ocean] + amount * ocean_b[ocean], 0.5)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def smooth_ocean(
    elev: np.ndarray,
    land: np.ndarray,
    *,
    sigma_a: float = 4.0,
    sigma_b: float = 9.0,
    amount: float = 0.88,
) -> np.ndarray:
    """Stronger seafloor-only smooth (coastline kept via masked blur)."""
    elev = elev.astype(np.float32)
    ocean = ~land
    mild = masked_gaussian(elev, ocean, sigma_a)
    strong = masked_gaussian(elev, ocean, sigma_b)
    blurred = 0.35 * mild + 0.65 * strong
    out = elev.copy()
    out[ocean] = (1.0 - amount) * elev[ocean] + amount * blurred[ocean]
    out[ocean] = np.minimum(out[ocean], 0.5)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def soft_band_weight(h: np.ndarray, lo: float, hi: float, edge: float) -> np.ndarray:
    """Smooth plateau weight ≈1 on [lo,hi], soft falloff over `edge` meters."""
    return smoothstep(lo - edge, lo, h) * (1.0 - smoothstep(hi, hi + edge, h))


def land_band_reshape(
    elev: np.ndarray,
    land: np.ndarray,
    *,
    max_elev_m: float = 8000.0,
    down_lo: float = 1000.0,
    down_hi: float = 1500.0,
    down_m: float = 160.0,
    up_lo: float = 2000.0,
    up_hi: float = 3000.0,
    up_m: float = 220.0,
    edge_m: float = 160.0,
) -> np.ndarray:
    """
    Smooth additive remap on land:
      lower ~[1000,1500] m, raise ~[2000,3000] m.
    Endpoints / non-band regions stay near identity.
    """
    elev = elev.astype(np.float32)
    h = np.clip((elev - 0.5) / 0.5, 0.0, 1.0) * max_elev_m
    w_down = soft_band_weight(h, down_lo, down_hi, edge_m)
    w_up = soft_band_weight(h, up_lo, up_hi, edge_m)
    # Peak of product-of-smoothsteps is 1 inside plateau; shape is C1 at edges.
    delta = (-down_m * w_down + up_m * w_up).astype(np.float32)
    h2 = h.copy()
    h2[land] = np.clip(h[land] + delta[land], 0.0, max_elev_m)
    out = elev.copy()
    out[land] = 0.5 + 0.5 * (h2[land] / max_elev_m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def smooth_elev_band(
    elev: np.ndarray,
    land: np.ndarray,
    *,
    max_elev_m: float = 8000.0,
    lo_m: float = 1000.0,
    hi_m: float = 2000.0,
    edge_m: float = 180.0,
    sigma_a: float = 2.8,
    sigma_b: float = 5.5,
    amount: float = 0.78,
) -> np.ndarray:
    """
    Stronger land-only smooth inside [lo_m, hi_m] with soft edges.
    amount < 1 leaves a little of the jittered irregularity.
    """
    elev = elev.astype(np.float32)
    h = np.clip((elev - 0.5) / 0.5, 0.0, 1.0) * max_elev_m
    # Soft membership in the band
    w = smoothstep(lo_m - edge_m, lo_m, h) * (1.0 - smoothstep(hi_m, hi_m + edge_m, h))
    w = np.where(land, w, 0.0).astype(np.float32)
    if float(w.max()) < 1e-6:
        return elev.copy()

    mild = masked_gaussian(elev, land, sigma_a)
    strong = masked_gaussian(elev, land, sigma_b)
    blurred = 0.35 * mild + 0.65 * strong
    a = (amount * w).astype(np.float32)
    out = elev * (1.0 - a) + blurred * a
    out = np.where(land, np.maximum(out, 0.5), elev)
    out = np.where(land, out, elev)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def band_roughness(elev: np.ndarray, land: np.ndarray, lo_m: float, hi_m: float, max_elev_m: float = 8000.0) -> float:
    """Mean |∇h| (m / px) inside elevation band on land."""
    h = np.clip((elev - 0.5) / 0.5, 0.0, 1.0) * max_elev_m
    band = land & (h >= lo_m) & (h <= hi_m)
    if not np.any(band):
        return 0.0
    dx = np.abs(np.roll(h, -1, axis=1) - h)
    dy = np.abs(np.roll(h, -1, axis=0) - h)
    g = np.hypot(dx, dy)
    return float(g[band].mean())


def lower_mid_elev(
    elev: np.ndarray,
    land: np.ndarray,
    *,
    max_elev_m: float = 8000.0,
    lo_m: float = 31.0,
    hi_m: float = 2000.0,
    drop_m: float = 50.0,
    jitter_m: float = 22.0,
    seed: int = 42,
) -> np.ndarray:
    """
    For land with lo_m < h <= hi_m: subtract ~drop_m with per-pixel jitter.
    Keeps results ≥ sea level.
    """
    out = elev.astype(np.float32).copy()
    h = np.clip((out - 0.5) / 0.5, 0.0, 1.0) * max_elev_m
    band = land & (h > lo_m) & (h <= hi_m)
    if not np.any(band):
        return out
    rng = np.random.default_rng(seed)
    delta = drop_m + rng.uniform(-jitter_m, jitter_m, size=int(band.sum())).astype(np.float32)
    h2 = h.copy()
    h2[band] = np.maximum(h[band] - delta, 0.0)
    out[land] = 0.5 + 0.5 * (h2[land] / max_elev_m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_gray01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def save_gray01(path: Path, v01: np.ndarray) -> None:
    g = np.clip(np.rint(v01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(g, mode="L").save(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    graphs = root / "graphs"
    full_path = graphs / "Terrain - Full Elevation.png"
    land_path = graphs / "Terrain - Land Mask.png"
    above_path = graphs / "Terrain - Elevation Above Sea Level.png"
    below_path = graphs / "Terrain - Elevation Below Sea Level.png"

    land = load_gray01(land_path) > 0.5

    bak = graphs / "Terrain - Full Elevation.pre_nl.png"
    if bak.is_file():
        elev = load_gray01(bak)
        print(f"Restored source ← {bak.name}")
    else:
        elev = load_gray01(full_path)
        Image.open(full_path).save(bak)
        print(f"Backup → {bak.name}")

    land_t = np.clip((elev - 0.5) / 0.5, 0.0, 1.0)
    ocean_d = np.clip((0.5 - elev) / 0.5, 0.0, 1.0)

    land_t2 = land_plain_hills_peak(land_t)
    ocean_d2 = np.zeros_like(ocean_d)
    ocean_d2[~land] = ocean_depth_t_sq(ocean_d[~land])

    elev2 = np.where(land, 0.5 + 0.5 * land_t2, 0.5 - 0.5 * ocean_d2).astype(np.float32)

    print("Full Elevation land height fraction t:")
    print(f"  before mean={land_t[land].mean():.4f}  after remap={land_t2[land].mean():.4f}")
    print(
        f"  endpoints land min/max before {land_t[land].min():.4f}/{land_t[land].max():.4f} "
        f"after remap {land_t2[land].min():.4f}/{land_t2[land].max():.4f}"
    )
    elev_b = land_t[land] * 8000.0
    elev_a = land_t2[land] * 8000.0
    print(
        f"  elev m mean {elev_b.mean():.0f}→{elev_a.mean():.0f}  "
        f"med {np.median(elev_b):.0f}→{np.median(elev_a):.0f}"
    )

    elev3 = adaptive_elev_smooth(elev2, land)
    d = elev3 - elev2
    land_t3 = np.clip((elev3 - 0.5) / 0.5, 0.0, 1.0)
    hi = land & (land_t2 >= 0.5)
    print("Adaptive smooth:")
    print(
        f"  |Δ| mean={np.abs(d).mean():.5f}  land={np.abs(d[land]).mean():.5f}  "
        f"ocean={np.abs(d[~land]).mean():.5f}"
    )
    if hi.any():
        print(
            f"  high land (t≥0.5): |Δ| mean={np.abs(d[hi]).mean():.5f}  "
            f"elev m {land_t2[hi].mean()*8000:.0f}→{land_t3[hi].mean()*8000:.0f}"
        )
    print(f"  land max t {land_t2[land].max():.4f}→{land_t3[land].max():.4f}")

    elev4 = lower_mid_elev(elev3, land)
    h3 = np.clip((elev3 - 0.5) / 0.5, 0.0, 1.0) * 8000.0
    h4 = np.clip((elev4 - 0.5) / 0.5, 0.0, 1.0) * 8000.0
    band = land & (h3 > 31.0) & (h3 <= 2000.0)
    print("Mid-elev −50 m (±jitter):")
    print(f"  band pixels={int(band.sum())}  mean h {h3[band].mean():.1f}→{h4[band].mean():.1f} m")
    print(f"  land mean elev {h3[land].mean():.0f}→{h4[land].mean():.0f} m")

    r_before = band_roughness(elev4, land, 1000.0, 2000.0)
    elev4b = smooth_elev_band(elev4, land, lo_m=1000.0, hi_m=2000.0, amount=0.78)
    r_after = band_roughness(elev4b, land, 1000.0, 2000.0)
    b12 = land & (h4 >= 1000.0) & (h4 <= 2000.0)
    h4b = np.clip((elev4b - 0.5) / 0.5, 0.0, 1.0) * 8000.0
    print("1–2 km band smooth (keep ~22% irregularity):")
    print(f"  pixels≈{int(b12.sum())}  mean |∇h| {r_before:.3f}→{r_after:.3f} m/px")
    if b12.any():
        print(f"  band mean elev {h4[b12].mean():.1f}→{h4b[b12].mean():.1f} m")

    # Extra pass around the 1 km contour (jitter seam)
    r1_before = band_roughness(elev4b, land, 750.0, 1250.0)
    elev4c = smooth_elev_band(
        elev4b,
        land,
        lo_m=750.0,
        hi_m=1250.0,
        edge_m=150.0,
        sigma_a=2.8,
        sigma_b=5.5,
        amount=0.78,
    )
    r1_after = band_roughness(elev4c, land, 750.0, 1250.0)
    b1 = land & (h4b >= 750.0) & (h4b <= 1250.0)
    h4c = np.clip((elev4c - 0.5) / 0.5, 0.0, 1.0) * 8000.0
    print("≈1 km contour band smooth (750–1250 m):")
    print(f"  pixels≈{int(b1.sum())}  mean |∇h| {r1_before:.3f}→{r1_after:.3f} m/px")
    if b1.any():
        print(f"  band mean elev {h4b[b1].mean():.1f}→{h4c[b1].mean():.1f} m")

    elev5 = mild_global_smooth(elev4c, land)
    d5 = elev5 - elev4c
    print("Mild global smooth:")
    print(
        f"  |Δ| mean={np.abs(d5).mean():.5f}  land={np.abs(d5[land]).mean():.5f}  "
        f"ocean={np.abs(d5[~land]).mean():.5f}"
    )

    # Raise 2–3 km / lower 1–1.5 km (smooth band weights)
    elev6 = land_band_reshape(elev5, land)
    h5 = np.clip((elev5 - 0.5) / 0.5, 0.0, 1.0) * 8000.0
    h6 = np.clip((elev6 - 0.5) / 0.5, 0.0, 1.0) * 8000.0
    b_down = land & (h5 >= 1000.0) & (h5 <= 1500.0)
    b_up = land & (h5 >= 2000.0) & (h5 <= 3000.0)
    print("Land band reshape (smooth −160 m @1–1.5 km, +220 m @2–3 km):")
    if b_down.any():
        print(f"  1–1.5 km mean {h5[b_down].mean():.0f}→{h6[b_down].mean():.0f} m")
    if b_up.any():
        print(f"  2–3 km mean {h5[b_up].mean():.0f}→{h6[b_up].mean():.0f} m")
    print(f"  land mean elev {h5[land].mean():.0f}→{h6[land].mean():.0f} m")

    elev7 = smooth_ocean(elev6, land)
    d7 = elev7 - elev6
    ocean = ~land
    print("Ocean / seafloor smooth:")
    print(f"  |Δ| ocean mean={np.abs(d7[ocean]).mean():.5f}  land={np.abs(d7[land]).mean():.5f}")

    land_t7 = np.clip((elev7 - 0.5) / 0.5, 0.0, 1.0)
    save_gray01(full_path, elev7)

    above = np.where(land, land_t7, 0.0).astype(np.float32)
    ocean_d7 = np.clip((0.5 - elev7) / 0.5, 0.0, 1.0)
    below = np.where(land, 1.0, ocean_d7).astype(np.float32)
    save_gray01(above_path, above)
    save_gray01(below_path, below)
    print(f"Wrote {full_path.name}, {above_path.name}, {below_path.name}")


if __name__ == "__main__":
    main()
