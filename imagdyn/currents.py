#!/usr/bin/env python3
"""
Ocean current fine-tuning filter (independent of the core temperature synthesizer).

Heuristic
---------
Detect continental coastlines on an equirectangular land mask, then apply a
signed temperature correction on coastal seas:

- Continent **east** coast (western-boundary current) → **warm**
- Continent **west** coast (eastern-boundary current) → **cold**

The warm/cold contrast is strongest at mid-latitudes. A Gaussian weight in
latitude peaks near |φ| ≈ 30° and falls smoothly toward the equator and poles.

Humidity is reserved for a later climate pass (placeholder only).

Usage::

    python -m imagdyn.currents
    python -m imagdyn.currents --dump-maps

Temperature synthesis folds this ΔT into base seawater SST targets
(see ``synthesize_temperatures``); humidity remains a placeholder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .temperature import (
    box_filter_wrap_lon,
    get_device,
    kernel_radius_px,
    latitude_grid,
    load_grayscale,
    release_torch_memory,
)


@dataclass
class CurrentCorrection:
    """Per-grid oceanic current adjustments.

    ``temperature_C`` is applied today. ``humidity`` is a reserved slot for a
    future moisture / rainfall pass (None or zeros until implemented).
    """

    temperature_C: torch.Tensor
    humidity: torch.Tensor | None = None  # placeholder (°q / RH units TBD)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_numpy(self) -> dict[str, np.ndarray | None]:
        return {
            "temperature_C": self.temperature_C.detach().cpu().numpy().astype(np.float32),
            "humidity": (
                None
                if self.humidity is None
                else self.humidity.detach().cpu().numpy().astype(np.float32)
            ),
        }


class OceanCurrentFilter:
    """Coastline edge-detect → mid-latitude weighted warm/cold current filter."""

    def __init__(
        self,
        *,
        peak_lat_deg: float = 30.0,
        lat_sigma_deg: float = 12.0,
        warm_delta_C: float = 4.5,
        cold_delta_C: float = -4.5,
        reach_km: float = 450.0,
        diffuse_passes: int = 5,
        land_bleed: float = 0.25,
        planet_radius_km: float = 6371.0,
        enabled: bool = True,
    ) -> None:
        self.peak_lat_deg = float(peak_lat_deg)
        self.lat_sigma_deg = max(float(lat_sigma_deg), 1e-3)
        self.warm_delta_C = float(warm_delta_C)
        self.cold_delta_C = float(cold_delta_C)
        self.reach_km = float(reach_km)
        self.diffuse_passes = max(1, int(diffuse_passes))
        self.land_bleed = float(np.clip(land_bleed, 0.0, 1.0))
        self.planet_radius_km = float(planet_radius_km)
        self.enabled = bool(enabled)

    # --- coastline edge detection -------------------------------------------------

    @staticmethod
    def detect_coast_edges(land: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (east_coast_ocean, west_coast_ocean) bool masks.

        East coast of a continent: ocean cell with land immediately to its **west**
        (western-boundary / warm current side).
        West coast: ocean cell with land immediately to its **east**.
        Longitude wraps.
        """
        land_b = land.bool()
        water = ~land_b
        land_west = torch.roll(land_b, shifts=1, dims=1)  # neighbor at −Δlon
        land_east = torch.roll(land_b, shifts=-1, dims=1)  # neighbor at +Δlon
        east_coast = water & land_west
        west_coast = water & land_east
        return east_coast, west_coast

    @staticmethod
    def signed_coast_source(land: torch.Tensor) -> torch.Tensor:
        """+1 on east-coast ocean seeds, −1 on west-coast ocean seeds."""
        east, west = OceanCurrentFilter.detect_coast_edges(land)
        return east.float() - west.float()

    # --- latitude weight ----------------------------------------------------------

    def latitude_weight(self, lat_deg: torch.Tensor) -> torch.Tensor:
        """Gaussian in |latitude|, peaking at ``peak_lat_deg`` (default 30°)."""
        abs_lat = lat_deg.abs()
        z = (abs_lat - self.peak_lat_deg) / self.lat_sigma_deg
        return torch.exp(-0.5 * z * z)

    # --- build / apply ------------------------------------------------------------

    def compute(
        self,
        land: torch.Tensor,
        lat_deg: torch.Tensor,
    ) -> CurrentCorrection:
        """
        Build the correction field from a land mask and 1-D or 2-D latitudes.

        ``land``: (H, W) bool/float. ``lat_deg``: (H,) or (H, W).
        """
        device = land.device
        land_b = land.bool()
        h, w = land_b.shape
        if lat_deg.ndim == 1:
            lat2 = lat_deg.to(device=device, dtype=torch.float32)[:, None].expand(h, w)
        else:
            lat2 = lat_deg.to(device=device, dtype=torch.float32)

        if not self.enabled:
            z = torch.zeros((h, w), device=device, dtype=torch.float32)
            return CurrentCorrection(
                temperature_C=z,
                humidity=None,
                meta={"enabled": False, **self._param_meta()},
            )

        seed = self.signed_coast_source(land_b)
        ry, rx = kernel_radius_px(h, w, self.reach_km, self.planet_radius_km)
        field = seed
        for _ in range(self.diffuse_passes):
            field = box_filter_wrap_lon(field, ry, rx)

        ocean = (~land_b).float()
        land_f = land_b.float()
        field = field * (ocean + self.land_bleed * land_f)
        peak = field.abs().amax().clamp_min(1e-6)
        field = (field / peak).clamp(-1.0, 1.0)

        lat_w = self.latitude_weight(lat2)
        # field > 0 → warm (east); field < 0 → cold (west).
        # |cold_delta_C| is the cooling magnitude (default cold_delta_C=-3).
        warm = torch.clamp(field, min=0.0) * self.warm_delta_C
        cold = torch.clamp(field, max=0.0) * abs(self.cold_delta_C)
        dT = (warm + cold) * lat_w

        east, west = self.detect_coast_edges(land_b)
        meta = {
            "enabled": True,
            **self._param_meta(),
            "east_coast_px": int(east.sum().item()),
            "west_coast_px": int(west.sum().item()),
            "dT_min_C": float(dT.min().item()),
            "dT_max_C": float(dT.max().item()),
            "dT_ocean_mean_C": float(dT[ocean.bool()].mean().item()) if bool(ocean.any()) else 0.0,
            "humidity": None,
        }
        return CurrentCorrection(temperature_C=dT, humidity=None, meta=meta)

    def apply(
        self,
        temperature_C: torch.Tensor,
        correction: CurrentCorrection | None = None,
        *,
        land: torch.Tensor | None = None,
        lat_deg: torch.Tensor | None = None,
        humidity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Add current ΔT to ``temperature_C``.

        Humidity is accepted and returned unchanged (placeholder for a future
        moisture filter). If ``correction`` is omitted, build one from land/lat.
        """
        if correction is None:
            if land is None or lat_deg is None:
                raise ValueError("apply() needs correction=… or land=… and lat_deg=…")
            correction = self.compute(land, lat_deg)

        out_t = temperature_C + correction.temperature_C
        out_h = humidity if humidity is not None else correction.humidity
        return out_t, out_h

    def apply_monthly(
        self,
        monthly_C: torch.Tensor,
        land: torch.Tensor,
        lat_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, CurrentCorrection]:
        """Apply the same geographic ΔT to each month in (12,H,W)."""
        corr = self.compute(land, lat_deg)
        out = monthly_C + corr.temperature_C[None]
        return out, corr

    def _param_meta(self) -> dict[str, Any]:
        return {
            "peak_lat_deg": self.peak_lat_deg,
            "lat_sigma_deg": self.lat_sigma_deg,
            "warm_delta_C": self.warm_delta_C,
            "cold_delta_C": self.cold_delta_C,
            "reach_km": self.reach_km,
            "diffuse_passes": self.diffuse_passes,
            "land_bleed": self.land_bleed,
            "planet_radius_km": self.planet_radius_km,
            "model": "coast_edge_east_warm_west_cold_gaussian_midlat",
        }


def _save_signed_rgb(path: Path, field: np.ndarray) -> None:
    """Visualize signed field: blue=cold/neg, gray=0, red=warm/pos."""
    v = np.clip(field, -1.0, 1.0)
    rgb = np.zeros((*v.shape, 3), dtype=np.uint8)
    pos = np.clip(v, 0.0, 1.0)
    neg = np.clip(-v, 0.0, 1.0)
    rgb[..., 0] = np.clip(40 + 200 * pos, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(40 + 40 * (1.0 - pos - neg), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(40 + 200 * neg, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from . import paths

    root = paths.ROOT
    p = argparse.ArgumentParser(
        description="Ocean current coastline filter (east=warm, west=cold, peak |lat|≈30°).",
    )
    p.add_argument(
        "--land-mask",
        type=Path,
        default=root / "graphs" / "Terrain - Land Mask.png",
    )
    p.add_argument(
        "--elevation",
        type=Path,
        default=root / "graphs" / "Terrain - Full Elevation.png",
        help="Used only if land mask is missing (elev > 0.5 → land)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "graphs" / "temperature",
    )
    p.add_argument("--peak-lat", type=float, default=30.0)
    p.add_argument("--lat-sigma", type=float, default=12.0)
    p.add_argument("--warm-c", type=float, default=3.0)
    p.add_argument("--cold-c", type=float, default=-3.0)
    p.add_argument("--reach-km", type=float, default=450.0)
    p.add_argument("--diffuse-passes", type=int, default=5)
    p.add_argument("--land-bleed", type=float, default=0.25)
    p.add_argument("--radius-km", type=float, default=6371.0)
    p.add_argument("--dump-maps", action="store_true", help="Write diagnostic PNGs + JSON")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from .assets import ensure_derived_terrain

    args = parse_args(argv)
    ensure_derived_terrain(seed_template=True)
    device = get_device(prefer_gpu=not args.cpu)

    if args.land_mask.is_file():
        land_np = load_grayscale(args.land_mask) > 0.5
    else:
        elev = load_grayscale(args.elevation)
        land_np = elev > 0.5

    land = torch.from_numpy(land_np.astype(np.bool_)).to(device)
    h, w = land.shape
    lat = latitude_grid(h, device)

    filt = OceanCurrentFilter(
        peak_lat_deg=args.peak_lat,
        lat_sigma_deg=args.lat_sigma,
        warm_delta_C=args.warm_c,
        cold_delta_C=args.cold_c,
        reach_km=args.reach_km,
        diffuse_passes=args.diffuse_passes,
        land_bleed=args.land_bleed,
        planet_radius_km=args.radius_km,
    )
    corr = filt.compute(land, lat)
    meta = corr.meta
    print(
        f"Ocean currents  device={device}  grid={w}x{h}  "
        f"east={meta.get('east_coast_px')}  west={meta.get('west_coast_px')}  "
        f"dT=[{meta.get('dT_min_C'):.2f}, {meta.get('dT_max_C'):.2f}] C"
    )
    print("Humidity correction: placeholder (not applied)")

    if args.dump_maps:
        out = args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        dT = corr.temperature_C.detach().cpu().numpy()
        amp = max(abs(args.warm_c), abs(args.cold_c), 1e-3)
        _save_signed_rgb(out / "Ocean Currents - dT sign.png", dT / amp)
        gray = np.clip(np.rint(128.0 + (dT / amp) * 127.0), 0, 255).astype(np.uint8)
        Image.fromarray(gray, mode="L").save(out / "Ocean Currents - dT.png")
        east, west = filt.detect_coast_edges(land)
        edge = np.zeros((h, w, 3), dtype=np.uint8)
        e_np = east.cpu().numpy()
        w_np = west.cpu().numpy()
        edge[e_np] = (220, 80, 60)
        edge[w_np] = (60, 100, 220)
        Image.fromarray(edge, mode="RGB").save(out / "Ocean Currents - coast edges.png")
        payload = {k: v for k, v in meta.items()}
        (out / "ocean_currents_meta.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote diagnostics under {out}")

    del corr, filt, land, lat
    release_torch_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
