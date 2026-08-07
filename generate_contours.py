#!/usr/bin/env python3
"""Generate Terrain - Contours.png from Full Elevation + Land Mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parent
    elev_path = root / "graphs" / "Terrain - Full Elevation.png"
    land_path = root / "graphs" / "Terrain - Land Mask.png"
    out_path = root / "graphs" / "Terrain - Contours.png"

    elev = np.asarray(Image.open(elev_path), dtype=np.float32)
    if elev.ndim == 3:
        elev = elev[..., 0]
    if elev.max() > 1.5:
        elev = elev / 255.0

    land = np.asarray(Image.open(land_path), dtype=np.float32)
    if land.ndim == 3:
        land = land[..., 0]
    land = land > 127

    max_elev_m = 8000.0
    elev_m = np.where(land, (elev - 0.5) / 0.5 * max_elev_m, 0.0).astype(np.float32)
    elev_m = np.clip(elev_m, 0.0, max_elev_m)

    # RGB canvas: muted bathymetry + land wash
    h, w = elev.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    depth01 = np.clip((0.5 - elev) / 0.5, 0.0, 1.0)
    rgb[..., 0] = np.where(land, 0.18, 0.04 + 0.10 * (1.0 - depth01))
    rgb[..., 1] = np.where(land, 0.20, 0.10 + 0.22 * (1.0 - depth01))
    rgb[..., 2] = np.where(land, 0.18, 0.18 + 0.35 * (1.0 - depth01))

    land_shade = np.clip(elev_m / max_elev_m, 0.0, 1.0)
    rgb[..., 0] = np.where(land, 0.16 + 0.28 * land_shade, rgb[..., 0])
    rgb[..., 1] = np.where(land, 0.18 + 0.22 * land_shade, rgb[..., 1])
    rgb[..., 2] = np.where(land, 0.14 + 0.16 * land_shade, rgb[..., 2])

    # Contour levels (m). Index every 1000 m drawn thicker.
    minor = np.arange(200, max_elev_m + 1, 200, dtype=np.float32)
    major = np.arange(1000, max_elev_m + 1, 1000, dtype=np.float32)

    def contour_at(level: float) -> np.ndarray:
        above = (elev_m >= level) & land
        edge = np.zeros_like(above)
        edge[:, 1:] |= above[:, 1:] & ~above[:, :-1]
        edge[:, :-1] |= above[:, :-1] & ~above[:, 1:]
        edge[1:, :] |= above[1:, :] & ~above[:-1, :]
        edge[:-1, :] |= above[:-1, :] & ~above[1:, :]
        return edge

    def dilate4(mask: np.ndarray) -> np.ndarray:
        out = mask.copy()
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        out[1:, :] |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        return out & land

    minor_edge = np.zeros((h, w), dtype=bool)
    for lv in minor:
        if float(lv) in set(float(x) for x in major):
            continue
        minor_edge |= contour_at(float(lv))

    major_edge = np.zeros((h, w), dtype=bool)
    for lv in major:
        major_edge |= dilate4(contour_at(float(lv)))

    coast = np.zeros((h, w), dtype=bool)
    coast[:, 1:] |= land[:, 1:] & ~land[:, :-1]
    coast[:, :-1] |= land[:, :-1] & ~land[:, 1:]
    coast[1:, :] |= land[1:, :] & ~land[:-1, :]
    coast[:-1, :] |= land[:-1, :] & ~land[1:, :]
    coast &= land

    # Paint lines — minor slightly brighter than original, no thicken
    rgb[minor_edge] = (0.78, 0.74, 0.60)
    rgb[major_edge] = (0.95, 0.90, 0.75)
    rgb[coast] = (0.85, 0.78, 0.55)

    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(out_path)
    print(f"Wrote {out_path}  minor={int(minor_edge.sum())} major={int(major_edge.sum())}")


if __name__ == "__main__":
    main()
