"""Detect graph assets and derive missing terrain products from Full Elevation."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from .io_gray import load_gray01, save_gray01, save_mask


@dataclass
class AssetStatus:
    full_elevation: bool = False
    land_mask: bool = False
    above_sea: bool = False
    below_sea: bool = False
    contours: bool = False
    satellite: bool = False
    temperature_annual: bool = False
    temperature_meta: bool = False
    temperature_months: list[bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.temperature_months is None:
            d["temperature_months"] = [False] * 12
        return d


def _exists(name: str, base: Path = paths.GRAPHS) -> bool:
    return (base / name).is_file()


def probe_assets(graphs: Path | None = None) -> AssetStatus:
    g = graphs or paths.GRAPHS
    months = [_exists(n, g / "temperature") if (g / "temperature").is_dir() else False for n in paths.MONTH_TEMP_NAMES]
    # Also allow temp files directly under temperature dir via paths
    temp = g / "temperature"
    return AssetStatus(
        full_elevation=_exists(paths.FULL_ELEV, g),
        land_mask=_exists(paths.LAND_MASK, g),
        above_sea=_exists(paths.ABOVE_SEA, g),
        below_sea=_exists(paths.BELOW_SEA, g),
        contours=_exists(paths.CONTOURS, g),
        satellite=_exists(paths.SATELLITE, g),
        temperature_annual=(temp / paths.TEMP_ANNUAL).is_file(),
        temperature_meta=(temp / paths.TEMP_META).is_file(),
        temperature_months=[(temp / n).is_file() for n in paths.MONTH_TEMP_NAMES],
    )


def write_assets_json(status: AssetStatus | None = None, graphs: Path | None = None) -> Path:
    g = graphs or paths.GRAPHS
    st = status or probe_assets(g)
    out = g / paths.ASSETS_JSON
    g.mkdir(parents=True, exist_ok=True)
    payload = {
        **st.to_dict(),
        "paths": {
            "satellite": paths.SATELLITE,
            "full_elevation": paths.FULL_ELEV,
            "land_mask": paths.LAND_MASK,
            "above_sea": paths.ABOVE_SEA,
            "below_sea": paths.BELOW_SEA,
            "contours": paths.CONTOURS,
            "temperature_dir": "temperature",
            "temperature_annual": f"temperature/{paths.TEMP_ANNUAL}",
            "temperature_meta": f"temperature/{paths.TEMP_META}",
            "temperature_months": [f"temperature/{n}" for n in paths.MONTH_TEMP_NAMES],
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def maybe_seed_full_elevation(graphs: Path | None = None) -> bool:
    """Copy Full Elevation from graphs/template/ if missing. Returns True if seeded."""
    g = graphs or paths.GRAPHS
    dest = g / paths.FULL_ELEV
    if dest.is_file():
        return False
    src = paths.TEMPLATE_DIR / paths.FULL_ELEV
    if src.is_file():
        g.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"Seeded {dest.name} ← template/")
        return True
    return False


def derive_from_full_elevation(
    elev01: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From full elevation (0.5 = sea level):
      land mask, above-sea (0..1 on land), below-sea (depth fraction on ocean).
    """
    land = elev01 > 0.5
    land_t = np.clip((elev01 - 0.5) / 0.5, 0.0, 1.0)
    ocean_d = np.clip((0.5 - elev01) / 0.5, 0.0, 1.0)
    above = np.where(land, land_t, 0.0).astype(np.float32)
    below = np.where(land, 1.0, ocean_d).astype(np.float32)
    return land, above, below


def ensure_derived_terrain(
    *,
    graphs: Path | None = None,
    force: bool = False,
    seed_template: bool = True,
) -> AssetStatus:
    """
    If only Full Elevation is present (or force), write Land Mask + Above/Below.
    Optionally seed Full Elevation from graphs/template/.
    """
    g = graphs or paths.GRAPHS
    if seed_template:
        maybe_seed_full_elevation(g)

    full = g / paths.FULL_ELEV
    if not full.is_file():
        raise FileNotFoundError(
            f"Missing {full}. Place a full elevation PNG there "
            f"or under {paths.TEMPLATE_DIR / paths.FULL_ELEV}."
        )

    land_p = g / paths.LAND_MASK
    above_p = g / paths.ABOVE_SEA
    below_p = g / paths.BELOW_SEA

    need = force or (not land_p.is_file()) or (not above_p.is_file()) or (not below_p.is_file())
    if need:
        elev = load_gray01(full)
        land, above, below = derive_from_full_elevation(elev)
        if force or not land_p.is_file():
            save_mask(land_p, land)
            print(f"Wrote {land_p.name}  land={100.0 * float(land.mean()):.1f}%")
        if force or not above_p.is_file():
            save_gray01(above_p, above)
            print(f"Wrote {above_p.name}")
        if force or not below_p.is_file():
            save_gray01(below_p, below)
            print(f"Wrote {below_p.name}")
    else:
        print("Derived terrain products already present (use --force to regenerate).")

    st = probe_assets(g)
    write_assets_json(st, g)
    print(f"Wrote {g / paths.ASSETS_JSON}")
    return st


def print_status(status: AssetStatus | None = None) -> None:
    st = status or probe_assets()
    months = st.temperature_months or [False] * 12
    rows = [
        ("Full Elevation", st.full_elevation),
        ("Land Mask", st.land_mask),
        ("Above Sea Level", st.above_sea),
        ("Below Sea Level", st.below_sea),
        ("Contours", st.contours),
        ("Satellite Color", st.satellite),
        ("Temperature annual", st.temperature_annual),
        ("Temperature meta", st.temperature_meta),
        (f"Temperature months ({sum(months)}/12)", all(months)),
    ]
    print("=== IMagDyn assets ===")
    for name, ok in rows:
        print(f"  [{'OK' if ok else '--'}] {name}")
