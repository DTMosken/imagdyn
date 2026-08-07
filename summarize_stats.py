#!/usr/bin/env python3
"""Summarize temperature / terrain statistics for magdyn."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
TEMP_DIR = ROOT / "graphs" / "temperature"
T_MIN, T_MAX = -60.0, 45.0
MAX_ELEV = 8000.0
MONTHS = [
    "01 January", "02 February", "03 March", "04 April",
    "05 May", "06 June", "07 July", "08 August",
    "09 September", "10 October", "11 November", "12 December",
]


def load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def gray_to_T(g: np.ndarray) -> np.ndarray:
    return T_MIN + (g / 255.0) * (T_MAX - T_MIN)


def elev_m(g01: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Signed elevation: land/ocean both linear."""
    v = g01
    out = np.empty_like(v)
    out[land] = np.clip((v[land] - 0.5) / 0.5 * MAX_ELEV, 0, MAX_ELEV)
    out[~land] = -np.clip((0.5 - v[~land]) / 0.5 * MAX_ELEV, 0, MAX_ELEV)
    return out


def pct(x: np.ndarray, qs=(5, 10, 25, 50, 75, 90, 95)) -> dict:
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def band_stats(T: np.ndarray, land: np.ndarray, lat: np.ndarray) -> dict:
    lat2 = lat[:, None]
    out = {}
    for name, mask_lat in [
        ("tropics_|lat|<23.5", np.abs(lat2) < 23.5),
        ("midlat_23.5-60", (np.abs(lat2) >= 23.5) & (np.abs(lat2) <= 60)),
        ("polar_|lat|>60", np.abs(lat2) > 60),
        ("eq_ocean_|lat|<10", (np.abs(lat2) < 10) & (~land)),
        ("polar_ocean_|lat|>60", (np.abs(lat2) > 60) & (~land)),
    ]:
        m = mask_lat if mask_lat.ndim == 2 else mask_lat
        if m.ndim == 1:
            m = m[:, None] & np.ones(land.shape[1], dtype=bool)
        # fix broadcasting
        if name.startswith("tropics") or name.startswith("midlat") or name.startswith("polar_|"):
            m = {
                "tropics_|lat|<23.5": np.abs(lat2) < 23.5,
                "midlat_23.5-60": (np.abs(lat2) >= 23.5) & (np.abs(lat2) <= 60),
                "polar_|lat|>60": np.abs(lat2) > 60,
            }[name]
            m = np.broadcast_to(m, land.shape)
            sel = T[m]
            out[name] = {
                "mean_C": float(sel.mean()),
                "land_mean_C": float(T[m & land].mean()) if (m & land).any() else None,
                "ocean_mean_C": float(T[m & ~land].mean()) if (m & ~land).any() else None,
            }
        else:
            if name.startswith("eq"):
                m = (np.abs(lat2) < 10) & (~land)
            else:
                m = (np.abs(lat2) > 60) & (~land)
            m = np.broadcast_to(m if m.shape == land.shape else m, land.shape) if False else m
            # rebuild properly
            pass
    # redo cleanly
    out = {}
    h, w = land.shape
    lat2d = np.broadcast_to(lat[:, None], (h, w))
    bands = {
        "tropics (|lat|<23.5)": np.abs(lat2d) < 23.5,
        "midlat (23.5–60)": (np.abs(lat2d) >= 23.5) & (np.abs(lat2d) <= 60),
        "polar (|lat|>60)": np.abs(lat2d) > 60,
    }
    for name, m in bands.items():
        out[name] = {
            "all_mean_C": float(T[m].mean()),
            "land_mean_C": float(T[m & land].mean()) if (m & land).any() else None,
            "ocean_mean_C": float(T[m & ~land].mean()) if (m & ~land).any() else None,
        }
    out["eq ocean (|lat|<10)"] = {
        "mean_C": float(T[(np.abs(lat2d) < 10) & (~land)].mean())
    }
    out["polar ocean (|lat|>60)"] = {
        "mean_C": float(T[(np.abs(lat2d) > 60) & (~land)].mean())
    }
    return out


def main() -> None:
    meta_path = TEMP_DIR / "temperature_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    elev_g = load_gray(ROOT / "graphs" / "Terrain - Full Elevation.png")
    if elev_g.max() > 1.5:
        elev01 = elev_g / 255.0
    else:
        elev01 = elev_g
    lm = load_gray(ROOT / "graphs" / "Terrain - Land Mask.png")
    land = lm > 127
    h, w = land.shape
    lat = 90.0 - (np.arange(h) + 0.5) * (180.0 / h)

    elev = elev_m(elev01, land)
    land_elev = elev[land]
    ocean_depth = -elev[~land]  # positive depth

    annual = gray_to_T(load_gray(TEMP_DIR / "Temperature - Annual Mean.png"))
    monthly = np.stack(
        [gray_to_T(load_gray(TEMP_DIR / f"Temperature - {name}.png")) for name in MONTHS],
        axis=0,
    )
    amp = np.abs(monthly[6] - monthly[0])  # Jul-Jan

    # coast jump sample
    from scipy import ndimage

    coast_land = land & ndimage.binary_dilation(~land, structure=ndimage.generate_binary_structure(2, 1))
    rng = np.random.default_rng(0)
    ys, xs = np.where(coast_land)
    idx = rng.choice(len(ys), size=min(8000, len(ys)), replace=False)
    jumps = []
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        vals = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, (x + dx) % w
                if 0 <= yy < h and not land[yy, xx]:
                    vals.append(annual[yy, xx])
        if vals:
            jumps.append(annual[y, x] - float(np.mean(vals)))
    jumps = np.asarray(jumps)

    freeze = float(meta.get("freeze_C", 0.0))
    frozen_month = [(monthly[m] <= freeze).mean() * 100 for m in range(12)]

    stats = {
        "grid": {"width": w, "height": h, "land_pct": float(land.mean() * 100)},
        "meta_summary": {
            "device": meta.get("device"),
            "s0_W_m2": meta.get("s0_W_m2"),
            "obliquity_deg": meta.get("obliquity_deg"),
            "greenhouse_factor": meta.get("greenhouse_factor"),
            "heat_transport": meta.get("heat_transport"),
            "ocean_sst_nudge": meta.get("ocean_sst_nudge"),
            "coast_blend_km": meta.get("coast_blend_km"),
            "gray_scale_C": meta.get("gray_scale_C", [T_MIN, T_MAX]),
        },
        "terrain_display": {
            "land_elev_m": {
                "mean": float(land_elev.mean()),
                "max": float(land_elev.max()),
                **pct(land_elev),
            },
            "ocean_depth_m_linear": {
                "note": f"depth = {MAX_ELEV} * (0.5-v)/0.5",
                "mean": float(ocean_depth.mean()),
                "median": float(np.median(ocean_depth)),
                "max": float(ocean_depth.max()),
                **pct(ocean_depth),
            },
        },
        "temperature_annual_C": {
            "global": {"mean": float(annual.mean()), "min": float(annual.min()), "max": float(annual.max()), **pct(annual)},
            "land": {"mean": float(annual[land].mean()), "min": float(annual[land].min()), "max": float(annual[land].max()), **pct(annual[land])},
            "ocean": {"mean": float(annual[~land].mean()), "min": float(annual[~land].min()), "max": float(annual[~land].max()), **pct(annual[~land])},
            "bands": band_stats(annual, land, lat),
        },
        "temperature_monthly_C": {
            name: {
                "mean": float(monthly[i].mean()),
                "land_mean": float(monthly[i][land].mean()),
                "ocean_mean": float(monthly[i][~land].mean()),
                "min": float(monthly[i].min()),
                "max": float(monthly[i].max()),
                "frozen_pct": float(frozen_month[i]),
            }
            for i, name in enumerate(MONTHS)
        },
        "seasonality_Jul_minus_Jan_C": {
            "land_amp_abs_mean": float(amp[land].mean()),
            "ocean_amp_abs_mean": float(amp[~land].mean()),
            "land_amp_p90": float(np.percentile(amp[land], 90)),
            "ocean_amp_p90": float(np.percentile(amp[~land], 90)),
            "midlat_land_amp_mean": float(
                amp[land & (np.abs(lat[:, None]) >= 30) & (np.abs(lat[:, None]) <= 60)].mean()
            ),
            "midlat_ocean_amp_mean": float(
                amp[(~land) & (np.abs(lat[:, None]) >= 30) & (np.abs(lat[:, None]) <= 60)].mean()
            ),
        },
        "coast_land_minus_ocean_annual_C": {
            "mean": float(jumps.mean()),
            "median": float(np.median(jumps)),
            "p10": float(np.percentile(jumps, 10)),
            "p90": float(np.percentile(jumps, 90)),
            "abs_max": float(np.max(np.abs(jumps))),
        },
        "open_water_frac_monthly": meta.get("open_water_frac_monthly"),
    }

    out = TEMP_DIR / "temperature_stats.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Pretty print
    print("=== Magdyn climate / terrain stats ===")
    print(f"Grid {w}×{h}  land {stats['grid']['land_pct']:.1f}%")
    print()
    print("Terrain (viewer decode)")
    te = stats["terrain_display"]["land_elev_m"]
    od = stats["terrain_display"]["ocean_depth_m_linear"]
    print(f"  land elev  mean {te['mean']:.0f} m  max {te['max']:.0f} m  p50 {te['p50']:.0f}")
    print(f"  ocean depth mean {od['mean']:.0f} m  med {od['median']:.0f} m  p90 {od['p90']:.0f}  max {od['max']:.0f}")
    print()
    ta = stats["temperature_annual_C"]
    print("Annual temperature (°C)")
    print(f"  global  mean {ta['global']['mean']:.1f}  [{ta['global']['min']:.1f}, {ta['global']['max']:.1f}]")
    print(f"  land    mean {ta['land']['mean']:.1f}  [{ta['land']['min']:.1f}, {ta['land']['max']:.1f}]")
    print(f"  ocean   mean {ta['ocean']['mean']:.1f}  [{ta['ocean']['min']:.1f}, {ta['ocean']['max']:.1f}]")
    for k, v in ta["bands"].items():
        if "mean_C" in v:
            print(f"  {k}: {v['mean_C']:.1f}")
        else:
            print(
                f"  {k}: all {v['all_mean_C']:.1f}  "
                f"land {v['land_mean_C']:.1f}  ocean {v['ocean_mean_C']:.1f}"
                if v.get("land_mean_C") is not None
                else f"  {k}: all {v['all_mean_C']:.1f}"
            )
    print()
    print("Monthly mean (global / land / ocean)  frozen%")
    for name in MONTHS:
        m = stats["temperature_monthly_C"][name]
        print(
            f"  {name[3:6]}  {m['mean']:5.1f} / {m['land_mean']:5.1f} / {m['ocean_mean']:5.1f}   "
            f"{m['frozen_pct']:4.1f}%"
        )
    print()
    s = stats["seasonality_Jul_minus_Jan_C"]
    print("Jul−Jan |amplitude| (°C)")
    print(f"  land mean {s['land_amp_abs_mean']:.1f}  p90 {s['land_amp_p90']:.1f}")
    print(f"  ocean mean {s['ocean_amp_abs_mean']:.1f}  p90 {s['ocean_amp_p90']:.1f}")
    print(f"  midlat land/ocean {s['midlat_land_amp_mean']:.1f} / {s['midlat_ocean_amp_mean']:.1f}")
    print()
    c = stats["coast_land_minus_ocean_annual_C"]
    print("Coast land−ocean ΔT annual (°C)")
    print(f"  mean {c['mean']:.2f}  med {c['median']:.2f}  p10/p90 {c['p10']:.2f}/{c['p90']:.2f}  |max| {c['abs_max']:.2f}")
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
