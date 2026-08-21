#!/usr/bin/env python3
"""Summarize temperature / terrain statistics for IMagDyn."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import paths
from .io_gray import write_text_atomic
from .params import ENCODE, PLANET
from .temperature import area_weighted_mean_np
from .timing import StepTimer

ROOT = paths.ROOT
TEMP_DIR = paths.TEMP_DIR
T_MIN, T_MAX = ENCODE.t_gray_min, ENCODE.t_gray_max
MAX_ELEV = PLANET.max_elev_m
MIN_ELEV = -PLANET.max_elev_m
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
    """Signed elevation (README): land 0…+MAX_ELEV, ocean 0…MIN_ELEV, both linear in gray."""
    v = g01
    out = np.empty_like(v)
    out[land] = np.clip((v[land] - 0.5) / 0.5 * MAX_ELEV, 0.0, MAX_ELEV)
    out[~land] = np.clip((0.5 - v[~land]) / 0.5, 0.0, 1.0) * MIN_ELEV
    return out


def _wmean(field: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if mask is not None and not np.any(mask):
        return None
    return float(area_weighted_mean_np(field, lat, mask=mask))


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
            "all_mean_C": _wmean(T, lat, m),
            "land_mean_C": _wmean(T, lat, m & land),
            "ocean_mean_C": _wmean(T, lat, m & ~land),
        }
    out["eq ocean (|lat|<10)"] = {
        "mean_C": _wmean(T, lat, (np.abs(lat2d) < 10) & (~land))
    }
    out["polar ocean (|lat|>60)"] = {
        "mean_C": _wmean(T, lat, (np.abs(lat2d) > 60) & (~land))
    }
    return out


def main() -> None:
    wall = StepTimer("summarize")
    meta_path = TEMP_DIR / "temperature_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    with wall.step("load"):
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

    with wall.step("coast"):
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

    with wall.step("stats"):
        freeze_land = float(meta.get("freeze_land_C", 0.0))
        freeze_ocean = float(meta.get("freeze_ocean_C", meta.get("freeze_C", -1.8)))
        frozen_month = [
            100.0
            * area_weighted_mean_np(
                (
                    (land & (monthly[m] <= freeze_land))
                    | ((~land) & (monthly[m] <= freeze_ocean))
                ).astype(np.float64),
                lat,
            )
            for m in range(12)
        ]

        stats = {
            "grid": {
                "width": w,
                "height": h,
                "land_pct": float(100.0 * area_weighted_mean_np(land.astype(np.float64), lat)),
            },
            "meta_summary": {
                "device": meta.get("device"),
                "s0_W_m2": meta.get("s0_W_m2"),
                "obliquity_deg": meta.get("obliquity_deg"),
                "greenhouse_factor": meta.get("greenhouse_factor"),
                "heat_capacity_land": meta.get("heat_capacity_land"),
                "heat_capacity_ocean": meta.get("heat_capacity_ocean"),
                "spinup_years": meta.get("spinup_years"),
                "freeze_land_C": meta.get("freeze_land_C"),
                "freeze_ocean_C": meta.get("freeze_ocean_C"),
                "gray_scale_C": meta.get("gray_scale_C", [T_MIN, T_MAX]),
            },
            "terrain_display": {
                "land_elev_m": {
                    "mean": _wmean(elev, lat, land),
                    "max": float(land_elev.max()) if land_elev.size else 0.0,
                    **pct(land_elev),
                },
                "ocean_depth_m_linear": {
                    "note": f"elev = min_elev_m * (0.5-v)/0.5  (min_elev_m={MIN_ELEV})",
                    "mean": _wmean(-elev, lat, ~land) if (~land).any() else 0.0,
                    "median": float(np.median(ocean_depth)) if ocean_depth.size else 0.0,
                    "max": float(ocean_depth.max()) if ocean_depth.size else 0.0,
                    **pct(ocean_depth),
                },
            },
            "temperature_annual_C": {
                "global": {
                    "mean": _wmean(annual, lat),
                    "min": float(annual.min()),
                    "max": float(annual.max()),
                    **pct(annual),
                },
                "land": {
                    "mean": _wmean(annual, lat, land),
                    "min": float(annual[land].min()) if land.any() else None,
                    "max": float(annual[land].max()) if land.any() else None,
                    **(pct(annual[land]) if land.any() else {}),
                },
                "ocean": {
                    "mean": _wmean(annual, lat, ~land),
                    "min": float(annual[~land].min()) if (~land).any() else None,
                    "max": float(annual[~land].max()) if (~land).any() else None,
                    **(pct(annual[~land]) if (~land).any() else {}),
                },
                "bands": band_stats(annual, land, lat),
            },
            "temperature_monthly_C": {
                name: {
                    "mean": _wmean(monthly[i], lat),
                    "land_mean": _wmean(monthly[i], lat, land),
                    "ocean_mean": _wmean(monthly[i], lat, ~land),
                    "min": float(monthly[i].min()),
                    "max": float(monthly[i].max()),
                    "frozen_pct": float(frozen_month[i]),
                }
                for i, name in enumerate(MONTHS)
            },
            "seasonality_Jul_minus_Jan_C": {
                "land_amp_abs_mean": _wmean(amp, lat, land),
                "ocean_amp_abs_mean": _wmean(amp, lat, ~land),
                "land_amp_p90": float(np.percentile(amp[land], 90)) if land.any() else None,
                "ocean_amp_p90": float(np.percentile(amp[~land], 90)) if (~land).any() else None,
                "midlat_land_amp_mean": _wmean(
                    amp,
                    lat,
                    land & (np.abs(lat[:, None]) >= 30) & (np.abs(lat[:, None]) <= 60),
                ),
                "midlat_ocean_amp_mean": _wmean(
                    amp,
                    lat,
                    (~land) & (np.abs(lat[:, None]) >= 30) & (np.abs(lat[:, None]) <= 60),
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
        write_text_atomic(
            out, json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
        )

    # Pretty print
    print("=== IMagDyn climate / terrain stats ===")
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

    # Always (re)generate wind_stats.json when annual temperature is available
    wind_stats_path = paths.WIND_DIR / paths.WIND_STATS
    ann_path = TEMP_DIR / "Temperature - Annual Mean.png"
    with wall.step("wind_stats"):
        if ann_path.is_file() and land.shape == annual.shape:
            try:
                from .wind import compute_and_write_wind_stats, elev01_to_meters, get_device

                print("  wind stats + waterworld 1D…", flush=True)
                elev_m_w = elev01_to_meters(elev01, land, max_elev_m=MAX_ELEV)
                device = get_device(prefer_gpu=True)
                wind_stats_path = compute_and_write_wind_stats(
                    annual.astype(np.float32),
                    land,
                    elev_m_w,
                    device=device,
                    out_dir=paths.WIND_DIR,
                )
                print(f"  wrote {wind_stats_path}", flush=True)
            except Exception as exc:  # noqa: BLE001 — stats should not abort summarize
                print(f"  wind stats skipped: {exc}", flush=True)
        else:
            print(
                "  wind stats skipped: need Temperature - Annual Mean.png "
                "matching terrain grid",
                flush=True,
            )

    if wind_stats_path.is_file():
        ws = json.loads(wind_stats_path.read_text(encoding="utf-8"))
        print()
        print("=== Wind / pressure stats ===")
        pg = ws.get("pressure_hpa", {}).get("global", {})
        sg = ws.get("speed_m_s", {}).get("global", {})
        if pg:
            print(
                f"  pressure mean {pg.get('mean', 0):.1f} hPa  "
                f"[{pg.get('min', 0):.1f}, {pg.get('max', 0):.1f}]"
            )
        if sg:
            print(
                f"  speed mean {sg.get('mean', 0):.2f} m/s  "
                f"p95 {sg.get('p95', 0):.2f}  max {sg.get('max', 0):.2f}"
            )
        ww_root = ws.get("waterworld_1d", {})
        cases = ww_root.get("cases") or {}
        if cases:
            print("  waterworld 1D (aquaplanet elev=0, all ocean):")
            for key in ("tropic_cancer", "equator", "tropic_capricorn"):
                c = cases.get(key) or {}
                sm = c.get("summary") or {}
                wwp = sm.get("pressure_hpa") or {}
                wws = sm.get("speed_m_s") or {}
                teq = c.get("thermal_equator_lat_deg")
                teq_s = f"{teq:.1f}°" if isinstance(teq, (int, float)) else "—"
                print(
                    f"    {c.get('label', key)}: Teq={teq_s}  "
                    f"p mean {wwp.get('mean', 0):.1f} hPa  "
                    f"speed mean {wws.get('mean', 0):.2f} m/s"
                )
        else:
            ww = ww_root.get("summary", {})
            if ww:
                wwp = ww.get("pressure_hpa", {})
                wws = ww.get("speed_m_s", {})
                print(
                    f"  waterworld 1D  p mean {wwp.get('mean', 0):.1f} hPa  "
                    f"speed mean {wws.get('mean', 0):.2f} m/s"
                )
        print(f"  (from {wind_stats_path})")
    print(wall.summary())


if __name__ == "__main__":
    main()
