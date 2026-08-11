"""Path constants for the IMagDyn project."""

from __future__ import annotations

from pathlib import Path

# Package lives at <repo>/imagdyn/; project root is parent.
ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "graphs"
TEMP_DIR = GRAPHS / "temperature"
TEMPLATE_DIR = GRAPHS / "template"
VIEWER_DIR = ROOT / "viewer"

FULL_ELEV = "Terrain - Full Elevation.png"
FULL_ELEV_BAK = "Terrain - Full Elevation.pre_nl.png"
LAND_MASK = "Terrain - Land Mask.png"
ABOVE_SEA = "Terrain - Elevation Above Sea Level.png"
BELOW_SEA = "Terrain - Elevation Below Sea Level.png"
CONTOURS = "Terrain - Contours.png"
SATELLITE = "Satellite Color.png"
ASSETS_JSON = "assets.json"

TEMP_ANNUAL = "Temperature - Annual Mean.png"
TEMP_META = "temperature_meta.json"
TEMP_STATS = "temperature_stats.json"

MONTH_TEMP_NAMES = [
    "Temperature - 01 January.png",
    "Temperature - 02 February.png",
    "Temperature - 03 March.png",
    "Temperature - 04 April.png",
    "Temperature - 05 May.png",
    "Temperature - 06 June.png",
    "Temperature - 07 July.png",
    "Temperature - 08 August.png",
    "Temperature - 09 September.png",
    "Temperature - 10 October.png",
    "Temperature - 11 November.png",
    "Temperature - 12 December.png",
]

WIND_DIR = GRAPHS / "wind"
WIND_META = "wind_meta.json"
WIND_STATS = "wind_stats.json"
PRESSURE_ANNUAL = "Pressure - Annual Mean.png"
# Packed color products (RGBA16 / RGB16) — no separate U/V float files.
WIND_UV_ANNUAL = "Wind - UV - Annual Mean.png"
WIND_TERRAIN_DOT_ANNUAL = "Wind - Terrain Dot - Annual Mean.png"

MONTH_NAMES_SHORT = [
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

MONTH_PRESSURE_NAMES = [f"Pressure - {n}.png" for n in MONTH_NAMES_SHORT]
MONTH_WIND_UV_NAMES = [f"Wind - UV - {n}.png" for n in MONTH_NAMES_SHORT]
MONTH_TERRAIN_DOT_NAMES = [f"Wind - Terrain Dot - {n}.png" for n in MONTH_NAMES_SHORT]


def graphs_path(*parts: str) -> Path:
    return GRAPHS.joinpath(*parts)


def temp_path(*parts: str) -> Path:
    return TEMP_DIR.joinpath(*parts)


def wind_path(*parts: str) -> Path:
    return WIND_DIR.joinpath(*parts)
