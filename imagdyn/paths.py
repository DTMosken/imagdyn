"""Path constants for the magdyn project."""

from __future__ import annotations

from pathlib import Path

# Package lives at <repo>/magdyn/; project root is parent.
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


def graphs_path(*parts: str) -> Path:
    return GRAPHS.joinpath(*parts)


def temp_path(*parts: str) -> Path:
    return TEMP_DIR.joinpath(*parts)
