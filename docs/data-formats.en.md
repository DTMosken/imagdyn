# Data formats

*English | [中文](data-formats.md)*

Raster and JSON conventions used by IMagDyn.

## Full elevation `graphs/Terrain - Full Elevation.png`

Single-channel grayscale; **0.5** (about gray 128) = sea level:

| Range | Meaning |
|-------|---------|
| `> 0.5` | Land → linear map to `0 … max_elev_m` (default **+8000 m**, see [`imagdyn/params.py`](../imagdyn/params.py)) |
| `< 0.5` | Ocean → linear map to `0 … min_elev_m` (default **−8000 m**) |

Land/ocean prefers `Terrain - Land Mask.png` (white = land). Without a mask, `ensure` / the viewer infer land from elevation `> 0.5`.

Derived layers (`ensure`):

- `Terrain - Land Mask.png`
- `Terrain - Above Sea Level.png` / `Terrain - Below Sea Level.png`
- Optional `Satellite Color.png`

Example seed: `graphs/template/` (real-Earth full elevation). Menu item `2 ensure` can copy it into `graphs/`.

## Temperature `graphs/temperature/*.png`

```text
gray = clip((T_C − T_MIN) / (T_MAX − T_MIN) × 255)
```

Defaults: `T_MIN = −60 °C`, `T_MAX = +45 °C` (`ENCODE` in [`imagdyn/params.py`](../imagdyn/params.py)). Same directory:

- `temperature_meta.json` — generation parameters and summary
- `temperature_stats.json` — zonal / land–ocean stats from `summarize`

Monthly files look like `Temperature - 01 January.png`; annual mean is `Temperature - Annual Mean.png`.

## Wind `graphs/wind/`

| Product | Format |
|---------|--------|
| `Wind - UV - *.png` | RGB8: U→R, V→G, pressure→B |
| `Wind - Terrain Dot - *.png` | RGB: terrain_dot→R,G (16-bit split), speed→B (8-bit) |
| `wind_meta.json` | Scales, pressure/speed legend, per-period belt / AMC summary |
| `wind_stats.json` | Pressure/speed stats + ideal waterworld 1D profiles |

Physical conventions (written to meta):

- Stored UV: `+U` east, `+V` south (image row direction)
- Pressure is sea-level pressure (SLP): land temperature is lapse-corrected before pressure anomalies

## Asset index

`graphs/assets.json` is maintained by `status` / `ensure`. The viewer reads it first and falls back to per-file probing.

Back to [README](../README.en.md).
