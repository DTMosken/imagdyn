# Wind and pressure

*English | [中文](wind.md)*

Module: [`imagdyn/wind.py`](../imagdyn/wind.py).

## Pipeline

1. Sea-levelize surface temperature (undo land lapse) → drive SLP.
2. **Planetary pressure belts** (longitude split into **36** sectors):
   - Sector-mean temperature → thermal equator;
   - Hadley-cell simulation: upper-level Rayleigh-drag angular-momentum path integral + thermal-wind `u_crit` → subtropical high-pressure belts;
   - Max `|dT̄/dφ|` poleward of the subtropical high → subpolar low-pressure belts;
   - Polar highs fixed at **±88°**;
   - Adjacent belts joined by a **half-period cosine**;
   - Sector results composited with **periodic longitude interpolation**; meta stores mean latitudes and `upper_amc`.
3. Local thermal pressure anomalies (weaker near the equator due to the **weak temperature-gradient approximation**) + dynamic land damping of high-pressure anomalies.
4. 2D diffusion (longitude wraps); **85°→88°** fade of east–west anomalies.
5. ∇p → Coriolis + **quadratic** surface-drag balance; neighborhood convolution of UV.
6. Terrain block / divert / lee (water elevation treated as 0; `|lat|≥88` forced to ocean).

Default map rotation period is **24 h** (equator).

## Products

See [Data formats · Wind](data-formats.en.md#wind-graphswind). `summarize` / `wind` write `wind_stats.json` (including 1D aquaplanet profiles for three subsolar cases: all ocean, elevation 0).

## Common CLI

```bash
python -m imagdyn wind --
python -m imagdyn wind -- --annual-only --cpu
python -m imagdyn temperature -- --no-wind
```

Back to [README](../README.en.md) · [Data formats](data-formats.en.md).
