# Temperature synthesis

*English | [中文](temperature.md)*

Modules: [`imagdyn/temperature.py`](../imagdyn/temperature.py), currents [`imagdyn/currents.py`](../imagdyn/currents.py).

## Pipeline

1. From full elevation and the land mask, get land height and a soft coastline.
2. Twelve months: daily-mean TOA insolation (obliquity default ±23.5°) → annual mean + heat transport → radiative-equilibrium temperature.
3. Maritime influence: mean-filter diffusion from open water, yielding `maritime`.
4. Continentality / seasonal sensitivity, polar-ocean damping, elevation correction.
5. Ocean temperature is diffused toward land sea-level targets; current ΔT is generated (warm east coasts / cold west coasts, Gaussian in latitude peaking near 30°).
6. Coastal land–sea mixing; water at or below the freeze threshold does not count as open water that month.

## Lake inertia

Among connected water bodies, inland lakes whose area is ≤ `--lake-max-area-km2` (default 20 000 km²) have thermal inertia reduced to `--lake-inertia` (default **0.6**). Open ocean uses `--ocean-inertia` (default 0.90).

## Common CLI

```bash
python -m imagdyn temperature -- --help
python -m imagdyn temperature -- --cpu --no-wind --no-currents
python -m imagdyn currents -- --dump-maps
```

Key flags: `--greenhouse`, `--heat-transport`, `--maritime-efold-km`, `--maritime-diffuse-passes`, `--ocean-sst-nudge`, `--lake-inertia`, `--lake-max-area-km2`.

Back to [README](../README.en.md) · [Data formats](data-formats.en.md).
