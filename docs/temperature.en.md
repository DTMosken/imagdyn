# Temperature synthesis

*English | [中文](temperature.md)*

Modules: [`imagdyn/temperature.py`](../imagdyn/temperature.py), currents [`imagdyn/currents.py`](../imagdyn/currents.py). Defaults live in [`imagdyn/params.py`](../imagdyn/params.py).

## Pipeline

1. From full elevation and the land mask, get land height, water depth, and a soft coast. East–west kernels stretch by ``1/cos(φ)`` so polar coasts are not too wide.
2. Initialize the map (default 280 K) and run ``spinup_years`` (default **6**) virtual years on the GPU. Each month-step: ice-albedo sigmoid → ``C_eff = C_base + C_latent`` → ``Q_abs = (1-α)Q + Q_curr + Q_transport`` → implicit update ``T_new = T + (Q_abs − σT⁴/G) / (C/Δt + 4σT³/G)`` → lift to sea-level T, metric-diffuse, subtract lapse. Current ΔT is computed once, then each month becomes ``Q_curr = (4σT³/G) ΔT`` on the world ocean only. Heat transport is ``Q_transport = λ (T̄_global − T_local)`` with ``λ = transport_lambda`` (default **3.8** W/m²/K); ``T̄_global`` is the area-weighted (``dA ∝ cos(φ)``) global mean.
3. Keep **only the last 12 months** for maps and statistics (spin-up is not written).
4. Coast: ``aa_blend_px`` (default 1 equatorial pixel) anti-alias only; no wide land–sea blend.

## Heat capacity and ice

- Land: ``heat_capacity_land``.
- World ocean: ``heat_capacity_ocean · I(z)`` with ``I(z) = I_shallow + (I_deep − I_shallow)(1 − e^{−z/d0})``, ``d0 = mix_depth_m`` (default 200 m).
- Inland lakes: spherical cell area ``dA ∝ cos(φ)``. Area ≤ ``lake_max_area_km2`` uses coefficient ``lake_inertia`` (default 0.45), ramping to **1** at ``lake_max_area_km2``. Freeze points: land/lakes **0.0 °C**, ocean **−1.8 °C**. Latent heat is a Gaussian virtual capacity at the freeze point (``δT`` default 0.8 °C).

Greenhouse ``G`` divides longwave (``OLR = σT⁴/G``); shortwave is not multiplied by ``G``.

## Common CLI

```bash
python -m imagdyn temperature -- --help
python -m imagdyn temperature -- --cpu --no-wind --no-currents
python -m imagdyn currents -- --dump-maps
```

Key flags: `--greenhouse`, `--transport-lambda`, `--heat-capacity-land`, `--heat-capacity-ocean`, `--spinup-years`, `--maritime-efold-km`, `--lake-inertia`, `--freeze-land-c`, `--freeze-ocean-c`.

Back to [README](../README.en.md) · [Data formats](data-formats.en.md).
