# IMagDyn

*English | [中文](README.md)*

An **interactive climate and wind toolkit** for custom maps: from a grayscale full-elevation image, generate land/ocean masks, contours, monthly and annual temperature, pressure and wind, then inspect them in the browser.

| | |
|--|--|
| Display name | **IMagDyn** |
| Python package | `imagdyn` |
| Entry points | `IMagDyn.cmd` (Windows) · `IMagDyn.sh` (Unix) |

Typical flow: place or seed a full-elevation map → `ensure` / `contours` → `temperature` (wind on by default) → `viewer`.

---

## Screenshots

#### Full elevation

![Full elevation](docs/screenshots/terrain-full-elevation.png)

#### Annual-mean temperature

![Annual-mean temperature](docs/screenshots/temperature-annual-mean.png)

#### Seasonal temperature (Jun / Sep / Dec)

| June | September | December |
|------|-----------|----------|
| ![June temperature](docs/screenshots/temperature-june.png) | ![September temperature](docs/screenshots/temperature-september.png) | ![December temperature](docs/screenshots/temperature-december.png) |

#### Annual-mean pressure

![Annual-mean pressure](docs/screenshots/wind-pressure-annual-mean.png)

#### Seasonal pressure (Jun / Sep / Dec)

| June | September | December |
|------|-----------|----------|
| ![June pressure](docs/screenshots/wind-pressure-june.png) | ![September pressure](docs/screenshots/wind-pressure-september.png) | ![December pressure](docs/screenshots/wind-pressure-december.png) |

#### Monsoon (June / December)

| June | December |
|------|----------|
| ![Monsoon June](docs/screenshots/monsoon-june.png) | ![Monsoon December](docs/screenshots/monsoon-december.png) |

---

## User guide

### Launch

**No arguments** opens the interactive menu; **with arguments** the subcommand runs directly.

```bash
# Windows (double-click works)
IMagDyn.cmd

# Linux / macOS / Git Bash
./IMagDyn.sh
```

Use a conda environment with the dependencies installed. Set the environment from the menu item **Environment** (written to `.imagdyn_env`); later launches enter it via `conda run`.

```bash
IMagDyn.cmd --lang zh
./IMagDyn.sh --lang en
python -m imagdyn menu --lang zh
```

`viewer.bat` starts only the map viewer. Legacy names `magdyn.cmd` / `magdyn.sh` still forward to the new entry points.

### Dependencies

- Python **3.10+**
- Libraries: `numpy`, `Pillow`, `scipy`, `torch`
- Recommended: CUDA

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # + pytest / flake8
python -m pytest
```

### Interactive menu

| Key | Action |
|-----|--------|
| 1 | `status` — check assets, write `graphs/assets.json` |
| 2 | `ensure` — derive Land Mask / Above / Below from full elevation (can seed from `graphs/template/`) |
| 3 | `contours` — land contours |
| 4 | `temperature` — 12 months + annual mean (wind on by default; `--no-wind` to skip) |
| 5 | `wind` — wind / pressure (reads existing temperature maps) |
| 6 | `summarize` — climate / terrain / wind stats (including `wind_stats.json`) |
| 7 | `viewer` — local HTTP viewer |
| 8 | `pipeline` — one-shot pipeline |
| 9 | **Environment** — Python / conda; set the env and relaunch |
| L | Switch 中文 / English (`.imagdyn_lang`) |
| 0 | Quit |

Missing dependencies print an error and a hint to activate the environment.

### Common commands

```bash
IMagDyn.cmd status
./IMagDyn.sh ensure
python -m imagdyn viewer --port 8765
python -m imagdyn temperature -- --cpu
python -m imagdyn wind --
python -m imagdyn pipeline --temperature
```

Arguments for the temperature / wind generators go after `--`, for example:

```bash
python -m imagdyn temperature -- --cpu --maritime-iters 2 --no-wind
python -m imagdyn wind -- --annual-only --cpu
```

### Recommended pipeline

```text
ensure → contours → temperature (with wind) → summarize → viewer
```

```bash
./IMagDyn.sh pipeline --temperature
./IMagDyn.sh pipeline --force --temperature   # force-rebuild derived terrain
./IMagDyn.sh pipeline --wind                  # wind only, if temperature already exists
```

| Command | Description |
|---------|-------------|
| `status` | List assets and write `assets.json` |
| `ensure` | Derive mask / Above / Below; `--force` rebuilds |
| `contours` | Minor 200 m, major 1000 m |
| `temperature` | Temperature maps; wind by default; `--no-wind`, `--cpu` |
| `wind` | Pressure / UV / terrain_dot PNGs + meta / stats |
| `summarize` | Terminal summary + `temperature_stats.json` / `wind_stats.json` |
| `viewer` | [http://127.0.0.1:8765/viewer/](http://127.0.0.1:8765/viewer/) (run `ensure` first) |
| `pipeline` | `ensure` → `contours`; optional `--temperature` / `--wind` |

### Inputs and outputs

- **Input:** `graphs/Terrain - Full Elevation.png` (0.5 ≈ sea level). An example can be seeded from `graphs/template/` via `ensure`.
- **Temperature:** `graphs/temperature/`
- **Wind:** `graphs/wind/`
- **Viewer:** `./IMagDyn.sh viewer` → layers, legend, graticule, wind arrows, hover / pin readout

After changing terrain, re-run `temperature` (and `summarize` / `wind` if needed), or layers may not match the new elevation.

Encoding, layer conventions, and viewer controls are in the technical notes linked below.

### Local config

| File | Purpose |
|------|---------|
| `.imagdyn_env` | Preferred conda environment name |
| `.imagdyn_lang` | `zh` / `en` |

Legacy `.magdyn_env` / `.magdyn_lang` are still read.

---

## Technical notes (brief)

Pipeline overview:

```text
Full Elevation → ensure / contours
              → temperature (radiation + land/ocean inertia + current SST)
              → wind (pressure belts + thermal pressure → geostrophic/friction balance → terrain)
              → summarize / viewer
```

| Topic | Summary | Details |
|-------|---------|---------|
| Data formats | Elevation gray, temperature gray, packed wind RGB, JSON metadata | [docs/data-formats.en.md](docs/data-formats.en.md) |
| Temperature | Daily-mean TOA insolation, heat-diffusion maritime buffer, small-lake inertia, current SST | [docs/temperature.en.md](docs/temperature.en.md) |
| Wind / pressure | 36 longitude sectors, cosine belts, Rayleigh AMC subtropical highs, quadratic drag | [docs/wind.en.md](docs/wind.en.md) |
| Viewer | Local static server, layers and wind arrows, readout pins | [docs/viewer.en.md](docs/viewer.en.md) |

### Repository layout

```text
IMagDyn/
├── IMagDyn.cmd / IMagDyn.sh
├── imagdyn/                 # Python package (cli / temperature / wind / …)
├── viewer/index.html
├── docs/                    # technical notes and screenshots/
└── graphs/                  # terrain, temperature, wind products
```

---

## License

This project uses a **dual license**. See [`NOTICE`](NOTICE):

| Material | License |
|----------|---------|
| **Source code** (`imagdyn/`, entry scripts, `viewer/`, …) | [GPLv3](LICENSE) |
| **Generated files** (e.g. derived maps and JSON under `graphs/`) | [CC BY-SA 4.0](LICENSE.CC-BY-SA-4.0) |

Copyright (C) 2026 DTMosken

Third-party maps or data used as input remain under their original licenses; this project does not grant rights to those inputs.
