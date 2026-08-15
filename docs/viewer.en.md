# Viewer

*English | [中文](viewer.md)*

Front end: [`viewer/index.html`](../viewer/index.html). `imagdyn viewer` serves it over local HTTP.

## Start

```bash
./IMagDyn.sh viewer
# → http://127.0.0.1:8765/viewer/
```

Optional `--port`. Derived terrain (`ensure`) is required first; temperature / wind layers load from whatever files exist under `graphs/`.

## Features

- **Layers:** satellite (hidden if missing), elevation, temperature, pressure, and so on
- **Legend:** always visible; color scale / land outline / contours / **graticule** / **wind arrows** can be toggled
- **Graticule:** 30° grid; equator, tropics (±23.5°), polar circles (±66.5°)
- **Contours:** minor 200 m, major 1000 m (thicker)
- **Wind:** overlay arrows; UV / pressure decoded from the packed PNGs (see [Data formats](data-formats.en.md))
- **Readout:** hover and pins (lat/lon, land/ocean, elevation or depth, temperature); pins can show an annual temperature curve
- **Assets:** prefer `graphs/assets.json`, otherwise probe files one by one
- **Language:** top-right button toggles **EN** / **中文**; stored in `localStorage` (`imagdyn_viewer_lang`), or `?lang=en` / `?lang=zh`

Screenshots: [README · Screenshots](../README.en.md#screenshots).

Back to [README](../README.en.md).
