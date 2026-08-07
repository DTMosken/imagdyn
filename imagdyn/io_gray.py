"""Grayscale image load / save helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def save_gray01(path: Path, v01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.clip(np.rint(v01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(g, mode="L").save(path)


def save_mask(path: Path, land: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.where(land, 255, 0).astype(np.uint8)
    Image.fromarray(g, mode="L").save(path)
