"""Grayscale image load / save helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image


def replace_atomic(
    path: Path,
    write_tmp: Callable[[Path], None],
    *,
    attempts: int = 8,
) -> None:
    """
    Write to a same-dir temp file then ``os.replace`` onto ``path``.

    Windows often raises OSError when overwriting a file the viewer (or
    antivirus) still has open; retry + replace is more reliable than writing
    straight to the final path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    last_err: OSError | None = None
    for i in range(max(int(attempts), 1)):
        try:
            write_tmp(tmp)
            os.replace(tmp, path)
            return
        except OSError as e:
            last_err = e
            try:
                if tmp.is_file():
                    tmp.unlink()
            except OSError:
                pass
            time.sleep(0.05 * (i + 1))
    assert last_err is not None
    raise last_err


def save_png_atomic(path: Path, img: Image.Image, *, attempts: int = 8) -> None:
    def _write(tmp: Path) -> None:
        img.save(tmp, format="PNG", compress_level=1)

    replace_atomic(path, _write, attempts=attempts)


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8", attempts: int = 8) -> None:
    def _write(tmp: Path) -> None:
        tmp.write_text(text, encoding=encoding)

    replace_atomic(path, _write, attempts=attempts)


def load_gray01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def save_gray01(path: Path, v01: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.clip(np.rint(v01 * 255.0), 0, 255).astype(np.uint8)
    save_png_atomic(path, Image.fromarray(g, mode="L"))


def save_mask(path: Path, land: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.where(land, 255, 0).astype(np.uint8)
    save_png_atomic(path, Image.fromarray(g, mode="L"))


def save_gray_png(path: Path, gray: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_png_atomic(path, Image.fromarray(gray, mode="L"))
