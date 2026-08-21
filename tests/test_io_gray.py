"""Tests for grayscale load / save helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from imagdyn.io_gray import load_gray01, save_gray01, save_gray_png, save_mask, write_text_atomic


def test_save_load_gray01_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "g.png"
    src = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    save_gray01(path, src)
    got = load_gray01(path)
    assert got.shape == src.shape
    np.testing.assert_allclose(got, src, atol=1 / 255)


def test_load_gray01_accepts_rgb(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "rgb.png"
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 128
    rgb[..., 1] = 200
    rgb[..., 2] = 50
    Image.fromarray(rgb, mode="RGB").save(path)
    got = load_gray01(path)
    assert got.ndim == 2
    np.testing.assert_allclose(got, 128 / 255.0, atol=1 / 255)


def test_save_mask(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    land = np.array([[True, False], [False, True]])
    save_mask(path, land)
    got = load_gray01(path)
    assert got[0, 0] == 1.0
    assert got[0, 1] == 0.0
    assert got[1, 1] == 1.0


def test_save_gray_png_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "Temperature - Annual Mean.png"
    save_gray_png(path, np.zeros((4, 4), dtype=np.uint8))
    save_gray_png(path, np.full((4, 4), 200, dtype=np.uint8))
    got = load_gray01(path)
    np.testing.assert_allclose(got, 200 / 255.0, atol=1 / 255)


def test_write_text_atomic_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "temperature_meta.json"
    write_text_atomic(path, '{"a": 1}\n')
    write_text_atomic(path, '{"a": 2}\n')
    assert path.read_text(encoding="utf-8") == '{"a": 2}\n'
