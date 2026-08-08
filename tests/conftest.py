"""Shared fixtures for IMagDyn tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from imagdyn import paths


@pytest.fixture
def tiny_elev01() -> np.ndarray:
    """4x6 full-elevation map: land (>0.5) left, ocean (<0.5) right."""
    elev = np.full((4, 6), 0.25, dtype=np.float32)
    elev[:, :3] = 0.75  # land
    elev[0, 0] = 1.0  # peak
    elev[0, 5] = 0.0  # deep ocean
    return elev


@pytest.fixture
def graphs_dir(tmp_path: Path, tiny_elev01: np.ndarray) -> Path:
    """Temporary graphs/ with a Full Elevation PNG."""
    g = tmp_path / "graphs"
    g.mkdir()
    arr = np.clip(np.rint(tiny_elev01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(g / paths.FULL_ELEV)
    return g
