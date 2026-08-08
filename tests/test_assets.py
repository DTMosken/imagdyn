"""Tests for asset probing and terrain derivation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imagdyn import paths
from imagdyn.assets import (
    derive_from_full_elevation,
    ensure_derived_terrain,
    probe_assets,
    write_assets_json,
)


def test_derive_from_full_elevation(tiny_elev01: np.ndarray) -> None:
    land, above, below = derive_from_full_elevation(tiny_elev01)
    assert land.dtype == bool or land.dtype == np.bool_
    assert land[:, :3].all()
    assert not land[:, 3:].any()
    assert above[0, 0] == pytest.approx(1.0)
    assert above[0, 5] == 0.0
    assert below[0, 5] == pytest.approx(1.0)
    assert below[0, 0] == pytest.approx(1.0)  # land → below channel is 1.0


def test_probe_and_write_assets_json(graphs_dir: Path) -> None:
    st = probe_assets(graphs_dir)
    assert st.full_elevation is True
    assert st.land_mask is False
    assert st.temperature_months is not None
    assert len(st.temperature_months) == 12

    out = write_assets_json(st, graphs_dir)
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["full_elevation"] is True
    assert data["paths"]["full_elevation"] == paths.FULL_ELEV
    assert len(data["paths"]["temperature_months"]) == 12


def test_ensure_derived_terrain(graphs_dir: Path) -> None:
    st = ensure_derived_terrain(graphs=graphs_dir, force=True, seed_template=False)
    assert st.full_elevation and st.land_mask and st.above_sea and st.below_sea
    assert (graphs_dir / paths.LAND_MASK).is_file()
    assert (graphs_dir / paths.ABOVE_SEA).is_file()
    assert (graphs_dir / paths.BELOW_SEA).is_file()
    assert (graphs_dir / paths.ASSETS_JSON).is_file()


def test_ensure_missing_full_raises(tmp_path: Path) -> None:
    g = tmp_path / "empty"
    g.mkdir()
    with pytest.raises(FileNotFoundError):
        ensure_derived_terrain(graphs=g, seed_template=False)
