"""Tests for imagdyn.paths."""

from __future__ import annotations

from imagdyn import paths


def test_root_contains_package_and_viewer() -> None:
    assert (paths.ROOT / "imagdyn").is_dir()
    assert (paths.VIEWER_DIR / "index.html").is_file()


def test_graphs_helpers() -> None:
    assert paths.graphs_path("assets.json") == paths.GRAPHS / "assets.json"
    assert paths.temp_path(paths.TEMP_ANNUAL) == paths.TEMP_DIR / paths.TEMP_ANNUAL


def test_month_temp_names_count() -> None:
    assert len(paths.MONTH_TEMP_NAMES) == 12
    assert paths.MONTH_TEMP_NAMES[0].startswith("Temperature - 01")
    assert paths.MONTH_TEMP_NAMES[11].startswith("Temperature - 12")
