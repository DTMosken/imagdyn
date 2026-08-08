"""Tests for CLI argument parsing (no interactive menu / HTTP server)."""

from __future__ import annotations

import sys

import pytest

from imagdyn.cli import build_parser, main


def test_build_parser_viewer_defaults() -> None:
    p = build_parser()
    args = p.parse_args(["viewer", "--no-open", "--port", "9001"])
    assert args.command == "viewer"
    assert args.port == 9001
    assert args.no_open is True


def test_build_parser_ensure_flags() -> None:
    p = build_parser()
    args = p.parse_args(["ensure", "--force", "--no-template"])
    assert args.command == "ensure"
    assert args.force is True
    assert args.no_template is True


def test_temperature_strips_double_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, list[str]] = {}

    def fake_temp_main() -> None:
        called["argv"] = list(sys.argv)

    monkeypatch.setattr("imagdyn.temperature.main", fake_temp_main)

    ec = main(["temperature", "--", "--cpu", "--downsample", "2"])
    assert ec == 0
    assert "--cpu" in called["argv"]
    assert "--downsample" in called["argv"]
    assert "--" not in called["argv"]


def test_reshape_missing_returns_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate missing local reshape module → exit code 1."""
    monkeypatch.setitem(sys.modules, "imagdyn.reshape", None)
    ec = main(["reshape"])
    assert ec == 1


def test_pipeline_parser() -> None:
    p = build_parser()
    args = p.parse_args(["pipeline", "--temperature", "--force"])
    assert args.command == "pipeline"
    assert args.temperature is True
    assert args.force is True
