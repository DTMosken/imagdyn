"""Defaults in imagdyn.params stay aligned with CLI / synthesizer signatures."""

from __future__ import annotations

import inspect
import sys

from imagdyn.currents import OceanCurrentFilter, parse_args as parse_currents_args
from imagdyn.params import (
    CURRENTS,
    ENCODE,
    PLANET,
    TEMPERATURE,
    WIND,
    temperature_call_kwargs,
    wind_field_kwargs,
)
from imagdyn.temperature import parse_args as parse_temp_args
from imagdyn.temperature import synthesize_temperatures
from imagdyn.wind import WindField, parse_args as parse_wind_args


def test_planet_and_encode_earthlike() -> None:
    assert PLANET.radius_km == 6371.0
    assert PLANET.obliquity_deg == 23.5
    assert PLANET.max_elev_m == 8000.0
    assert ENCODE.t_gray_min < ENCODE.t_gray_max


def test_temperature_call_kwargs_match_synthesize() -> None:
    skip = {"elev01", "land_np", "device"}
    names = {
        n
        for n, p in inspect.signature(synthesize_temperatures).parameters.items()
        if n not in skip
    }
    assert set(temperature_call_kwargs()) == names


def test_wind_field_kwargs_match_init() -> None:
    names = {
        n
        for n in inspect.signature(WindField.__init__).parameters
        if n != "self"
    }
    assert set(wind_field_kwargs()) == names


def test_temperature_cli_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["temperature"])
    args = parse_temp_args()
    assert args.s0 == PLANET.s0
    assert args.obliquity == PLANET.obliquity_deg
    assert args.max_elev_m == PLANET.max_elev_m
    assert args.lapse == PLANET.lapse_k_per_km
    assert args.radius_km == PLANET.radius_km
    assert args.greenhouse == TEMPERATURE.greenhouse_factor
    assert args.transport_lambda == TEMPERATURE.transport_lambda
    assert args.lake_inertia == TEMPERATURE.lake_inertia
    assert args.heat_capacity_land == TEMPERATURE.heat_capacity_land
    assert args.heat_capacity_ocean == TEMPERATURE.heat_capacity_ocean
    assert args.spinup_years == TEMPERATURE.spinup_years
    assert args.freeze_ocean_c == TEMPERATURE.freeze_ocean_C
    assert args.freeze_land_c == TEMPERATURE.freeze_land_C
    assert args.current_warm_c == CURRENTS.warm_delta_C
    assert args.current_cold_c == CURRENTS.cold_delta_C
    assert args.t_gray_min == ENCODE.t_gray_min
    assert args.t_gray_max == ENCODE.t_gray_max
    assert args.currents is TEMPERATURE.currents
    assert args.wind is TEMPERATURE.sync_wind


def test_wind_cli_defaults() -> None:
    args = parse_wind_args([])
    assert args.max_elev_m == PLANET.max_elev_m
    assert args.lapse == PLANET.lapse_k_per_km
    assert args.t_spin_hours == PLANET.t_spin_hours
    assert args.radius_km == PLANET.radius_km


def test_currents_cli_and_filter_defaults() -> None:
    args = parse_currents_args([])
    assert args.peak_lat == CURRENTS.peak_lat_deg
    assert args.warm_c == CURRENTS.warm_delta_C
    assert args.cold_c == CURRENTS.cold_delta_C
    assert args.radius_km == PLANET.radius_km
    filt = OceanCurrentFilter()
    assert filt.warm_delta_C == CURRENTS.warm_delta_C
    assert filt.planet_radius_km == PLANET.radius_km


def test_windfield_defaults_from_params() -> None:
    wf = WindField()
    assert wf.p0_hpa == WIND.p0_hpa
    assert wf.belt_lon_sectors == WIND.belt_lon_sectors
    assert wf.planet_radius_km == PLANET.radius_km
    assert wf.lapse_k_per_km == PLANET.lapse_k_per_km
    assert wf.t_spin_s == PLANET.t_spin_hours * 3600.0
