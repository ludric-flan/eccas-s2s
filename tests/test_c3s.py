"""
Offline tests for the C3S/CDS request builder. No network or CDS credentials required
(cdsapi is only imported inside the download functions).
"""
import pytest

from eccas_s2s.io import c3s


def test_area_to_cds_conversion():
    # (lon_min, lon_max, lat_min, lat_max) -> [North, West, South, East]
    assert c3s._area_to_cds((4, 36, -21, 25)) == [25, 4, -21, 36]


def test_build_c3s_request_forecast():
    dataset, req = c3s.build_c3s_request(
        "ecmwf", "PRCP", ["2026"], init_month=6, area=(4, 36, -21, 25),
    )
    assert dataset == "seasonal-original-single-levels"
    assert req["originating_centre"] == "ecmwf"
    assert req["system"] == "51"                  # default system for ECMWF
    assert req["variable"] == ["total_precipitation"]
    assert req["year"] == ["2026"]
    assert req["month"] == ["06"]
    assert req["area"] == [25, 4, -21, 36]
    assert req["leadtime_hour"][0] == "24"


def test_build_c3s_request_explicit_system_and_hindcast_years():
    _, req = c3s.build_c3s_request(
        "ukmo", "PRCP", [str(y) for y in range(1993, 1996)],
        init_month=6, area=(4, 36, -21, 25), system=610,
    )
    assert req["system"] == "610"
    assert req["year"] == ["1993", "1994", "1995"]


def test_build_c3s_request_unknown_centre_requires_system():
    with pytest.raises(KeyError):
        c3s.build_c3s_request("acme", "PRCP", ["2026"], 6, (4, 36, -21, 25))
