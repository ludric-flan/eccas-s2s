"""
Offline tests for the IRIDL observed-rainfall readers (CHIRPS, TAMSAT) and the ERA5
request builder. No network required.
"""
import pandas as pd

from eccas_s2s.io import iridl, chirps, tamsat, era5


def test_iridl_token_formatting():
    assert iridl.fmt_lon(4) == "(4E)"
    assert iridl.fmt_lon(-10) == "(10W)"
    assert iridl.fmt_lat(25) == "(25N)"
    assert iridl.fmt_lat(-21) == "(21S)"
    assert iridl.fmt_month("2000-06-15") == "(Jun 2000)"
    assert iridl.xy_range((4, 36, -21, 25)) == (
        "/X/(4E)/(36E)/RANGEEDGES/Y/(21S)/(25N)/RANGEEDGES"
    )


def test_decode_months_since_custom_epoch():
    idx = iridl.decode_months_since([0, 12], "months since 1991-01-01")
    assert idx[0] == pd.Timestamp("1991-01-01")
    assert idx[1] == pd.Timestamp("1992-01-01")


def test_build_chirps_url():
    url = chirps.build_chirps_url((4, 36, -21, 25), 1991, 2020)
    assert url == (
        "https://iridl.ldeo.columbia.edu/SOURCES"
        "/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation"
        "/X/(4E)/(36E)/RANGEEDGES/Y/(21S)/(25N)/RANGEEDGES"
        "/T/(Jan%201991)/(Dec%202020)/RANGEEDGES"
        "/data.nc"
    )


def test_build_tamsat_url_dods():
    url = tamsat.build_tamsat_url((4, 36, -21, 25), 2000, 2000, fmt="dods")
    assert url.startswith(
        "https://iridl.ldeo.columbia.edu/SOURCES"
        "/.Reading/.Meteorology/.TAMSAT/.v3p1/.monthly/.rfe"
    )
    assert url.endswith("/dods")


def test_build_era5_request_single_and_pressure():
    ds, req = era5.build_era5_request(["SST", "SLP"], [2020], [6, 7, 8], (4, 36, -21, 25))
    assert ds == "reanalysis-era5-single-levels-monthly-means"
    assert req["variable"] == ["sea_surface_temperature", "mean_sea_level_pressure"]
    assert req["month"] == ["06", "07", "08"]
    assert req["area"] == [25, 4, -21, 36]

    dsp, reqp = era5.build_era5_request(["U"], [2020], [1], (4, 36, -21, 25),
                                        level_type="pressure", pressure_level="850")
    assert dsp == "reanalysis-era5-pressure-levels-monthly-means"
    assert reqp["pressure_level"] == ["850"]
