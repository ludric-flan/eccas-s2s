"""
Tests for the in-situ station reader (CDT / CPT) and its defensive handling of a
missing or empty station directory. Fixtures are synthetic and self-contained.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from eccas_s2s.io import obs_stations as obs


# ---- synthetic, format-compliant fixtures ---------------------------------------
def _write_cdt_csv(path):
    rows = [
        ["ID", "StationA", "StationB"],
        ["LON", "15.05", "18.38"],
        ["LAT", "12.11", "9.15"],
        ["DAILY/ELEV", "295", "365"],
        ["19910101", "0.0", "1.2"],
        ["19910102", "3.5", "-999.0"],   # missing sentinel -> NaN
        ["19910103", "0.0", "10.0"],
    ]
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def _write_cpt_csv(path):
    rows = [
        ["STATION", "StationA", "StationB"],
        ["LAT", "12.11", "9.15"],
        ["LON", "15.05", "18.38"],
        ["1991", "850.0", "1200.0"],
        ["1992", "910.0", "1100.0"],
    ]
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


# ---- CDT --------------------------------------------------------------------------
def test_parse_cdt(tmp_path):
    f = tmp_path / "stations_cdt.csv"
    _write_cdt_csv(f)
    ds = obs.read_station_file(str(f))

    assert list(ds.station.values) == ["StationA", "StationB"]
    assert ds.sizes == {"time": 3, "station": 2}
    np.testing.assert_allclose(ds.longitude.values, [15.05, 18.38])
    np.testing.assert_allclose(ds.latitude.values, [12.11, 9.15])
    assert "elevation" in ds.coords
    # missing sentinel became NaN
    assert np.isnan(ds["prcp"].sel(station="StationB").isel(time=1).item())
    assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("1991-01-01")


# ---- CPT --------------------------------------------------------------------------
def test_parse_cpt(tmp_path):
    f = tmp_path / "stations_cpt.csv"
    _write_cpt_csv(f)
    ds = obs.read_station_file(str(f))

    assert ds.sizes == {"time": 2, "station": 2}
    np.testing.assert_allclose(ds.longitude.values, [15.05, 18.38])
    np.testing.assert_allclose(ds.latitude.values, [12.11, 9.15])
    # yearly rows dated YYYY-08-01 by default
    assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("1991-08-01")
    assert ds["prcp"].sel(station="StationA").isel(time=0).item() == 850.0


def test_format_autodetection(tmp_path):
    cdt, cpt = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_cdt_csv(cdt)
    _write_cpt_csv(cpt)
    assert obs._detect_format(obs._read_table(str(cdt))) == "cdt"
    assert obs._detect_format(obs._read_table(str(cpt))) == "cpt"


# ---- defensive behavior -----------------------------------------------------------
def test_missing_directory_returns_none_with_warning():
    with pytest.warns(UserWarning, match="Station directory not found"):
        assert obs.load_stations("/no/such/dir/stations") is None


def test_empty_directory_returns_none_with_warning(tmp_path):
    with pytest.warns(UserWarning, match="No station files"):
        assert obs.load_stations(str(tmp_path), pattern="*.csv") is None


def test_load_stations_directory(tmp_path):
    _write_cdt_csv(tmp_path / "one.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a valid load must not warn
        ds = obs.load_stations(str(tmp_path), pattern="*.csv")
    assert ds is not None
    assert ds.sizes["station"] == 2


def test_duplicate_coords_are_nudged():
    ds = obs._to_dataset(
        ["A", "B"], lon=[10.0, 10.0], lat=[5.0, 5.0],
        times=pd.to_datetime(["1991-01-01"]), values=[[1.0, 2.0]],
    )
    # identical input coords must be made distinct
    assert ds.longitude.values[0] != ds.longitude.values[1]
