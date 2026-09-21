"""Tests of the ERA5 temperature reference: continuity, calendar means, conservative regridding."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.obs.era5 import open_era5_daily, process_daily
from eccas_s2s.obs.regrid import conservative_to_degree


def _daily(start="1995-12-01", end="1996-03-31", nlat=8, nlon=4):
    t = pd.date_range(start, end)
    lat = np.round(-1.875 + 0.25 * np.arange(nlat), 4)
    lon = np.round(10.125 + 0.25 * np.arange(nlon), 4)
    base = np.arange(len(t), dtype="float32")[:, None, None] * np.ones((1, nlat, nlon), "float32")
    return xr.Dataset({"tmean": (("time", "latitude", "longitude"), base),
                       "tmax": (("time", "latitude", "longitude"), base + 5),
                       "tmin": (("time", "latitude", "longitude"), base - 5)},
                      coords={"time": t, "latitude": lat, "longitude": lon})


def test_process_daily_means_and_qc():
    qc, dek, mon = process_daily(_daily())
    # December 1995: values 0..30 -> monthly mean 15, dekad 1 (0..9) -> 4.5
    assert float(mon["tmean"].sel(time="1995-12-01").isel(latitude=0, longitude=0)) == pytest.approx(15.0)
    assert float(dek["tmean"].sel(time="1995-12-01").isel(latitude=0, longitude=0)) == pytest.approx(4.5)
    assert float(dek["tmax"].sel(time="1995-12-01").isel(latitude=0, longitude=0)) == pytest.approx(9.5)
    # leap February 1996 has a 9-day third dekad
    feb3 = dek.sel(time="1996-02-21").isel(latitude=0, longitude=0)
    assert not np.isnan(float(feb3["tmean"]))
    assert set(qc["variable"]) == {"tmean", "tmax", "tmin"}
    assert (qc["days_present"] == qc["days_expected"]).all()
    assert int(qc["missing_values"].sum()) == 0


def test_incomplete_dekad_is_nan():
    ds = _daily(start="1995-12-01", end="1995-12-25")     # third dekad incomplete
    _, dek, mon = process_daily(ds)
    assert np.isnan(float(dek["tmean"].sel(time="1995-12-21").isel(latitude=0, longitude=0)))
    assert np.isnan(float(mon["tmean"].sel(time="1995-12-01").isel(latitude=0, longitude=0)))


def test_open_era5_daily_rejects_gaps(tmp_path):
    _daily(start="1995-01-01", end="1995-01-10").to_netcdf(tmp_path / "era5_t2m_daily_1995.nc")
    _daily(start="1995-01-15", end="1995-01-20").to_netcdf(tmp_path / "era5_t2m_daily_1996.nc")
    with pytest.raises(ValueError, match="manquant"):
        open_era5_daily(tmp_path / "era5_t2m_daily_*.nc")


def test_conservative_regridding_preserves_values_and_gradients():
    lat = np.arange(-2, 2.01, 0.25)
    lon = np.arange(10, 13.01, 0.25)
    const = xr.DataArray(np.full((len(lat), len(lon)), 3.0), dims=("latitude", "longitude"),
                         coords={"latitude": lat, "longitude": lon})
    out = conservative_to_degree(const, 1.0)
    assert float(out.min()) == pytest.approx(3.0) and float(out.max()) == pytest.approx(3.0)
    np.testing.assert_allclose(out["latitude"].values, [-1.5, -0.5, 0.5, 1.5])
    grad = xr.DataArray(np.tile(lon, (len(lat), 1)), dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})
    assert float(conservative_to_degree(grad, 1.0).sel(latitude=0.5, longitude=11.5)) == pytest.approx(11.5, abs=1e-3)
