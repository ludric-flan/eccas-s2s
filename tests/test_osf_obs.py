"""Tests of the observation layer (phase P1): CHIRPS streaming totals, block averaging, normals."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.core.periods import Period, build_periods
from eccas_s2s.obs.chirps import process_daily
from eccas_s2s.obs.climatology import calendar_keys, normals, normals_for_periods, obs_period_totals
from eccas_s2s.obs.regrid import block_average, match_model_grid


def _daily(start="1995-11-01", end="1996-03-31", value=1.0, nlat=40, nlon=20):
    """Synthetic 0.05° field with edges on integer degrees: rain = `value` mm every day."""
    t = pd.date_range(start, end)
    lat = np.round(-1.975 + 0.05 * np.arange(nlat), 4)       # edges -2.0 .. 0.0
    lon = np.round(10.025 + 0.05 * np.arange(nlon), 4)       # edges 10.0 .. 11.0
    data = np.full((len(t), nlat, nlon), value, dtype="float32")
    return xr.DataArray(data, dims=("time", "latitude", "longitude"),
                        coords={"time": t, "latitude": lat, "longitude": lon}, name="precip")


def test_process_daily_totals_and_qc():
    da = _daily()
    qc, dek, mon = process_daily(da)
    assert list(qc["days_present"]) == list(qc["days_expected"])
    # Feb 1996 is a leap month: D3 = 21..29 = 9 days
    assert float(dek.sel(time="1996-02-21").isel(latitude=0, longitude=0)) == 9.0
    assert float(dek.sel(time="1995-12-21").isel(latitude=0, longitude=0)) == 11.0
    assert float(mon.sel(time="1996-02-01").isel(latitude=0, longitude=0)) == 29.0
    assert list(dek["dekad"].values[:3]) == [1, 2, 3]


def test_missing_day_makes_total_nan_and_is_reported():
    da = _daily()
    da[5, 0, 0] = np.nan                                   # 1995-11-06, one land cell
    qc, dek, mon = process_daily(da, land_mask=xr.ones_like(da.isel(time=0, drop=True), dtype=bool))
    assert np.isnan(float(dek.sel(time="1995-11-01").isel(latitude=0, longitude=0)))
    assert np.isnan(float(mon.sel(time="1995-11-01").isel(latitude=0, longitude=0)))
    assert float(mon.sel(time="1995-11-01").isel(latitude=0, longitude=1)) == 30.0
    assert qc.loc[(qc.year == 1995) & (qc.month == 11), "missing_land_values"].item() == 1


def test_block_average_exact_and_aligned():
    da = _daily().isel(time=0)
    rng = np.random.default_rng(0)
    da = da.copy(data=rng.random(da.shape).astype("float32"))
    out = block_average(da, 1.0)
    assert out.sizes == {"latitude": 2, "longitude": 1}
    np.testing.assert_allclose(out["latitude"].values, [-1.5, -0.5])
    np.testing.assert_allclose(float(out.isel(latitude=0, longitude=0)),
                               float(da.isel(latitude=slice(0, 20), longitude=slice(0, 20)).mean()), rtol=1e-6)


def test_block_average_rejects_misaligned_grid():
    da = _daily().isel(time=0)
    shifted = da.assign_coords(latitude=da["latitude"] + 0.025)
    with pytest.raises(ValueError, match="aligné"):
        block_average(shifted, 1.0)


def test_block_average_min_valid_fraction():
    da = _daily().isel(time=0).copy()
    da[:20, :] = np.nan
    da[0, 0] = 1.0                                          # 1 valid cell out of 400
    out = block_average(da, 1.0, min_valid=0.5)
    assert np.isnan(float(out.isel(latitude=0, longitude=0)))


def test_match_model_grid():
    obs = block_average(_daily().isel(time=0), 1.0)
    model = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("latitude", "longitude"),
                         coords={"latitude": [-2.5, -1.5, -0.5], "longitude": [9.5, 10.5, 11.5, 12.5]})
    m = match_model_grid(model, obs)
    assert m.sizes == obs.sizes and float(m.sel(latitude=-1.5, longitude=10.5)) == 5.0


def test_obs_period_totals_season_across_years():
    """Periods are built on the forecast year (2026) and resolved for past years."""
    _, dek, mon = process_daily(_daily(start="1995-09-01", end="1997-03-31"))
    periods = build_periods(pd.Timestamp("2026-09-01"), 181, scales=("season",))
    tot = obs_period_totals(dek, mon, periods, years=[1995, 1996, 1997])
    djf = tot.sel(period="season_m3").isel(latitude=0, longitude=0)
    assert djf["calendar_key"].item() == "season_12"
    assert float(djf.sel(year=1995)) == 31 + 31 + 29          # DJF 1995-96 (leap February)
    assert float(djf.sel(year=1996)) == 31 + 31 + 28          # DJF 1996-97
    assert np.isnan(float(djf.sel(year=1997)))                # beyond the archive -> NaN


def test_normals_weibull_percentiles():
    years = list(range(1991, 2000))                         # 9 years, value = year index + 1 mm
    t = pd.DatetimeIndex([pd.Timestamp(y, 1, 1) for y in years])
    mon = xr.DataArray(np.arange(1, 10, dtype="float32")[:, None, None] * np.ones((1, 1, 2), "float32"),
                       dims=("time", "latitude", "longitude"),
                       coords={"time": t, "latitude": [0.0], "longitude": [0.0, 1.0]})
    dek = mon.isel(time=slice(0, 0))
    ds = normals(dek, mon, (1991, 1999), [50, 90], "weibull", min_years=5, keys=["month_01"])
    # Weibull: p-quantile at rank p*(n+1) -> P50 = 5th value = 5, P90 = rank 9 = 9
    np.testing.assert_allclose(ds["quantile"].sel(calendar_key="month_01").isel(latitude=0, longitude=0), [5, 9])
    assert int(ds["n_years"].sel(calendar_key="month_01").isel(latitude=0, longitude=0)) == 9
    sub = normals_for_periods(ds, [Period("month", 1, 0)])
    assert list(sub["period"].values) == ["month_m0"]


def test_calendar_keys_count():
    keys = calendar_keys()
    assert len(keys) == 60 and keys[0] == "dekad_01_1" and keys[-1] == "season_12"


def test_block_average_accepts_float32_coordinates():
    """CHIRPS stores coordinates in float32 (5.024994...): nesting must still be recognised."""
    da = _daily().isel(time=0)
    da = da.assign_coords(latitude=da["latitude"].astype("float32") + np.float32(-6e-6),
                          longitude=da["longitude"].astype("float32") + np.float32(-6e-6))
    out = block_average(da, 1.0)
    np.testing.assert_allclose(out["latitude"].values, [-1.5, -0.5])


def test_open_chirps_daily_multi_files(tmp_path):
    from eccas_s2s.obs.chirps import open_chirps_daily
    a = _daily(start="1990-12-25", end="1990-12-31").to_dataset()
    b = _daily(start="1991-01-01", end="1991-01-05").to_dataset()
    a.to_netcdf(tmp_path / "a_1990.nc"); b.to_netcdf(tmp_path / "b_1991.nc")
    da = open_chirps_daily([tmp_path / "a_*.nc", tmp_path / "b_1991.nc"])
    assert da.sizes["time"] == 12 and str(da.time.values[0])[:10] == "1990-12-25"
    # a gap at the junction is refused
    _daily(start="1991-01-03", end="1991-01-05").to_dataset().to_netcdf(tmp_path / "c.nc")
    with pytest.raises(ValueError, match="manquant"):
        open_chirps_daily([tmp_path / "a_1990.nc", tmp_path / "c.nc"])
    # duplicated days are refused
    with pytest.raises(ValueError, match="double"):
        open_chirps_daily([tmp_path / "b_1991.nc", tmp_path / "c.nc"])
