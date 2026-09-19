"""Tests of monthly aggregation (C3S monthly statistics, NMME) and lagged start dates."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.core.monthly import aggregate_monthly
from eccas_s2s.core.periods import build_periods
from eccas_s2s.io.c3s_read import _nominal_init
from eccas_s2s.operations.nmme_totals import monthly_values


def _monthly(years=(1995, 2026), values=(20.0, 21.0, 22.0, 23.0, 24.0, 25.0)):
    data = np.tile(np.asarray(values, dtype=float)[None, :, None], (len(years), 1, 1))
    return xr.DataArray(data, dims=("year", "month_offset", "longitude"),
                        coords={"year": list(years), "month_offset": range(len(values)), "longitude": [0.0]})


def test_season_mean_weighted_by_days_and_leap_year():
    periods = build_periods(pd.Timestamp("2026-09-01"), 181, scales=("month", "season"))
    out = aggregate_monthly(_monthly(), periods, how="mean")
    djf = out.sel(period="season_m3").isel(longitude=0)
    # DJF = offsets 3,4,5 -> 23, 24, 25 ; weights 31, 31, 28 (2027) or 29 (1996)
    assert float(djf.sel(year=2026)) == pytest.approx((23 * 31 + 24 * 31 + 25 * 28) / 90)
    assert float(djf.sel(year=1995)) == pytest.approx((23 * 31 + 24 * 31 + 25 * 29) / 91)
    assert float(out.sel(period="month_m0", year=2026).isel(longitude=0)) == 20.0


def test_season_sum_and_missing_month():
    periods = build_periods(pd.Timestamp("2026-09-01"), 181, scales=("season",))
    out = aggregate_monthly(_monthly(values=(1, 2, 3, 4, 5)), periods, how="sum")   # no Feb
    assert float(out.sel(period="season_m0", year=2026).isel(longitude=0)) == 6.0
    assert np.isnan(float(out.sel(period="season_m3", year=2026).isel(longitude=0)))


def test_decades_refused_from_monthly():
    periods = build_periods(pd.Timestamp("2026-09-01"), 181, scales=("decade",))
    with pytest.raises(ValueError, match="décades"):
        aggregate_monthly(_monthly(), periods)


def test_nominal_init_of_lagged_starts():
    assert _nominal_init(pd.Timestamp("2026-09-01")) == pd.Timestamp("2026-09-01")
    assert _nominal_init(pd.Timestamp("2026-08-22")) == pd.Timestamp("2026-09-01")
    assert _nominal_init(pd.Timestamp("1995-12-25")) == pd.Timestamp("1996-01-01")


def test_nmme_rate_to_monthly_total():
    t = pd.DatetimeIndex(["2026-09-01", "2027-02-01"])
    da = xr.DataArray([1e-5, 1e-5], dims="target", coords={"target": t})
    tot = monthly_values(da, "prate")
    np.testing.assert_allclose(tot.values, [1e-5 * 86400 * 30, 1e-5 * 86400 * 28])
    temp = monthly_values(xr.DataArray([300.0, 273.15], dims="target", coords={"target": t}), "tmp2m")
    np.testing.assert_allclose(temp.values, [26.85, 0.0])
