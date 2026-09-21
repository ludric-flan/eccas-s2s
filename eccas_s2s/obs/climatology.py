"""
Observed period totals and observed normals (workflow step E1, decisions D4/D12).

Two uses of the observations, kept separate on purpose:

* **period totals** for every year (the predictand ``Y`` of the calibration and
  the verifying observation), selected from the dekadal/monthly archive;
* **normals 1991-2020** per grid point and per *calendar* period: mean,
  standard deviation and percentiles (P5 ... P95, terciles). They define the
  common category thresholds of all models (D12) and the observed anomalies.

Normals are indexed by :attr:`Period.calendar_key` (``dekad_MM_D``, ``month_MM``,
``season_MM``) so they are computed once and shared by all cycles. A season
starting in month MM of year Y uses the months of Y and, when it crosses the
year, of Y+1 (e.g. the 1991-2020 DJF normal spans DJF 1991-92 ... DJF 2020-21).
During cross-validation the thresholds are recomputed without the verified year
(see the calibration phase); the files written here hold the full-normal values
used for operational forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.core.periods import Period, _shift_month


def calendar_keys() -> list[str]:
    """The 60 calendar periods: 36 dekads, 12 months, 12 three-month seasons."""
    keys = [f"dekad_{m:02d}_{d}" for m in range(1, 13) for d in (1, 2, 3)]
    keys += [f"month_{m:02d}" for m in range(1, 13)]
    keys += [f"season_{m:02d}" for m in range(1, 13)]
    return keys


def _key_dates(key: str, year: int) -> list[pd.Timestamp]:
    """Archive time stamps making up a calendar period starting in ``year``."""
    parts = key.split("_")
    month = int(parts[1])
    if parts[0] == "dekad":
        return [pd.Timestamp(year, month, {1: 1, 2: 11, 3: 21}[int(parts[2])])]
    if parts[0] == "month":
        return [pd.Timestamp(year, month, 1)]
    return [pd.Timestamp(*_shift_month(year, month, k), 1) for k in range(3)]


def total_for_key(dekads: xr.DataArray, months: xr.DataArray, key: str, year: int,
                  how: str = "sum") -> xr.DataArray:
    """
    Observed value of a calendar period starting in ``year`` (NaN if not in the archive).

    ``how="sum"`` for totals (precipitation) and ``how="mean"`` for means
    (temperature); a seasonal mean is weighted by the number of days of its
    three months.
    """
    source = dekads if key.startswith("dekad") else months
    stamps = _key_dates(key, year)
    available = pd.DatetimeIndex(source["time"].values)
    if not all(t in available for t in stamps):
        return xr.full_like(source.isel(time=0, drop=True), np.nan)
    sel = source.sel(time=stamps).drop_vars("dekad", errors="ignore")
    if how == "sum":
        return sel.sum("time", skipna=False)
    if how != "mean":
        raise ValueError("how doit valoir 'sum' ou 'mean'")
    if len(stamps) == 1:
        return sel.isel(time=0, drop=True)
    w = xr.DataArray([t.days_in_month for t in stamps], dims="time", coords={"time": sel["time"]})
    return (sel * w).sum("time", skipna=False) / w.sum()


def obs_period_totals(dekads: xr.DataArray, months: xr.DataArray,
                      periods: list[Period], years, how: str = "sum") -> xr.DataArray:
    """
    Observed totals of the cycle's periods for each initialisation year.

    Output dims ``(year, period, latitude, longitude)`` with the same ``period``
    keys as :func:`eccas_s2s.core.daily.aggregate_periods`, so model and
    observation arrays align directly. ``year`` is the initialisation year: the
    DJF period of an init in September 1995 is DJF 1995-96.
    """
    per_period = []
    for p in periods:
        per_year = []
        for y in years:
            start = p.dates(int(y))[0]
            per_year.append(total_for_key(dekads, months, p.calendar_key, start.year, how))
        per_period.append(xr.concat(per_year, dim=pd.Index(list(years), name="year")))
    out = xr.concat(per_period, dim=pd.Index([p.key for p in periods], name="period"))
    out = out.assign_coords(
        scale=("period", [p.scale for p in periods]),
        calendar_key=("period", [p.calendar_key for p in periods]))
    out.name = dekads.name or "obs"
    out.attrs = {**dekads.attrs, "aggregation": how}
    return out.transpose("year", "period", "latitude", "longitude")


def normals(dekads: xr.DataArray, months: xr.DataArray, years: tuple[int, int],
            percentiles, method: str = "weibull", min_years: int = 25,
            keys: list[str] | None = None, progress=None, how: str = "sum") -> xr.Dataset:
    """
    Observed normals per grid point and calendar period.

    Returns a Dataset with ``mean``, ``std`` (ddof=1) and ``n_years`` on
    ``(calendar_key, latitude, longitude)`` and ``quantile`` on
    ``(percentile, calendar_key, latitude, longitude)``. Cells with fewer than
    ``min_years`` valid years are NaN.
    """
    keys = keys or calendar_keys()
    y0, y1 = years
    q = np.asarray(percentiles, dtype=float) / 100.0
    means, stds, counts, quants = [], [], [], []
    for key in keys:
        if progress:
            progress(key)
        sample = xr.concat([total_for_key(dekads, months, key, y, how) for y in range(y0, y1 + 1)],
                           dim="year").load()
        n = sample.notnull().sum("year")
        ok = n >= min_years
        arr = sample.values
        with np.errstate(all="ignore"), _quiet_nan_warnings():
            qv = np.nanquantile(arr, q, axis=0, method=method)
        means.append(sample.mean("year").where(ok))
        stds.append(sample.std("year", ddof=1).where(ok))
        counts.append(n)
        quants.append(xr.DataArray(qv, dims=("percentile", "latitude", "longitude"),
                                   coords={"percentile": np.asarray(percentiles, dtype=float),
                                           "latitude": sample["latitude"],
                                           "longitude": sample["longitude"]}).where(ok))
    idx = pd.Index(keys, name="calendar_key")
    ds = xr.Dataset({
        "mean": xr.concat(means, dim=idx),
        "std": xr.concat(stds, dim=idx),
        "n_years": xr.concat(counts, dim=idx).astype("int16"),
        "quantile": xr.concat(quants, dim=idx).transpose("percentile", "calendar_key", ...),
    })
    units = dekads.attrs.get("units", "mm")
    for v in ("mean", "std", "quantile"):
        ds[v] = ds[v].astype("float32")
        ds[v].attrs["units"] = units
    ds.attrs.update({"normal_period": f"{y0}-{y1}", "percentile_method": method,
                     "min_years": min_years, "aggregation": how})
    return ds


class _quiet_nan_warnings:
    """Silence 'All-NaN slice' warnings (ocean cells) inside a block."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def normals_for_periods(norm: xr.Dataset, periods: list[Period]) -> xr.Dataset:
    """Select the normals of a cycle's periods, re-indexed by ``Period.key``."""
    sel = norm.sel(calendar_key=[p.calendar_key for p in periods])
    sel = sel.rename({"calendar_key": "period"}).assign_coords(period=[p.key for p in periods])
    return sel.assign_coords(calendar_key=("period", [p.calendar_key for p in periods]))
