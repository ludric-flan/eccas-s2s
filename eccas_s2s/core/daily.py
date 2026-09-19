"""
Daily fields on the *lead-day* axis and their aggregation over target periods.

Date attribution (important)
----------------------------
C3S ``seasonal-original-single-levels`` precipitation (``tp``) is accumulated since
the initialisation time. The field at step = k days (valid at init + k days, 00 UTC)
holds the rain fallen from init to init + k days. The daily total of lead-day
index ``i`` (0 = initialisation day) is therefore::

    rain[i] = tp(step = i+1 days) - tp(step = i days),   with tp(step = 0) = 0

so the 24 h step, stamped on the 2nd of the month at 00 UTC, is the rain of the
**1st**. The legacy pipeline indexed daily totals by their ``valid_time`` and so
shifted every period by one day (a "decade" labelled 2-10 June was rain of 1-9
June). The OSF chain attributes each total to the day it fell on, which is what
daily observations such as CHIRPS use.

All functions work on a common layout::

    (year, number, lead_day, latitude, longitude)

where ``year`` is the initialisation year (one value for a forecast, 24 for a
C3S hindcast).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.core.periods import Period


def to_year_dim(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """
    Replace the initialisation-time coordinate by an integer ``year`` dimension.

    A forecast has a scalar ``time``: it becomes a length-1 ``year`` dimension. A
    hindcast has ``time`` as a dimension (one initialisation per year).
    """
    if time_dim in da.dims:
        years = pd.DatetimeIndex(da[time_dim].values).year
        da = da.assign_coords({time_dim: years.values}).rename({time_dim: "year"})
    else:
        year = int(pd.Timestamp(da[time_dim].values).year) if time_dim in da.coords else None
        if year is None:
            raise ValueError("Aucune coordonnée d'initialisation « time » trouvée.")
        da = da.drop_vars(time_dim).expand_dims(year=[year])
    return da


def _lead_days_from_steps(steps: np.ndarray) -> np.ndarray:
    days = steps / np.timedelta64(1, "D")
    if not np.allclose(days, np.round(days)):
        raise ValueError("Les échéances ne sont pas des multiples entiers d'un jour.")
    return np.round(days).astype(int)


def daily_precip_from_accumulated(tp: xr.DataArray, step_dim: str = "step") -> xr.DataArray:
    """
    Convert precipitation accumulated since initialisation (m) to daily totals (mm).

    Returns a DataArray with ``step`` replaced by ``lead_day`` (0 = initialisation
    day). Small negative increments (GRIB packing noise) are set to 0 and counted
    in the ``n_negative_clipped`` attribute.
    """
    days = _lead_days_from_steps(tp[step_dim].values)
    if days[0] == 0:                       # a step-0 field, if present, is zero by definition
        tp = tp.isel({step_dim: slice(1, None)})
        days = days[1:]
    if days[0] != 1 or np.any(np.diff(days) != 1):
        raise ValueError(
            f"Échéances quotidiennes non contiguës à partir de 1 jour (reçu {days[:5]}...)."
        )

    mm = tp * 1000.0
    first = mm.isel({step_dim: slice(0, 1)})
    rest = mm.diff(step_dim, label="upper")
    daily = xr.concat([first, rest], dim=step_dim)

    n_neg = int((daily < 0).sum())
    daily = daily.clip(min=0)

    daily = daily.assign_coords({step_dim: days - 1}).rename({step_dim: "lead_day"})
    for c in ("valid_time", "surface"):
        if c in daily.coords:
            daily = daily.drop_vars(c)
    daily.name = "precip"
    daily.attrs = {
        "long_name": "daily total precipitation",
        "units": "mm day-1",
        "lead_day_convention": "0 = initialisation day; total of the 24 h starting 00 UTC",
        "n_negative_clipped": n_neg,
    }
    return daily


def lead_day_dates(init_year: int, init_month: int, n_days: int) -> pd.DatetimeIndex:
    """Calendar dates of lead days 0..n_days-1 for one initialisation year."""
    return pd.date_range(pd.Timestamp(init_year, init_month, 1), periods=n_days, freq="D")


def aggregate_periods(daily: xr.DataArray, periods: list[Period],
                      how: str = "sum", init_month: int | None = None) -> xr.DataArray:
    """
    Aggregate a daily field over each target period, for every initialisation year.

    Parameters
    ----------
    daily : DataArray with dims including ``year`` and ``lead_day``.
    periods : periods from :func:`eccas_s2s.core.periods.build_periods`.
    how : ``"sum"`` (precipitation totals) or ``"mean"`` (temperatures).

    Returns
    -------
    DataArray with ``lead_day`` replaced by a ``period`` dimension (``Period.key``).
    A period not fully covered for a given year (e.g. Feb 29 pushing it beyond
    the horizon) is NaN for that year: missing days are never ignored.
    """
    if how not in ("sum", "mean"):
        raise ValueError("how doit valoir 'sum' ou 'mean'.")
    if init_month is None:
        init_month = periods[0].init_month
    n_lead = daily.sizes["lead_day"]
    lead0 = int(daily["lead_day"].values[0])
    if lead0 != 0:
        raise ValueError("lead_day doit commencer à 0 (jour d'initialisation).")

    per_period = []
    for p in periods:
        if p.init_month != init_month:
            raise ValueError("Toutes les périodes doivent partager le mois d'initialisation.")
        per_year = []
        for y in daily["year"].values:
            i0, i1 = p.day_slice(int(y))
            seg = daily.sel(year=y).isel(lead_day=slice(i0, i1 + 1))
            if i1 >= n_lead:
                agg = xr.full_like(seg.isel(lead_day=0, drop=True), np.nan)
            elif how == "sum":
                agg = seg.sum("lead_day", skipna=False)
            else:
                agg = seg.mean("lead_day", skipna=False)
            per_year.append(agg)
        per_period.append(xr.concat(per_year, dim="year"))

    out = xr.concat(per_period, dim=pd.Index([p.key for p in periods], name="period"))
    out = out.assign_coords(
        scale=("period", [p.scale for p in periods]),
        month_offset=("period", [p.month_offset for p in periods]),
        decade=("period", [p.decade or 0 for p in periods]),
    )
    out.attrs = {**daily.attrs, "aggregation": how}
    out.attrs.pop("n_negative_clipped", None)
    if how == "sum" and daily.attrs.get("units", "").startswith("mm"):
        out.attrs["units"] = "mm"
    return out
