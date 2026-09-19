"""
Monthly fields on the ``month_offset`` axis and their aggregation over months
and seasons (used for the C3S monthly statistics and NMME).

``month_offset`` 0 is the initialisation month. A season is aggregated from its
three months: totals are summed, means are averaged with weights equal to the
number of days of each month (which differ between years for February).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.core.periods import Period


def aggregate_monthly(monthly: xr.DataArray, periods: list[Period], how: str = "mean") -> xr.DataArray:
    """
    Values of month and season periods from ``(year, ..., month_offset, ...)`` data.

    Decades are not possible from monthly data and raise an error. A period with a
    missing month is NaN (no partial seasons).
    """
    if any(p.scale == "decade" for p in periods):
        raise ValueError("les décades ne peuvent pas être calculées à partir de moyennes mensuelles")
    if how not in ("sum", "mean"):
        raise ValueError("how doit valoir 'sum' ou 'mean'")
    offsets = set(int(o) for o in monthly["month_offset"].values)
    per_period = []
    for p in periods:
        wanted = list(range(p.month_offset, p.month_offset + p.n_months))
        if not set(wanted) <= offsets:
            per_period.append(xr.full_like(monthly.isel(month_offset=0, drop=True), np.nan))
            continue
        sel = monthly.sel(month_offset=wanted)
        if how == "sum":
            per_period.append(sel.sum("month_offset", skipna=False))
            continue
        years = monthly["year"].values
        w = np.array([[pd.Timestamp(*_ym(int(y), p.init_month, o), 1).days_in_month for o in wanted]
                      for y in years], dtype=float)
        weights = xr.DataArray(w, dims=("year", "month_offset"), coords={"year": years, "month_offset": wanted})
        per_period.append((sel * weights).sum("month_offset", skipna=False) / weights.sum("month_offset"))
    out = xr.concat(per_period, dim=pd.Index([p.key for p in periods], name="period"))
    out = out.assign_coords(scale=("period", [p.scale for p in periods]),
                            calendar_key=("period", [p.calendar_key for p in periods]))
    out.attrs = {**monthly.attrs, "aggregation": how}
    return out


def _ym(init_year: int, init_month: int, offset: int) -> tuple[int, int]:
    idx = init_month - 1 + offset
    return init_year + idx // 12, idx % 12 + 1
