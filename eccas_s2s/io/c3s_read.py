"""
Read C3S GRIB files into the common OSF layout ``(year, number, lead_day, lat, lon)``.

Raw files are treated as immutable: cfgrib's ``.idx`` index files are written to a
separate cache directory (``$ECCAS_S2S_CACHE`` or ``~/.cache/eccas-s2s/cfgrib``),
never next to the raw data. Without an index, cfgrib rescans the whole file on
every opening; with it, only the first opening is slow.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.core.daily import daily_precip_from_accumulated, to_year_dim

LAYOUT = ("year", "number", "lead_day", "latitude", "longitude")


def cfgrib_index_path(path: str | Path) -> str:
    """cfgrib index template for ``path`` inside the cache directory."""
    cache = Path(os.environ.get("ECCAS_S2S_CACHE", Path.home() / ".cache" / "eccas-s2s")) / "cfgrib"
    cache.mkdir(parents=True, exist_ok=True)
    p = Path(path).resolve()
    tag = f"{p.parent.name}_{p.name}"          # e.g. 202609_c3s_ecmwf_..._09.grib
    return str(cache / (tag + ".{short_hash}.idx"))


def open_c3s_grib(path: str | Path) -> xr.Dataset:
    """Open a C3S GRIB file with cfgrib, latitude sorted ascending."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": cfgrib_index_path(path)})
    rename = {k: v for k, v in (("lat", "latitude"), ("lon", "longitude")) if k in ds.coords}
    if rename:
        ds = ds.rename(rename)
    return ds.sortby("latitude")


def load_c3s_precip_daily(path: str | Path) -> xr.DataArray:
    """
    Daily precipitation (mm/day) of a C3S forecast or hindcast GRIB file.

    Returns dims ``(year, number, lead_day, latitude, longitude)``; lead day 0 is
    the initialisation day (see :mod:`eccas_s2s.core.daily`).
    """
    ds = open_c3s_grib(path)
    if "tp" not in ds:
        raise KeyError(f"Variable « tp » absente de {path} (variables : {list(ds.data_vars)}).")
    daily = daily_precip_from_accumulated(ds["tp"])
    daily = to_year_dim(daily)
    daily.attrs["source_file"] = str(Path(path).resolve())
    return daily.transpose(*LAYOUT)


def _nominal_init(start: pd.Timestamp) -> pd.Timestamp:
    """Nominal (1st-of-month) initialisation of a start date: lagged starts belong to the next month."""
    start = pd.Timestamp(start).normalize()
    return start if start.day == 1 else (start + pd.offsets.MonthBegin(1))


def load_c3s_monthly(path: str | Path, var: str = "t2m", init_month: int | None = None) -> xr.DataArray:
    """
    Monthly means of a C3S ``seasonal-monthly-single-levels`` GRIB file.

    Returns dims ``(year, number, month_offset, latitude, longitude)``:
    ``month_offset`` 0 is the initialisation month. In this dataset a monthly
    value is stamped at the **end** of its month (valid_time = first day of the
    next month); the target month is recovered from it.

    Lagged ensembles (UKMO, BoM: several start dates before the 1st) are merged:
    every start date is attached to its nominal 1st-of-month initialisation and
    all its members are stacked in ``number``. Temperatures are converted to °C.
    """
    ds = open_c3s_grib(path)
    da = ds[var]
    if "time" not in da.dims:
        da = da.expand_dims(time=[ds["time"].values])
    vt = ds["valid_time"]
    if "time" not in vt.dims:
        vt = vt.expand_dims(time=da["time"].values)
    starts = pd.DatetimeIndex(da["time"].values)
    nominal = pd.DatetimeIndex([_nominal_init(s) for s in starts])
    if init_month is not None:
        keep = nominal.month == init_month
        da, vt, starts, nominal = da.isel(time=keep), vt.isel(time=keep), starts[keep], nominal[keep]

    by_year = {}
    for i, (s, nom) in enumerate(zip(starts, nominal)):
        field = da.isel(time=i)
        # cfgrib puts the steps of all start dates on one axis: keep this start's own steps
        present = field.notnull().any([d for d in field.dims if d != "step"]).values
        field = field.isel(step=np.flatnonzero(present))
        vts = pd.DatetimeIndex(vt.isel(time=i).values[present])
        target = (vts - pd.Timedelta(days=1)).to_period("M")
        offsets = [(t.year - nom.year) * 12 + t.month - nom.month for t in target]
        if len(set(offsets)) != len(offsets):
            raise ValueError(f"{Path(path).name} : mois cibles en double pour le démarrage {s.date()}")
        field = field.assign_coords(step=offsets).rename(step="month_offset")
        field = field.where(field["month_offset"] >= 0, drop=True)
        by_year.setdefault(nom.year, []).append(field)

    years = sorted(by_year)
    per_year = []
    for y in years:
        members = []
        for field in by_year[y]:
            valid = field.notnull().any(["month_offset", "latitude", "longitude"])
            members.append(field.isel(number=np.flatnonzero(valid.values)))
        stacked = xr.concat(members, dim="number", join="outer")
        stacked = stacked.assign_coords(number=np.arange(stacked.sizes["number"]))
        per_year.append(stacked)
    out = xr.concat(per_year, dim=pd.Index(years, name="year"), join="outer")
    for c in ("time", "valid_time", "surface"):
        if c in out.coords:
            out = out.drop_vars(c)
    if var in ("t2m", "mx2t24", "mn2t24"):
        out = out - 273.15
        out.attrs = {"units": "degC"}
    out.attrs.update({"source_file": str(Path(path).resolve()),
                      "n_start_dates": int(len(starts)), "lagged_ensemble": bool((starts.day != 1).any())})
    out.name = var
    return out.transpose("year", "number", "month_offset", "latitude", "longitude")


def load_c3s_daily_last24h(path: str | Path) -> xr.DataArray:
    """
    Daily maximum or minimum 2 m temperature (°C) of a C3S daily GRIB file.

    ``maximum/minimum_2m_temperature_in_the_last_24_hours`` at step k days covers
    the 24 h ending at init + k days, i.e. lead day k-1 (same convention as the
    precipitation: lead day 0 = initialisation day).
    """
    ds = open_c3s_grib(path)
    name = [v for v in ds.data_vars][0]
    da = ds[name]
    days = (da["step"].values / np.timedelta64(1, "D")).round().astype(int)
    if days[0] != 1 or np.any(np.diff(days) != 1):
        raise ValueError(f"échéances quotidiennes non contiguës dans {path}")
    da = (da - 273.15).assign_coords(step=days - 1).rename(step="lead_day")
    for c in ("valid_time", "surface"):
        if c in da.coords:
            da = da.drop_vars(c)
    da = to_year_dim(da)
    da.name = name
    da.attrs = {"units": "degC", "source_file": str(Path(path).resolve()),
                "lead_day_convention": "0 = initialisation day (24 h ending at 00 UTC of lead day + 1)"}
    return da.transpose(*LAYOUT)
