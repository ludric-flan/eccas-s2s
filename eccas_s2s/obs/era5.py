"""
ERA5 daily temperature reference: reading, quality control and calendar means
(workflow step E1).

Input: the yearly files written by
:mod:`eccas_s2s.operations.download_era5_hourly` (``tmean``, ``tmax``, ``tmin``
in °C, UTC days, 0.25°). They are read year by year and turned into dekadal and
monthly **means**, the archive shared by all cycles — the temperature
counterpart of the CHIRPS totals.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VARIABLES = ("tmean", "tmax", "tmin")
#: daily values outside this range (°C) are reported by the quality control.
PLAUSIBLE_RANGE = (-15.0, 60.0)


def expand_paths(paths) -> list[Path]:
    """Expand a path, a glob pattern or a list of them into sorted existing files."""
    import glob
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files = []
    for item in items:
        hits = sorted(glob.glob(str(item)))
        if not hits:
            raise FileNotFoundError(f"aucun fichier ERA5 pour {item}")
        files += [Path(h) for h in hits]
    return files


def open_era5_daily(paths) -> xr.Dataset:
    """
    Open the ERA5 daily files lazily; duplicated or missing days raise an error.

    Returns a Dataset with ``tmean``, ``tmax`` and ``tmin`` (°C) on
    ``(time, latitude, longitude)``, latitude ascending.
    """
    files = expand_paths(paths)
    ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords", chunks={"time": 62})
    missing = [v for v in VARIABLES if v not in ds]
    if missing:
        raise KeyError(f"variables absentes des fichiers ERA5 : {missing}")
    ds = ds.sortby(["time", "latitude"])
    t = pd.DatetimeIndex(ds["time"].values)
    if t.has_duplicates:
        raise ValueError(f"jours en double dans ERA5 : {t[t.duplicated()][:3].tolist()}")
    expected = pd.date_range(t[0], t[-1], freq="D")
    if len(expected) != len(t):
        gap = expected.difference(t)
        raise ValueError(f"{len(gap)} jour(s) manquant(s) dans ERA5, ex. {gap[:3].tolist()}")
    ds.attrs["source_files"] = ";".join(str(f.resolve()) for f in files)
    return ds


def process_daily(ds: xr.Dataset, progress=None):
    """
    One pass over the daily files: quality control + dekadal and monthly means.

    Returns ``(qc, dekads, months)``. ``qc`` has one row per month and variable
    (days expected/present, missing values, minimum, maximum, values outside the
    plausible range). A mean is NaN when a day of the period is missing
    (``skipna=False``).
    """
    times = pd.DatetimeIndex(ds["time"].values)
    years = sorted(set(times.year))
    qc_rows, dek, mon, dek_time, dek_idx, mon_time = [], [], [], [], [], []
    for year in years:
        if progress:
            progress(year)
        block = ds.sel(time=str(year)).load()
        bt = pd.DatetimeIndex(block["time"].values)
        for month in sorted(set(bt.month)):
            sel = block.sel(time=f"{year}-{month:02d}")
            ndays = calendar.monthrange(year, month)[1]
            complete = sel.sizes["time"] == ndays
            for v in VARIABLES:
                da = sel[v]
                qc_rows.append({
                    "year": year, "month": month, "variable": v,
                    "days_expected": ndays, "days_present": int(sel.sizes["time"]),
                    "missing_values": int(da.isnull().sum()),
                    "min": round(float(da.min()), 2), "max": round(float(da.max()), 2),
                    "outside_range": int(((da < PLAUSIBLE_RANGE[0]) | (da > PLAUSIBLE_RANGE[1])).sum()),
                })
            for d, (d0, d1) in enumerate(((1, 10), (11, 20), (21, ndays)), start=1):
                seg = sel.sel(time=slice(pd.Timestamp(year, month, d0), pd.Timestamp(year, month, d1)))
                ok = seg.sizes["time"] == d1 - d0 + 1
                dek.append(seg.mean("time", skipna=False) if ok
                           else xr.full_like(sel.isel(time=0, drop=True), np.nan))
                dek_time.append(pd.Timestamp(year, month, d0))
                dek_idx.append(d)
            mon.append(sel.mean("time", skipna=False) if complete
                       else xr.full_like(sel.isel(time=0, drop=True), np.nan))
            mon_time.append(pd.Timestamp(year, month, 1))

    dekads = xr.concat(dek, dim=pd.DatetimeIndex(dek_time, name="time")).assign_coords(
        dekad=("time", dek_idx))
    months = xr.concat(mon, dim=pd.DatetimeIndex(mon_time, name="time"))
    for out, label in ((dekads, "dekadal"), (months, "monthly")):
        for v in VARIABLES:
            out[v] = out[v].astype("float32")
            out[v].attrs = {"units": "degC", "long_name": f"{label} mean of daily {v} (ERA5)"}
        out.attrs = {"source": "ERA5 hourly 2 m temperature, daily statistics, averaged"}
    return pd.DataFrame(qc_rows), dekads, months
