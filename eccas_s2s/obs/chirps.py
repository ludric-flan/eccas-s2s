"""
CHIRPS daily precipitation: reading, quality control and calendar totals
(workflow step E1, Draft Framework §3.1.6).

The daily file is large (28 GB for 1991-2025 at 0.05° over the ECCAS box), so
every function here streams it **one month at a time**. The calendar totals
(dekads and months) are computed once and stored as a derived archive shared by
all forecast cycles; a cycle then only *selects* the periods it needs.

Date convention: CHIRPS ``time`` denotes the day the rain fell ("time variable
denotes the first day of the given day"), which matches the OSF lead-day
convention for C3S (see :mod:`eccas_s2s.core.daily`).
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.obs.regrid import snap_coords

#: daily totals above this value (mm) are reported as suspicious in the QC.
SUSPICIOUS_DAILY_MM = 300.0

_ENCODING = {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(-9999.0)}


def open_chirps_daily(path: str | Path) -> xr.DataArray:
    """Open the CHIRPS daily file lazily, as ``precip`` (mm/day), latitude ascending."""
    ds = xr.open_dataset(path, chunks={"time": 31})
    name = "precip" if "precip" in ds else list(ds.data_vars)[0]
    rename = {k: v for k, v in (("lat", "latitude"), ("lon", "longitude")) if k in ds.coords}
    da = ds[name].rename(rename) if rename else ds[name]
    da = snap_coords(da.sortby("latitude"))
    da.name = "precip"
    da.attrs.setdefault("units", "mm/day")
    return da


def _month_slices(times: pd.DatetimeIndex):
    """Yield (year, month, start, end) for every calendar month in ``times``."""
    months = pd.period_range(times[0], times[-1], freq="M")
    for m in months:
        start = m.start_time.normalize()
        end = m.end_time.normalize()
        yield m.year, m.month, start, end


def _month_qc(block: xr.DataArray, year: int, month: int, land_mask: xr.DataArray) -> dict:
    return {
        "year": year, "month": month,
        "days_expected": calendar.monthrange(year, month)[1],
        "days_present": int(block.sizes["time"]),
        "missing_land_values": int((block.isnull() & land_mask).sum()),
        "negative_values": int((block < 0).sum()),
        "max_daily_mm": float(block.max()),
        "suspicious_values": int((block > SUSPICIOUS_DAILY_MM).sum()),
    }


def process_daily(da: xr.DataArray, land_mask: xr.DataArray | None = None, progress=None):
    """
    One streaming pass over a daily precipitation field: QC + calendar totals.

    Returns ``(qc, dekads, months)``:

    * ``qc`` — one row per month: days expected/present, missing land values,
      negative values, maximum daily value, values above 300 mm/day;
    * ``dekads`` — dekadal totals, ``time`` = first day of the dekad (1, 11, 21),
      with a ``dekad`` coordinate (1, 2, 3);
    * ``months`` — monthly totals, ``time`` = first day of the month.

    A total is NaN at a cell if any day of the period is missing there
    (``skipna=False``). ``land_mask`` (True = cell of the domain) defaults to the
    cells valid on the first day. ``progress(year, month)`` is called per month.
    """
    times = pd.DatetimeIndex(da["time"].values)
    if land_mask is None:
        land_mask = da.isel(time=0).notnull().load()
    template = xr.full_like(da.isel(time=0, drop=True), np.nan, dtype="float32").load()

    qc_rows, dek_list, dek_time, dek_idx, mon_list, mon_time = [], [], [], [], [], []
    for year, month, start, end in _month_slices(times):
        if progress:
            progress(year, month)
        block = da.sel(time=slice(start, end)).load()
        qc_rows.append(_month_qc(block, year, month, land_mask))
        ndays = calendar.monthrange(year, month)[1]
        for d, (d0, d1) in enumerate(((1, 10), (11, 20), (21, ndays)), start=1):
            seg = block.sel(time=slice(pd.Timestamp(year, month, d0), pd.Timestamp(year, month, d1)))
            tot = seg.sum("time", skipna=False) if seg.sizes["time"] == d1 - d0 + 1 else template
            dek_list.append(tot.astype("float32"))
            dek_time.append(pd.Timestamp(year, month, d0))
            dek_idx.append(d)
        mtot = block.sum("time", skipna=False) if block.sizes["time"] == ndays else template
        mon_list.append(mtot.astype("float32"))
        mon_time.append(pd.Timestamp(year, month, 1))

    dekads = xr.concat(dek_list, dim=pd.DatetimeIndex(dek_time, name="time"))
    dekads = dekads.assign_coords(dekad=("time", dek_idx))
    months = xr.concat(mon_list, dim=pd.DatetimeIndex(mon_time, name="time"))
    for out, label in ((dekads, "dekadal"), (months, "monthly")):
        out.name = "precip"
        out.attrs = {"long_name": f"{label} total precipitation", "units": "mm",
                     "source": "CHIRPS v2.0 daily, summed with skipna=False"}
    qc = pd.DataFrame(qc_rows)
    qc["land_cells"] = int(land_mask.sum())
    return qc, dekads, months


def write_totals(da: xr.DataArray, path: str | Path, attrs: dict | None = None) -> Path:
    """Write a totals archive as compressed NetCDF-4 (one time step per chunk)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = da.to_dataset()
    ds.attrs.update(attrs or {})
    enc = {"precip": {**_ENCODING, "chunksizes": (1, da.sizes["latitude"], da.sizes["longitude"])}}
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)                       # atomic: never leave a half-written archive
    return path
