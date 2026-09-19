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
