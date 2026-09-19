"""
NMME ensemble-mean forecasts and hindcasts from the NOAA/CPC server.

Layout of the server (the files served to CPT)::

    <base_url>/<mon><YYYY>ic/<MODEL>/<MODEL>.<var>.<YYYYMM>.ENSMEAN.fcst.nc

One directory per initialisation month holds, for every model, one file per
initialisation year (hindcast years and the current forecast) and variable:
``prate`` (precipitation rate, mm/s), ``tmp2m`` (2 m temperature, K) and
``tmpsfc`` (surface temperature / SST, K). Only the **ensemble mean** is
provided. Each file is global, 1° (cell centres on integer degrees), with a
``target`` axis of forecast months ("months since 1960-01-01").

Unlike IRIDL, this server needs no credentials and answers quickly, which is
why eccas-s2s uses it (the same source as WASS2S and CPT).
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.io._base import http_download

DEFAULT_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/International/nmme/netcdf"
_FILE = re.compile(r'href="(?P<name>(?P<model>[^".]+(?:\.\d)?[^".]*)\.(?P<var>[a-z0-9]+)\.'
                   r'(?P<ym>\d{6})\.ENSMEAN\.fcst\.nc)"')


def init_folder(init_date) -> str:
    """Server folder of an initialisation, e.g. 2026-09-01 -> 'sep2026ic'."""
    d = pd.Timestamp(init_date)
    return f"{d.strftime('%b').lower()}{d.year}ic"


def file_url(base_url: str, init_date, model: str, var: str, year: int) -> str:
    d = pd.Timestamp(init_date)
    return f"{base_url}/{init_folder(d)}/{model}/{model}.{var}.{year}{d.month:02d}.ENSMEAN.fcst.nc"


def list_available(base_url: str, init_date, model: str) -> pd.DataFrame:
    """Files listed on the server for one model and initialisation (variable, year, name)."""
    url = f"{base_url}/{init_folder(init_date)}/{model}/"
    with urllib.request.urlopen(url, timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    rows = [{"variable": m["var"], "year": int(m["ym"][:4]), "name": m["name"]}
            for m in _FILE.finditer(html)]
    return pd.DataFrame(rows).drop_duplicates()


def download_file(base_url: str, init_date, model: str, var: str, year: int, dest_dir) -> Path:
    """Download one file (kept as delivered: global, immutable). Returns its path."""
    dest = Path(dest_dir) / model / f"{model}.{var}.{year}{pd.Timestamp(init_date).month:02d}.ENSMEAN.fcst.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return Path(http_download(file_url(base_url, init_date, model, var, year), str(dest)))


def _months_since_1960(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp("1960-01-01") + pd.DateOffset(months=int(round(v)))
                             for v in np.atleast_1d(values)])


def open_nmme_file(path) -> xr.DataArray:
    """
    Open one CPC NMME file as ``(target, latitude, longitude)``.

    ``target`` becomes the first day of each forecast month; the initialisation
    month is stored in the ``init`` attribute; latitudes are sorted ascending and
    longitudes converted to -180..180.
    """
    ds = xr.open_dataset(path, decode_times=False)
    da = ds["fcst"]
    init = _months_since_1960(ds["initial_time"].values)[0]
    da = da.assign_coords(target=_months_since_1960(ds["target"].values))
    da = da.rename({"lat": "latitude", "lon": "longitude"})
    lon = da["longitude"].values.astype("float64")
    da = da.assign_coords(longitude=np.where(lon > 180, lon - 360, lon)).sortby(["latitude", "longitude"])
    da.attrs = {**da.attrs, "init": str(init.date()), "source_file": str(Path(path).resolve())}
    return da
