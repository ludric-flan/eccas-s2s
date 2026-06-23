"""
NMME (North American Multi-Model Ensemble) acquisition from the IRI Data Library.

No credentials are required: the IRIDL is a public OPeNDAP / "Ingrid" data server.
We let the server do the spatial and temporal subsetting, so only the small Central
Africa box is transferred instead of the global field.

--------------------------------------------------------------------------------------
How an IRIDL ("Ingrid") URL is built  --  read this to understand the download
--------------------------------------------------------------------------------------
An IRIDL URL is a *pipeline of operations* appended to a dataset path, left to right.
Each ``/Word/`` is either a dimension name, a value, or an operator. Example
(the exact URL this module generates, line-broken for clarity)::

    https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME      <- server + NMME root
        /.NCEP-CFSv2          <- model
        /.HINDCAST            <- dataset: HINDCAST (reforecasts) or FORECAST (real-time)
        /.MONTHLY             <- temporal resolution
        /.prec                <- variable (prec = precip in mm/day; tref = 2 m temp)
        /X/(8E)/(31E)/RANGEEDGES/     <- subset longitude to the 8E..31E edges
        /Y/(15S)/(25N)/RANGEEDGES/    <- subset latitude  to the 15S..25N edges
        /S/(0000 1 Jun 2000)/VALUES/  <- pick the start (initialization) time
        /data.nc                      <- export format (data.nc = NetCDF; dods = OPeNDAP)

Key ideas:
* ``X``/``Y`` are longitude/latitude; ``(8E)``, ``(31E)``, ``(15S)``, ``(25N)`` are
  human-readable coordinates. ``RANGEEDGES`` keeps everything between the two edges.
* ``S`` is the forecast start time, expressed as ``(HHMM day Mon Year)``. ``VALUES``
  selects the nearest matching grid point(s). ``S`` is stored as
  *"months since 1960-01-01"* on a non-standard ``360`` calendar (see ``decode_S``).
* ``L`` is the lead in months with half-month centres: ``0.5`` = first target month,
  ``1.5`` = second, ... Optionally subset with ``L/(0.5)/(2.5)/RANGEEDGES/``.
* ``M`` is the ensemble member.
* Spaces in the URL must be percent-encoded as ``%20`` (handled here).
* Swap the ``data.nc`` suffix for ``dods`` to open the same query lazily over OPeNDAP
  instead of downloading a file.

Dimension order of the array is ``prec[S][L][M][Y][X]``.
"""
import os
import urllib.parse

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.io._base import http_download

IRIDL_ROOT = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"

#: friendly name -> exact IRIDL model path token (the ``.<TOKEN>`` segment).
#: Tokens must match IRIDL spelling exactly; availability varies by model/variable.
NMME_MODELS = {
    "ncep_cfsv2":     "NCEP-CFSv2",
    "cmc1_cancm3":    "CMC1-CanCM3",
    "cmc2_cancm4":    "CMC2-CanCM4",
    "gem_nemo":       "GEM-NEMO",
    "cancm4i":        "CanCM4i",
    "gfdl_spear":     "GFDL-SPEAR",
    "nasa_geoss2s":   "NASA-GEOSS2S",
    "cola_ccsm4":     "COLA-RSMAS-CCSM4",
    "ncar_cesm1":     "NCAR-CESM1",
}

#: friendly variable name -> IRIDL variable token (+ canonical output name).
NMME_VARIABLES = {
    "PRCP": ("prec", "prec"),   # mm/day
    "TEMP": ("tref", "tref"),   # K (2 m temperature)
}

_MONTH_EPOCH = pd.Timestamp("1960-01-01")


# ----------------------------------------------------------------------------------
# coordinate token formatting
# ----------------------------------------------------------------------------------
def _fmt_lon(v):
    return f"({abs(v):g}{'E' if v >= 0 else 'W'})"


def _fmt_lat(v):
    return f"({abs(v):g}{'N' if v >= 0 else 'S'})"


def _fmt_start(date):
    """Format a date as the IRIDL start token, e.g. '(0000 1 Jun 2000)'."""
    d = pd.Timestamp(date)
    return f"(0000 {d.day} {d.strftime('%b %Y')})"


def decode_S(months_since_1960):
    """Decode IRIDL ``S`` values ('months since 1960-01-01') to month-start Timestamps."""
    out = [_MONTH_EPOCH + pd.DateOffset(months=int(round(float(m))))
           for m in np.atleast_1d(months_since_1960)]
    return pd.DatetimeIndex(out)


# ----------------------------------------------------------------------------------
# URL construction
# ----------------------------------------------------------------------------------
def build_nmme_url(model, variable, start, area,
                   dataset="HINDCAST", leads=None, fmt="data.nc"):
    """
    Build the IRIDL URL for an NMME monthly subset.

    Parameters
    ----------
    model : str         friendly key in NMME_MODELS (e.g. "ncep_cfsv2") or a raw IRIDL token.
    variable : str      key in NMME_VARIABLES ("PRCP", "TEMP").
    start : date-like   forecast initialization month (any day; month is what matters).
    area : tuple        (lon_min, lon_max, lat_min, lat_max) — matches config.MAP_EXTENT.
    dataset : str       "HINDCAST" (reforecasts) or "FORECAST" (real-time archive).
    leads : tuple|None  (l_min, l_max) in months to subset L; None keeps all leads.
    fmt : str           "data.nc" (download a NetCDF) or "dods" (OPeNDAP endpoint).
    """
    token = NMME_MODELS.get(model, model)
    if variable not in NMME_VARIABLES:
        raise KeyError(f"variable must be one of {list(NMME_VARIABLES)}")
    var_token, _ = NMME_VARIABLES[variable]
    lon_min, lon_max, lat_min, lat_max = area

    url = (f"{IRIDL_ROOT}/.{token}/.{dataset}/.MONTHLY/.{var_token}"
           f"/X/{_fmt_lon(lon_min)}/{_fmt_lon(lon_max)}/RANGEEDGES"
           f"/Y/{_fmt_lat(lat_min)}/{_fmt_lat(lat_max)}/RANGEEDGES")
    if leads is not None:
        url += f"/L/({leads[0]})/({leads[1]})/RANGEEDGES"
    url += f"/S/{_fmt_start(start)}/VALUES"
    url += "/dods" if fmt == "dods" else "/data.nc"

    # percent-encode spaces (and only spaces) inside the value tokens
    return url.replace(" ", "%20")


# ----------------------------------------------------------------------------------
# normalization
# ----------------------------------------------------------------------------------
def normalize_nmme(ds, add_valid_time=True):
    """
    Normalize a raw IRIDL NMME dataset to canonical coords.

    Renames X/Y/M -> longitude/latitude/number, decodes S -> ``start_time``, keeps
    ``L`` as ``lead`` (months), and optionally adds a 2-D ``valid_time(start_time, lead)``.
    """
    ds = ds.rename({k: v for k, v in
                    {"X": "longitude", "Y": "latitude", "M": "number"}.items()
                    if k in ds.variables})

    if "S" in ds.variables:
        start = decode_S(ds["S"].values)
        ds = ds.rename({"S": "start_time"})
        ds = ds.assign_coords(start_time=("start_time", start))
    if "L" in ds.variables:
        ds = ds.rename({"L": "lead"})

    if add_valid_time and "start_time" in ds.coords and "lead" in ds.coords:
        st = pd.DatetimeIndex(ds["start_time"].values)
        leads = np.asarray(ds["lead"].values, dtype=float)
        vt = np.empty((st.size, leads.size), dtype="datetime64[ns]")
        for i, s in enumerate(st):
            for j, l in enumerate(leads):
                # L=0.5 -> first target month; offset = round(L - 0.5)
                vt[i, j] = (s + pd.DateOffset(months=int(round(l - 0.5)))).to_datetime64()
        ds = ds.assign_coords(valid_time=(("start_time", "lead"), vt))

    return ds


# ----------------------------------------------------------------------------------
# public entry points
# ----------------------------------------------------------------------------------
def download_nmme(model, variable, start, area, dir_to_save,
                  dataset="HINDCAST", leads=None, force_download=False):
    """Download an NMME monthly subset to a NetCDF file and return its path."""
    url = build_nmme_url(model, variable, start, area, dataset=dataset,
                         leads=leads, fmt="data.nc")
    token = NMME_MODELS.get(model, model)
    tag = pd.Timestamp(start).strftime("%Y%m")
    dest = os.path.join(dir_to_save, f"nmme_{token}_{variable}_{dataset}_{tag}.nc")
    return http_download(url, dest, force_download=force_download)


def open_nmme(path_or_url, add_valid_time=True):
    """Open a downloaded file (or a /dods URL) and return a normalized Dataset."""
    ds = xr.open_dataset(path_or_url, decode_times=False)
    return normalize_nmme(ds, add_valid_time=add_valid_time)


def load_nmme(model, variable, start, area, dir_to_save,
              dataset="HINDCAST", leads=None, force_download=False):
    """Convenience: download a subset then open it, normalized. Returns a Dataset."""
    path = download_nmme(model, variable, start, area, dir_to_save,
                         dataset=dataset, leads=leads, force_download=force_download)
    return open_nmme(path)
