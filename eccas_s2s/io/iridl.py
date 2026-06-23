"""
Shared helpers for the IRI Data Library ("Ingrid") server.

Several public datasets (NMME, CHIRPS, TAMSAT, ...) live on the IRIDL and share the same
URL grammar and the same awkward time encoding ("months since 1960-01-01" on a
non-standard '360' calendar that xarray/cftime refuse to auto-decode). This module
centralizes:

* coordinate-token formatting and percent-encoding,
* the X/Y server-side subset path segment,
* opening a downloaded file / OPeNDAP endpoint with the time axis decoded by hand.

See ``nmme.py`` for a fully worked explanation of how an Ingrid URL is assembled.
"""
import re

import numpy as np
import pandas as pd
import xarray as xr

IRIDL_BASE = "https://iridl.ldeo.columbia.edu/SOURCES"


# ----------------------------------------------------------------------------------
# token formatting
# ----------------------------------------------------------------------------------
def fmt_lon(v):
    """Longitude -> Ingrid token, e.g. 4 -> '(4E)', -10 -> '(10W)'."""
    return f"({abs(v):g}{'E' if v >= 0 else 'W'})"


def fmt_lat(v):
    """Latitude -> Ingrid token, e.g. 25 -> '(25N)', -21 -> '(21S)'."""
    return f"({abs(v):g}{'N' if v >= 0 else 'S'})"


def fmt_month(date):
    """Date -> month token for T/S selection, e.g. '(Jun 2000)'."""
    return f"({pd.Timestamp(date).strftime('%b %Y')})"


def encode(url):
    """Percent-encode the spaces inside Ingrid value tokens."""
    return url.replace(" ", "%20")


def xy_range(area):
    """Server-side longitude/latitude subset segment for ``area``.

    ``area`` is (lon_min, lon_max, lat_min, lat_max), matching ``config.MAP_EXTENT``.
    """
    lon_min, lon_max, lat_min, lat_max = area
    return (f"/X/{fmt_lon(lon_min)}/{fmt_lon(lon_max)}/RANGEEDGES"
            f"/Y/{fmt_lat(lat_min)}/{fmt_lat(lat_max)}/RANGEEDGES")


# ----------------------------------------------------------------------------------
# time decoding
# ----------------------------------------------------------------------------------
def decode_months_since(values, units="months since 1960-01-01"):
    """Decode integer 'months since <epoch>' values to month-start Timestamps."""
    m = re.search(r"months since (\d{4})-(\d{1,2})-(\d{1,2})", units)
    epoch = (pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
             if m else pd.Timestamp("1960-01-01"))
    return pd.DatetimeIndex(
        [epoch + pd.DateOffset(months=int(round(float(v))))
         for v in np.atleast_1d(values)]
    )


def open_iridl(path_or_url, time_dim="T", rename_time_to=None, rename_xy=True):
    """
    Open an IRIDL file / ``/dods`` endpoint and decode its time axis manually.

    Parameters
    ----------
    time_dim : str          name of the time coordinate to decode ("T" or "S").
    rename_time_to : str    rename the decoded time dim (e.g. "time"); None keeps it.
    rename_xy : bool        rename X/Y -> longitude/latitude.
    """
    ds = xr.open_dataset(path_or_url, decode_times=False)
    if time_dim in ds.variables:
        units = ds[time_dim].attrs.get("units", "")
        if "months since" in units:
            decoded = decode_months_since(ds[time_dim].values, units)
            ds = ds.assign_coords({time_dim: (ds[time_dim].dims, decoded.values)})
        if rename_time_to and rename_time_to != time_dim:
            ds = ds.rename({time_dim: rename_time_to})
    if rename_xy:
        ds = ds.rename({k: v for k, v in {"X": "longitude", "Y": "latitude"}.items()
                        if k in ds.variables})
    return ds
