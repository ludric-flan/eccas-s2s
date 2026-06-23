"""
TAMSAT v3.1 monthly rainfall acquisition from the IRI Data Library.

TAMSAT (University of Reading) is an Africa-tuned satellite rainfall estimate, a useful
second observed product alongside CHIRPS. Served credential-free on the IRIDL with
server-side subsetting. Monthly values are accumulations in mm/month.

Time axis: 'months since 1960-01-01' on a '360' calendar (decoded by io.iridl).
"""
import os

import pandas as pd

from eccas_s2s.io._base import http_download
from eccas_s2s.io import iridl

#: IRIDL path to TAMSAT v3.1 monthly rainfall estimate (variable token included).
TAMSAT_SOURCE = "/.Reading/.Meteorology/.TAMSAT/.v3p1/.monthly/.rfe"


def build_tamsat_url(area, year_start, year_end, fmt="data.nc"):
    """Build the IRIDL URL for a TAMSAT monthly subset over ``area`` and a year range."""
    url = iridl.IRIDL_BASE + TAMSAT_SOURCE + iridl.xy_range(area)
    url += (f"/T/{iridl.fmt_month(pd.Timestamp(int(year_start), 1, 1))}"
            f"/{iridl.fmt_month(pd.Timestamp(int(year_end), 12, 1))}/RANGEEDGES")
    url += "/dods" if fmt == "dods" else "/data.nc"
    return iridl.encode(url)


def download_tamsat(area, year_start, year_end, dir_to_save, force_download=False):
    """Download a TAMSAT monthly subset to NetCDF and return its path."""
    url = build_tamsat_url(area, year_start, year_end, fmt="data.nc")
    dest = os.path.join(dir_to_save, f"tamsat_v3p1_monthly_{year_start}_{year_end}.nc")
    return http_download(url, dest, force_download=force_download)


def open_tamsat(path_or_url):
    """Open a TAMSAT file / dods endpoint, normalized (time, latitude, longitude)."""
    return iridl.open_iridl(path_or_url, time_dim="T", rename_time_to="time")


def load_tamsat(area, year_start, year_end, dir_to_save, force_download=False):
    """Convenience: download then open a TAMSAT monthly subset. Returns a Dataset."""
    path = download_tamsat(area, year_start, year_end, dir_to_save,
                           force_download=force_download)
    return open_tamsat(path)
