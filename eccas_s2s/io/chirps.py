"""
CHIRPS v2.0 monthly rainfall acquisition from the IRI Data Library.

CHIRPS (UCSB Climate Hazards Group) is a long, high-resolution (0.05deg) satellite+gauge
rainfall product — a primary observed dataset for Central Africa where gauge networks
are sparse. As with NMME, the IRIDL serves it with no credentials and lets us subset the
ECCAS box server-side. Monthly values are accumulations in mm.

Time axis: 'months since 1960-01-01' on a '360' calendar (decoded by io.iridl).
"""
import os

import pandas as pd

from eccas_s2s.io._base import http_download
from eccas_s2s.io import iridl

#: IRIDL path to CHIRPS v2.0 monthly global precipitation (variable token included).
CHIRPS_SOURCE = "/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation"


def build_chirps_url(area, year_start, year_end, fmt="data.nc"):
    """Build the IRIDL URL for a CHIRPS monthly subset over ``area`` and a year range."""
    url = iridl.IRIDL_BASE + CHIRPS_SOURCE + iridl.xy_range(area)
    url += (f"/T/{iridl.fmt_month(pd.Timestamp(int(year_start), 1, 1))}"
            f"/{iridl.fmt_month(pd.Timestamp(int(year_end), 12, 1))}/RANGEEDGES")
    url += "/dods" if fmt == "dods" else "/data.nc"
    return iridl.encode(url)


def download_chirps(area, year_start, year_end, dir_to_save, force_download=False):
    """Download a CHIRPS monthly subset to NetCDF and return its path."""
    url = build_chirps_url(area, year_start, year_end, fmt="data.nc")
    dest = os.path.join(dir_to_save, f"chirps_v2p0_monthly_{year_start}_{year_end}.nc")
    return http_download(url, dest, force_download=force_download)


def open_chirps(path_or_url):
    """Open a CHIRPS file / dods endpoint, normalized (time, latitude, longitude)."""
    return iridl.open_iridl(path_or_url, time_dim="T", rename_time_to="time")


def load_chirps(area, year_start, year_end, dir_to_save, force_download=False):
    """Convenience: download then open a CHIRPS monthly subset. Returns a Dataset."""
    path = download_chirps(area, year_start, year_end, dir_to_save,
                           force_download=force_download)
    return open_chirps(path)
