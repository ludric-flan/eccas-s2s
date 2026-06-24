"""
In-situ station rainfall ingestion in CDT / CPT formats.

CAPC does not yet have a centralized in-situ station archive, so this reader is written
defensively: a missing station directory (or no matching files) is logged as a warning
and yields ``None`` / an empty result, so the rest of the eccas-s2s pipeline keeps
running on gridded data alone.

Supported layouts (stations in columns; ``.csv`` or ``.xlsx``)
-------------------------------------------------------------
CDT (daily/dekadal/monthly, one row per date)::

    ID          Ndjamena  Sarh_(Tchad)  ...     <- row 0: station names
    LON          15.05      18.38        ...     <- row 1: longitude
    LAT          12.11       9.15        ...     <- row 2: latitude
    DAILY/ELEV    (elev)     (elev)      ...     <- row 3: elevation (optional)
    19910101      0.0        0.0         ...     <- data rows, dates as YYYYMMDD
    19910102      ...

CPT (one row per year, e.g. a seasonal total)::

    STATION   Stn1   Stn2  ...                   <- header
    LAT       12.1    9.1  ...                    <- row 0: latitude
    LON       15.0   18.4  ...                    <- row 1: longitude
    1991      123.4  98.7  ...                    <- data rows, index = year

Both normalize to an xarray ``Dataset`` with a ``prcp(time, station)`` variable and
``longitude(station)`` / ``latitude(station)`` (plus ``elevation`` for CDT) coordinates.
This mirrors the WAS tool's CDT/CPT handling but is an independent implementation.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

#: sentinel values treated as missing in station files.
MISSING_VALUES = (-999.0, -99.0, -9999.0)


def _read_table(path):
    """Read a CDT/CPT file (.csv or .xlsx) as raw strings, no header inference."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, header=None, dtype=str)
    return pd.read_csv(path, header=None, dtype=str)


def _adjust_duplicate_coords(values):
    """Nudge exactly-duplicated station coordinates so each station stays distinct.

    Identical (lon, lat) pairs would collapse onto one point; add a tiny increasing
    epsilon to repeats (same intent as the WAS tool's ``adjust_duplicates``).
    """
    seen = {}
    out = []
    for v in values:
        if v in seen:
            seen[v] += 1
            out.append(v + 1e-6 * seen[v])
        else:
            seen[v] = 0
            out.append(v)
    return np.array(out, dtype=float)


def _to_dataset(stations, lon, lat, times, values, elevation=None):
    """Assemble the canonical station Dataset from parsed pieces."""
    lon = _adjust_duplicate_coords(lon)
    lat = _adjust_duplicate_coords(lat)
    data = np.asarray(values, dtype=float)
    for mv in MISSING_VALUES:
        data = np.where(np.isclose(data, mv), np.nan, data)

    coords = {
        "time": ("time", pd.DatetimeIndex(times)),
        "station": ("station", np.asarray(stations, dtype=object)),
        "longitude": ("station", lon),
        "latitude": ("station", lat),
    }
    if elevation is not None:
        coords["elevation"] = ("station", np.asarray(elevation, dtype=float))

    ds = xr.Dataset(
        {"prcp": (("time", "station"), data)},
        coords=coords,
    )
    ds["prcp"].attrs = {"long_name": "Station precipitation", "units": "mm"}
    return ds


def parse_cdt(df):
    """Parse a CDT-format DataFrame (raw strings) to the canonical station Dataset."""
    stations = df.iloc[0, 1:].tolist()
    lon = pd.to_numeric(df.iloc[1, 1:], errors="coerce").to_numpy()
    lat = pd.to_numeric(df.iloc[2, 1:], errors="coerce").to_numpy()

    # row 3 is elevation if its label looks like ELEV/DAILY/DEKADAL/MONTHLY, else data
    label3 = str(df.iloc[3, 0]).upper()
    if any(tok in label3 for tok in ("ELEV", "DAILY", "DEKAD", "MONTH")):
        elevation = pd.to_numeric(df.iloc[3, 1:], errors="coerce").to_numpy()
        data_start = 4
    else:
        elevation = None
        data_start = 3

    date_labels = df.iloc[data_start:, 0].astype(str)
    times = pd.to_datetime(date_labels, format="%Y%m%d", errors="coerce")
    valid = times.notna().to_numpy()
    values = df.iloc[data_start:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
    return _to_dataset(stations, lon, lat, times[valid], values[valid], elevation)


def parse_cpt(df, month_day="08-01"):
    """Parse a CPT-format DataFrame (raw strings) to the canonical station Dataset.

    Yearly rows are dated ``YYYY-<month_day>`` (default 1 August) so the time axis is
    a real datetime, consistent with the CDT path.
    """
    stations = df.iloc[0, 1:].tolist()
    # CPT row order is LAT then LON
    lat = pd.to_numeric(df.iloc[1, 1:], errors="coerce").to_numpy()
    lon = pd.to_numeric(df.iloc[2, 1:], errors="coerce").to_numpy()

    year_labels = df.iloc[3:, 0].astype(str)
    times = pd.to_datetime(year_labels + f"-{month_day}", format="%Y-%m-%d", errors="coerce")
    valid = times.notna().to_numpy()
    values = df.iloc[3:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
    return _to_dataset(stations, lon, lat, times[valid], values[valid])


def _detect_format(df):
    """Return 'cdt' or 'cpt' from the first-column row labels."""
    first = str(df.iloc[0, 0]).strip().upper()
    if first == "ID":
        return "cdt"
    if first in ("STATION", "STN"):
        return "cpt"
    # fall back on the row-1/2 labels
    labels = {str(df.iloc[i, 0]).strip().upper() for i in range(min(3, len(df)))}
    if "LON" in labels and "LAT" in labels:
        # CDT has LON on row 1; CPT has LAT on row 1
        return "cdt" if str(df.iloc[1, 0]).strip().upper() == "LON" else "cpt"
    raise ValueError(f"Unrecognized station file layout (first cell={first!r})")


def read_station_file(path, fmt=None, month_day="08-01"):
    """Read one CDT/CPT station file and return the canonical Dataset."""
    df = _read_table(path)
    fmt = fmt or _detect_format(df)
    if fmt == "cdt":
        return parse_cdt(df)
    if fmt == "cpt":
        return parse_cpt(df, month_day=month_day)
    raise ValueError(f"Unknown fmt {fmt!r}; expected 'cdt' or 'cpt'")


def load_stations(station_dir, pattern="*.csv", fmt=None, month_day="08-01"):
    """
    Load station files from ``station_dir`` into one combined Dataset.

    Returns ``None`` (with a warning) if the directory is missing or contains no
    matching files — so callers can simply skip station data when unavailable::

        ds = load_stations(cfg.STATION_DIR)
        if ds is not None:
            ...  # blend with gridded obs

    Multiple files are concatenated along ``station``; CSV and XLSX patterns can be
    combined by passing e.g. ``pattern=("*.csv", "*.xlsx")``.
    """
    if not station_dir or not os.path.isdir(station_dir):
        warnings.warn(
            f"Station directory not found: {station_dir!r}. "
            "Continuing without in-situ station data.",
            stacklevel=2,
        )
        return None

    patterns = (pattern,) if isinstance(pattern, str) else tuple(pattern)
    files = sorted({f for p in patterns for f in glob.glob(os.path.join(station_dir, p))})
    if not files:
        warnings.warn(
            f"No station files matching {patterns} in {station_dir!r}. "
            "Continuing without in-situ station data.",
            stacklevel=2,
        )
        return None

    datasets = []
    for path in files:
        try:
            datasets.append(read_station_file(path, fmt=fmt, month_day=month_day))
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the batch
            warnings.warn(f"Skipping unreadable station file {path!r}: {exc}", stacklevel=2)

    if not datasets:
        warnings.warn(
            f"All station files in {station_dir!r} failed to parse. "
            "Continuing without in-situ station data.",
            stacklevel=2,
        )
        return None

    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="station")
