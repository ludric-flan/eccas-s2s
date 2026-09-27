"""
Write netCDF files other tools can actually open (CF-1.8).

The chain's own reader is xarray, which is happy with almost anything — string
dimensions, auxiliary coordinates, object dtypes. Other tools are not, and the
files are meant to be shared: CDO refuses a file whose dimensions carry
``NC_STRING`` coordinates ("Unsupported file structure"), ncview reports
``unknown data type (12)`` for the same reason, and a 4-D variable with no time
axis is skipped by both.

This module puts a dataset in a form those tools understand, without losing
anything the chain uses:

* a **time axis**. A period (``season_m1``, ``decade_m2_d3``) is a window of the
  calendar, so it becomes a real ``time`` coordinate — the first day of the
  window — with ``time_bnds`` giving its exact start and end. That is the CF way
  of saying "this value is an aggregate over that window", and it is what makes
  the file open in CDO, ncview, GrADS or QGIS. The original key stays as
  ``period_key``, so nothing is lost and the chain can read it back;
* **numeric category indices**. ``category`` becomes 0, 1, 2 with
  ``flag_values`` and ``flag_meanings`` (CF §3.5), the names staying in
  ``category_name``;
* **no text variable at all**. Labels, period keys and category names travel in
  **global attributes**, one pipe-separated list per axis (``period_key``,
  ``label``, ``label_fr``, ``scale``…), in the order of that axis. A character
  array on the time dimension is still read as a "character coordinate" by CDO,
  which then fails on it (``cdf_get_vara_text: Index exceeds dimension bound``);
  an attribute is read by everything and lost by nothing;
* **axis attributes** on latitude and longitude, and no ``_FillValue`` on a
  coordinate, which is invalid in CF and confuses several readers.

:func:`from_cf` is the inverse for the chain's own code: it restores the string
``period`` dimension so that ``sel(period="season_m1")`` keeps working.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TIME = "time"
PERIOD = "period"
BOUNDS = "nv"
#: text coordinates carried alongside the periods; they become plain variables,
#: never auxiliary coordinates, because a string auxiliary coordinate is exactly
#: what ncview and CDO choke on
TEXT_COORDS = ("period_key", "label", "label_fr", "scale", "calendar_key", "category_name")


def _char_encoding(ds: xr.Dataset) -> dict:
    """Write every text variable as a character array (netCDF-3 compatible)."""
    enc = {}
    for name, var in ds.variables.items():
        if var.dtype.kind in ("U", "O", "S"):
            enc[name] = {"dtype": "S1", "_FillValue": None}
    return enc


def to_cf(ds: xr.Dataset, init_year: int | None = None, init_month: int | None = None,
          period_dim: str = PERIOD) -> xr.Dataset:
    """
    CF-ready copy of ``ds``: time axis, numeric flags, character strings.

    ``init_year`` and ``init_month`` are those of the cycle; they turn each
    period key into the calendar window it designates for the forecast year. A
    dataset without a period dimension is returned with its coordinate
    attributes fixed, which is all it needs.
    """
    out = ds.copy()

    labels: dict[str, list[str]] = {}
    if period_dim in out.dims and out[period_dim].dtype.kind in ("U", "O"):
        keys = [str(k) for k in out[period_dim].values]
        out = out.assign_coords({period_dim: np.arange(len(keys), dtype="int32")})
        out[period_dim].attrs = {"long_name": "index of the target period",
                                 "comment": "clés dans l'attribut global period_key"}
        labels["period_key"] = keys
        if init_year is not None and init_month is not None:
            starts, ends = [], []
            from eccas_s2s.core.periods import period_from_key

            for key in keys:
                period = period_from_key(key, init_month)
                first, last = period.dates(init_year)
                starts.append(first)
                # CF bounds are half-open: the end is the first instant after the window
                ends.append(last + pd.Timedelta(days=1))
            out = out.assign_coords({TIME: (period_dim, pd.DatetimeIndex(starts))})
            out[TIME].attrs = {"standard_name": "time", "axis": "T", "bounds": "time_bnds",
                               "long_name": "first day of the target period"}
            bounds = np.stack([pd.DatetimeIndex(starts).values,
                               pd.DatetimeIndex(ends).values], axis=1)
            out["time_bnds"] = xr.DataArray(bounds, dims=(period_dim, BOUNDS))
            out = out.swap_dims({period_dim: TIME}).drop_vars(period_dim)
            # a bounds variable carries no coordinates of its own (CF §7.1); leaving
            # them makes CDO report an inconsistent variable definition
            out["time_bnds"].encoding["coordinates"] = None
            out["time_bnds"].attrs.pop("coordinates", None)

    # every other textual axis (the calendar keys of the observed normals, the
    # categories) becomes an integer index, its values kept in an attribute
    for dim in [d for d in out.dims if d in out.coords and out[d].dtype.kind in ("U", "O")]:
        names = [str(v) for v in out[dim].values]
        out = out.assign_coords({dim: np.arange(len(names), dtype="int32")})
        out[dim].attrs = {"long_name": f"index of {dim}",
                          "flag_values": np.arange(len(names), dtype="int32"),
                          "flag_meanings": " ".join(n.replace(" ", "_") for n in names)}
        labels[dim] = names

    # textual auxiliary coordinates and variables: their content moves to the
    # global attributes, aligned with the axis they described
    for name in list(out.coords) + list(out.data_vars):
        if name in out.dims or name not in out.variables:
            continue
        var = out[name]
        if var.dtype.kind in ("U", "O") and var.ndim <= 1:
            labels[name] = [str(v) for v in np.atleast_1d(var.values)]
            out = out.drop_vars(name)

    for name, values in labels.items():
        out.attrs[name] = " | ".join(values)
    if labels:
        out.attrs["text_axes"] = " ".join(sorted(labels))
        # a `coordinates` attribute still naming a variable we just moved into the
        # global attributes makes CDO warn on every read ("Variable not found")
        dropped = set(labels)
        for var in out.variables.values():
            for holder in (var.attrs, var.encoding):
                listed = holder.get("coordinates")
                if not listed:
                    continue
                kept = [c for c in str(listed).split() if c not in dropped]
                if kept:
                    holder["coordinates"] = " ".join(kept)
                else:
                    holder.pop("coordinates", None)

    for name, attrs in (("latitude", {"units": "degrees_north", "standard_name": "latitude",
                                      "axis": "Y", "long_name": "latitude"}),
                        ("longitude", {"units": "degrees_east", "standard_name": "longitude",
                                       "axis": "X", "long_name": "longitude"})):
        if name in out.coords:
            out[name].attrs.update(attrs)
    if "year" in out.coords:
        out["year"].attrs.setdefault("long_name", "initialisation year of the hindcast")
    if "number" in out.coords:
        out["number"].attrs.setdefault("long_name", "ensemble member")

    out.attrs.setdefault("Conventions", "CF-1.8")
    return out


def from_cf(ds: xr.Dataset, period_dim: str = PERIOD) -> xr.Dataset:
    """
    Restore the chain's own view: a string ``period`` dimension.

    The inverse of :func:`to_cf` for internal readers — everything the chain
    selects by (``sel(period="season_m1")``, ``sel(category="AN")``) works again,
    while the file on disk stays CF.
    """
    out = ds
    stored = {name: [v.strip() for v in str(out.attrs[name]).split("|")]
              for name in str(out.attrs.get("text_axes", "")).split() if name in out.attrs}
    if not stored:
        return out

    keys = stored.pop("period_key", None)
    if keys is not None:
        axis = TIME if TIME in out.dims else period_dim
        if axis in out.dims and out.sizes[axis] == len(keys):
            out = out.assign_coords({period_dim: (axis, keys)})
            if axis != period_dim:
                out = out.swap_dims({axis: period_dim})
    for name, values in stored.items():
        if name in out.dims and out.sizes[name] == len(values):
            out = out.assign_coords({name: values})          # a textual axis
        elif period_dim in out.dims and out.sizes[period_dim] == len(values):
            out = out.assign_coords({name: (period_dim, values)})
    return out


def open_cf(path, **kwargs) -> xr.Dataset:
    """Open a file written by :func:`save` and restore the chain's view."""
    return from_cf(xr.open_dataset(path, **kwargs))


def save(ds: xr.Dataset, path, init_year: int | None = None, init_month: int | None = None,
         complevel: int = 4, float32: bool = True) -> Path:
    """
    Write ``ds`` as a CF-1.8 netCDF, atomically.

    The file is written beside its destination and renamed, so a reader never
    sees a half-written file; floats are stored in single precision (a score map
    has no use for the extra digits) and compressed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = to_cf(ds, init_year, init_month)

    encoding = _char_encoding(out)
    for name, var in out.data_vars.items():
        if var.dtype.kind == "f":
            encoding.setdefault(name, {})
            encoding[name].update({"zlib": True, "complevel": complevel})
            if float32:
                encoding[name]["dtype"] = "float32"
    for name in out.coords:
        if out[name].dtype.kind == "f":
            encoding.setdefault(name, {})["_FillValue"] = None
    if TIME in out.coords:
        encoding.setdefault(TIME, {}).update({"units": "days since 1970-01-01",
                                              "dtype": "float64", "_FillValue": None})
        if "time_bnds" in out.variables:
            encoding.setdefault("time_bnds", {}).update(
                {"units": "days since 1970-01-01", "dtype": "float64", "_FillValue": None})

    tmp = path.with_suffix(".tmp.nc")
    out.to_netcdf(tmp, encoding=encoding, format="NETCDF4_CLASSIC"
                  if not _has_strings(out) else "NETCDF4")
    tmp.replace(path)
    return path


def _has_strings(ds: xr.Dataset) -> bool:
    """NETCDF4_CLASSIC cannot hold variable-length strings; char arrays are fine."""
    return False
