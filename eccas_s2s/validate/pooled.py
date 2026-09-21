"""
Pooled grid-point pairs of a zone, for the reliability and ROC diagrams.

A reliability diagram needs the forecast probabilities to be *sorted into bins*
and each bin to hold enough cases. On a zone index there are only as many cases
as hindcast years (24 for the C3S reference period): ten bins over 24 points
give observed frequencies pinned at 0 or 1, which says nothing about the
reliability of the system.

The reference chain of the CAPC-AC solves this the usual way
(``compute_pooled_diagrams_v2.R``): the pairs of **every grid point of the zone**
are pooled, so a bin holds hundreds of cases. This is legitimate here because
the events are tercile categories, whose base rate is 1/3 at every grid point by
construction — pooling therefore does not mix events of different frequency, as
it would for a fixed millimetre threshold.

Two properties of the pooled sample are carried to the diagrams and must be kept
in mind when reading them:

* the grid points of a zone are **not independent** (one season is one large-scale
  situation), so the pooled N is not an effective sample size: the confidence
  intervals are computed by resampling whole **years** (block bootstrap, the
  grid points travelling with their year), never individual pairs;
* a bin drawing on very few distinct years is a three-sample estimate however
  many pairs it holds; such bins are marked as under-populated and the diagram
  does not draw a line through them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

CATEGORIES = ("BN", "NN", "AN")
MAX_PIXELS = 400


def zone_cells(mask: xr.DataArray, max_pixels: int = MAX_PIXELS) -> xr.DataArray:
    """
    Cells of a zone, thinned to at most ``max_pixels`` by a regular stride.

    A regular stride (rather than a random draw) keeps the sample spatially
    balanced and makes the selection reproducible without a seed.
    """
    stacked = mask.stack(cell=("latitude", "longitude"))
    cells = stacked.where(stacked, drop=True)["cell"]
    n = int(cells.size)
    if n > max_pixels:
        step = int(np.ceil(n / max_pixels))
        cells = cells.isel(cell=slice(None, None, step))
    return cells


def pooled_frame(ds: xr.Dataset, mask: xr.DataArray, max_pixels: int = MAX_PIXELS
                 ) -> pd.DataFrame:
    """
    Long table of the pairs of a zone: one row per grid point, year and period.

    Columns: ``period, year, cell, lat, lon, ensmean, obs, obs_cat`` and, when
    the system has members, ``pBN, pNN, pAN``. Rows with a missing observation
    or a missing forecast are dropped.
    """
    cells = zone_cells(mask, max_pixels)
    keep = [v for v in ("ensmean", "obs", "obs_cat", "prob", "fcst_cat") if v in ds]
    sub = ds[keep].stack(cell=("latitude", "longitude")).sel(cell=cells)

    periods = list(sub["period"].values) if "period" in sub.dims else ["single"]
    years = sub["year"].values
    lat = np.asarray([c[0] for c in sub["cell"].values])
    lon = np.asarray([c[1] for c in sub["cell"].values])
    rows = []
    for p in periods:
        sel = sub.sel(period=p) if "period" in sub.dims else sub
        sel = sel.transpose("year", "cell", ..., missing_dims="ignore")
        n_year, n_cell = len(years), len(lat)
        row = {"period": p,
               "year": np.repeat(years, n_cell),
               "cell": np.tile(np.arange(n_cell), n_year),
               "lat": np.tile(lat, n_year), "lon": np.tile(lon, n_year),
               "ensmean": np.asarray(sel["ensmean"].values).ravel(),
               "obs": np.asarray(sel["obs"].values).ravel(),
               "obs_cat": np.asarray(sel["obs_cat"].values).ravel()}
        if "prob" in sel:
            for k, c in enumerate(CATEGORIES):
                row[f"p{c}"] = np.asarray(
                    sel["prob"].sel(category=c).transpose("year", "cell").values).ravel()
        elif "fcst_cat" in sel:
            row["fcst_cat"] = np.asarray(sel["fcst_cat"].values).ravel()
        rows.append(pd.DataFrame(row))

    frame = pd.concat(rows, ignore_index=True)
    frame = frame[np.isfinite(frame["obs"]) & np.isfinite(frame["ensmean"])
                  & np.isfinite(frame["obs_cat"])]
    return frame.reset_index(drop=True)
