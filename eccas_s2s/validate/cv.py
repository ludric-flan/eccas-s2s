"""
Cross-validation helpers (decision D10: leave-one-year-out).

Every quantity fitted on the hindcast — category thresholds, climatologies,
later the calibration coefficients — must be computed **without the year being
verified** (Draft Framework §4.3). These helpers do that on the ``year``
dimension of an :class:`xarray.DataArray`.

``loyo_quantile`` pools the ensemble members with the years, as the raw tercile
probabilities of the products do: the thresholds describe the model climate,
not the climate of one member.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

YEAR = "year"


def loyo_mean(da: xr.DataArray, dims=None, year_dim: str = YEAR) -> xr.DataArray:
    """
    Mean over ``dims`` excluding, for each year, that year itself.

    With ``n`` years, the leave-one-out mean is ``(n * mean - year) / (n - 1)``,
    computed in one pass instead of ``n`` passes.
    """
    dims = list(dims or [d for d in da.dims if d != year_dim])
    others = [d for d in dims if d != year_dim]
    per_year = da.mean(others) if others else da
    n = per_year.sizes[year_dim]
    if n < 2:
        raise ValueError("au moins deux années sont nécessaires pour le LOYO")
    total = per_year.sum(year_dim)
    return (total - per_year) / (n - 1)


def loyo_quantile(da: xr.DataArray, q, dims=None, year_dim: str = YEAR,
                  method: str = "weibull") -> xr.DataArray:
    """
    Quantiles of ``da`` computed for each year on the other years.

    ``dims`` are the dimensions pooled with the years (e.g. ``["number"]`` to
    pool the ensemble members). Returns a DataArray with the original ``year``
    dimension plus a ``quantile`` dimension when several ``q`` are given.
    """
    qs = np.atleast_1d(np.asarray(q, dtype=float))
    dims = list(dims or [])
    stack_dims = [year_dim] + dims
    years = da[year_dim].values
    out = []
    for i in range(len(years)):
        sample = da.isel({year_dim: [j for j in range(len(years)) if j != i]})
        out.append(sample.quantile(qs, dim=stack_dims, method=method).drop_vars("quantile"))
    res = xr.concat(out, dim=da[year_dim])
    res = res.assign_coords(quantile=("quantile", qs))
    return res.squeeze("quantile", drop=True) if len(qs) == 1 else res


def loyo_year_mask(years, verified_year: int) -> np.ndarray:
    """Boolean mask of the training years for one verified year."""
    years = np.asarray(years)
    return years != verified_year
