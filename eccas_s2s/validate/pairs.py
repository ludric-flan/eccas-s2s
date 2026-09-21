"""
Forecast / observation pairs of a hindcast, ready to be scored (step E4).

For one model, one variable and one set of periods, this module builds — on the
**model grid**, year by year — everything the scores need:

* ``ensmean`` and ``obs``: ensemble mean and observation (deterministic scores);
* ``prob`` (BN, NN, AN) and ``obs_cat``: tercile probabilities and observed
  category (probabilistic scores);
* ``ens_sd``: ensemble spread, used by the CRPS of the R package.

Two different climatologies on purpose, as in the reference chain:

* the **observed** categories use the observation's own terciles, computed
  leave-one-year-out (decision D10) — an "above normal" is defined by the
  observations, never by the model;
* the **forecast** probabilities count the members beyond the terciles of the
  *model's own* hindcast, also leave-one-year-out. Raw model values are biased,
  so their probabilities can only be defined against the model climate. This is
  what makes the result a fair **raw** baseline for the calibration (E5).

A model whose ensemble has very few members (UKMO with 2, BoM with 3) can only
produce coarse probabilities; this is a property of the data, reported by the
scores, not something corrected here.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from eccas_s2s.validate.cv import loyo_quantile

CATEGORIES = ("BN", "NN", "AN")
TERCILES = (1 / 3, 2 / 3)


def observed_categories(obs: xr.DataArray, year_dim: str = "year",
                        method: str = "weibull") -> tuple[xr.DataArray, xr.DataArray]:
    """
    Observed tercile category (0, 1, 2) and the leave-one-year-out thresholds.

    Ties with a threshold (frequent when many years are dry) fall in the lower
    category, as ``<`` / ``>=`` comparisons do.
    """
    q = loyo_quantile(obs, TERCILES, year_dim=year_dim, method=method)
    q33 = q.sel(quantile=TERCILES[0], drop=True)
    q67 = q.sel(quantile=TERCILES[1], drop=True)
    cat = xr.where(obs < q33, 0, xr.where(obs >= q67, 2, 1)).where(obs.notnull())
    cat.name = "obs_cat"
    cat.attrs = {"long_name": "observed tercile category (0 BN, 1 NN, 2 AN)",
                 "thresholds": "leave-one-year-out terciles of the observation"}
    return cat, q


def tercile_probabilities(ensemble: xr.DataArray, year_dim: str = "year",
                          member_dim: str = "number", method: str = "weibull") -> xr.DataArray:
    """
    Raw tercile probabilities: fraction of members beyond the model's own
    leave-one-year-out terciles (members pooled with the years).
    """
    q = loyo_quantile(ensemble, TERCILES, dims=[member_dim], year_dim=year_dim, method=method)
    q33 = q.sel(quantile=TERCILES[0], drop=True)
    q67 = q.sel(quantile=TERCILES[1], drop=True)
    p_bn = (ensemble < q33).mean(member_dim)
    p_an = (ensemble >= q67).mean(member_dim)
    p_nn = 1.0 - p_bn - p_an
    prob = xr.concat([p_bn, p_nn, p_an],
                     dim=xr.DataArray(list(CATEGORIES), dims="category", name="category"))
    prob = prob.where(ensemble.notnull().any(member_dim))
    prob.name = "prob"
    prob.attrs = {"long_name": "raw tercile probabilities (member counting)",
                  "thresholds": "leave-one-year-out terciles of the model hindcast",
                  "n_members": int(ensemble.sizes[member_dim])}
    return prob


def build_pairs(hindcast: xr.DataArray, obs: xr.DataArray, year_dim: str = "year",
                member_dim: str = "number") -> xr.Dataset:
    """
    Assemble the pairs of one model and one variable, on the common years.

    ``hindcast`` has dims ``(year, number, period, lat, lon)`` (or no ``number``
    for an ensemble-mean system such as NMME) and ``obs`` ``(year, period, lat,
    lon)`` on the same grid. Only the years present in both are kept.
    """
    years = np.intersect1d(hindcast[year_dim].values, obs[year_dim].values)
    if len(years) < 3:
        raise ValueError(f"trop peu d'années communes ({len(years)}) pour vérifier")
    fc = hindcast.sel({year_dim: years})
    ob = obs.sel({year_dim: years})

    has_members = member_dim in fc.dims
    ensmean = fc.mean(member_dim) if has_members else fc
    ens_sd = fc.std(member_dim, ddof=1) if has_members else xr.full_like(ensmean, np.nan)
    obs_cat, obs_q = observed_categories(ob, year_dim)
    if has_members:
        prob = tercile_probabilities(fc, year_dim, member_dim)
    else:
        # no members: a "probability" can only be the deterministic category of
        # the ensemble mean, kept for completeness and flagged in the attributes
        q = loyo_quantile(ensmean, TERCILES, year_dim=year_dim)
        cat = xr.where(ensmean < q.sel(quantile=TERCILES[0], drop=True), 0,
                       xr.where(ensmean >= q.sel(quantile=TERCILES[1], drop=True), 2, 1))
        prob = xr.concat([(cat == k).astype(float) for k in range(3)],
                         dim=xr.DataArray(list(CATEGORIES), dims="category", name="category"))
        prob.attrs = {"long_name": "deterministic category of the ensemble mean (0/1)",
                      "note": "system without members: no probabilistic information"}

    ds = xr.Dataset({"ensmean": ensmean, "ens_sd": ens_sd, "obs": ob,
                     "obs_cat": obs_cat, "prob": prob,
                     "obs_q33": obs_q.sel(quantile=TERCILES[0], drop=True),
                     "obs_q67": obs_q.sel(quantile=TERCILES[1], drop=True)})
    ds.attrs = {"n_years": int(len(years)), "years": f"{years[0]}-{years[-1]}",
                "has_members": int(has_members),
                "n_members": int(fc.sizes[member_dim]) if has_members else 0}
    return ds


def zone_index(ds: xr.Dataset, mask: xr.DataArray, lat_dim: str = "latitude") -> xr.Dataset:
    """
    Area-average the pairs over a zone, then rebuild the categories of the index.

    Averaging probabilities over a zone would mix grid points with different
    signs; the reference chain therefore averages the **physical quantities**
    (forecast members and observation) and recomputes the categories on the
    index. Here the ensemble mean and the observation are averaged with a
    cos(latitude) weight, and the tercile categories of the index are rebuilt
    leave-one-year-out; the probabilities are the (already computed) mean of the
    grid-point probabilities, which keeps the zone score comparable with the maps.
    """
    weights = np.cos(np.deg2rad(ds[lat_dim])).where(mask)
    out = {}
    for v in ("ensmean", "ens_sd", "obs", "prob"):
        out[v] = ds[v].weighted(weights.fillna(0)).mean([d for d in mask.dims])
    idx = xr.Dataset(out)
    idx["obs_cat"], _ = observed_categories(idx["obs"])
    idx.attrs = {**ds.attrs, "zone_cells": int(mask.sum())}
    return idx
