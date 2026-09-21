"""
Verification scores computed per grid point (workflow step E4/E9, Draft §5.2).

These are the vectorised versions used for **maps** and for the eligibility
criterion of §3.3. The zone-average scores of the reference implementation are
computed in R with the ``verification`` package (see
:mod:`eccas_s2s.validate.r_bridge`); the two are cross-checked in the tests.

Conventions
-----------
* deterministic scores compare the ensemble mean with the observation over the
  ``year`` dimension;
* probabilistic scores take a ``category`` dimension (BN, NN, AN) of forecast
  probabilities and the observed category index (0, 1, 2);
* the reference of the skill scores is the climatological forecast (1/3 in each
  category), as in the Draft Framework.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

YEAR = "year"
CATEGORIES = ("BN", "NN", "AN")


# --------------------------------------------------------------- deterministic
def _rank_along_year(da: xr.DataArray, year_dim: str) -> xr.DataArray:
    """Average ranks along ``year`` (ties share their mean rank); NaNs stay NaN."""
    from scipy.stats import rankdata

    def _rank(a):
        out = np.full(a.shape, np.nan)
        flat, oflat = a.reshape(-1, a.shape[-1]), out.reshape(-1, a.shape[-1])
        for i in range(flat.shape[0]):
            ok = np.isfinite(flat[i])
            if ok.sum() >= 2:
                oflat[i, ok] = rankdata(flat[i, ok])
        return out

    return xr.apply_ufunc(_rank, da, input_core_dims=[[year_dim]], output_core_dims=[[year_dim]],
                          dask="parallelized", output_dtypes=[float])


def deterministic_scores(forecast: xr.DataArray, obs: xr.DataArray,
                         year_dim: str = YEAR) -> xr.Dataset:
    """
    Bias, MAE, RMSE, MSESS, Pearson, Spearman and ACC per grid point.

    ``forecast`` is the ensemble mean. The anomaly correlation (ACC) uses
    anomalies relative to the mean of the verification sample, so for a single
    period it is identical to the Pearson correlation; both are kept because
    the Draft Framework lists them separately.
    """
    valid = forecast.notnull() & obs.notnull()
    f = forecast.where(valid)
    o = obs.where(valid)
    n = valid.sum(year_dim)

    err = f - o
    bias = err.mean(year_dim)
    mae = abs(err).mean(year_dim)
    mse = (err ** 2).mean(year_dim)
    o_mean = o.mean(year_dim)
    mse_clim = ((o - o_mean) ** 2).mean(year_dim)

    fa = f - f.mean(year_dim)
    oa = o - o_mean
    denom = np.sqrt((fa ** 2).sum(year_dim) * (oa ** 2).sum(year_dim))
    pearson = (fa * oa).sum(year_dim) / denom.where(denom > 0)

    rf = _rank_along_year(f, year_dim)
    ro = _rank_along_year(o, year_dim)
    rfa = rf - rf.mean(year_dim)
    roa = ro - ro.mean(year_dim)
    rden = np.sqrt((rfa ** 2).sum(year_dim) * (roa ** 2).sum(year_dim))
    spearman = (rfa * roa).sum(year_dim) / rden.where(rden > 0)

    ds = xr.Dataset({
        "n_years": n,
        "bias": bias,
        "mae": mae,
        "rmse": np.sqrt(mse),
        "rmse_clim": np.sqrt(mse_clim),
        "msess": (1 - mse / mse_clim.where(mse_clim > 0)),
        "pearson": pearson,
        "acc": pearson,
        "spearman": spearman,
    })
    ds = ds.where(n >= 3)
    ds["n_years"] = n
    for v, long_name in (("bias", "mean error (forecast - observation)"),
                         ("mae", "mean absolute error"), ("rmse", "root mean square error"),
                         ("rmse_clim", "root mean square error of the climatological forecast"),
                         ("msess", "mean square error skill score vs climatology"),
                         ("pearson", "Pearson correlation"), ("acc", "anomaly correlation"),
                         ("spearman", "Spearman rank correlation")):
        ds[v].attrs["long_name"] = long_name
    return ds


# -------------------------------------------------------------- probabilistic
def _observed_matrix(obs_cat: xr.DataArray, n_cat: int, cat_dim: str) -> xr.DataArray:
    cats = xr.DataArray(np.arange(n_cat), dims=cat_dim, coords={cat_dim: list(CATEGORIES[:n_cat])})
    return (obs_cat == cats).astype(float)


def rps_scores(prob: xr.DataArray, obs_cat: xr.DataArray, year_dim: str = YEAR,
               cat_dim: str = "category") -> xr.Dataset:
    """
    Ranked probability score and skill score (reference: climatology 1/K).

    ``prob`` has a ``category`` dimension in increasing order (BN, NN, AN);
    ``obs_cat`` holds the observed category index. The sum of squared
    differences of the cumulative distributions is divided by ``K - 1``, the
    convention of Wilks and of the R ``verification`` package, so that a
    forecast that always puts all its probability on the wrong extreme category
    scores 1.
    """
    n_cat = prob.sizes[cat_dim]
    obs_mat = _observed_matrix(obs_cat, n_cat, cat_dim)
    valid = prob.notnull().all(cat_dim) & obs_cat.notnull()
    cum_f = prob.cumsum(cat_dim)
    cum_o = obs_mat.cumsum(cat_dim)
    norm = n_cat - 1
    rps = ((cum_f - cum_o) ** 2).sum(cat_dim).where(valid).mean(year_dim) / norm

    clim = xr.full_like(prob, 1.0 / n_cat)
    rps_clim = (((clim.cumsum(cat_dim) - cum_o) ** 2).sum(cat_dim)).where(valid).mean(year_dim) / norm
    n = valid.sum(year_dim)
    ds = xr.Dataset({"rps": rps, "rps_clim": rps_clim,
                     "rpss": 1 - rps / rps_clim.where(rps_clim > 0), "n_years": n})
    ds = ds.where(n >= 3)
    ds["n_years"] = n
    ds["rpss"].attrs["long_name"] = "ranked probability skill score vs climatology"
    return ds


def brier_scores(prob_event: xr.DataArray, event: xr.DataArray, climatology: float | xr.DataArray,
                 year_dim: str = YEAR) -> xr.Dataset:
    """Brier score and skill score of one binary event (reference: ``climatology``)."""
    valid = prob_event.notnull() & event.notnull()
    p = prob_event.where(valid)
    o = event.where(valid)
    bs = ((p - o) ** 2).mean(year_dim)
    bs_clim = ((climatology - o) ** 2).mean(year_dim)
    n = valid.sum(year_dim)
    ds = xr.Dataset({"bs": bs, "bs_clim": bs_clim,
                     "bss": 1 - bs / bs_clim.where(bs_clim > 0), "base_rate": o.mean(year_dim),
                     "n_years": n})
    ds = ds.where(n >= 3)
    ds["n_years"] = n
    return ds


def roc_area(prob_event: xr.DataArray, event: xr.DataArray, year_dim: str = YEAR) -> xr.DataArray:
    """
    Area under the ROC curve, as the two-alternatives-forced-choice estimate.

    The 2AFC form (Mann-Whitney U) is exact for a finite sample and needs no
    threshold binning: it is the probability that the forecast probability of an
    event year exceeds that of a non-event year, ties counting a half. NaN where
    the sample has no event or no non-event.
    """
    def _auc(p, e):
        out = np.full(p.shape[:-1], np.nan)
        for idx in np.ndindex(p.shape[:-1]):
            pi, ei = p[idx], e[idx]
            ok = np.isfinite(pi) & np.isfinite(ei)
            pi, ei = pi[ok], ei[ok]
            pos, neg = pi[ei == 1], pi[ei == 0]
            if pos.size == 0 or neg.size == 0:
                continue
            greater = (pos[:, None] > neg[None, :]).sum()
            ties = (pos[:, None] == neg[None, :]).sum()
            out[idx] = (greater + 0.5 * ties) / (pos.size * neg.size)
        return out

    da = xr.apply_ufunc(_auc, prob_event, event, input_core_dims=[[year_dim], [year_dim]],
                        dask="parallelized", output_dtypes=[float])
    da.attrs["long_name"] = "ROC area (2AFC estimate)"
    return da


def tercile_skill(prob: xr.DataArray, obs_cat: xr.DataArray, year_dim: str = YEAR,
                  cat_dim: str = "category") -> xr.Dataset:
    """RPS/RPSS plus, for each category, the Brier skill score and the ROC area."""
    ds = rps_scores(prob, obs_cat, year_dim, cat_dim)
    bss, roc = [], []
    for k, name in enumerate(CATEGORIES[:prob.sizes[cat_dim]]):
        event = (obs_cat == k).astype(float).where(obs_cat.notnull())
        p = prob.isel({cat_dim: k})
        bss.append(brier_scores(p, event, 1.0 / prob.sizes[cat_dim], year_dim)["bss"])
        roc.append(roc_area(p, event, year_dim))
    cats = list(CATEGORIES[:prob.sizes[cat_dim]])
    ds["bss"] = xr.concat(bss, dim=xr.DataArray(cats, dims=cat_dim, name=cat_dim))
    ds["roc_area"] = xr.concat(roc, dim=xr.DataArray(cats, dims=cat_dim, name=cat_dim))
    ds["bss"].attrs["long_name"] = "Brier skill score vs climatology, per category"
    return ds
