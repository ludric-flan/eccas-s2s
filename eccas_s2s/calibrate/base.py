"""
Calibration: common interface and predictive distributions (workflow E5, phase P3).

The chain calibrates **one model at a time**, on the observation grid, and every
product is then read from a single object: the **predictive distribution**. A
rainfall total, an anomaly, a tercile category, a probability of exceeding a
threshold and an event probability are five questions asked of the same
distribution — which is exactly the logic of the CAPC-AC method note, where each
indicator has its own method but they all end in a distribution one can question.

Two families are implemented:

* :class:`EnsembleDistribution` — a set of corrected members (bias correction,
  quantile mapping). Nothing is assumed about the shape of the distribution;
  probabilities are member frequencies, which is coarse when the ensemble is
  small (UKMO has 2 members) but never wrong.
* :class:`NormalDistribution` — mean and spread of a fitted law (NGR/EMOS, and
  the regressions of phase P4). Probabilities are exact, at the price of an
  assumed shape; for rainfall the fit is done on a transformed variable.

Every calibrator exposes the same two entry points:

``fit_predict_loyo(ensemble, obs)``
    the out-of-sample prediction of each hindcast year, the year being left out
    of the fit (decision D10) — this is what the scores of phase P2 need to
    compare raw and calibrated on the same footing;
``fit(ensemble, obs)`` then ``predict(ensemble)``
    the operational path: parameters fitted once on the whole hindcast, frozen,
    archived, then applied to the real-time forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

YEAR = "year"
MEMBER = "number"
CATEGORIES = ("BN", "NN", "AN")


class PredictiveDistribution:
    """What every calibrated product is read from (see the module docstring)."""

    def mean(self) -> xr.DataArray:
        raise NotImplementedError

    def prob_below(self, threshold: xr.DataArray) -> xr.DataArray:
        """P(X < threshold), with ``threshold`` broadcast over the grid."""
        raise NotImplementedError

    def quantile(self, p: float) -> xr.DataArray:
        raise NotImplementedError

    def tercile_probs(self, q33: xr.DataArray, q67: xr.DataArray) -> xr.DataArray:
        """
        Probabilities of the three categories, on the **observed** thresholds.

        The thresholds come from the observation (decision D12), never from the
        model: that is what makes the categories comparable between models.
        """
        p_bn = self.prob_below(q33)
        p_an = 1.0 - self.prob_below(q67)
        p_nn = (1.0 - p_bn - p_an).clip(0.0, 1.0)
        prob = xr.concat([p_bn, p_nn, p_an],
                         dim=xr.DataArray(list(CATEGORIES), dims="category", name="category"))
        return (prob / prob.sum("category")).rename("prob")


@dataclass
class EnsembleDistribution(PredictiveDistribution):
    """Corrected members; probabilities are member frequencies."""

    members: xr.DataArray
    member_dim: str = MEMBER

    def mean(self) -> xr.DataArray:
        return self.members.mean(self.member_dim)

    def spread(self) -> xr.DataArray:
        return self.members.std(self.member_dim, ddof=1)

    def prob_below(self, threshold: xr.DataArray) -> xr.DataArray:
        return (self.members < threshold).mean(self.member_dim)

    def quantile(self, p: float) -> xr.DataArray:
        return self.members.quantile(p, dim=self.member_dim, method="weibull").drop_vars("quantile")


@dataclass
class NormalDistribution(PredictiveDistribution):
    """
    Gaussian predictive law (NGR/EMOS, regressions).

    ``transform`` names the space the law lives in: ``None`` for a variable used
    as is (temperature), or a transform object for rainfall, whose back
    transformation is applied to every quantity read from the distribution.
    """

    mu: xr.DataArray
    sigma: xr.DataArray
    transform: object | None = None

    def _back(self, values: xr.DataArray) -> xr.DataArray:
        return values if self.transform is None else self.transform.inverse(values)

    def mean(self) -> xr.DataArray:
        # in a transformed space the back-transformed mean is a median; it is the
        # value the chain publishes as "the forecast", and the label says so.
        return self._back(self.mu)

    def prob_below(self, threshold: xr.DataArray) -> xr.DataArray:
        from scipy.special import ndtr

        t = threshold if self.transform is None else self.transform.forward(threshold)
        z = (t - self.mu) / self.sigma.clip(min=1e-9)
        return xr.apply_ufunc(ndtr, z, dask="parallelized", output_dtypes=[float])

    def quantile(self, p: float) -> xr.DataArray:
        from scipy.special import ndtri

        return self._back(self.mu + float(ndtri(p)) * self.sigma)


class Calibrator:
    """Base class: a name, a fit, a prediction, and the LOYO loop for free."""

    name = "base"
    #: does the method need the individual members, or only their mean?
    needs_members = True

    def fit(self, ensemble: xr.DataArray, obs: xr.DataArray, year_dim: str = YEAR,
            member_dim: str = MEMBER) -> dict:
        raise NotImplementedError

    def predict(self, ensemble: xr.DataArray, params: dict, member_dim: str = MEMBER
                ) -> PredictiveDistribution:
        raise NotImplementedError

    def loyo_params(self, ensemble: xr.DataArray, obs: xr.DataArray,
                    year_dim: str = YEAR, member_dim: str = MEMBER) -> dict:
        """
        Parameters of every leave-one-year-out fold, stacked along the year axis.

        Fitting fold by fold but predicting **once** keeps a method's code free of
        the cross-validation loop: ``predict`` simply sees parameters that happen
        to vary with the year, and xarray broadcasts them.
        """
        years = list(ensemble[year_dim].values)
        folds = []
        for y in years:
            keep = [v for v in years if v != y]
            params = self.fit(ensemble.sel({year_dim: keep}), obs.sel({year_dim: keep}),
                              year_dim, member_dim)
            folds.append(_expand_year(params, year_dim, y))
        return _stack_folds(folds, year_dim)

    def fit_predict_loyo(self, ensemble: xr.DataArray, obs: xr.DataArray,
                         year_dim: str = YEAR, member_dim: str = MEMBER
                         ) -> PredictiveDistribution:
        """Predict every hindcast year with parameters fitted without that year (D10)."""
        params = self.loyo_params(ensemble, obs, year_dim, member_dim)
        return self.predict(ensemble, params, member_dim)


def _expand_year(params, year_dim: str, year):
    """Tag every parameter of a fold with the year it predicts."""
    if isinstance(params, dict):
        return {k: _expand_year(v, year_dim, year) for k, v in params.items()}
    if isinstance(params, (list, tuple)):
        return [_expand_year(v, year_dim, year) for v in params]
    if isinstance(params, xr.DataArray):
        return params.expand_dims({year_dim: [year]})
    return params


def _stack_folds(folds: list, year_dim: str):
    """Concatenate the folds along the year axis, whatever the parameter tree."""
    first = folds[0]
    if isinstance(first, dict):
        return {k: _stack_folds([f[k] for f in folds], year_dim) for k in first}
    if isinstance(first, list):
        return [_stack_folds([f[i] for f in folds], year_dim) for i in range(len(first))]
    if isinstance(first, xr.DataArray):
        return xr.concat(folds, dim=year_dim)
    return first


def loyo_sum(da: xr.DataArray, year_dim: str = YEAR) -> xr.DataArray:
    """Sum over the years **except** the current one (vectorised leave-one-out)."""
    total = da.sum(year_dim)
    return total - da


def loyo_mean_and_var(da: xr.DataArray, year_dim: str = YEAR) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Leave-one-out mean and (unbiased) variance, in closed form.

    Recomputing them year by year would cost as many passes as there are years;
    the sums of ``x`` and ``x²`` give both in one pass, which is what makes the
    bias corrections and the quantile mappings affordable on a 0.25° grid.
    """
    n = da.sizes[year_dim] - 1
    s1 = loyo_sum(da, year_dim)
    s2 = loyo_sum(da ** 2, year_dim)
    mean = s1 / n
    var = (s2 - n * mean ** 2) / (n - 1)
    return mean, var.clip(min=0.0)


def to_ensemble(da: xr.DataArray, member_dim: str = MEMBER) -> xr.DataArray:
    """Give a member-less field a size-one member dimension (NMME)."""
    return da if member_dim in da.dims else da.expand_dims({member_dim: [0]})


def sanitize(values: xr.DataArray, variable: str) -> xr.DataArray:
    """Rainfall cannot be negative; a corrected total is clipped at zero."""
    return values.clip(min=0.0) if variable == "precip" else values


__all__ = ["CATEGORIES", "Calibrator", "EnsembleDistribution", "NormalDistribution",
           "PredictiveDistribution", "loyo_mean_and_var", "loyo_sum", "sanitize", "to_ensemble"]
