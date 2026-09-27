"""
Logistic calibration: terciles, thresholds and events (workflow E5c).

The CAPC-AC method note asks for a *probabilistic* calibration wherever the
product is a probability — a tercile, an exceedance, an event. Raw member
counting answers the question but badly: a model that says 70 % and is right 40 %
of the time is overconfident, and phase P2 showed exactly that on the CEEAC
(RPSS ≲ 0 while the ROC area stays above 0.5 — the signal is there, its
expression is not).

Two methods, both fitted per grid point and leave-one-year-out:

**Logistic regression** (:class:`TercileLogistic`) models the probability of
being below a threshold as ``P = σ(a + b·x)``, where ``x`` is the standardised
ensemble mean. One fit per threshold. Simple, robust on 24 years, and it
*learns* how much of the model's signal deserves to be believed: a model with no
skill converges to ``b ≈ 0``, hence to the climatological probability, which is
the honest answer.

**Extended logistic regression** (:class:`ExtendedLogistic`, Wilks 2009) adds the
threshold itself as a covariate: ``P(y ≤ q) = σ(a + c·g(q) + b·x)``. One fit
covers every threshold at once, the probabilities are automatically coherent
(never decreasing with the threshold), and the model becomes a full predictive
distribution — which is what the chain needs to publish quintiles, percentiles
and arbitrary thresholds from a single calibration.

Fitting is a Newton (IRLS) iteration vectorised over grid points: the whole
0.25° CEEAC mask and its 24 folds take about two seconds per period.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from eccas_s2s.calibrate.base import (YEAR, MEMBER, CATEGORIES, Calibrator,
                                      PredictiveDistribution, to_ensemble)

EPS = 1e-9
RIDGE = 1e-3          # keeps the Newton step finite when a cell is degenerate
MAX_ITER = 12          # Newton converges well before that (measured: identical probabilities)
TOL = 1e-7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logistic_fit(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """
    Newton/IRLS fit of a logistic regression, vectorised over the leading axes.

    ``X`` has shape ``(..., n, p)`` and ``y`` ``(..., n)``; the returned
    coefficients have shape ``(..., p)``. A small ridge term keeps the Hessian
    invertible where a cell has no variation at all (an always-dry desert point,
    a category that never occurs in the training years).
    """
    *lead, n, p = X.shape
    beta = np.zeros((*lead, p))
    eye = np.eye(p) * ridge
    for _ in range(MAX_ITER):
        eta = np.einsum("...np,...p->...n", X, beta)
        mu = _sigmoid(eta)
        w = np.clip(mu * (1.0 - mu), 1e-6, None)
        grad = np.einsum("...np,...n->...p", X, y - mu) - ridge * beta
        hess = np.einsum("...np,...n,...nq->...pq", X, w, X) + eye
        step = np.linalg.solve(hess, grad[..., None])[..., 0]
        beta = beta + step
        if np.nanmax(np.abs(step)) < TOL:
            break
    return beta


def _standardised_predictor(ensemble: xr.DataArray, year_dim: str, member_dim: str
                            ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Ensemble mean, plus the mean and spread used to standardise it."""
    x = to_ensemble(ensemble, member_dim).mean(member_dim)
    mu = x.mean(year_dim)
    sd = x.std(year_dim, ddof=1)
    return x, mu, sd.where(sd > EPS, 1.0)


class _LogisticDistribution(PredictiveDistribution):
    """
    Predictive distribution defined by its probabilities of non-exceedance.

    A logistic calibration gives probabilities, not values: ``mean()`` returns
    the deterministic forecast it was fitted on (the corrected ensemble mean),
    and ``quantile`` is only available for the extended form, which knows every
    threshold.
    """

    def __init__(self, prob_fn, mean_value: xr.DataArray, quantile_fn=None):
        self._prob_fn = prob_fn
        self._mean = mean_value
        self._quantile_fn = quantile_fn

    def mean(self) -> xr.DataArray:
        return self._mean

    def prob_below(self, threshold: xr.DataArray) -> xr.DataArray:
        return self._prob_fn(threshold)

    def quantile(self, p: float) -> xr.DataArray:
        if self._quantile_fn is None:
            raise NotImplementedError("la régression logistique simple ne donne pas de quantile ;"
                                      " utiliser la forme étendue (ELR)")
        return self._quantile_fn(p)


class TercileLogistic(Calibrator):
    """
    Probability of each tercile by logistic regression on the ensemble mean.

    Two fits per grid point — ``P(y < q33)`` and ``P(y < q67)`` — then the middle
    category by difference. The thresholds are the **observed** ones (D12), and
    inside a LOYO fold they are recomputed without the year being predicted.
    """

    name = "logistic"
    needs_members = False

    def __init__(self, variable: str = "precip", quantiles=(1 / 3, 2 / 3)):
        self.variable = variable
        self.quantiles = tuple(quantiles)

    def _fit_threshold(self, x_std: xr.DataArray, obs: xr.DataArray, q: xr.DataArray,
                       year_dim: str) -> xr.DataArray:
        below = (obs < q).astype(float)
        ones = xr.ones_like(x_std)
        X = xr.concat([ones, x_std], dim="coef").transpose(..., year_dim, "coef")
        beta = xr.apply_ufunc(
            logistic_fit, X, below.transpose(..., year_dim),
            input_core_dims=[[year_dim, "coef"], [year_dim]], output_core_dims=[["coef"]],
            dask="parallelized", output_dtypes=[float])
        return beta.assign_coords(coef=["intercept", "slope"])

    def fit(self, ensemble: xr.DataArray, obs: xr.DataArray, year_dim: str = YEAR,
            member_dim: str = MEMBER) -> dict:
        x, mu, sd = _standardised_predictor(ensemble, year_dim, member_dim)
        x_std = (x - mu) / sd
        qs = [obs.quantile(q, dim=year_dim, method="weibull").drop_vars("quantile")
              for q in self.quantiles]
        betas = [self._fit_threshold(x_std, obs, q, year_dim) for q in qs]
        return {"mu": mu, "sd": sd, "betas": betas, "thresholds": qs}

    def predict(self, ensemble: xr.DataArray, params: dict, member_dim: str = MEMBER
                ) -> PredictiveDistribution:
        x = to_ensemble(ensemble, member_dim).mean(member_dim)
        x_std = (x - params["mu"]) / params["sd"]

        def prob_below(threshold: xr.DataArray) -> xr.DataArray:
            """Probability at the fitted threshold closest to the one asked for."""
            diffs = [float(abs((threshold - q).mean())) for q in params["thresholds"]]
            beta = params["betas"][int(np.argmin(diffs))]
            eta = beta.sel(coef="intercept") + beta.sel(coef="slope") * x_std
            return xr.apply_ufunc(_sigmoid, eta, dask="parallelized", output_dtypes=[float])

        return _LogisticDistribution(prob_below, x)


class ExtendedLogistic(Calibrator):
    """
    Extended logistic regression (Wilks 2009): one fit, every threshold.

    ``P(y ≤ q) = σ(a + c·g(q) + b·x)`` with ``g`` the square root for rainfall
    (which tames the skewness of a total) and the identity for a temperature.
    The training sample stacks the years and a set of reference quantiles of the
    observed climatology, so the threshold is learned as a covariate.
    """

    name = "elr"
    needs_members = False

    def __init__(self, variable: str = "precip",
                 train_quantiles=(0.1, 0.2, 1 / 3, 0.5, 2 / 3, 0.8, 0.9)):
        self.variable = variable
        self.train_quantiles = tuple(train_quantiles)

    # ------------------------------------------------------------ transform
    def _g(self, q):
        return np.sqrt(np.clip(q, 0.0, None)) if self.variable == "precip" else q

    def _g_inv(self, g):
        return np.clip(g, 0.0, None) ** 2 if self.variable == "precip" else g

    def fit(self, ensemble: xr.DataArray, obs: xr.DataArray, year_dim: str = YEAR,
            member_dim: str = MEMBER) -> dict:
        x, mu, sd = _standardised_predictor(ensemble, year_dim, member_dim)
        x_std = (x - mu) / sd
        qs = xr.concat([obs.quantile(q, dim=year_dim, method="weibull").drop_vars("quantile")
                        for q in self.train_quantiles],
                       dim=xr.DataArray(list(self.train_quantiles), dims="q", name="q"))
        # sample = (year, threshold): one line per year and per reference quantile
        below = (obs < qs).astype(float).stack(sample=(year_dim, "q"))
        gq = self._g(qs).broadcast_like((obs * 0 + qs)).stack(sample=(year_dim, "q"))
        xx = x_std.broadcast_like((obs * 0 + qs)).stack(sample=(year_dim, "q"))
        ones = xr.ones_like(xx)
        X = xr.concat([ones, gq, xx], dim="coef").transpose(..., "sample", "coef")
        beta = xr.apply_ufunc(
            logistic_fit, X.reset_index("sample", drop=True),
            below.transpose(..., "sample").reset_index("sample", drop=True),
            input_core_dims=[["sample", "coef"], ["sample"]], output_core_dims=[["coef"]],
            dask="parallelized", output_dtypes=[float])
        beta = beta.assign_coords(coef=["intercept", "threshold", "slope"])
        return {"mu": mu, "sd": sd, "beta": beta}

    def predict(self, ensemble: xr.DataArray, params: dict, member_dim: str = MEMBER
                ) -> PredictiveDistribution:
        x = to_ensemble(ensemble, member_dim).mean(member_dim)
        x_std = (x - params["mu"]) / params["sd"]
        beta = params["beta"]
        base = beta.sel(coef="intercept") + beta.sel(coef="slope") * x_std
        slope_q = beta.sel(coef="threshold")

        def prob_below(threshold: xr.DataArray) -> xr.DataArray:
            eta = base + slope_q * self._g(threshold)
            return xr.apply_ufunc(_sigmoid, eta, dask="parallelized", output_dtypes=[float])

        def quantile(p: float) -> xr.DataArray:
            # invert the logistic in the transformed threshold space
            g = (np.log(p / (1.0 - p)) - base) / slope_q.where(abs(slope_q) > EPS, np.nan)
            return xr.apply_ufunc(self._g_inv, g, dask="parallelized", output_dtypes=[float])

        return _LogisticDistribution(prob_below, quantile(0.5), quantile)


def logistic_calibrators(variable: str) -> list[Calibrator]:
    return [TercileLogistic(variable), ExtendedLogistic(variable)]
