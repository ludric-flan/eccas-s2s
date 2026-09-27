"""
NGR / EMOS: the ensemble's own spread becomes a calibrated uncertainty (E5c).

Bias corrections move the members; the logistic family recalibrates
probabilities from the ensemble *mean* alone. NGR (non-homogeneous Gaussian
regression, also called EMOS) uses both: it fits

    y ~ N(a + b·x̄,  c² + d²·s²)

where ``x̄`` is the ensemble mean and ``s²`` its variance. The mean is corrected
like a regression — which shrinks a model with little skill towards climatology,
exactly what the RPSS of phase P2 was asking for — and the **spread is
recalibrated**: ``d`` says how much of the ensemble's dispersion is real
information about the uncertainty of the day, and ``c`` adds the irreducible
part the ensemble does not see. A systematically under-dispersed ensemble (the
usual case) comes out with an honest width.

Parameters are fitted by minimising the **CRPS**, which for a normal law has a
closed form and an analytic gradient — so the fit is a vectorised Adam over the
whole grid rather than one optimisation per cell: about two seconds per period
for the 0.25° CEEAC mask and its 24 folds.

Rainfall is fitted in a **square-root space** (a standard choice for EMOS on
precipitation): it tames the skewness of a total, keeps the fit Gaussian, and the
back transformation returns quantities that cannot be negative. The distribution
object carries the transform, so every product reads the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from eccas_s2s.calibrate.base import (YEAR, MEMBER, Calibrator, NormalDistribution,
                                      to_ensemble)

EPS = 1e-6
INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------- transforms
class Identity:
    """No transform: temperature is fitted as it is."""

    name = "identity"

    def forward(self, x):
        return x

    def inverse(self, x):
        return x


@dataclass
class SqrtTransform:
    """Square root for rainfall; the inverse clips at zero (no negative total)."""

    name: str = "sqrt"

    def forward(self, x):
        return np.sqrt(np.clip(x, 0.0, None)) if not isinstance(x, xr.DataArray) \
            else x.clip(min=0.0) ** 0.5

    def inverse(self, x):
        return np.clip(x, 0.0, None) ** 2 if not isinstance(x, xr.DataArray) \
            else x.clip(min=0.0) ** 2


def transform_for(variable: str):
    return SqrtTransform() if variable == "precip" else Identity()


# --------------------------------------------------------------- CRPS and fit
def _phi(z):
    return np.exp(-0.5 * z ** 2) * INV_SQRT_2PI


def _Phi(z):
    from scipy.special import ndtr

    return ndtr(z)


def crps_normal(y, mu, sigma):
    """CRPS of ``N(mu, sigma)`` against ``y`` (Gneiting et al. 2005)."""
    sigma = np.maximum(sigma, EPS)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - INV_SQRT_PI)


def _fit_emos(x, s2, y, n_iter=120, lr=0.05):
    """
    Minimise the mean CRPS over the last axis, vectorised over the leading ones.

    ``x`` ensemble mean, ``s2`` ensemble variance, ``y`` observation, all of
    shape ``(..., n)``. Returns ``(a, b, c, d)``, each of shape ``(...)``.

    Start from the least-squares regression of ``y`` on ``x`` (closed form) and
    from the residual spread: the optimiser then only has to share the variance
    between its constant part ``c²`` and the part carried by the ensemble
    ``d²·s²``. Measured on ECMWF/SON: 80, 120 and 200 iterations give the same
    mean CRPS to 0.04 %, so the default is 120 — 200 was paying twice the time
    for nothing.
    """
    xm = x.mean(-1, keepdims=True)
    ym = y.mean(-1, keepdims=True)
    cov = ((x - xm) * (y - ym)).mean(-1, keepdims=True)
    var = ((x - xm) ** 2).mean(-1, keepdims=True)
    b = np.where(var > EPS, cov / np.maximum(var, EPS), 0.0)
    a = ym - b * xm
    resid = y - (a + b * x)
    c = np.sqrt(np.maximum((resid ** 2).mean(-1, keepdims=True), EPS))
    d = np.full_like(c, 0.3)
    par = np.concatenate([a, b, c, d], axis=-1)          # (..., 4)

    m1 = np.zeros_like(par)
    m2 = np.zeros_like(par)
    for it in range(1, n_iter + 1):
        a, b, c, d = (par[..., k:k + 1] for k in range(4))
        mu = a + b * x
        var_pred = np.maximum(c ** 2 + d ** 2 * s2, EPS)
        sigma = np.sqrt(var_pred)
        z = (y - mu) / sigma
        Phi, phi = _Phi(z), _phi(z)
        d_mu = -(2.0 * Phi - 1.0)                        # ∂CRPS/∂mu
        d_sig = 2.0 * phi - INV_SQRT_PI                  # ∂CRPS/∂sigma
        g = np.concatenate([
            d_mu.mean(-1, keepdims=True),
            (d_mu * x).mean(-1, keepdims=True),
            (d_sig * c / sigma).mean(-1, keepdims=True),
            (d_sig * d * s2 / sigma).mean(-1, keepdims=True)], axis=-1)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g ** 2
        step = lr * (m1 / (1 - 0.9 ** it)) / (np.sqrt(m2 / (1 - 0.999 ** it)) + 1e-8)
        par = par - step
        par[..., 2:] = np.abs(par[..., 2:])              # c and d enter squared
    return par[..., 0], par[..., 1], par[..., 2], par[..., 3]


class NGR(Calibrator):
    """
    Non-homogeneous Gaussian regression (EMOS), fitted by CRPS minimisation.

    ``spread_from_ensemble=False`` gives the homoscedastic special case
    ``d = 0``: the width no longer depends on the day. It is the honest fallback
    for a system whose ensemble carries no usable spread information — two
    members (UKMO) or none at all (NMME).
    """

    name = "ngr"

    def __init__(self, variable: str = "precip", spread_from_ensemble: bool = True,
                 n_iter: int = 120):
        self.variable = variable
        self.transform = transform_for(variable)
        self.spread_from_ensemble = bool(spread_from_ensemble)
        self.n_iter = int(n_iter)
        self.name = "ngr" if spread_from_ensemble else "ngr_const"
        self.needs_members = spread_from_ensemble

    # ------------------------------------------------------------------ fit
    def _predictors(self, ensemble: xr.DataArray, member_dim: str):
        ens = self.transform.forward(to_ensemble(ensemble, member_dim))
        x = ens.mean(member_dim)
        s2 = (ens.var(member_dim, ddof=1) if ens.sizes[member_dim] > 1
              else xr.zeros_like(x))
        if not self.spread_from_ensemble:
            s2 = xr.zeros_like(x)
        return x, s2

    def fit(self, ensemble: xr.DataArray, obs: xr.DataArray, year_dim: str = YEAR,
            member_dim: str = MEMBER) -> dict:
        x, s2 = self._predictors(ensemble, member_dim)
        y = self.transform.forward(obs)
        a, b, c, d = xr.apply_ufunc(
            _fit_emos, x, s2, y,
            input_core_dims=[[year_dim], [year_dim], [year_dim]],
            output_core_dims=[[], [], [], []], kwargs={"n_iter": self.n_iter},
            dask="parallelized", output_dtypes=[float] * 4)
        return {"a": a, "b": b, "c": c, "d": d}

    def predict(self, ensemble: xr.DataArray, params: dict, member_dim: str = MEMBER
                ) -> NormalDistribution:
        x, s2 = self._predictors(ensemble, member_dim)
        mu = params["a"] + params["b"] * x
        sigma = (params["c"] ** 2 + params["d"] ** 2 * s2).clip(min=EPS) ** 0.5
        return NormalDistribution(mu, sigma, self.transform)


def ngr_calibrators(variable: str, has_members: bool = True) -> list[NGR]:
    """EMOS with the ensemble spread when there is one, the constant form otherwise."""
    if not has_members:
        return [NGR(variable, spread_from_ensemble=False)]
    return [NGR(variable, True), NGR(variable, False)]
