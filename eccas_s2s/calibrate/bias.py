"""
Bias corrections: the deterministic family of the CAPC-AC method note (E5a).

For a **rainfall total** the note recommends an additive or multiplicative
correction, or a linear scaling; for an **anomaly**, a correction followed by a
standardisation. All three are closed-form, so their leave-one-year-out version
is computed in one pass with :func:`eccas_s2s.calibrate.base.loyo_mean_and_var`
instead of one fit per year — that is what keeps a 0.25° grid affordable.

What each one does to the ensemble, written on the members so that the
probabilities that follow stay consistent with the corrected mean:

===================  ==========================================================
``mean``             adds the mean error: ``x - (μ_mod - μ_obs)``. Keeps the
                     spread of the model untouched.
``ratio``            multiplies by ``μ_obs / μ_mod``. The natural correction for
                     a rainfall total (a 40 % excess is a factor, not an offset)
                     and it cannot produce a negative total.
``scaling``          corrects mean **and** variability:
                     ``μ_obs + (x - μ_mod) · σ_obs/σ_mod``. A model whose
                     interannual variability is too weak — the usual case for
                     seasonal rainfall — has its anomalies inflated to the
                     observed amplitude.
===================  ==========================================================

A word of caution that the maps make visible: ``scaling`` improves the spread of
the distribution but can degrade the RMSE, because inflating an anomaly that
points the wrong way makes the error larger. The chain therefore keeps the three
corrections, scores them all, and lets the selection (E5, ``selection.py``)
decide per model, variable and period.
"""
from __future__ import annotations

import xarray as xr

from eccas_s2s.calibrate.base import (YEAR, MEMBER, Calibrator, EnsembleDistribution,
                                      loyo_mean_and_var, sanitize, to_ensemble)

EPS = 1e-6


class BiasCorrection(Calibrator):
    """Mean, ratio or mean-and-variance correction of the members."""

    def __init__(self, method: str = "mean", variable: str = "precip"):
        if method not in ("mean", "ratio", "scaling"):
            raise ValueError(f"correction inconnue : {method}")
        self.method = method
        self.variable = variable
        self.name = f"bias_{method}"

    # ------------------------------------------------------------------ fit
    def fit(self, ensemble: xr.DataArray, obs: xr.DataArray, year_dim: str = YEAR,
            member_dim: str = MEMBER) -> dict:
        ens = to_ensemble(ensemble, member_dim)
        mod_mean = ens.mean([year_dim, member_dim])
        obs_mean = obs.mean(year_dim)
        params = {"mod_mean": mod_mean, "obs_mean": obs_mean}
        if self.method == "scaling":
            # variability of the *ensemble mean* against the observation: the
            # spread between members is not an interannual variability.
            params["mod_sd"] = ens.mean(member_dim).std(year_dim, ddof=1)
            params["obs_sd"] = obs.std(year_dim, ddof=1)
        return params

    def predict(self, ensemble: xr.DataArray, params: dict, member_dim: str = MEMBER
                ) -> EnsembleDistribution:
        ens = to_ensemble(ensemble, member_dim)
        if self.method == "mean":
            out = ens - (params["mod_mean"] - params["obs_mean"])
        elif self.method == "ratio":
            out = ens * (params["obs_mean"] / params["mod_mean"].clip(min=EPS))
        else:
            ratio = params["obs_sd"] / params["mod_sd"].clip(min=EPS)
            out = params["obs_mean"] + (ens - params["mod_mean"]) * ratio
        return EnsembleDistribution(sanitize(out, self.variable), member_dim)

    # ------------------------------------------------- vectorised LOYO
    def fit_predict_loyo(self, ensemble: xr.DataArray, obs: xr.DataArray,
                         year_dim: str = YEAR, member_dim: str = MEMBER
                         ) -> EnsembleDistribution:
        """Closed-form leave-one-out: one pass instead of one fit per year."""
        ens = to_ensemble(ensemble, member_dim)
        ens_mean_y = ens.mean(member_dim)                       # ensemble mean per year
        mod_mean, mod_var = loyo_mean_and_var(ens_mean_y, year_dim)
        obs_mean, obs_var = loyo_mean_and_var(obs, year_dim)

        if self.method == "mean":
            out = ens - (mod_mean - obs_mean)
        elif self.method == "ratio":
            out = ens * (obs_mean / mod_mean.clip(min=EPS))
        else:
            ratio = (obs_var ** 0.5) / (mod_var ** 0.5).clip(min=EPS)
            out = obs_mean + (ens - mod_mean) * ratio
        return EnsembleDistribution(sanitize(out, self.variable), member_dim)


def bias_calibrators(variable: str) -> list[BiasCorrection]:
    """The corrections worth trying for a variable (a ratio has no sense in °C)."""
    methods = ("mean", "ratio", "scaling") if variable == "precip" else ("mean", "scaling")
    return [BiasCorrection(m, variable) for m in methods]
