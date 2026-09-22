"""
Quantile mapping: EQM and QDM (workflow E5a, CAPC-AC method note).

The bias corrections of :mod:`eccas_s2s.calibrate.bias` move the mean, or the
mean and the amplitude. Quantile mapping goes further: it makes the **whole
distribution** of the model match the observed one — a model that rains too
often but too weakly is corrected on its dry tail and on its wet tail at the
same time, which a single factor cannot do.

Two variants, and the difference matters for a forecast:

**EQM** (empirical quantile mapping) replaces a forecast value by the observed
value of the same rank: ``x → F_obs⁻¹(F_mod(x))``. It imposes the observed
climatology, which is exactly what one wants for a hindcast — and also its main
risk in operations: a forecast more extreme than anything in the training sample
is pulled back inside it.

**QDM** (quantile delta mapping) corrects the same quantiles but **preserves the
relative change** the model forecasts: the model's departure from its own
climatology is kept, then transported onto the observed distribution. A forecast
outside the model's historical range stays outside the observed one.

Both are computed leave-one-year-out (D10): the year being corrected never
contributes to the two climatologies it is mapped between. Everything is
vectorised over the grid points — the cost is a sort per cell and per fold.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from eccas_s2s.calibrate.base import (YEAR, MEMBER, Calibrator, EnsembleDistribution,
                                      sanitize, to_ensemble)

EPS = 1e-6


def _plotting_positions(n: int) -> np.ndarray:
    """Weibull positions ``i/(n+1)``, the convention used everywhere in the chain."""
    return np.arange(1, n + 1) / (n + 1)


def _map_quantiles(x: np.ndarray, mod_sample: np.ndarray, obs_sample: np.ndarray) -> np.ndarray:
    """
    Map ``x`` through the model climatology onto the observed one (last axis).

    Sorted samples and linear interpolation of the plotting positions; values
    beyond the sample are handled by the tail rule of ``np.interp`` (constant),
    which the QDM variant corrects with its delta.
    """
    p_mod = _plotting_positions(mod_sample.shape[-1])
    p_obs = _plotting_positions(obs_sample.shape[-1])
    mod_sorted = np.sort(mod_sample, axis=-1)
    obs_sorted = np.sort(obs_sample, axis=-1)
    # the output is filled through its flat view, so it is allocated flat and
    # reshaped at the end: `np.empty_like(x).reshape(...)` can return a *copy*
    # when x is not contiguous, and the writes would then be lost (they were).
    flat_x = np.ascontiguousarray(x).reshape(-1, x.shape[-1])
    flat_m = np.ascontiguousarray(mod_sorted).reshape(-1, mod_sorted.shape[-1])
    flat_o = np.ascontiguousarray(obs_sorted).reshape(-1, obs_sorted.shape[-1])
    if not (flat_x.shape[0] == flat_m.shape[0] == flat_o.shape[0]):
        raise ValueError(f"cellules désalignées : {flat_x.shape[0]}, {flat_m.shape[0]}, "
                         f"{flat_o.shape[0]} — le mapping porterait sur la mauvaise maille")
    flat_out = np.empty_like(flat_x)
    for i in range(flat_x.shape[0]):                 # one cell (and fold) at a time
        p = np.interp(flat_x[i], flat_m[i], p_mod)
        flat_out[i] = np.interp(p, p_obs, flat_o[i])
    return flat_out.reshape(x.shape)


class QuantileMapping(Calibrator):
    """EQM (``delta=False``) or QDM (``delta=True``), leave-one-year-out."""

    def __init__(self, delta: bool = False, variable: str = "precip"):
        self.delta = bool(delta)
        self.variable = variable
        self.name = "qdm" if delta else "eqm"

    def fit(self, ensemble: xr.DataArray, obs: xr.DataArray, year_dim: str = YEAR,
            member_dim: str = MEMBER) -> dict:
        ens = to_ensemble(ensemble, member_dim)
        # the training sample is a bag of values: its year and member labels are
        # dropped, otherwise xarray tries to align it with the year being predicted
        mod = ens.stack(sample=(year_dim, member_dim)).reset_index("sample", drop=True)
        ref = obs.rename({year_dim: "sample_obs"}).reset_index("sample_obs", drop=True)
        return {"mod_sample": mod, "obs_sample": ref}

    def predict(self, ensemble: xr.DataArray, params: dict, member_dim: str = MEMBER
                ) -> EnsembleDistribution:
        ens = to_ensemble(ensemble, member_dim)
        out = xr.apply_ufunc(
            _map_quantiles, ens, params["mod_sample"], params["obs_sample"],
            input_core_dims=[[member_dim], ["sample"], ["sample_obs"]],
            output_core_dims=[[member_dim]], dask="parallelized", output_dtypes=[float])
        if self.delta:
            # keep the model's departure from its own climatology
            ref = xr.apply_ufunc(
                lambda m: np.median(m, axis=-1, keepdims=True), params["mod_sample"],
                input_core_dims=[["sample"]], output_core_dims=[["one"]],
                dask="parallelized", output_dtypes=[float]).isel(one=0, drop=True)
            if self.variable == "precip":
                # the delta is a ratio: it is capped, otherwise a cell whose model
                # climatology is nearly dry turns a normal forecast into a flood
                ratio = (ens / ref.clip(min=EPS)).clip(min=0.1, max=10.0)
                out = out * ratio
            else:
                out = out + (ens - ref)
        return EnsembleDistribution(sanitize(out.transpose(*ens.dims), self.variable), member_dim)

    def fit_predict_loyo(self, ensemble: xr.DataArray, obs: xr.DataArray,
                         year_dim: str = YEAR, member_dim: str = MEMBER
                         ) -> EnsembleDistribution:
        ens = to_ensemble(ensemble, member_dim)
        years = list(ens[year_dim].values)
        parts = []
        for y in years:
            keep = [v for v in years if v != y]
            params = self.fit(ens.sel({year_dim: keep}), obs.sel({year_dim: keep}),
                              year_dim, member_dim)
            parts.append(self.predict(ens.sel({year_dim: [y]}), params, member_dim).members)
        return EnsembleDistribution(xr.concat(parts, dim=year_dim), member_dim)


def qmap_calibrators(variable: str) -> list[QuantileMapping]:
    return [QuantileMapping(False, variable), QuantileMapping(True, variable)]
