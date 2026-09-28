"""
Principal component regression: a large-scale field predicts every grid point (E5b, E7).

This is the second route of the chain, and it answers a different question from
the local calibration of phase P3. There, the predictor of a cell was *that same
cell* of the interpolated model, so the correction could only rescale what the
model already said at that place; here the predictor is a **field** — the sea
surface temperature the model forecasts for the target season — and every
observation cell is regressed on the leading patterns of that field. Two cells
inside the same model box can therefore receive different anomalies, which is
precisely the sub-grid structure the local methods cannot create.

How many modes, and why it is decided rather than chosen
--------------------------------------------------------
The predictor is truncated to a few EOFs, and the number is the only real
parameter. Three rules bound it:

* **North et al. (1982)**: an eigenvalue carries a sampling error λ√(2/N); two
  modes closer than that are degenerate — their order swaps from one sample to
  the next, so a truncation *inside* such a pair fits a pattern that changes
  between leave-one-year-out folds. The admissible cuts are those that keep a
  degenerate group whole;
* **the length of the record**: the fit uses 23 years (24 minus the verified
  one). Beyond roughly N/4 predictors a least-squares regression starts fitting
  the noise, which caps the truncation near 5;
* **nested cross-validation**: among the admissible cuts, the one used for a
  given year is chosen on the *other* years only. Choosing it once on all years
  and reusing it everywhere would leak the verified year into the choice.

The predictand is **not** reduced. Its spectrum is flat over Central Africa —
14 %, 12 %, 8 %, then a continuum — so projecting rainfall onto three modes would
throw away two thirds of the field before the regression starts. Regressing each
cell separately costs one least-squares solve for the whole grid at once, which
is cheaper than the EOF it would replace.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from eccas_s2s.calibrate.base import Calibrator, NormalDistribution

YEAR = "year"


def north_errors(variance: np.ndarray, n_years: int) -> np.ndarray:
    """Sampling error of each explained-variance fraction (North et al. 1982)."""
    return np.asarray(variance) * np.sqrt(2.0 / float(n_years))


def admissible_truncations(variance: np.ndarray, n_years: int, max_modes: int = 5) -> list[int]:
    """
    Numbers of modes that do not cut inside a degenerate group.

    A cut after mode *k* is admissible when the gap between λ_k and λ_{k+1}
    exceeds the sampling error of λ_k: the two are then distinguishable, and the
    subspace kept is stable from one fold to the next.
    """
    variance = np.asarray(variance, dtype=float)
    err = north_errors(variance, n_years)
    keep = []
    for k in range(min(max_modes, len(variance) - 1)):
        if variance[k] - variance[k + 1] > err[k]:
            keep.append(k + 1)
    return keep or [1]


def _eof(field: np.ndarray, n_modes: int):
    """EOF of a (year, cell) matrix already weighted and centred; returns (pcs, patterns, var)."""
    centred = field - field.mean(0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    return u[:, :n_modes] * s[:n_modes], vt[:n_modes], var


@dataclass
class PCRegression(Calibrator):
    """
    Regression of every observation cell on the EOFs of a predictor field.

    ``predictor`` has dims ``(year, latitude, longitude)`` — the forecast field,
    usually the ensemble-mean SST of the target season. It is weighted by the
    cosine of latitude before the EOF, as a field on a sphere must be, and
    standardised so that a basin with a large seasonal cycle does not dominate
    the decomposition.
    """

    predictor: xr.DataArray = None
    name: str = "pcr"
    needs_members: bool = False
    max_modes: int = 5
    #: truncations tried inside the inner loop; filled from the spectrum
    candidates: list = field(default_factory=list)

    def _prepare(self, obs: xr.DataArray):
        """
        Predictor and predictand as plain matrices, once for the whole run.

        The predictor is weighted by the cosine of latitude — a field on a sphere
        — and each sea point standardised, so that a basin with a large seasonal
        cycle does not dominate the decomposition simply by its amplitude. The
        matrices are built **once**: the leave-one-year-out loops then index rows,
        instead of restacking twenty thousand points six hundred times.
        """
        years = [int(y) for y in obs[YEAR].values]
        da = self.predictor.sel({YEAR: years})
        weights = np.sqrt(np.cos(np.deg2rad(da["latitude"])))
        flat = (da * weights).stack(cell=("latitude", "longitude")).transpose(YEAR, "cell")
        x = flat.values
        x = x[:, np.isfinite(x).all(0)]
        sd = x.std(0, ddof=1)
        x = (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)

        target = obs.stack(cell=("latitude", "longitude")).transpose(YEAR, "cell")
        y = target.values
        return years, x, y, np.isfinite(y).all(0), target.isel({YEAR: 0}, drop=True)

    def _fit_predict(self, x, y, valid, shape, train_idx, test_idx, n_modes: int):
        """Fit on the training rows and predict the test rows, for one truncation."""
        x_train = x[train_idx]
        mean = x_train.mean(0)
        pcs, patterns, _ = _eof(x_train, n_modes)
        pcs_test = (x[test_idx] - mean) @ patterns.T

        design = np.column_stack([np.ones(len(train_idx)), pcs])
        coef, *_ = np.linalg.lstsq(design, np.nan_to_num(y[np.ix_(train_idx, np.flatnonzero(valid))]),
                                   rcond=None)
        test_design = np.column_stack([np.ones(len(test_idx)), pcs_test])
        predicted = np.full((len(test_idx), y.shape[1]), np.nan)
        predicted[:, valid] = test_design @ coef
        fitted = design @ coef
        residual = np.full(y.shape[1], np.nan)
        residual[valid] = np.sqrt(((np.nan_to_num(y[np.ix_(train_idx, np.flatnonzero(valid))])
                                    - fitted) ** 2).mean(0))
        return predicted, residual, shape

    def fit_predict_loyo(self, ensemble: xr.DataArray, obs: xr.DataArray):
        """
        Leave-one-year-out prediction, the truncation chosen inside each fold.

        ``ensemble`` is ignored — a PCR reads the predictor field, not the model's
        rainfall; it is accepted so that the method has the same signature as
        every other calibrator and can enter the same register.
        """
        years, x, y, valid, shape = self._prepare(obs)
        n = len(years)
        spectrum = _eof(x, min(self.max_modes + 3, n - 1))[2]
        self.candidates = admissible_truncations(spectrum, n, self.max_modes)

        mus, sigmas, chosen = [], [], []
        for i, year in enumerate(years):
            train = [j for j in range(n) if j != i]
            best, best_error = self.candidates[0], np.inf
            for n_modes in self.candidates:          # inner loop: the verified year is out
                errors = []
                for k in train[::3]:                 # a third of the folds: enough, and thrifty
                    inner = [j for j in train if j != k]
                    pred, _, _ = self._fit_predict(x, y, valid, shape, inner, [k], n_modes)
                    errors.append(np.nanmean((pred[0] - y[k]) ** 2))
                if np.mean(errors) < best_error:
                    best, best_error = n_modes, float(np.mean(errors))
            pred, residual, _ = self._fit_predict(x, y, valid, shape, train, [i], best)
            mus.append(shape.copy(data=pred[0]).unstack("cell").expand_dims({YEAR: [year]}))
            sigmas.append(shape.copy(data=residual).unstack("cell").expand_dims({YEAR: [year]}))
            chosen.append(best)
        mu = xr.concat(mus, dim=YEAR)
        sigma = xr.concat(sigmas, dim=YEAR)
        mu.attrs["modes_by_year"] = ", ".join(str(c) for c in chosen)
        mu.attrs["candidate_truncations"] = ", ".join(str(c) for c in self.candidates)
        return NormalDistribution(mu=mu, sigma=sigma.clip(min=1e-6))
