"""
Raw products: the conventions of the reference chain, written once (E8).

Every raw product of the CAPC-AC — tercile and quintile probabilities, median
exceedance, anomalies, fixed-threshold exceedance, anomaly-threshold
probabilities, SPI — is computed here **exactly as** ``run_forecast_v2`` computes
it (``s2s_processing_v2.py``, ``s2s_spi.py``). Re-deriving these rules from
memory is how the chain lost time twice, so they live in one module, each with
the line of the reference it reproduces.

The rules, and why each one is what it is:

``model_climatology``
    Thresholds come from the **hindcast of the model itself**, pooling members
    and years (``sample = (number, year)``). A raw model is biased, so its
    categories can only be defined against its own climate; comparing raw
    members with observed thresholds measures the bias, not the forecast.

``tercile_probabilities``
    ``BN`` is ``< lower``, ``AN`` is ``> upper``, ``NN`` is the closed interval
    in between — the three add up to one because the comparisons are strict on
    the outside and inclusive in the middle.

``exceedance_probabilities``
    Fixed millimetre thresholds use ``>=``: "at least 200 mm" includes 200 mm.
    (The median and quintile products use ``>``, as in the reference.)

``anomaly``
    Forecast minus the hindcast mean over ``(number, year)``; the published map
    is the ensemble mean of the member anomalies.

``spi``
    Gamma fitted by the method of moments on the **non-zero** pooled hindcast
    samples, a zero-probability ``p0``, then ``SPI = Φ⁻¹(p0 + (1-p0)·F)`` per
    member; the deterministic SPI is the mean of the member SPIs, and the
    classes are counted on the members.

One difference, deliberate and documented: in **verification** the thresholds
are recomputed leave-one-year-out (decision D10), while in **operations** they
come from the whole hindcast. Pass the thresholds you want; the functions never
choose for you.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

YEAR = "year"
MEMBER = "number"
CATEGORIES = ("BN", "NN", "AN")
#: mm below which a total counts as zero for the gamma fit (s2s_config_v2).
SPI_ZERO_THR = 0.1
SPI_MIN_NONZERO = 10
SPI_CLASS_BOUND = 1.0
SPI_CLAMP = 3.0


def model_climatology(hindcast: xr.DataArray, year_dim: str = YEAR, member_dim: str = MEMBER,
                      dry_threshold_mm: float = 1.0) -> xr.Dataset:
    """
    Climatological thresholds of a model, pooled over members and years.

    Reproduces ``compute_climatological_thresholds``: terciles 1/3 and 2/3,
    quintiles 0.20 and 0.80, median, mean, and the dry mask ``mean < 1 mm``.
    """
    dims = [d for d in (year_dim, member_dim) if d in hindcast.dims]
    sample = hindcast.stack(sample=dims) if len(dims) > 1 else hindcast.rename({dims[0]: "sample"})
    q = sample.quantile([1 / 3, 2 / 3, 0.20, 0.80, 0.50], dim="sample")
    clim_mean = hindcast.mean(dims)
    out = xr.Dataset({
        "tercile_lower": q.sel(quantile=1 / 3, drop=True),
        "tercile_upper": q.sel(quantile=2 / 3, drop=True),
        "quintile_lower": q.sel(quantile=0.20, drop=True),
        "quintile_upper": q.sel(quantile=0.80, drop=True),
        "median": q.sel(quantile=0.50, drop=True),
        "clim_mean": clim_mean,
        "dry_mask": clim_mean < dry_threshold_mm,
    })
    out.attrs["reference"] = "s2s_processing_v2.compute_climatological_thresholds"
    return out


def tercile_probabilities(forecast: xr.DataArray, thresholds: xr.Dataset,
                          member_dim: str = MEMBER) -> xr.DataArray:
    """Fraction of members in each tercile (``compute_forecast_probabilities``)."""
    n = forecast.sizes[member_dim]
    low, up = thresholds["tercile_lower"], thresholds["tercile_upper"]
    p_bn = (forecast < low).sum(member_dim) / n
    p_an = (forecast > up).sum(member_dim) / n
    p_nn = ((forecast >= low) & (forecast <= up)).sum(member_dim) / n
    prob = xr.concat([p_bn, p_nn, p_an],
                     dim=xr.DataArray(list(CATEGORIES), dims="category", name="category"))
    prob.name = "prob"
    prob.attrs = {"long_name": "tercile probabilities (member counting)",
                  "thresholds": "model climatology, pooled members and years"}
    return prob


def quintile_and_median_probabilities(forecast: xr.DataArray, thresholds: xr.Dataset,
                                      member_dim: str = MEMBER) -> xr.Dataset:
    """P(< P20), P(> P80) and P(> median), as in the reference."""
    n = forecast.sizes[member_dim]
    return xr.Dataset({
        "prob_low20": (forecast < thresholds["quintile_lower"]).sum(member_dim) / n,
        "prob_high20": (forecast > thresholds["quintile_upper"]).sum(member_dim) / n,
        "prob_exceed_median": (forecast > thresholds["median"]).sum(member_dim) / n,
    })


def anomaly(forecast: xr.DataArray, clim_mean: xr.DataArray) -> xr.DataArray:
    """Member anomalies against the model climatology (``compute_climatology_and_anomaly``)."""
    out = forecast - clim_mean
    out.attrs = {**forecast.attrs, "long_name": "anomaly vs model climatology"}
    return out


def exceedance_probabilities(forecast: xr.DataArray, thresholds_mm, member_dim: str = MEMBER
                             ) -> xr.DataArray:
    """
    P(total ≥ threshold) for a list of fixed millimetre thresholds.

    ``>=`` and not ``>``: "at least 200 mm" includes exactly 200 mm
    (``compute_exceedance_probabilities``).
    """
    n = forecast.sizes[member_dim]
    thr = xr.DataArray(list(thresholds_mm), dims="threshold",
                       coords={"threshold": list(thresholds_mm)})
    prob = (forecast >= thr).sum(member_dim) / n
    prob.name = "prob_exceed"
    prob.attrs = {"long_name": "probability of exceeding a fixed threshold",
                  "units": "fraction (0-1)"}
    return prob


def anomaly_threshold_probabilities(member_anomaly: xr.DataArray, threshold_mm: float,
                                    member_dim: str = MEMBER) -> xr.Dataset:
    """P(anomaly > +t), P(|anomaly| ≤ t), P(anomaly < −t) (``compute_anomaly_threshold_probabilities``)."""
    n = member_anomaly.sizes[member_dim]
    thr = float(threshold_mm)
    ds = xr.Dataset({
        "prob_athr_above": (member_anomaly > thr).sum(member_dim) / n,
        "prob_athr_normal": ((member_anomaly >= -thr) & (member_anomaly <= thr)).sum(member_dim) / n,
        "prob_athr_below": (member_anomaly < -thr).sum(member_dim) / n,
    })
    ds.attrs["threshold_mm"] = thr
    return ds


# ------------------------------------------------------------------------ SPI
def fit_gamma(samples: np.ndarray, zero_thr: float = SPI_ZERO_THR,
              min_nonzero: int = SPI_MIN_NONZERO):
    """
    Gamma by the method of moments on the non-zero samples (``fit_gamma_per_pixel``).

    ``samples`` has the pooled (member, year) axis **first**. Returns
    ``(alpha, theta, p0, valid)``; ``alpha = μ²/σ²`` and ``theta = σ²/μ`` on the
    non-zero part, ``p0`` the proportion of zeros, and ``valid`` False where
    there are fewer than ``min_nonzero`` wet samples — a desert point cannot
    carry a gamma fit.
    """
    valid_count = np.sum(~np.isnan(samples), axis=0)
    zero_count = np.sum((samples <= zero_thr) & ~np.isnan(samples), axis=0)
    nonzero_count = valid_count - zero_count
    with np.errstate(divide="ignore", invalid="ignore"):
        p0 = np.where(valid_count > 0, zero_count / valid_count, 0.0)
    wet = np.where(samples > zero_thr, samples, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_nz = np.nanmean(wet, axis=0)
        var_nz = np.nanvar(wet, axis=0, ddof=1)
        alpha = np.where((var_nz > 0) & (mean_nz > 0), mean_nz ** 2 / var_nz, np.nan)
        theta = np.where((var_nz > 0) & (mean_nz > 0), var_nz / mean_nz, np.nan)
    valid = ((nonzero_count >= min_nonzero) & np.isfinite(alpha) & (alpha > 0)
             & np.isfinite(theta) & (theta > 0))
    return alpha, theta, p0, valid


def cumul_to_spi(values: np.ndarray, alpha, theta, p0, valid,
                 zero_thr: float = SPI_ZERO_THR, clamp: float = SPI_CLAMP) -> np.ndarray:
    """Totals to SPI through the fitted gamma and the zero probability (``cumul_to_spi``)."""
    from scipy import stats

    fc = np.asarray(values, dtype=float)
    spi = np.full(fc.shape, np.nan)
    H = np.full(fc.shape, np.nan)
    valid_b, alpha_b = np.broadcast_to(valid, fc.shape), np.broadcast_to(alpha, fc.shape)
    theta_b, p0_b = np.broadcast_to(theta, fc.shape), np.broadcast_to(p0, fc.shape)
    dry = valid_b & (fc <= zero_thr) & ~np.isnan(fc)
    wet = valid_b & (fc > zero_thr) & ~np.isnan(fc)
    H[dry] = p0_b[dry]
    if np.any(wet):
        F = stats.gamma.cdf(fc[wet], a=alpha_b[wet], loc=0, scale=theta_b[wet])
        H[wet] = p0_b[wet] + (1.0 - p0_b[wet]) * F
    finite = ~np.isnan(H)
    spi[finite] = stats.norm.ppf(np.clip(H, 1e-6, 1 - 1e-6)[finite])
    return np.clip(spi, -clamp, clamp) if clamp else spi


def spi_products(hindcast: xr.DataArray, forecast: xr.DataArray, year_dim: str = YEAR,
                 member_dim: str = MEMBER, class_bound: float = SPI_CLASS_BOUND) -> xr.Dataset:
    """
    Member SPI, deterministic SPI and the three SPI classes (``compute_spi_products``).

    The gamma is fitted on the pooled hindcast of the **model**, so the SPI says
    where the forecast sits in that model's own distribution — the same logic as
    the tercile probabilities.
    """
    dims = [d for d in (year_dim, member_dim) if d in hindcast.dims]
    pooled = hindcast.stack(sample=dims).transpose("sample", ...)
    alpha, theta, p0, valid = fit_gamma(pooled.values)
    spi_members = xr.apply_ufunc(
        lambda v: cumul_to_spi(v, alpha, theta, p0, valid), forecast,
        dask="parallelized", output_dtypes=[float])
    det = spi_members.mean(member_dim)
    n = forecast.sizes[member_dim]
    ds = xr.Dataset({
        "spi_members": spi_members,
        "spi": det,
        "prob_spi_dry": (spi_members < -class_bound).sum(member_dim) / n,
        "prob_spi_normal": ((spi_members >= -class_bound)
                            & (spi_members <= class_bound)).sum(member_dim) / n,
        "prob_spi_wet": (spi_members > class_bound).sum(member_dim) / n,
    })
    ds.attrs.update({"reference": "s2s_spi.compute_spi_products",
                     "class_bound": float(class_bound),
                     "gamma_fit": "method of moments on non-zero pooled hindcast samples"})
    return ds
