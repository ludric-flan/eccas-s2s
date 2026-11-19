"""
The products themselves, for the forecast year (workflow E8).

Phases P2 and P3 measured what the products are worth; this module produces
them. It deliberately reads the **same functions** as the verification
(:mod:`eccas_s2s.validate.product_scores`): a probability published on a map and
the probability that was scored must be the same number, or the skill printed
beside a product describes something else than the product.

One thing changes between verification and production, and only one. In a
hindcast the thresholds are recomputed **without the verified year**, because
that year is being predicted; in a real forecast there is no such year — the
target is 2026 and the normal is the observed 1991–2020 one (decision D12). The
context below therefore builds the thresholds once, from the observed normal,
and every model of the cycle is read against those same thresholds. That is what
makes the models comparable, and what makes their products averageable.

The multi-model then averages the **products**, not the distributions: the mean
of eight tercile probabilities is a tercile probability, while the mean of eight
predictive laws is a mixture whose spread is wider than any of them — and the
bulletins publish probabilities, not mixtures.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from eccas_s2s.products.catalogue import CATEGORIES, EVENT, VALUE, catalogue
from eccas_s2s.products.raw_products import SPI_CLASS_BOUND, cumul_to_spi, fit_gamma
from eccas_s2s.validate.product_scores import OWN_QUANTILE

YEAR = "year"
MEMBER = "number"
TERCILES = (1 / 3, 2 / 3)
CATEGORY_NAMES = ("BN", "NN", "AN")


class ForecastContext:
    """
    The observed thresholds a forecast is read against, built once per period.

    ``normal`` is the observed series of that calendar period over the reference
    normal (1991–2020): every quantile, every millimetre threshold and the gamma
    of the SPI come from it. No leave-one-year-out here — the forecast year is
    not in the sample, which is precisely what makes it a forecast.
    """

    def __init__(self, normal: xr.DataArray, variable: str, scale: str, thresholds: dict,
                 year_dim: str = YEAR):
        self.normal = normal
        self.variable, self.scale, self.year_dim = variable, scale, year_dim
        self.products = catalogue(variable, scale, thresholds)
        quantiles = normal.quantile([TERCILES[0], TERCILES[1], 0.20, 0.50, 0.80],
                                    dim=year_dim, method="weibull")
        self.q33 = quantiles.sel(quantile=TERCILES[0], drop=True)
        self.q67 = quantiles.sel(quantile=TERCILES[1], drop=True)
        self.quantile_values = {0.20: quantiles.sel(quantile=0.20, drop=True),
                                0.50: quantiles.sel(quantile=0.50, drop=True),
                                0.80: quantiles.sel(quantile=0.80, drop=True),
                                TERCILES[0]: self.q33, TERCILES[1]: self.q67}
        self.mean = normal.mean(year_dim)
        self._spi = None

    @property
    def spi_thresholds(self):
        """Rainfall totals matching SPI −1 and +1 in the observed normal."""
        if self._spi is None:
            from scipy import stats

            stacked = self.normal.transpose(self.year_dim, ...)
            alpha, theta, p0, valid = fit_gamma(stacked.values)
            bounds = []
            for spi in (-SPI_CLASS_BOUND, SPI_CLASS_BOUND):
                h = stats.norm.cdf(spi)
                p = np.where(valid & (h > p0), (h - p0) / np.maximum(1.0 - p0, 1e-9), np.nan)
                x = np.where(np.isfinite(p) & (p > 0),
                             stats.gamma.ppf(np.clip(p, 1e-9, 1 - 1e-9), a=alpha, scale=theta), 0.0)
                shape = stacked.isel({self.year_dim: 0}, drop=True)
                bounds.append(shape.copy(data=np.where(valid, x, np.nan)))
            self._spi = tuple(bounds)
        return self._spi

    def event_threshold(self, product):
        """Observed threshold of an event, and whether the event lies above it."""
        if product.name == "categorie_BN":
            return self.q33, False
        if product.name == "categorie_AN":
            return self.q67, True
        if product.name == "quintile_bas":
            return self.quantile_values[0.20], False
        if product.name == "quintile_haut":
            return self.quantile_values[0.80], True
        if product.name == "depassement_median":
            return self.quantile_values[0.50], True
        if product.name.startswith("depassement_"):
            return xr.full_like(self.mean, float(product.parameter)), True
        raise KeyError(product.name)


def product_values(dist, ctx: ForecastContext, products=None, hindcast=None) -> dict:
    """
    Every product of the catalogue, as the map that will be published.

    A value product (total, anomaly, SPI) returns a value; a three-category
    product returns the three probabilities; an event returns the probability of
    that event. The reading convention is the one of the verification: a raw
    ensemble reads its quantile products in **its own** climatology
    (``uses_own_climatology``), because its values are biased, while a calibrated
    distribution — already on the observation scale — reads everything against
    the observed thresholds.

    For a raw forecast that own climatology comes from ``hindcast``, the model's
    own past runs, exactly as the reference chain does: a single forecast year
    has no distribution of its own to be read against. Without it a raw forecast
    cannot produce quantile products at all, and the function says so rather than
    quietly using the observed thresholds, which would publish the model's bias
    as a probability.
    """
    own = bool(getattr(dist, "uses_own_climatology", False))
    member_dim = getattr(dist, "member_dim", MEMBER)
    wanted = set(products) if products else None
    out = {}
    clim = None
    if own:
        if hindcast is None:
            raise ValueError("une prévision brute exige le hindcast du modèle "
                             "pour ses seuils (climatologie propre)")
        from eccas_s2s.products.raw_products import model_climatology
        clim = model_climatology(hindcast)

    for product in ctx.products:
        if wanted is not None and product.name not in wanted:
            continue
        if product.kind == VALUE:
            if product.name == "cumul":
                out[product.name] = dist.mean()
            elif product.name == "anomalie":
                # a raw anomaly is taken from the model's own mean, so that its
                # bias cancels; a calibrated one from the observed normal
                out[product.name] = dist.mean() - (clim["clim_mean"] if own else ctx.mean)
            elif product.name == "spi":
                out[product.name] = _spi_value(dist, ctx, member_dim, own, hindcast)
        elif product.kind == CATEGORIES:
            if product.name == "terciles":
                out[product.name] = (_own_terciles(dist, clim, member_dim) if own
                                     else dist.tercile_probs(ctx.q33, ctx.q67))
            else:                                            # classes SPI
                out[product.name] = _spi_classes(dist, ctx, member_dim, own, hindcast)
        else:                                                # événement
            threshold, above = ctx.event_threshold(product)
            if own and product.name in OWN_QUANTILE:
                threshold = _own_threshold(product.name, clim)
                p_below = (dist.members < threshold).mean(member_dim)
            else:
                p_below = dist.prob_below(threshold)
            out[product.name] = (1.0 - p_below) if above else p_below
        if product.name in out:
            out[product.name].attrs = {"product": product.name, "label": product.label,
                                       "kind": product.kind, "event": product.event,
                                       "own_climatology": int(own)}
    return out


#: quantile products, and the threshold of the model climatology they read
OWN_THRESHOLD = {"categorie_BN": "tercile_lower", "categorie_AN": "tercile_upper",
                 "quintile_bas": "quintile_lower", "quintile_haut": "quintile_upper",
                 "depassement_median": "median"}


def _own_threshold(product_name: str, clim: xr.Dataset) -> xr.DataArray:
    """Threshold of the model's own climatology for a quantile product."""
    return clim[OWN_THRESHOLD[product_name]]


def _own_terciles(dist, clim: xr.Dataset, member_dim: str) -> xr.DataArray:
    """Tercile probabilities counted against the model's own hindcast terciles."""
    from eccas_s2s.products.raw_products import tercile_probabilities

    return tercile_probabilities(dist.members, clim, member_dim=member_dim)


def _spi_value(dist, ctx: ForecastContext, member_dim: str, own: bool,
               hindcast=None) -> xr.DataArray:
    """Deterministic SPI of the forecast, in the distribution it belongs to."""
    if own and hindcast is not None:
        pooled = hindcast.stack(sample=[d for d in (YEAR, member_dim) if d in hindcast.dims])
        alpha, theta, p0, valid = fit_gamma(pooled.transpose("sample", ...).values)
        spi = xr.apply_ufunc(lambda v: cumul_to_spi(v, alpha, theta, p0, valid), dist.members)
        return spi.mean(member_dim)
    stacked = ctx.normal.transpose(ctx.year_dim, ...)
    alpha, theta, p0, valid = fit_gamma(stacked.values)
    return xr.apply_ufunc(lambda v: cumul_to_spi(v, alpha, theta, p0, valid), dist.mean())


def _spi_classes(dist, ctx: ForecastContext, member_dim: str, own: bool,
                 hindcast=None) -> xr.DataArray:
    """Probabilities of the dry / normal / wet SPI classes."""
    if own and hindcast is not None:
        pooled = hindcast.stack(sample=[d for d in (YEAR, member_dim) if d in hindcast.dims])
        alpha, theta, p0, valid = fit_gamma(pooled.transpose("sample", ...).values)
        spi = xr.apply_ufunc(lambda v: cumul_to_spi(v, alpha, theta, p0, valid), dist.members)
        p_dry = (spi < -SPI_CLASS_BOUND).mean(member_dim)
        p_wet = (spi > SPI_CLASS_BOUND).mean(member_dim)
    else:
        low, high = ctx.spi_thresholds
        p_dry = dist.prob_below(low)
        p_wet = 1.0 - dist.prob_below(high)
    p_normal = (1.0 - p_dry - p_wet).clip(0, 1)
    prob = xr.concat([p_dry, p_normal, p_wet],
                     dim=xr.DataArray(list(CATEGORY_NAMES), dims="category", name="category"))
    return (prob / prob.sum("category")).rename("prob")


def combine(products_by_model: dict, weights: dict | None = None) -> dict:
    """
    Multi-model product: the weighted mean of the models' **products**.

    The mean of eight tercile probabilities is a tercile probability, and it is
    what a bulletin publishes; the mean of eight predictive laws would be a
    mixture, wider than any of its members. Weights default to equal — the
    reference of the reference chain — and are renormalised over the models that
    actually have the product, so a model missing a scale (NCEP has no dekads)
    does not silently count as zero.
    """
    names = sorted({name for products in products_by_model.values() for name in products})
    out = {}
    for name in names:
        contributions, used = [], []
        for model, products in products_by_model.items():
            if name in products and products[name] is not None:
                # a fitted distribution leaves its own trace on the field — the
                # coefficients of an EMOS, the thresholds of a logistic model.
                # Two models fitted with two different methods therefore carry
                # different stray coordinates, which no concatenation can align:
                # only the axes of the field itself are kept
                contributions.append(products[name].reset_coords(drop=True))
                used.append(model)
        if not contributions:
            continue
        w = np.array([float((weights or {}).get(m, 1.0)) for m in used], dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(used))
        w = w / w.sum()
        stacked = xr.concat(contributions, dim=xr.DataArray(used, dims="model", name="model"))
        combined = (stacked * xr.DataArray(w, dims="model", coords={"model": used})).sum("model")
        if "category" in combined.dims:                    # a set of probabilities stays one
            combined = combined / combined.sum("category")
        combined.attrs = {**contributions[0].attrs, "models": ", ".join(used),
                          "weights": ", ".join(f"{m}={v:.3f}" for m, v in zip(used, w))}
        out[name] = combined
    return out
