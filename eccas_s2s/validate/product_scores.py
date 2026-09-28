"""
Scores product by product (workflow E4 and E5, register of recommended methods).

The chain does not deliver "a forecast": it delivers a **catalogue of products**
— a rainfall total, a tercile map, a "more than 200 mm" map, an SPI class map —
and each one is a different operational question. So each one is judged with the
metric that fits it (:mod:`eccas_s2s.products.catalogue`), and that metric is the
one that decides, in phase P2 whether the raw product can be trusted and in phase
P3 which calibration to recommend.

Everything is read from a single predictive distribution, so the same code scores
the raw ensemble and every calibrated method. Two families of products are read
differently, and the difference is deliberate:

* a product defined by a **position in the distribution** (terciles, P20, median,
  P80, SPI classes) is read in the **model's own** leave-one-year-out climatology
  when the distribution says so (``uses_own_climatology``, the raw ensemble).
  Counting raw members against the observed thresholds would measure the model's
  bias, not its information — a model 28 % too dry would announce "below normal"
  every year;
* a product defined in **absolute units** (the total itself, its anomaly, "more
  than 200 mm") is read as it is, against the observed threshold. Its raw scores
  show the bias in full, which is exactly what phase P2 has to establish.

A calibrated distribution already sits on the observation scale, so it uses the
observed thresholds for both families.

The observed side always comes from the observation, leave-one-year-out
(decision D10), so that no product is judged against a threshold its own year
helped define.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.products.catalogue import CATEGORIES, EVENT, VALUE, catalogue
from eccas_s2s.products.raw_products import SPI_CLASS_BOUND, cumul_to_spi, fit_gamma
from eccas_s2s.validate.cv import loyo_mean, loyo_quantile
from eccas_s2s.validate.pairs import observed_categories
from eccas_s2s.validate.scores import brier_scores, deterministic_scores, groc, roc_area, rps_scores

YEAR = "year"
MEMBER = "number"
TERCILES = (1 / 3, 2 / 3)
CATEGORY_NAMES = ("BN", "NN", "AN")

#: every metric a product of each kind can receive (``--metrics all``)
METRICS_OF_KIND = {
    VALUE: ("msess", "acc", "pearson", "spearman", "rmse", "mae", "bias"),
    CATEGORIES: ("rpss", "rpss_fair", "groc", "roc_area"),
    EVENT: ("bss", "bss_fair", "roc_skill"),
}
#: metrics whose *low* values are the good ones (no "fraction positive")
LOWER_IS_BETTER = ("rmse", "mae", "bias")
#: value a metric takes when the forecast has no skill. It is 0 for every skill
#: score, but **0.5** for an area under the ROC curve: counting the cells where
#: a GROC is "above zero" would count the whole map and say nothing.
NO_SKILL_VALUE = {"groc": 0.5, "roc_area": 0.5}
#: sampling-debiased companion of a member-counted score
FAIR_OF = {"rpss": "rpss_fair", "bss": "bss_fair"}
#: discrimination companion, computed with the deciding metric for every product.
#: The register needs the pair: the deciding metric says whether the product is
#: usable **as delivered**, the discrimination says whether there is signal that a
#: calibration could recover. A product with a negative RPSS and a GROC above 0.5
#: is overconfident, not uninformative — it belongs on the phase-P3 work list.
DISCRIMINATION_OF = {VALUE: "acc", CATEGORIES: "groc", EVENT: "roc_skill"}
#: quantile position of the products read in the model's own climatology
OWN_QUANTILE = {"categorie_BN": TERCILES[0], "categorie_AN": TERCILES[1],
                "quintile_bas": 0.20, "quintile_haut": 0.80, "depassement_median": 0.50}


def wanted_metrics(product, mode: str = "deciding", has_members: bool = True) -> tuple[str, ...]:
    """
    Metrics to compute for one product.

    ``deciding`` (the default of an operational run) gives the metric that
    decides, plus two companions the register needs: the sampling-debiased
    version when the probabilities are counted on members — the pair is what
    makes models of 20 and 51 members comparable — and the discrimination
    (:data:`DISCRIMINATION_OF`), which separates "no signal" from "signal
    present but overconfident". ``reported`` adds the metrics the catalogue
    lists beside the deciding one, and ``all`` every metric compatible with the
    product's nature.
    """
    if mode == "all":
        metrics = METRICS_OF_KIND[product.kind]
    elif mode == "reported":
        metrics = (product.metric, *product.reported)
    elif mode == "deciding":
        metrics = (product.metric,)
    else:                                   # an explicit list given by the user
        metrics = tuple(mode)
        return tuple(m for m in metrics if m in METRICS_OF_KIND[product.kind])
    fair = FAIR_OF.get(product.metric)
    if fair and has_members and fair not in metrics:
        metrics = (*metrics, fair)
    discrimination = DISCRIMINATION_OF[product.kind]
    if discrimination not in metrics and (has_members or product.kind == VALUE):
        metrics = (*metrics, discrimination)
    return tuple(dict.fromkeys(metrics))


# --------------------------------------------------------------------- SPI
def loyo_spi(values: xr.DataArray, year_dim: str = YEAR, pool_dims=()) -> xr.DataArray:
    """
    SPI of each year, the gamma being fitted **without that year**.

    ``pool_dims`` are pooled with the years for the fit (``["number"]`` for a
    model, nothing for the observation), which is the convention of the
    reference chain: the model's SPI says where the forecast sits in the model's
    own distribution, the observed SPI where the observation sits in its own.
    """
    pool_dims = [d for d in pool_dims if d in values.dims]
    years = list(values[year_dim].values)
    space = [d for d in values.dims if d not in (year_dim, *pool_dims)]
    out = []
    for y in years:
        sample = values.sel({year_dim: [v for v in years if v != y]})
        stacked = sample.stack(sample=[year_dim, *pool_dims]) if pool_dims else sample
        # the pooled axis first, the grid keeping its own shape: fit_gamma then
        # returns parameters shaped like the grid, which broadcast over members
        stacked = stacked.transpose("sample" if pool_dims else year_dim, *space)
        alpha, theta, p0, valid = fit_gamma(stacked.values)
        this = values.sel({year_dim: y}).transpose(..., *space)
        spi = xr.apply_ufunc(lambda v: cumul_to_spi(v, alpha, theta, p0, valid), this,
                             dask="parallelized", output_dtypes=[float])
        out.append(spi.expand_dims({year_dim: [y]}))
    spi = xr.concat(out, dim=year_dim)
    spi.attrs = {"long_name": "standardised precipitation index (LOYO gamma)",
                 "pooled_with_years": ", ".join(pool_dims) or "none",
                 "space_dims": ", ".join(space)}
    return spi


def _loyo_gamma_thresholds(obs: xr.DataArray, year_dim: str = YEAR,
                           bound: float = SPI_CLASS_BOUND) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Rainfall totals matching SPI = −bound and +bound, observed side, LOYO.

    Used for a distribution without members (a calibrated law): the SPI classes
    become two ordinary thresholds, "below the total whose SPI is −1" and "above
    the one whose SPI is +1".
    """
    from scipy import stats

    years = list(obs[year_dim].values)
    lows, highs = [], []
    for y in years:
        sample = obs.sel({year_dim: [v for v in years if v != y]})
        stacked = sample.transpose(year_dim, ...).values
        alpha, theta, p0, valid = fit_gamma(stacked.reshape(stacked.shape[0], -1))
        out = []
        for spi in (-bound, bound):
            h = stats.norm.cdf(spi)
            p = np.where(valid & (h > p0), (h - p0) / np.maximum(1.0 - p0, 1e-9), np.nan)
            x = np.where(np.isfinite(p) & (p > 0),
                         stats.gamma.ppf(np.clip(p, 1e-9, 1 - 1e-9), a=alpha, scale=theta), 0.0)
            out.append(np.where(valid, x, np.nan).reshape(stacked.shape[1:]))
        shape = {d: obs[d] for d in obs.dims if d != year_dim}
        lows.append(xr.DataArray(out[0], coords=shape, dims=list(shape)).expand_dims({year_dim: [y]}))
        highs.append(xr.DataArray(out[1], coords=shape, dims=list(shape)).expand_dims({year_dim: [y]}))
    return xr.concat(lows, dim=year_dim), xr.concat(highs, dim=year_dim)


def _event_threshold(product, obs: xr.DataArray, q33: xr.DataArray, q67: xr.DataArray,
                     year_dim: str = YEAR) -> tuple[xr.DataArray, bool]:
    """Observed threshold of an event, and whether the event is *above* it."""
    if product.name == "categorie_BN":
        return q33, False
    if product.name == "categorie_AN":
        return q67, True
    if product.name in ("quintile_bas", "quintile_haut", "depassement_median"):
        q = loyo_quantile(obs, (float(product.parameter),), year_dim=year_dim)
        # loyo_quantile drops the `quantile` axis when a single one is asked for
        thr = q.sel(quantile=float(product.parameter), drop=True) if "quantile" in q.dims else q
        return thr, product.name != "quintile_bas"
    if product.name.startswith("depassement_"):
        return xr.full_like(obs, float(product.parameter)), True
    raise KeyError(product.name)


class ObservationContext:
    """
    Everything the scoring needs from the observation, computed once.

    The observed categories, the thresholds of every event and the observed SPI
    depend on the observation alone — not on the forecast being scored. Ten
    calibration methods share one context, which is what makes scoring the whole
    catalogue affordable (the SPI alone is 24 gamma fits per period).

    The **leave-one-year-out climatology** lives here too, and it is not a
    detail: a forecast fitted without year *i* carries the training mean, which
    is perfectly anticorrelated with the value of that year. A method that relies
    on that mean — that is, a method with little signal — is therefore punished
    twice, while the raw forecast, which fits nothing, is untouched. Referring
    both sides of a value product to the same LOYO climatology removes the
    artefact; without it, "no method beats the raw anomaly" is a conclusion about
    the metric, not about the methods.
    """

    def __init__(self, obs: xr.DataArray, variable: str, scale: str, thresholds: dict,
                 mask: xr.DataArray | None = None, year_dim: str = YEAR):
        self.obs = obs
        self.variable, self.scale, self.year_dim = variable, scale, year_dim
        self.valid = mask if mask is not None else obs.notnull().any(year_dim)
        self.obs_cat, obs_q = observed_categories(obs, year_dim)
        self.q33 = obs_q.sel(quantile=TERCILES[0], drop=True)
        self.q67 = obs_q.sel(quantile=TERCILES[1], drop=True)
        self.products = catalogue(variable, scale, thresholds)
        # climatology of each year fitted without that year: the reference the
        # anomalies of a value product are taken from, on both sides
        self.climatology = loyo_mean(obs, dims=[], year_dim=year_dim)
        self.events = {p.name: _event_threshold(p, obs, self.q33, self.q67, year_dim)
                       for p in self.products if p.kind == EVENT}
        self._spi = None
        self._spi_thresholds = None

    @property
    def spi(self) -> xr.DataArray:
        """Observed SPI, LOYO gamma (computed on first use)."""
        if self._spi is None:
            self._spi = loyo_spi(self.obs, self.year_dim)
        return self._spi

    @property
    def spi_thresholds(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Rainfall totals of SPI −1 and +1, for a distribution without members."""
        if self._spi_thresholds is None:
            self._spi_thresholds = _loyo_gamma_thresholds(self.obs, self.year_dim)
        return self._spi_thresholds

    @property
    def spi_classes(self) -> xr.DataArray:
        """Observed SPI class index (0 dry, 1 normal, 2 wet)."""
        spi = self.spi
        return xr.where(spi < -SPI_CLASS_BOUND, 0,
                        xr.where(spi > SPI_CLASS_BOUND, 2, 1)).where(spi.notnull())


# ------------------------------------------------------- forecast quantities
def product_probabilities(dist, ctx: ObservationContext, product, cache: dict | None = None):
    """
    Forecast probability and observed occurrence of one probabilistic product.

    Returned as ``(prob, observed)``: for a three-category product, ``prob`` has
    a ``category`` axis and ``observed`` is the category index; for an event,
    ``prob`` is the probability of the event and ``observed`` its 0/1 occurrence.
    The maps and the reliability/ROC diagrams both read this function, so a
    diagram can never be drawn from probabilities other than the ones scored.
    """
    own = bool(getattr(dist, "uses_own_climatology", False))
    obs, year_dim = ctx.obs, ctx.year_dim
    cache = {} if cache is None else cache

    if product.name == "terciles":
        return dist.tercile_probs(ctx.q33, ctx.q67), ctx.obs_cat

    if product.name == "spi_classes":
        if own and hasattr(dist, "members"):
            member_dim = getattr(dist, "member_dim", MEMBER)
            # the members' SPI costs 24 gamma fits: the value product and the
            # classes share the one computation
            spi_m = cache.get("spi_members")
            if spi_m is None:
                spi_m = cache["spi_members"] = loyo_spi(dist.members, year_dim,
                                                        pool_dims=[member_dim])
            p_dry = (spi_m < -SPI_CLASS_BOUND).mean(member_dim)
            p_wet = (spi_m > SPI_CLASS_BOUND).mean(member_dim)
        else:
            low, high = ctx.spi_thresholds
            p_dry = dist.prob_below(low)
            p_wet = 1.0 - dist.prob_below(high)
        p_nn = (1.0 - p_dry - p_wet).clip(0, 1)
        prob = xr.concat([p_dry, p_nn, p_wet],
                         dim=xr.DataArray(list(CATEGORY_NAMES), dims="category", name="category"))
        return (prob / prob.sum("category")).rename("prob"), ctx.spi_classes

    thr, above = ctx.events[product.name]
    if own and product.name in OWN_QUANTILE:
        # a position in the distribution: read in the model's own climatology
        p_below = dist.prob_below_own_quantile(OWN_QUANTILE[product.name], year_dim)
    else:
        p_below = dist.prob_below(thr)
    prob = (1.0 - p_below) if above else p_below
    occurred = ((obs > thr) if above else (obs < thr)).astype(float).where(obs.notnull())
    return prob.rename("prob"), occurred


def _value_pair(dist, ctx: ObservationContext, product, cache: dict):
    """Forecast and observation of a value product, on the product's own scale."""
    if product.name != "spi":
        return dist.mean(), ctx.obs
    if "spi" not in cache:
        own = bool(getattr(dist, "uses_own_climatology", False))
        if own and hasattr(dist, "members"):
            member_dim = getattr(dist, "member_dim", MEMBER)
            spi_m = cache.get("spi_members")
            if spi_m is None:
                spi_m = cache["spi_members"] = loyo_spi(dist.members, ctx.year_dim,
                                                        pool_dims=[member_dim])
            # the reference chain averages the members' SPI, not the SPI of the mean
            cache["spi"] = spi_m.mean(member_dim)
        else:
            cache["spi"] = loyo_spi(dist.mean(), ctx.year_dim)
    return cache["spi"], ctx.spi


def product_maps(dist, ctx: ObservationContext, mode: str = "deciding",
                 n_members: int | None = None, kinds=None, products=None) -> dict[str, xr.Dataset]:
    """
    Score maps of every product of the catalogue, for one forecast and one period.

    Returns ``{product name: Dataset}``, each dataset holding the requested
    metrics on the grid (plus ``category`` for the per-category ROC area). The
    metric that decides carries the attribute ``deciding``. ``products`` restricts
    the catalogue by name — phase P3 uses it to score a method only on the
    products it can answer (:mod:`eccas_s2s.calibrate.products`).
    """
    year_dim = ctx.year_dim
    has_members = hasattr(dist, "members") and dist.members.sizes.get(
        getattr(dist, "member_dim", MEMBER), 1) > 1
    out: dict[str, xr.Dataset] = {}
    det = None
    cache: dict = {}

    for product in ctx.products:
        if kinds is not None and product.kind not in kinds:
            continue
        if products is not None and product.name not in products:
            continue
        metrics = wanted_metrics(product, mode, has_members)
        if not metrics:
            continue
        fields: dict[str, xr.DataArray] = {}

        if product.kind == VALUE:
            forecast, reference = _value_pair(dist, ctx, product, cache)
            if product.name in ("cumul", "anomalie"):
                det = det if det is not None else deterministic_scores(
                    forecast, reference, year_dim, climatology=ctx.climatology)
                scores = det
            else:                                     # SPI: scored on the SPI scale
                scores = deterministic_scores(forecast, reference, year_dim,
                                              climatology=loyo_mean(reference, dims=[],
                                                                    year_dim=year_dim))
            fields = {m: scores[m] for m in metrics if m in scores}
            n_years = scores["n_years"]

        elif product.kind == CATEGORIES:
            prob, observed = product_probabilities(dist, ctx, product, cache)
            rps = rps_scores(prob, observed, year_dim,
                             n_members=n_members if has_members else None)
            for m in metrics:
                if m in rps:
                    fields[m] = rps[m]
                elif m == "groc":
                    fields[m] = groc(prob, observed, year_dim)
                elif m == "roc_area":
                    per_cat = [roc_area((prob.sel(category=c)),
                                        (observed == i).astype(float).where(observed.notnull()),
                                        year_dim) for i, c in enumerate(prob["category"].values)]
                    fields[m] = xr.concat(per_cat, dim=prob["category"])
            n_years = rps["n_years"]

        else:                                          # EVENT
            prob, occurred = product_probabilities(dist, ctx, product, cache)
            brier = brier_scores(prob, occurred, occurred.mean(year_dim), year_dim,
                                 n_members=n_members if has_members else None)
            for m in metrics:
                if m in brier:
                    fields[m] = brier[m]
                elif m == "roc_skill":
                    # area - 0.5: positive means "better than chance", like every
                    # other metric of the register
                    fields[m] = roc_area(prob, occurred, year_dim) - 0.5
            fields.setdefault("base_rate", brier["base_rate"])
            n_years = brier["n_years"]

        if not fields:
            continue
        ds = xr.Dataset(fields)
        ds["n_years"] = n_years
        ds.attrs = {"product": product.name, "label": product.label, "kind": product.kind,
                    "deciding": product.metric, "event": product.event,
                    "own_climatology": int(bool(getattr(dist, "uses_own_climatology", False))
                                           and (product.name in OWN_QUANTILE
                                                or product.name in ("terciles", "spi_classes",
                                                                    "spi")))}
        out[product.name] = ds
    return out


def score_products(dist, obs, variable: str = "", scale: str = "", thresholds: dict | None = None,
                   mask: xr.DataArray | None = None, year_dim: str = YEAR,
                   mode: str = "reported", n_members: int | None = None) -> pd.DataFrame:
    """
    Tidy table of the products: one row per product and metric.

    Each row carries the median over the mask and the area fraction where the
    metric is positive — the two numbers the register of recommended methods and
    the eligibility criterion (Draft §3.3) are built from.
    """
    from eccas_s2s.core.geo import fraction_above

    ctx = obs if isinstance(obs, ObservationContext) else ObservationContext(
        obs, variable, scale, thresholds or {}, mask, year_dim)
    maps = product_maps(dist, ctx, mode, n_members)
    rows = []
    for product in ctx.products:
        ds = maps.get(product.name)
        if ds is None:
            continue
        for metric, field in ds.data_vars.items():
            if metric in ("n_years", "base_rate"):
                continue
            if "category" in field.dims:
                for cat in field["category"].values:
                    rows.append(_row(product, f"{metric}_{cat}", field.sel(category=cat),
                                     ctx.valid, fraction_above))
            else:
                rows.append(_row(product, metric, field, ctx.valid, fraction_above))
    return pd.DataFrame(rows)


def _row(product, metric: str, field: xr.DataArray, valid, fraction_above) -> dict:
    base = metric.split("_")[0]
    good_when_positive = base not in LOWER_IS_BETTER
    no_skill = NO_SKILL_VALUE.get(metric, NO_SKILL_VALUE.get(base, 0.0))
    return {"product": product.name, "label": product.label, "kind": product.kind,
            "metric": metric, "deciding": metric == product.metric,
            "median": float(field.where(valid).median()),
            "no_skill_value": no_skill,
            "fraction_positive": fraction_above(field, valid, no_skill)
            if good_when_positive else np.nan}
