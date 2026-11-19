"""
Multi-model products, raw and calibrated (workflow E6 + E8, phase P6).

What is combined here are the **products**, not the distributions: the tercile
probabilities of eight models are averaged into one tercile probability, their
totals into one total. That is what the reference chain does, what the bulletins
publish, and what keeps the result readable — the mean of eight predictive laws
would be a mixture wider than any of its members, and no forecaster would know
what to do with it.

Two multi-models are produced side by side:

* the **raw** one, from each model's own ensemble, on the 1° model grid. Each
  model reads its quantile products in its own hindcast climatology, so its bias
  cancels before the averaging;
* the **calibrated** one, on the observation grid at 0.25°, where each model
  contributes the product computed with **the method recommended for that
  product** (``registry/calibration_methods.csv``) — the whole point of the
  register of phase P3.

Three weighting schemes are computed for each, and they are not equivalent in
status: **equal weights are the reference**, the one the chain publishes unless
the others are shown to be better; the square root of the ensemble size and the
skill-proportional weights are experiments, kept in the output so that the
verification can decide.

Dekads are produced with the seven models that have them — NCEP has no daily
hindcast, so it has no dekads, and its absence renormalises the weights rather
than counting as a zero.
"""
from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.calibrate.base import RawEnsemble
from eccas_s2s.core.geo import mask_like
from eccas_s2s.core.periods import (add_period_arguments, announce_subset,
                                    selected_periods)
from eccas_s2s.io.netcdf import save as save_cf
from eccas_s2s.operations import c3s_totals, obs_chirps, obs_era5
from eccas_s2s.operations.calibrate_hindcast import (calibrators_for, model_on_grid,
                                                     observation_on_grid)
from eccas_s2s.operations.skill_raw import (AGGREGATION, OBS_VARIABLE, hindcast_streams,
                                            horizon_days)
from eccas_s2s.obs.climatology import obs_period_totals
from eccas_s2s.obs.regrid import match_model_grid
from eccas_s2s.products.forecast import ForecastContext, combine, product_values
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

#: the schemes, and what each says. "equal" is the reference of the chain.
SCHEMES = ("equal", "sqrt_members", "skill")


def multimodel_dir(cfg, kind: str = "raw"):
    """``<output_root>/mme/<cycle>/<kind>/`` — raw and calibrated side by side."""
    return cfg.output_root / "mme" / cfg.cycle_id / kind


#: a skill weighting needs several models on the positive side to mean anything
MIN_POSITIVE = 3


def weights_from_skill(summary: pd.DataFrame, models, variable: str, scale: str,
                       product: str, period: str, min_positive: int = MIN_POSITIVE):
    """
    Weights proportional to skill — and the guard that keeps them from degenerating.

    The rule of the Draft, ``w ∝ max(skill, 0)``, assumes skills that are mostly
    positive. Ours are not: the median RPSS of the raw terciles runs from −0.128
    to +0.006, so flooring at zero handed 0.976 of the weight to the single model
    that happened to land just above zero — a selection by sampling noise, not a
    weighting.

    So the deciding metric is used only when at least ``min_positive`` models are
    positive on it. Otherwise the **discrimination** takes over (GROC above 0.5,
    ROC skill, correlation): it is positive nearly everywhere in our register, it
    is what says a model carries information, and it degrades gracefully. When
    neither qualifies, the caller falls back to equal weights — which is the
    reference anyway.

    Returns ``(weights, source)``; an empty mapping means "use equal weights".
    """
    if summary is None or summary.empty:
        return {}, "equal (pas de synthèse)"
    common = summary[(summary["variable"] == variable) & (summary["scale"] == scale)
                     & (summary["product"] == product) & (summary["period"] == period)]
    for column, label, floor in (("deciding", "métrique décisive", 0.0),
                                 ("discrimination", "discrimination", 0.0)):
        if column not in common.columns:
            continue
        rows = common[common[column]]
        values, metrics = {}, []
        for model in models:
            got = rows[rows["model"] == model]
            if not len(got):
                values[model] = 0.0
                continue
            value = float(got["median"].iloc[0])
            metric = str(got["metric"].iloc[0])
            metrics.append(metric)
            # an area under the ROC curve has no skill at 0.5, not at 0
            reference = 0.5 if metric.split("_")[0] in ("groc", "roc") else floor
            values[model] = max(value - reference, 0.0)
        if sum(v > 0 for v in values.values()) >= min_positive:
            # the metric is named in the source: a weight is only readable if one
            # knows what it was measured on
            name = sorted(set(metrics))[0] if metrics else ""
            return values, f"{label} ({name}, {product})" if name else label
    return {}, "equal (skill non estimable)"


def _forecast_and_hindcast(cfg, model: str, variable: str, scale: str, period: str):
    """Forecast members and hindcast of one model for one period, on its own stream."""
    groups = hindcast_streams(cfg, "c3s", model, variable, [scale])
    stream = "monthly" if "monthly" in groups else "daily"
    if scale not in groups.get(stream, []):
        return None, None
    fc = c3s_totals.load_totals(cfg, model, "forecast", variable, stream)
    hc = c3s_totals.load_totals(cfg, model, "hindcast", variable, stream)
    keys = {str(k) for k in fc["period"].values}
    if period not in keys:
        return None, None
    forecast = fc.sel(period=period, drop=True).isel(year=0, drop=True).load()
    hindcast = hc.sel(period=period, drop=True).load()
    return forecast, hindcast


def _annual_normal(cfg, variable: str, resolution: str):
    """
    Mean annual total of each cell over the reference normal, on the chosen grid.

    The seasonality of a cell — what share of its year a period carries — is a
    property of the climate, so it is read on the **observation**, for the raw
    products as for the calibrated ones. It is what the relative dry mask
    compares the period climatology with, cell by cell.
    """
    y0, y1 = [int(v) for v in cfg.reference_period("obs_normal")]
    if variable == "precip":
        _, months = obs_chirps.load_archives(cfg, resolution)
    else:
        _, mon_ds = obs_era5.load_archives(cfg, resolution)
        months = mon_ds[OBS_VARIABLE[variable]]
    months = months.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    how = AGGREGATION[variable]
    yearly = months.groupby("time.year")
    yearly = yearly.sum("time") if how == "sum" else yearly.mean("time")
    return yearly.mean("year").load()


def _observed_normal(cfg, variable: str, scale: str, periods, resolution: str):
    """Observed series of the periods over the reference normal, on the chosen grid."""
    years = list(range(*[int(v) for v in cfg.reference_period("obs_normal")]))
    years.append(years[-1] + 1)
    if variable == "precip":
        dekads, months = obs_chirps.load_archives(cfg, resolution)
    else:
        dek_ds, mon_ds = obs_era5.load_archives(cfg, resolution)
        name = OBS_VARIABLE[variable]
        dekads, months = dek_ds[name], mon_ds[name]
    return obs_period_totals(dekads, months, periods, years, how=AGGREGATION[variable]).load()


def run(config: str, variables=("precip",), scales=None, models=None, kinds=("raw", "calibrated"),
        selection=None, schemes=SCHEMES, dry_method=None) -> RunContext:
    cfg = load_cycle(config)
    scales = list(scales) if scales else cfg.scales
    selected = [m for m in cfg.c3s_models if models is None or m in models]

    with RunContext(cfg, step="multimodel") as ctx:
        announce_subset(cfg, ctx, selection)
        ctx.record_parameter("kinds", list(kinds))
        ctx.record_parameter("schemes", list(schemes))
        ctx.record_parameter("models", selected)
        ctx.record_parameter("dry_mask_method",
                             dry_method or cfg.thresholds.get("precip", {})
                             .get("dry_mask", {}).get("method", "iri"))
        summaries = {k: _read_summary(cfg, k) for k in kinds}
        rows = []

        for variable in variables:
            for scale in scales:
                periods = cfg.periods_for(max(horizon_days(cfg, "c3s", m) for m in selected),
                                          (scale,), selection=selection)
                if not periods:
                    continue
                for kind in kinds:
                    resolution = "1p0" if kind == "raw" else "0p25"
                    normal = _observed_normal(cfg, variable, scale, periods, resolution)
                    mask = mask_like(cfg.raw["paths"]["shapefile"],
                                     normal.isel(year=0, period=0, drop=True))
                    normal = normal.where(mask)
                    annual = _annual_normal(cfg, variable, resolution).where(mask)
                    for period in periods:
                        rows += _one_period(cfg, ctx, variable, scale, period, kind, selected,
                                            normal.sel(period=period.key, drop=True), mask,
                                            summaries.get(kind), schemes, dry_method,
                                            annual=annual)
                        gc.collect()

        if rows:
            table = pd.DataFrame(rows)
            path = cfg.output_root / "mme" / cfg.cycle_id / "mme_products_summary.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            keys = ["kind", "variable", "scale", "period", "product", "scheme"]
            if path.exists():        # a partial run completes the table, never replaces it
                old = pd.read_csv(path)
                if set(keys) <= set(old.columns):
                    old = old[~old.set_index(keys).index.isin(table.set_index(keys).index)]
                    table = pd.concat([old, table], ignore_index=True)
            table.sort_values(keys).to_csv(path, index=False)
            ctx.record_output(path, role="mme_summary")
            ctx.log.info("%d combinaison(s) écrite(s)", len(table))
    return ctx


def _read_summary(cfg, kind: str):
    """Hindcast summary of a tree, for the skill weights."""
    name = "skill_raw_summary.csv" if kind == "raw" else "skill_calibrated_summary.csv"
    path = cfg.output_root / "skill" / cfg.cycle_id / kind / name
    return pd.read_csv(path) if path.exists() else None


def _recommended(cfg) -> dict:
    """``(model, variable, scale, product) -> method`` from the register of phase P3."""
    path = cfg.path_of("output_root") / "registry" / "calibration_methods.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {(r["model"], r["variable"], r["scale"], r["product"]): r["method"]
            for _, r in table.iterrows()}


def _one_period(cfg, ctx, variable, scale, period, kind, models, normal, mask,
                summary, schemes, dry_method=None, annual=None) -> list[dict]:
    """Products of every model for one period, then the three weighted combinations."""
    fctx = ForecastContext(normal, variable, scale, cfg.thresholds.get(variable, {}))
    per_model, members_count, methods = {}, {}, {}   # methods: (modèle, produit) -> méthode
    recommended = _recommended(cfg) if kind == "calibrated" else {}

    for model in models:
        forecast, hindcast = _forecast_and_hindcast(cfg, model, variable, scale, period.key)
        if forecast is None:
            continue
        members_count[model] = int(forecast.sizes.get("number", 1))
        if kind == "raw":
            # the model grid is wider than the observed one (55×50 against 45×30
            # at 1°): the products are restricted to the observation cells here,
            # explicitly, so that the forecast, the normal and the masks live on
            # the same grid instead of meeting through an implicit inner join
            target = normal.isel(year=0, drop=True)
            forecast = match_model_grid(forecast, target)
            hindcast = match_model_grid(hindcast, target)
            per_model[model] = product_values(RawEnsemble(forecast), fctx, hindcast=hindcast)
        else:
            got, by_product = _calibrated_products(cfg, model, variable, scale, period,
                                                   forecast, hindcast, fctx, recommended, ctx)
            if not got:
                continue
            per_model[model] = got
            # the method each product came from: it belongs on the map, because a
            # calibrated multi-model is not one method applied to eight models,
            # but as many methods as the register recommends
            for name, method in by_product.items():
                methods[(model, name)] = method
    if not per_model:
        ctx.warn(f"{kind} {variable} {scale} {period.key} : aucun modèle disponible")
        return []

    # dry mask of the period, with the convention of the configuration — the IRI
    # one by default (normal of the period < 15 % of the annual normal or < 30 mm,
    # for 3 months, scaled to the length of the period). A rainfall category
    # means nothing where the season brings almost nothing, and the maps grey
    # those cells out. The run can switch convention through thresholds.yaml.
    from eccas_s2s.products.masks import dry_mask as dry_mask_of, dry_mask_settings

    settings = dry_mask_settings(cfg.thresholds, method=dry_method)
    mme_mean = combine({m: {"cumul": p["cumul"]} for m, p in per_model.items()})["cumul"]
    # the relative criterion is a statement about the climate of each cell — the
    # share of its year this period carries — so it is read on the observed
    # normal for the raw products as for the calibrated ones; the absolute one
    # is read on the forecast itself
    dry = dry_mask_of(mme_mean, settings, climatology_mean=normal.mean("year"),
                      scale=scale, annual_mean=annual,
                      period_days=period.n_days(cfg.init_date.year)).where(mask)

    out_dir = multimodel_dir(cfg, kind) / variable / scale
    rows = []
    for scheme in schemes:
        weights, source = _weights(scheme, per_model, members_count, summary,
                                   variable, scale, period)
        combined = combine(per_model, weights)
        for name, field in combined.items():
            # every published field is restricted to the CEEAC mask, like every
            # other output of the chain: a probability over Nigeria is not ours
            field = field.where(mask)
            ds = field.to_dataset(name=name)
            ds.attrs = {**ctx.netcdf_attrs(), "variable": variable, "scale": scale,
                        "period": period.key, "label": period.label(cfg.init_date.year),
                        "kind": kind, "scheme": scheme, "product": name,
                        "models": field.attrs.get("models", ""),
                        "weights": field.attrs.get("weights", ""),
                        "weight_source": source,
                        "methods": _methods_of(methods, name),
                        "init_date": str(cfg.init_date.date()),
                        "grid": "modèle 1°" if kind == "raw" else "observation 0,25°"}
            ds["dry_mask"] = dry.astype("int8")
            path = save_cf(ds, out_dir / scheme / name / f"{period.key}.nc",
                           init_year=cfg.init_date.year, init_month=cfg.init_date.month)
            ctx.record_output(path, role="mme_product", kind=kind, scheme=scheme,
                              product=name, period=period.key)
            rows.append({"kind": kind, "variable": variable, "scale": scale,
                         "period": period.key, "label": period.label(cfg.init_date.year),
                         "product": name, "scheme": scheme,
                         "n_models": len(field.attrs.get("models", "").split(", ")),
                         "models": field.attrs.get("models", ""),
                         "weights": field.attrs.get("weights", ""),
                         "weight_source": source,
                         "methods": _methods_of(methods, name),
                         "mean": float(field.where(mask).mean())
                         if "category" not in field.dims else np.nan})
    ctx.log.info("%s %s %s %s : %d modèle(s), %d produit(s) × %d schéma(s)", kind, variable,
                 scale, period.key, len(per_model), len(combined), len(schemes))
    return rows


def _methods_of(methods: dict, product: str) -> str:
    """``modèle=méthode`` for one product, in the order the models were read."""
    return ", ".join(f"{model}={method}" for (model, name), method in methods.items()
                     if name == product)


def _weights(scheme: str, per_model: dict, members: dict, summary, variable, scale, period):
    """Weights of one scheme, and what they were built from."""
    if scheme == "equal":
        return None, "poids égaux"
    if scheme == "sqrt_members":
        return {m: float(np.sqrt(members.get(m, 1))) for m in per_model}, "√(membres)"
    got, source = weights_from_skill(summary, list(per_model), variable, scale,
                                     "terciles", period.key)
    return (got or None), source


def _calibrated_products(cfg, model, variable, scale, period, forecast, hindcast,
                         fctx, recommended, ctx):
    """
    Each product of a model, forecast with the method recommended **for it**.

    The register of phase P3 recommends a method per model *and per product*, and
    the recommendations differ widely: at the seasonal scale it asks for
    ``ngr_const`` on the terciles, ``elr`` on the quintiles and the thresholds,
    ``bias_ratio`` on the anomaly and **``raw``** on the SPI, for which no method
    beats the interpolated forecast. Applying the tercile method to everything —
    which this function used to do — publishes products the register explicitly
    rejects: the SPI so built scores −0,5 in MSESS where the interpolated
    forecast scores +0,11.

    So the products are grouped by the method they ask for, each method is fitted
    once on the whole hindcast — there is no year to leave out in a real forecast
    — and every product is read from its own distribution. Returns the products
    and, product by product, the method they came from, which the maps print
    under the models.
    """
    obs = observation_on_grid(cfg, variable, [period], [int(y) for y in hindcast["year"].values])
    obs = obs.sel(period=period.key, drop=True)
    ens_hind = model_on_grid(hindcast, obs)
    ens_fc = model_on_grid(forecast, obs)
    cals = {c.name: c for c in calibrators_for(variable, has_members=True)}

    wanted: dict[str, list] = {}
    for product in fctx.products:
        method = recommended.get((model, variable, scale, product.name), "raw")
        if method not in cals:
            ctx.warn(f"{model} {scale} {period.key} : méthode « {method} » inconnue "
                     f"pour {product.name}, prévision brute interpolée")
            method = "raw"
        wanted.setdefault(method, []).append(product.name)

    products, methods = {}, {}
    for method, names in wanted.items():
        cal = cals[method]
        dist = (RawEnsemble(ens_fc) if method == "raw"
                else cal.predict(ens_fc, cal.fit(ens_hind, obs)))
        # the interpolated forecast keeps reading its quantile products in its
        # own climatology, exactly as phase P3 scored it: the hindcast on the
        # observation grid is what that climatology is built from
        got = product_values(dist, fctx, products=names,
                             hindcast=ens_hind if method == "raw" else None)
        products.update(got)
        methods.update({name: method for name in got})
        del dist
    return products, methods


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variables", nargs="+", default=["precip"],
                    choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--kinds", nargs="+", default=["raw", "calibrated"],
                    choices=["raw", "calibrated"])
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES))
    ap.add_argument("--dry-mask", choices=["iri", "relative", "absolute"],
                    help="convention du masque sec (défaut : celle de thresholds.yaml)")
    add_period_arguments(ap)
    args = ap.parse_args(argv)
    run(args.config, args.variables, args.scales, args.models, args.kinds,
        selection=selected_periods(args), schemes=args.schemes, dry_method=args.dry_mask)


if __name__ == "__main__":
    main()
