"""
Hindcast skill of the multi-model (workflow E6, phase P6).

This step does to the multi-model exactly what phase P2 does to a raw model and
phase P3 to a calibrated one — the same functions, the same trees, the same
tables — so that a combination and the models it combines can be compared by
reading one column instead of two files:

* the hindcast pairs come from :func:`eccas_s2s.operations.skill_raw.prepare_pairs`
  for the raw tree, and from the interpolation and the recommended method of
  :mod:`eccas_s2s.operations.calibrate_hindcast` for the calibrated one;
* every model produces its products with :func:`eccas_s2s.validate.product_scores.forecast_field`,
  they are combined by :func:`eccas_s2s.products.forecast.combine` — the very
  function the operational run uses — and the combination is scored by
  :func:`eccas_s2s.validate.product_scores.score_fields`, which is the scoring of
  phases P2 and P3 applied one step later;
* the maps are written by :func:`eccas_s2s.operations.skill_raw.write_product_netcdf`
  and the rows by :func:`~eccas_s2s.operations.skill_raw.product_summary_rows`,
  into ``skill/<cycle>/{raw,calibrated}/`` beside the models, under the system
  name ``mme`` and the model name of the weighting scheme;
* the reliability and ROC diagrams are drawn family by family by the same R
  bridge as :mod:`eccas_s2s.operations.skill_diagrams`, in the same
  ``diagrams/`` branch.

The multi-model is therefore scored, mapped and diagrammed by the chain that
scores the models, and its rows land in ``skill_raw_summary.csv`` and
``skill_calibrated_summary.csv``. It is kept out of the **registers** alone
(``products_eligibility.csv``, ``calibration_methods.csv``): those answer "which
model may enter the workflow" and "which method to apply to it", and a
multi-model is an output of the workflow, not an input to it.

Example::

    python scripts/run_skill_multimodel.py --config config/cycle_202609.yaml \\
        --variables precip --scales season --kinds raw calibrated
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.calibrate.base import RawEnsemble
from eccas_s2s.core.periods import (add_period_arguments, announce_subset,
                                    selected_periods)
from eccas_s2s.products.catalogue import FAMILY_LABEL, families
from eccas_s2s.products.forecast import combine
from eccas_s2s.operations.calibrate_hindcast import (calibrators_for, model_on_grid,
                                                     observation_on_grid)
from eccas_s2s.operations.multimodel import SCHEMES, weights_from_skill
from eccas_s2s.operations.skill_raw import (_load_hindcast, hindcast_streams, horizon_days,
                                            prepare_pairs, product_summary_rows, score_paths,
                                            skill_dir, write_product_netcdf)
from eccas_s2s.provenance import RunContext
from eccas_s2s.registry import recommended_methods
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.pooled import MAX_PIXELS, SERIES_LABEL, family_frame
from eccas_s2s.validate.product_scores import (ObservationContext, forecast_field,
                                               observed_of, score_fields)
from eccas_s2s.validate.r_bridge import RNotAvailable, run_zone_diagrams

N_BOOT = 300
SYSTEM = "c3s"
#: the system name the multi-model is filed under, beside ``c3s`` and ``nmme``
MME_SYSTEM = "mme"
#: method level of the calibrated tree: each model enters with the method its
#: register recommends, so the combination has no single method of its own
MME_METHOD = "recommended"
#: the scheme whose diagrams are drawn — equal weights are the reference of the
#: chain, and three reliability curves of the same data would compare fits, not
#: forecasts
DIAGRAM_SCHEME = "equal"


def _summary_path(cfg, kind: str) -> Path:
    name = "skill_raw_summary.csv" if kind == "raw" else "skill_calibrated_summary.csv"
    return skill_dir(cfg, kind) / name


def _read_summary(cfg, kind: str):
    """Summary of a tree, read for the skill weights (as in the forecast run)."""
    path = _summary_path(cfg, kind)
    return pd.read_csv(path) if path.exists() else None


def _recommended(cfg) -> dict:
    """``(model, variable, scale, product) -> method`` from the register of the cycle (P3)."""
    return recommended_methods(cfg)


def _weights(scheme: str, fields: dict, members: dict, summary, variable, scale, period_key):
    """Weights of one scheme over the models that contributed, and their source."""
    if scheme == "equal":
        return None, "poids égaux"
    if scheme == "sqrt_members":
        return {m: float(np.sqrt(members.get(m, 1))) for m in fields}, "√(membres)"
    got, source = weights_from_skill(summary, list(fields), variable, scale,
                                     "terciles", period_key)
    return (got or None), source


# ------------------------------------------------------------------ hindcasts
def _raw_inputs(cfg, ctx, variable: str, scale: str, models, selection):
    """
    Members and observation of every model on the 1° grid, through ``prepare_pairs``.

    The same call phase P2 makes, model by model: the same stream (monthly for
    the lagged ensembles), the same interpolation onto the observation grid, the
    same CEEAC mask. The models are then restricted to the years they **share**,
    because a combination can only be verified on years every member covers.
    """
    per_model, periods, streams = {}, None, {}
    for model in models:
        groups = hindcast_streams(cfg, SYSTEM, model, variable, [scale])
        stream = next((s for s, sc in groups.items() if scale in sc), None)
        if stream is None:
            continue
        try:
            pairs, model_periods = prepare_pairs(cfg, SYSTEM, model, variable, [scale], ctx,
                                                 selection, stream)
        except (FileNotFoundError, KeyError) as exc:
            ctx.warn(f"{model} {variable} {scale} : données absentes ({exc})")
            continue
        if "members" not in pairs:
            ctx.warn(f"{model} {variable} {scale} : pas de membres, exclu du multi-modèle brut")
            continue
        per_model[model], streams[model] = pairs, stream
        periods = (model_periods if periods is None
                   else [p for p in periods if p.key in {q.key for q in model_periods}])
    if not per_model:
        return {}, [], None, {}
    years = sorted(set.intersection(*[{int(y) for y in p["year"].values}
                                      for p in per_model.values()]))
    per_model = {m: p.sel(year=years) for m, p in per_model.items()}
    obs = next(iter(per_model.values()))["obs"].sel(year=years)
    ctx.log.info("  brut : %d modèle(s), %d année(s) communes (%d-%d)",
                 len(per_model), len(years), years[0], years[-1])
    return per_model, periods or [], obs, streams


def _calibrated_inputs(cfg, ctx, variable: str, scale: str, models, periods):
    """Hindcasts of every model and the observation on its own 0,25° grid."""
    hindcasts, streams = {}, {}
    for model in models:
        groups = hindcast_streams(cfg, SYSTEM, model, variable, [scale])
        stream = next((s for s, sc in groups.items() if scale in sc), None)
        if stream is None:
            continue
        try:
            hind = _load_hindcast(cfg, SYSTEM, model, variable, stream).load()
        except (FileNotFoundError, KeyError) as exc:
            ctx.warn(f"{model} {variable} {scale} : données absentes ({exc})")
            continue
        hindcasts[model], streams[model] = hind, stream
    if not hindcasts:
        return {}, None, {}
    years = sorted(set.intersection(*[{int(y) for y in h["year"].values}
                                      for h in hindcasts.values()]))
    keys = {p.key for p in periods}
    hindcasts = {m: h.sel(year=years,
                          period=[k for k in h["period"].values if str(k) in keys])
                 for m, h in hindcasts.items()}
    obs = observation_on_grid(cfg, variable, periods, years, ctx)
    obs = obs.sel(period=[p.key for p in periods])
    ctx.log.info("  calibré : %d modèle(s), %d année(s) communes (%d-%d)",
                 len(hindcasts), len(years), years[0], years[-1])
    return hindcasts, obs, streams


def _model_fields(cfg, ctx, kind: str, model: str, data, obs_period, octx, variable: str,
                  scale: str, period_key: str, recommended: dict):
    """
    The products of one model over the hindcast, and the method each came from.

    A raw model gives them all from its own ensemble. A calibrated one gives
    each product from the method the register of phase P3 recommends **for that
    product**: the recommendations differ widely — ``ngr_const`` on the terciles,
    ``elr`` on the quintiles, ``raw`` on the SPI, where no method beats the
    interpolated forecast — and applying the tercile method to everything scores
    a product the register never recommended. The products are therefore grouped
    by method, each method is fitted once leave-one-year-out, and every product
    is read from its own distribution.
    """
    if kind == "raw":
        members = data["members"].sel(period=period_key, drop=True)
        dist = RawEnsemble(members)
        cache: dict = {}
        fields = {p.name: forecast_field(dist, octx, p, cache) for p in octx.products}
        return fields, {p.name: "raw" for p in octx.products}, int(members.sizes.get("number", 1))

    ens = model_on_grid(data.sel(period=period_key, drop=True), obs_period)
    cals = {c.name: c for c in calibrators_for(variable, has_members="number" in ens.dims)}
    wanted: dict[str, list] = {}
    for product in octx.products:
        method = recommended.get((model, variable, scale, product.name), "raw")
        if method not in cals:
            ctx.warn(f"{model} {scale} {period_key} : méthode « {method} » inconnue "
                     f"pour {product.name}, prévision brute interpolée")
            method = "raw"
        wanted.setdefault(method, []).append(product)

    fields, methods = {}, {}
    for method, products in wanted.items():
        dist = cals[method].fit_predict_loyo(ens, obs_period)
        cache = {}
        for product in products:
            fields[product.name] = forecast_field(dist, octx, product, cache)
            methods[product.name] = method
        del dist, cache
        gc.collect()
    n_members = int(ens.sizes.get("number", 1))
    del ens
    return fields, methods, n_members


# ---------------------------------------------------------------------- run
def _one_scale(cfg, ctx, variable: str, scale: str, kind: str, models, schemes, selection,
               metrics: str, diagrams: bool, n_boot: int, max_pixels: int,
               diagram_scheme: str) -> list[dict]:
    """One scale, every period: the models are combined, scored, mapped and diagrammed."""
    out = skill_dir(cfg, kind)
    periods = cfg.periods_for(max(horizon_days(cfg, SYSTEM, m) for m in models),
                              (scale,), selection=selection)
    if not periods:
        return []
    summary = _read_summary(cfg, kind)
    recommended = _recommended(cfg) if kind == "calibrated" else {}

    if kind == "raw":
        data, periods, obs, streams = _raw_inputs(cfg, ctx, variable, scale, models, selection)
    else:
        data, obs, streams = _calibrated_inputs(cfg, ctx, variable, scale, models, periods)
    if not data or obs is None:
        return []
    periods = [p for p in periods if p.key in {str(k) for k in obs["period"].values}]
    mask = obs.notnull().any(["year", "period"])
    thresholds = cfg.thresholds.get(variable, {})
    labels = {p.key: (p.label(cfg.init_date.year), p.label_fr(cfg.init_date.year, with_dates=True))
              for p in periods}
    years = [int(y) for y in obs["year"].values]
    stream = ", ".join(sorted(set(streams.values())))

    rows: list[dict] = []
    by_product: dict = {}
    n_figures = 0
    for period in periods:
        key = period.key
        ref = obs.sel(period=key, drop=True)
        octx = ObservationContext(ref, variable, scale, thresholds, mask, "year")
        fields, members, methods = {}, {}, {}
        for model in list(data):
            got, used_methods, n_members = _model_fields(cfg, ctx, kind, model, data[model],
                                                         ref, octx, variable, scale, key,
                                                         recommended)
            fields[model], members[model] = got, n_members
            for name, method in used_methods.items():
                methods[(model, name)] = method
            gc.collect()
        if not fields:
            continue

        for scheme in schemes:
            weights, source = _weights(scheme, fields, members, summary, variable, scale, key)
            combined = combine(fields, weights)
            used = list(fields)
            total = int(sum(members.get(m, 1) for m in used))
            maps = score_fields(combined, octx, metrics, total)
            for name, ds in maps.items():
                ds.attrs.update({"models": ", ".join(used), "scheme": scheme,
                                 "weight_source": source, "n_models": len(used),
                                 "methods": _methods_of(methods, name)})
                by_product.setdefault((name, scheme), []).append(
                    ds.expand_dims({"period": [key]}))
                rows += _rows(ds, mask, variable, scale, key, name, period, labels, scheme,
                              stream, total, len(used), _methods_of(methods, name))
            if diagrams and scheme == diagram_scheme:
                n_figures += _draw_diagrams(cfg, ctx, out, variable, scale, kind, scheme,
                                           period, combined, octx, mask, thresholds, years,
                                           n_boot, max_pixels)
            del combined, maps
        del fields
        gc.collect()

    # one netCDF per metric, on the branch of phases P2 and P3
    for (name, scheme), parts in by_product.items():
        merged = xr.concat(parts, dim="period", combine_attrs="override")
        pkeys = [str(k) for k in merged["period"].values]
        merged = merged.assign_coords(label=("period", [labels[k][0] for k in pkeys]),
                                      label_fr=("period", [labels[k][1] for k in pkeys]),
                                      scale=("period", [scale] * len(pkeys)))
        merged.attrs = {**ctx.netcdf_attrs(), **parts[0].attrs, "system": MME_SYSTEM,
                        "model": scheme, "variable": variable, "scale": scale,
                        "kind_of_run": f"multi-model {kind} hindcast skill",
                        "n_years": len(years), "stream": stream,
                        "hindcast_period": f"{min(years)}-{max(years)}",
                        "grid": "modèle 1°" if kind == "raw" else "observation 0,25°",
                        "init_date": str(cfg.init_date.date()),
                        "cross_validation": cfg.cv_scheme}
        branch = name if kind == "raw" else f"{name}/{MME_METHOD}"
        write_product_netcdf(merged, out / "netcdf", MME_SYSTEM, scheme, variable, scale,
                             branch, ctx)
    ctx.log.info("%s %s %s : %d produit(s) × %d schéma(s) sur %d période(s), %d diagramme(s)",
                 kind, variable, scale, len({n for n, _ in by_product}), len(schemes),
                 len(periods), n_figures)
    del data, obs, by_product
    gc.collect()
    return rows


def _methods_of(methods: dict, product: str) -> str:
    """``modèle=méthode`` for one product, in the order the models were read."""
    return ", ".join(f"{model}={method}" for (model, name), method in methods.items()
                     if name == product)


def _rows(ds: xr.Dataset, mask, variable: str, scale: str, key: str, product: str, period,
          labels: dict, scheme: str, stream: str, n_members: int, n_models: int,
          methods: str) -> list[dict]:
    """Summary rows of one product and period, in the columns of phases P2 and P3."""
    single = ds.expand_dims({"period": [key]}).assign_coords(
        label=("period", [labels[key][0]]), label_fr=("period", [labels[key][1]]))
    single.attrs = {**ds.attrs, "n_members": n_members}
    rows = product_summary_rows(single, mask, MME_SYSTEM, scheme, variable, scale, product,
                                {key: period.month_offset}, stream)
    for row in rows:
        row.update({"n_models": n_models, "method": MME_METHOD, "methods": methods})
    return rows


def _draw_diagrams(cfg, ctx, out: Path, variable: str, scale: str, kind: str, scheme: str,
                   period, combined: dict, octx, mask, thresholds: dict, years,
                   n_boot: int, max_pixels: int) -> int:
    """Reliability and ROC of the combination, family by family, as in phase P2."""
    from eccas_s2s.operations.skill_diagrams import KIND_LABEL, VARIABLE_LABEL

    drawn = 0
    for family, products in families(variable, scale, thresholds).items():
        series = []
        for product in products:
            if product.name not in combined:
                continue
            prob, observed = combined[product.name], observed_of(octx, product)
            if "category" in prob.dims:
                names = SERIES_LABEL.get(family, {})
                for index, cat in enumerate(prob["category"].values):
                    event = (observed == index).astype(float).where(observed.notnull())
                    series.append((str(cat), names.get(str(cat), str(cat)),
                                   prob.sel(category=cat), event))
            else:
                series.append((product.name, product.label, prob, observed))
        if not series:
            continue
        frame = family_frame(series, mask, period.key, period.label_fr(cfg.init_date.year),
                             max_pixels=max_pixels, drop_degenerate=family == "seuils_cumul")
        if frame.empty:
            ctx.warn(f"mme {kind} {variable} {scale} {family} {period.key} : "
                     "aucune paire exploitable")
            continue
        branch = family if kind == "raw" else f"{family}/{MME_METHOD}"
        zdir = score_paths(out / "diagrams", MME_SYSTEM, scheme, variable, scale, branch)
        label = (f"Multi-modèle ({scheme}) — {VARIABLE_LABEL.get(variable, variable)}"
                 f" — Init : {cfg.init_date.date()} | "
                 f"Hindcast Period : {min(years)}-{max(years)}")
        try:
            figures = run_zone_diagrams(frame, zdir, label, n_boot=n_boot,
                                        kind=KIND_LABEL.get(kind, kind),
                                        family=FAMILY_LABEL.get(family, family))
        except RNotAvailable as exc:
            ctx.warn(f"diagrammes mme {kind} {variable} {scale} {family} : {exc}")
            return drawn
        drawn += len(figures)
        for f in figures:
            ctx.record_output(f, role="skill_diagram", system=MME_SYSTEM, model=scheme,
                              variable=variable, scale=scale, family=family,
                              period=period.key)
    return drawn


def _merge_rows(cfg, ctx, kind: str, rows: list[dict]) -> None:
    """
    Merge the multi-model rows into the summary table of the tree.

    The same file the models are written to, with the same keys, so that a
    reader compares a combination and its members by filtering one column. The
    registers are rebuilt from that table elsewhere and ignore the ``mme``
    system: they say which **model** may enter the workflow, and with which
    method — a question a multi-model does not answer.
    """
    table = pd.DataFrame(rows)
    if table.empty:
        return
    path = _summary_path(cfg, kind)
    keys = (["system", "model", "variable", "scale", "period", "product", "metric"]
            + (["method"] if kind == "calibrated" else []))
    if path.exists():
        old = pd.read_csv(path)
        if set(keys) <= set(old.columns):
            old = old[~old.set_index(keys).index.isin(table.set_index(keys).index)]
            table = pd.concat([old, table], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.sort_values(keys).to_csv(path, index=False)
    ctx.record_output(path, role="summary")
    ctx.log.info("%s : %d ligne(s) multi-modèle dans %s", kind, len(rows), path.name)


def run(config: str, variables=("precip",), scales=("month", "season"),
        kinds=("raw", "calibrated"), models=None, schemes=SCHEMES, selection=None,
        metrics: str = "deciding", n_boot: int = N_BOOT, max_pixels: int = MAX_PIXELS,
        diagrams: bool = True, diagram_scheme: str = DIAGRAM_SCHEME) -> RunContext:
    cfg = load_cycle(config)
    selected = [m for m in cfg.c3s_models if models is None or m in models]
    schemes = [s for s in schemes if s in SCHEMES]

    with RunContext(cfg, step="skill_multimodel") as ctx:
        announce_subset(cfg, ctx, selection)
        ctx.record_parameter("variables", list(variables))
        ctx.record_parameter("scales", list(scales))
        ctx.record_parameter("kinds", list(kinds))
        ctx.record_parameter("schemes", list(schemes))
        ctx.record_parameter("models", selected)
        ctx.record_parameter("metrics", metrics)
        ctx.record_parameter("cross_validation", cfg.cv_scheme)

        for kind in kinds:
            rows: list[dict] = []
            for variable in variables:
                for scale in scales:
                    rows += _one_scale(cfg, ctx, variable, scale, kind, selected, schemes,
                                       selection, metrics, diagrams, n_boot, max_pixels,
                                       diagram_scheme)
            _merge_rows(cfg, ctx, kind, rows)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variables", nargs="+", default=["precip"],
                    choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--scales", nargs="+", default=["month", "season"],
                    choices=["decade", "month", "season"])
    ap.add_argument("--kinds", nargs="+", default=["raw", "calibrated"],
                    choices=["raw", "calibrated"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES))
    ap.add_argument("--metrics", nargs="+", default=["deciding"],
                    help="métriques par produit : deciding (défaut), reported, all, ou une liste")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--no-diagrams", action="store_true",
                    help="ne pas tracer les diagrammes de fiabilité et ROC")
    add_period_arguments(ap)
    args = ap.parse_args(argv)
    run(args.config, args.variables, args.scales, args.kinds, args.models, args.schemes,
        selection=selected_periods(args),
        metrics=args.metrics[0] if len(args.metrics) == 1 else tuple(args.metrics),
        n_boot=args.n_boot, max_pixels=args.max_pixels, diagrams=not args.no_diagrams)


if __name__ == "__main__":
    main()
