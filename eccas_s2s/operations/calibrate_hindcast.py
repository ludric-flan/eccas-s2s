"""
Calibration of the hindcasts, method by method (workflow E5, phase P3).

For one model, one variable and one scale, this step:

1. brings the model onto the **observation grid** (0.25°, decision of
   22/09/2026) — CHIRPS ``p25`` for rainfall, ERA5 for temperature — by bilinear
   interpolation of the members, which is the "interpolated forecast" baseline
   the downscaling of E7 must beat;
2. fits every method of :mod:`eccas_s2s.calibrate` **leave-one-year-out**
   (decision D10) and predicts each hindcast year out of sample;
3. writes, per method, the quantities the products and the scores need: the
   deterministic forecast, the tercile probabilities on the **observed**
   thresholds (D12), and the ensemble spread when the method has one.

The output feeds :mod:`eccas_s2s.operations.skill_calibrated`, which scores it
with exactly the metrics and diagrams of phase P2 — the only way to say whether
a calibration is worth using.

Output, one file per method::

    <output_root>/calibrated/<cycle>/<system>_<model>/<scale>/<variable>/<method>.nc

Example::

    python scripts/run_calibrate_hindcast.py --config config/cycle_202609.yaml \\
        --systems c3s --models ecmwf --variables precip --scales season
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.calibrate.base import CATEGORIES, RawForecast
from eccas_s2s.calibrate.bias import bias_calibrators
from eccas_s2s.calibrate.logistic import logistic_calibrators
from eccas_s2s.calibrate.ngr import ngr_calibrators
from eccas_s2s.calibrate.qmap import qmap_calibrators
from eccas_s2s.core.geo import mask_like
from eccas_s2s.io.netcdf import save as save_cf
from eccas_s2s.core.periods import (add_period_arguments, announce_subset,
                                    selected_periods)
from eccas_s2s.operations import obs_chirps, obs_era5
from eccas_s2s.calibrate.products import methods_for
from eccas_s2s.operations.skill_raw import (AGGREGATION, MIN_FRACTION, OBS_VARIABLE,
                                            SYSTEM_SCALES, _load_hindcast, hindcast_streams,
                                            horizon_days, product_summary_rows, skill_dir,
                                            split_skill_name, write_product_netcdf)
from eccas_s2s.obs.climatology import obs_period_totals
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.cv import loyo_quantile
from eccas_s2s.validate.comparison import compare_maps, gain_map, recommend
from eccas_s2s.validate.product_scores import ObservationContext, product_maps

TERCILES = (1 / 3, 2 / 3)
#: the method families of the CAPC-AC note, in the order the report lists them.
FAMILIES = ("raw", "bias", "qmap", "logistic", "ngr")


def calibrated_dir(cfg, cycle_kind: str = "calibrated") -> Path:
    return cfg.output_root / cycle_kind / cfg.cycle_id


def calibrators_for(variable: str, has_members: bool, families=FAMILIES) -> list:
    """
    The methods worth trying for a variable, following the CAPC-AC note.

    A ratio correction has no meaning in °C; quantile mapping and EMOS need an
    ensemble to say anything about the spread, so a member-less system (NMME)
    keeps the deterministic corrections, the logistic family — which only uses
    the ensemble mean — and the constant-spread EMOS.
    """
    out = []
    if "raw" in families:
        out.append(RawForecast(variable))       # the baseline, on the same grid
    if "bias" in families:
        out += bias_calibrators(variable)
    if "qmap" in families:
        out += qmap_calibrators(variable) if has_members else []
    if "logistic" in families:
        out += logistic_calibrators(variable)
    if "ngr" in families:
        out += ngr_calibrators(variable, has_members)
    return out


def observation_on_grid(cfg, variable: str, periods, years, ctx=None) -> xr.DataArray:
    """Observed values of the periods on the reference grid of the variable (0.25°)."""
    if variable == "precip":
        dekads, months = obs_chirps.load_archives(cfg)          # base resolution of the cycle
    else:
        dek_ds, mon_ds = obs_era5.load_archives(cfg, "0p25")
        name = OBS_VARIABLE[variable]
        dekads, months = dek_ds[name], mon_ds[name]
    obs = obs_period_totals(dekads, months, periods, years, how=AGGREGATION[variable]).load()
    mask = mask_like(cfg.raw["paths"]["shapefile"], obs.isel(year=0, period=0, drop=True))
    if ctx:
        ctx.log.info("    grille d'observation : %d mailles CEEAC sur %d",
                     int(mask.sum()), int(mask.size))
    return obs.where(mask)


def model_on_grid(hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """
    Bring the model (1°) onto the observation grid by bilinear interpolation.

    This is not a downscaling: no local information is added, the field is only
    read where the observation lives. It is the reference every calibration is
    measured against — a method that does not beat the interpolated forecast is
    not worth its parameters (workflow E7).
    """
    target = {"latitude": obs["latitude"], "longitude": obs["longitude"]}
    out = hindcast.interp(target, method="linear")
    edge = hindcast.interp(target, method="nearest")            # borders of the domain
    return out.fillna(edge)


def _summary(dist, obs: xr.DataArray, variable: str, year_dim: str = "year") -> xr.Dataset:
    """
    What a calibrated forecast is published and scored on.

    The tercile thresholds are those of the **observation**, recomputed without
    the year being predicted, exactly as in phase P2 — otherwise "raw" and
    "calibrated" would not be answering the same question.
    """
    q = loyo_quantile(obs, TERCILES, year_dim=year_dim)
    q33 = q.sel(quantile=TERCILES[0], drop=True)
    q67 = q.sel(quantile=TERCILES[1], drop=True)
    ds = xr.Dataset({"forecast": dist.mean(), "prob": dist.tercile_probs(q33, q67),
                     "obs_q33": q33, "obs_q67": q67})
    spread = getattr(dist, "spread", None)
    if callable(spread):
        ds["spread"] = spread()
    elif hasattr(dist, "sigma"):
        ds["spread"] = dist.sigma
    ds["forecast"].attrs["long_name"] = f"calibrated {variable} forecast"
    return ds


def run(config: str, systems=("c3s",), variables=("precip",), models=None, scales=("month", "season"),
        families=FAMILIES, selection=None, metrics: str = "deciding",
        min_improved: float = 0.5) -> RunContext:
    cfg = load_cycle(config)
    out_root = calibrated_dir(cfg)
    skill_root = skill_dir(cfg, "calibrated")

    # a system switched off in the configuration is skipped, not attempted
    systems = [s for s in systems if s in cfg.enabled_systems()]
    if not systems:
        raise ValueError("aucun système actif : voir `enabled` dans la configuration du cycle")

    with RunContext(cfg, step="calibrate_hindcast") as ctx:
        announce_subset(cfg, ctx, selection)
        ctx.record_parameter("systems", list(systems))
        ctx.record_parameter("variables", list(variables))
        ctx.record_parameter("scales", list(scales))
        ctx.record_parameter("families", list(families))
        ctx.record_parameter("cross_validation", cfg.cv_scheme)
        ctx.record_parameter("metrics", metrics)
        ctx.record_parameter("min_improved", min_improved)
        summary_rows_all: list = []
        comparison_rows: list = []

        for system in systems:
            candidates = (list(cfg.c3s_models) if system == "c3s"
                          else list(cfg.raw["systems"]["nmme"]["models"]))
            sys_scales = [s for s in scales if s in SYSTEM_SCALES[system]]
            for model in [m for m in candidates if models is None or m in models]:
                for variable in variables:
                    if system == "nmme" and variable in ("tmax", "tmin"):
                        continue
                    groups = hindcast_streams(cfg, system, model, variable, sys_scales)
                    for stream, stream_scales in groups.items():
                        if not stream_scales:
                            continue
                        try:
                            hind = _load_hindcast(cfg, system, model, variable, stream).load()
                        except (FileNotFoundError, KeyError) as exc:
                            ctx.warn(f"{system} {model} {variable} ({stream}) : "
                                     f"données absentes ({exc})")
                            continue
                        keep = [p for p in hind["period"].values
                                if str(hind["scale"].sel(period=p).values) in stream_scales]
                        hind = hind.sel(period=keep)
                        periods = cfg.periods_for(horizon_days(cfg, system, model),
                                                  tuple(stream_scales), selection=selection)
                        periods = [p for p in periods if p.key in set(keep)]
                        years = [int(y) for y in hind["year"].values]
                        obs = observation_on_grid(cfg, variable, periods, years, ctx)
                        obs = obs.sel(period=[p.key for p in periods])
                        has_members = "number" in hind.dims
                        cals = calibrators_for(variable, has_members, families)
                        ctx.log.info("%s %s %s : %d période(s), %d années, %d méthode(s)",
                                     system, model, variable, len(periods), len(years), len(cals))

                        for scale in sorted({p.scale for p in periods}):
                            keys = [p.key for p in periods if p.scale == scale]
                            results = {c.name: [] for c in cals}
                            by_product: dict = {}
                            scored: dict = {}
                            timings: dict[str, float] = {}
                            for key in keys:                      # period by period: memory
                                ens = model_on_grid(hind.sel(period=key, drop=True), obs)
                                ref = obs.sel(period=key, drop=True)
                                # observed side computed once for all the methods
                                obs_ctx = ObservationContext(ref, variable, scale,
                                                             cfg.thresholds.get("precip", {}))
                                n_members = int(ens.sizes.get("number", 0))
                                for cal in cals:
                                    t0 = time.perf_counter()
                                    dist = cal.fit_predict_loyo(ens, ref)
                                    summary = _summary(dist, ref, variable).load()
                                    # only the products this method can answer
                                    # (eccas_s2s.calibrate.products)
                                    wanted = [pr.name for pr in obs_ctx.products
                                              if cal.name in {getattr(c, "name", "")
                                                              for c in methods_for(pr, cals)}]
                                    maps = product_maps(dist, obs_ctx, mode=metrics,
                                                        n_members=n_members, products=wanted)
                                    for pname, ds_map in maps.items():
                                        by_product.setdefault((pname, cal.name), []).append(
                                            ds_map.expand_dims({"period": [key]}))
                                    timings[cal.name] = timings.get(cal.name, 0.0) + (
                                        time.perf_counter() - t0)
                                    results[cal.name].append(summary.expand_dims(period=[key]))
                                    del maps
                                del ens
                                gc.collect()
                            folder = (out_root / f"{system}_{model}" / scale / variable)
                            folder.mkdir(parents=True, exist_ok=True)
                            labels = {p.key: (p.label(cfg.init_date.year),
                                              p.label_fr(cfg.init_date.year, with_dates=True))
                                      for p in periods}
                            for name, parts in results.items():
                                ds = xr.concat(parts, dim="period")
                                ds = ds.assign_coords(
                                    label=("period", [labels[k][0] for k in keys]),
                                    label_fr=("period", [labels[k][1] for k in keys]),
                                    scale=("period", [scale] * len(keys)))
                                ds.attrs.update({**ctx.netcdf_attrs(), "system": system,
                                                 "model": model, "variable": variable,
                                                 "scale": scale, "method": name,
                                                 "kind": "calibrated hindcast (LOYO)",
                                                 "n_years": len(years),
                                                 "hindcast_period": f"{min(years)}-{max(years)}",
                                                 "grid": "observation 0.25°",
                                                 "cross_validation": cfg.cv_scheme})
                                path = save_cf(ds, folder / f"{name}.nc",
                                               init_year=cfg.init_date.year,
                                               init_month=cfg.init_date.month)
                                ctx.record_output(path, role="calibrated_hindcast", system=system,
                                                  model=model, variable=variable, scale=scale,
                                                  method=name)
                            ctx.log.info("  %s %s : %d méthode(s) écrite(s) dans %s",
                                         scale, variable, len(results), folder)
                            # --- scores des produits calibrés : même arbre qu'en P2,
                            # avec un niveau <méthode> de plus
                            mask = ref.notnull()
                            scale_periods = [p for p in periods if p.scale == scale]
                            lead_of = {p.key: p.month_offset for p in scale_periods}
                            for (pname, mname), parts in by_product.items():
                                merged = xr.concat(parts, dim="period", combine_attrs="override")
                                pkeys = [str(k) for k in merged["period"].values]
                                merged = merged.assign_coords(
                                    label=("period", [labels[k][0] for k in pkeys]),
                                    label_fr=("period", [labels[k][1] for k in pkeys]),
                                    scale=("period", [scale] * len(pkeys)))
                                merged.attrs = {**ctx.netcdf_attrs(), **parts[0].attrs,
                                                "system": system, "model": model,
                                                "variable": variable, "method": mname,
                                                "kind_of_run": "calibrated hindcast skill",
                                                "n_years": len(years),
                                                "hindcast_period": f"{min(years)}-{max(years)}",
                                                "grid": "observation 0.25°",
                                                "stream": stream,
                                                "init_date": str(cfg.init_date.date()),
                                                "cross_validation": cfg.cv_scheme}
                                write_product_netcdf(merged, skill_root / "netcdf", system,
                                                     model, variable, scale,
                                                     f"{pname}/{mname}", ctx)
                                rows = product_summary_rows(merged, mask, system, model,
                                                            variable, scale, pname, lead_of,
                                                            stream)
                                for row in rows:
                                    row["method"] = mname
                                summary_rows_all += rows
                                scored[(pname, mname)] = merged

                            # --- brut contre calibré, maille par maille ---------------
                            # le "brut" de la comparaison est la prévision interpolée sur
                            # la grille d'observation (méthode `raw`) : comparer avec les
                            # cartes de P2, calculées à 1°, mélangerait l'effet de la
                            # calibration et celui du changement de grille.
                            raw_of = {pname: ds for (pname, mname), ds in scored.items()
                                      if mname == "raw"}
                            for (pname, mname), ds in scored.items():
                                if mname == "raw" or pname not in raw_of:
                                    continue
                                ref_ds = raw_of[pname]
                                deciding = ds.attrs.get("deciding", "")
                                common = [v for v in ds.data_vars
                                          if v in ref_ds and v not in ("n_years", "base_rate")
                                          and "category" not in ds[v].dims]
                                for metric in common:
                                    gains = gain_map(ds[metric], ref_ds[metric], metric)
                                    gains = gains.assign_coords(
                                        label_fr=ds["label_fr"]).to_dataset(
                                            name=f"gain_{metric}")
                                    gains.attrs = {**ds.attrs, "product": pname,
                                                   "method": mname, "metric": metric,
                                                   "baseline": "raw (interpolé 0,25°)"}
                                    write_product_netcdf(gains, skill_root / "netcdf", system,
                                                         model, variable, scale,
                                                         f"{pname}/{mname}", ctx)
                                    for key in [str(k) for k in ds["period"].values]:
                                        row = compare_maps(ds[metric].sel(period=key),
                                                           ref_ds[metric].sel(period=key),
                                                           metric, mask, method=mname,
                                                           product=pname, period=key)
                                        row.update({"system": system, "model": model,
                                                    "variable": variable, "scale": scale,
                                                    "stream": stream,
                                                    "deciding": metric == deciding,
                                                    "label_fr": labels[key][1]})
                                        comparison_rows.append(row)
                            ctx.log.info("  temps par méthode (s) : %s",
                                         ", ".join(f"{k} {v:.0f}" for k, v in
                                                   sorted(timings.items(), key=lambda kv: -kv[1])))
                            ctx.record_parameter(f"timings_{scale}_{variable}",
                                                 {k: round(v, 1) for k, v in timings.items()})
                        del hind, obs
                        gc.collect()

        _write_calibrated_outputs(cfg, ctx, skill_root, summary_rows_all, comparison_rows,
                                  min_improved)
    return ctx


def _write_calibrated_outputs(cfg, ctx, skill_root, summary_rows, comparison_rows,
                              min_improved: float = 0.5) -> None:
    """Summary of the calibrated scores, comparison with the raw baseline, and register."""
    keys = ["system", "model", "variable", "scale", "period", "product", "method", "metric"]
    if summary_rows:
        table = pd.DataFrame(summary_rows)
        path = skill_root / "skill_calibrated_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_csv(path)
            if set(keys) <= set(old.columns):
                old = old[~old.set_index(keys).index.isin(table.set_index(keys).index)]
                table = pd.concat([old, table], ignore_index=True)
        table.sort_values(keys).to_csv(path, index=False)
        ctx.record_output(path, role="summary")

    if not comparison_rows:
        return
    comparison = pd.DataFrame(comparison_rows)
    path = skill_root / "comparison_raw_calibrated.csv"
    comparison.sort_values(["system", "model", "variable", "scale", "product", "metric",
                            "method", "period"]).to_csv(path, index=False)
    ctx.record_output(path, role="comparison")

    # la méthode recommandée se décide sur la métrique qui décide pour le produit
    register = recommend(comparison[comparison["deciding"]], min_improved=min_improved)
    reg_dir = cfg.path_of("output_root") / "registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    path = reg_dir / "calibration_methods.csv"
    register.to_csv(path, index=False)
    ctx.record_output(path, role="calibration_register")
    counts = register["method"].value_counts().to_dict()
    ctx.log.info("méthodes recommandées : %s",
                 ", ".join(f"{v}x {k}" for k, v in counts.items()))
    ctx.record_parameter("recommended_methods", counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--systems", nargs="+", default=["c3s"], choices=["c3s", "nmme"])
    ap.add_argument("--variables", nargs="+", default=["precip"],
                    choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--scales", nargs="+", default=["month", "season"],
                    choices=["decade", "month", "season"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--metrics", nargs="+", default=["deciding"],
                    help="métriques par produit : deciding (défaut), reported, all, ou une liste")
    ap.add_argument("--min-improved", type=float, default=0.5,
                    help="part du masque qu'une méthode doit améliorer pour être recommandée")
    add_period_arguments(ap)
    args = ap.parse_args(argv)
    run(args.config, args.systems, args.variables, args.models, args.scales, args.families,
        selection=selected_periods(args),
        metrics=args.metrics[0] if len(args.metrics) == 1 else tuple(args.metrics),
        min_improved=args.min_improved)


if __name__ == "__main__":
    main()
