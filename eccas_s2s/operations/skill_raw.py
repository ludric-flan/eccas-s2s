"""
Raw hindcast skill of every model (workflow step E4, Draft §3.3).

For each system (C3S, NMME), model, variable and period, the raw forecast is
compared with the observation over the whole hindcast, leave-one-year-out:

* **maps** (Python, :mod:`eccas_s2s.validate.scores`): bias, MAE, RMSE, MSESS,
  Pearson, Spearman, ACC, RPS/RPSS, and per category the Brier skill score and
  the ROC area;
* **zone scores** (R ``verification`` package, :mod:`eccas_s2s.validate.r_bridge`):
  the same families plus CRPS, the Brier decomposition, the ROC p-value,
  Heidke/Peirce/Gerrity and the reliability bins, for the whole domain and for
  the three zones;
* **eligibility** (§3.3): a model is eligible for a variable, scale and period
  when its skill is positive on at least ``min_fraction`` of the domain for at
  least one deterministic score **and** one probabilistic score.

Everything is raw: no bias correction, no calibration. These numbers are the
reference that the calibration of phase P3 must beat.

Outputs in ``<output_root>/skill/<YYYYMM>/raw/``::

    maps/<system>_<model>_<variable>_skill.nc
    zones/<system>_<model>_<variable>/<zone>/{deterministic,tercile,category}_scores.csv
    skill_raw_summary.csv
    models_eligibility.csv

Example::

    python scripts/run_skill_raw.py --config config/cycle_202609.yaml --variables precip
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.core.periods import build_periods
from eccas_s2s.obs.climatology import obs_period_totals
from eccas_s2s.obs.regrid import conservative_to_degree, match_model_grid
from eccas_s2s.operations import c3s_totals, nmme_totals, obs_chirps, obs_era5
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.pairs import build_pairs, zone_index
from eccas_s2s.validate.r_bridge import RNotAvailable, check_packages, pairs_to_frame, run_zone_scores
from eccas_s2s.validate.scores import deterministic_scores, tercile_skill
from eccas_s2s.validate.zones import DOMAIN, fraction_positive, zone_masks

#: how the observation of each variable is aggregated over a period.
AGGREGATION = {"precip": "sum", "t2m": "mean", "tmax": "mean", "tmin": "mean"}
#: eligibility: minimum area fraction with positive skill (Draft §3.3: 5-10 %).
MIN_FRACTION = 0.05
DETERMINISTIC_CRITERION = "pearson"
PROBABILISTIC_CRITERION = "rpss"


def skill_dir(cfg) -> Path:
    return cfg.output_root / "skill" / cfg.cycle_id / "raw"


def _cached_integer_grid_archive(cfg, variable: str, ctx=None) -> xr.Dataset:
    """
    Monthly observation archive remapped once onto the 1° grid **centred on
    integer degrees** (the NMME grid), and cached on disk.

    NMME is monthly, so only the monthly archive is needed. Remapping the
    0.05°/0.25° archive once is both cheaper and more faithful than remapping
    every period of every year.
    """
    src = "chirps" if variable == "precip" else "era5"
    path = cfg.data_root / "derived" / "obs" / src / f"{src}_1p0int_months_{variable}.nc"
    if path.exists():
        return xr.open_dataset(path)
    if ctx:
        ctx.log.info("    construction du cache observation sur la grille NMME (%s)", variable)
    if variable == "precip":
        _, months = obs_chirps.load_archives(cfg, "p05")
        da = months
    else:
        _, months = obs_era5.load_archives(cfg, "0p25")
        da = months[variable]
    chunks = [da.isel(time=slice(k, k + 60)).load() for k in range(0, da.sizes["time"], 60)]
    coarse = xr.concat([conservative_to_degree(c, 1.0, target_offset=0.0) for c in chunks],
                       dim="time")
    ds = coarse.astype("float32").to_dataset(name=variable)
    ds.attrs["regridding"] = "conservative to 1 deg centred on integer degrees (NMME grid)"
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding={variable: {"zlib": True, "complevel": 4}})
    tmp.replace(path)
    return xr.open_dataset(path)


def _observation_for(cfg, variable: str, periods, years, model_grid: xr.DataArray,
                     ctx=None) -> xr.DataArray:
    """
    Observed values of the periods, on the 1° grid that matches the model.

    C3S models share the 1° grid (cells centred on half degrees) of the
    pre-computed archives. NMME sits on a 1° grid centred on integer degrees, so
    the monthly archive is remapped conservatively once (cached) and the periods
    are built from it; NMME has no dekads, so nothing is lost. The observation
    covers a smaller area than the model, so it is the **model** that is
    restricted to these cells afterwards.
    """
    integer_centred = bool(np.isclose(float(model_grid["latitude"].values[0]) % 1, 0))
    how = AGGREGATION[variable]
    if integer_centred:
        months = _cached_integer_grid_archive(cfg, variable, ctx)[variable]
        dekads = months.isel(time=slice(0, 0))        # no dekads on this grid
    elif variable == "precip":
        dekads, months = obs_chirps.load_archives(cfg, "1p0")
    else:
        dek_ds, mon_ds = obs_era5.load_archives(cfg, "1p0")
        dekads, months = dek_ds[variable], mon_ds[variable]
    return obs_period_totals(dekads, months, periods, years, how=how).load()


def _load_hindcast(cfg, system: str, model: str, variable: str) -> xr.DataArray:
    """Hindcast period values of one model (C3S keeps its members, NMME is an ensemble mean)."""
    if system == "c3s":
        return c3s_totals.load_totals(cfg, model, "hindcast", variable)
    return nmme_totals.load_totals(cfg, model, variable, "hindcast")


def run(config: str, systems=("c3s", "nmme"), variables=("precip",), models=None,
        scales=None, min_fraction: float = MIN_FRACTION, skip_r: bool = False) -> RunContext:
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    scales = list(scales) if scales else cfg.scales

    with RunContext(cfg, step="skill_raw") as ctx:
        ctx.record_parameter("systems", list(systems))
        ctx.record_parameter("variables", list(variables))
        ctx.record_parameter("scales", scales)
        ctx.record_parameter("min_fraction", min_fraction)
        ctx.record_parameter("cross_validation", cfg.cv_scheme)
        if not skip_r:
            try:
                ctx.record_parameter("r", check_packages())
            except RNotAvailable as exc:
                ctx.warn(f"scores de zone désactivés : {exc}")
                skip_r = True

        summary, eligibility = [], []
        for system in systems:
            if system == "c3s":
                candidates = list(cfg.c3s_models)
            else:
                candidates = list(cfg.raw["systems"]["nmme"]["models"])
            selected = [m for m in candidates if models is None or m in models]

            for model in selected:
                for variable in variables:
                    if system == "nmme" and variable in ("tmax", "tmin"):
                        continue                       # NMME is monthly: no daily extremes
                    try:
                        hind = _load_hindcast(cfg, system, model, variable).load()
                    except (FileNotFoundError, KeyError) as exc:
                        ctx.warn(f"{system} {model} {variable} : données absentes ({exc})")
                        continue

                    keep = [p for p in hind["period"].values
                            if str(hind["scale"].sel(period=p).values) in scales]
                    hind = hind.sel(period=keep)
                    periods = build_periods(cfg.init_date, max(
                        (cfg.c3s_models[model].max_lead_days if system == "c3s" else 400), 1),
                        scales=tuple(scales))
                    periods = [p for p in periods if p.key in set(keep)]
                    years = [int(y) for y in hind["year"].values]
                    ctx.log.info("%s %s %s : %d périodes, %d années", system, model, variable,
                                 len(periods), len(years))

                    obs = _observation_for(cfg, variable, periods, years,
                                           hind.isel(period=0), ctx)
                    obs = obs.sel(period=[p.key for p in periods])
                    hind = match_model_grid(hind.sel(period=[p.key for p in periods]),
                                            obs.isel(year=0, period=0, drop=True))
                    pairs = build_pairs(hind, obs)

                    det = deterministic_scores(pairs["ensmean"], pairs["obs"])
                    prob = tercile_skill(pairs["prob"], pairs["obs_cat"])
                    maps = xr.merge([det, prob])
                    maps = maps.assign_coords(
                        label=("period", [p.label(cfg.init_date.year) for p in periods]),
                        scale=("period", [p.scale for p in periods]))
                    maps.attrs.update({**ctx.netcdf_attrs(), "system": system, "model": model,
                                       "variable": variable, "kind": "raw hindcast skill",
                                       "n_years": pairs.attrs["n_years"],
                                       "n_members": pairs.attrs["n_members"],
                                       "cross_validation": cfg.cv_scheme})
                    path = out / "maps" / f"{system}_{model}_{variable}_skill.nc"
                    tmp = path.with_suffix(".tmp.nc")
                    maps.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4}
                                                  for v in maps.data_vars})
                    tmp.replace(path)
                    ctx.record_output(path, role="skill_maps", system=system, model=model,
                                      variable=variable)

                    masks = zone_masks(pairs["obs"], cfg.domains,
                                       valid=pairs["obs"].notnull().all("year").any("period"))
                    for p in periods:
                        row = {"system": system, "model": model, "variable": variable,
                               "scale": p.scale, "period": p.key,
                               "label": p.label(cfg.init_date.year),
                               "lead_month": p.month_offset, "n_years": pairs.attrs["n_years"],
                               "n_members": pairs.attrs["n_members"]}
                        for score in ("pearson", "spearman", "acc", "bias", "rmse", "msess", "rpss"):
                            row[f"{score}_domain_median"] = float(
                                maps[score].sel(period=p.key).where(masks[DOMAIN]).median())
                        row["frac_pearson_positive"] = fraction_positive(
                            maps[DETERMINISTIC_CRITERION].sel(period=p.key), masks[DOMAIN])
                        row["frac_rpss_positive"] = fraction_positive(
                            maps[PROBABILISTIC_CRITERION].sel(period=p.key), masks[DOMAIN])
                        for cat in maps["category"].values:
                            row[f"roc_{cat}_domain_median"] = float(
                                maps["roc_area"].sel(period=p.key, category=cat)
                                .where(masks[DOMAIN]).median())
                        row["eligible"] = bool(row["frac_pearson_positive"] >= min_fraction
                                               and row["frac_rpss_positive"] >= min_fraction)
                        summary.append(row)

                    if not skip_r:
                        for zone, mask in masks.items():
                            idx = zone_index(pairs, mask)
                            frame = pairs_to_frame(idx)
                            zdir = out / "zones" / f"{system}_{model}_{variable}" / zone
                            try:
                                run_zone_scores(frame, zdir, f"{system}|{model}|{variable}|{zone}")
                            except RNotAvailable as exc:
                                ctx.warn(f"scores R {system} {model} {variable} {zone} : {exc}")
                    del hind, obs, pairs, maps
                    gc.collect()

        table = pd.DataFrame(summary)
        if not table.empty:
            f = out / "skill_raw_summary.csv"
            table.to_csv(f, index=False)
            ctx.record_output(f, role="summary")
            elig = (table.groupby(["system", "model", "variable", "scale"])["eligible"]
                    .agg(["sum", "count"]).reset_index()
                    .rename(columns={"sum": "periods_eligible", "count": "periods"}))
            elig["eligible"] = elig["periods_eligible"] > 0
            reg = cfg.path_of("output_root") / "registry"
            reg.mkdir(parents=True, exist_ok=True)
            f = reg / "models_eligibility.csv"
            elig.to_csv(f, index=False)
            ctx.record_output(f, role="eligibility")
            ctx.record_parameter("n_rows", len(table))
            eligibility = elig.to_dict("records")
        ctx.record_parameter("eligibility", eligibility)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--systems", nargs="+", default=["c3s", "nmme"], choices=["c3s", "nmme"])
    ap.add_argument("--variables", nargs="+", default=["precip"],
                    choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--min-fraction", type=float, default=MIN_FRACTION)
    ap.add_argument("--skip-r", action="store_true", help="ne pas calculer les scores de zone en R")
    args = ap.parse_args(argv)
    run(args.config, args.systems, args.variables, args.models, args.scales,
        args.min_fraction, args.skip_r)


if __name__ == "__main__":
    main()
