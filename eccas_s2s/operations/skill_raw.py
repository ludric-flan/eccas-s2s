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
  least one deterministic score **and** one probabilistic score; a system
  delivered as an ensemble mean (NMME) has no raw probability and is judged on
  the deterministic criterion alone.

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
#: model variable -> name of the same variable in the observation archives
#: (the ERA5 archive calls the mean temperature "tmean").
OBS_VARIABLE = {"precip": "precip", "t2m": "tmean", "tmax": "tmax", "tmin": "tmin"}
#: scales each system can be scored on. NMME is distributed on the NOAA/CPC
#: server as *monthly ensemble means*: there is no daily field (hence no dekad)
#: and no member (hence no raw probability — only a calibration, phase P3, can
#: give NMME probabilities).
SYSTEM_SCALES = {"c3s": ("decade", "month", "season"), "nmme": ("month", "season")}
#: eligibility: minimum area fraction with positive skill (Draft §3.3: 5-10 %).
MIN_FRACTION = 0.05
DETERMINISTIC_CRITERION = "pearson"
PROBABILISTIC_CRITERION = "rpss"


def split_skill_name(name: str) -> tuple[str, str, str]:
    """
    ``<system>_<model>_<variable>`` from a file or directory name.

    Naive splitting breaks on the model names that hold an underscore
    (``meteo_france``, ``NASA_GEOS5v2``, ``GEM5.2_NEMO``): the system is taken
    from the left, the variable from the right, the rest is the model.
    """
    system, rest = name.replace("_skill", "").split("_", 1)
    model, variable = rest.rsplit("_", 1)
    return system, model, variable


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
    name = OBS_VARIABLE[variable]
    path = cfg.data_root / "derived" / "obs" / src / f"{src}_1p0int_months_{name}.nc"
    if path.exists():
        return xr.open_dataset(path)
    if ctx:
        ctx.log.info("    construction du cache observation sur la grille NMME (%s)", variable)
    if variable == "precip":
        _, months = obs_chirps.load_archives(cfg, "p05")
        da = months
    else:
        _, months = obs_era5.load_archives(cfg, "0p25")
        da = months[name]
    chunks = [da.isel(time=slice(k, k + 60)).load() for k in range(0, da.sizes["time"], 60)]
    coarse = xr.concat([conservative_to_degree(c, 1.0, target_offset=0.0) for c in chunks],
                       dim="time")
    ds = coarse.astype("float32").to_dataset(name=name)
    ds.attrs["regridding"] = "conservative to 1 deg centred on integer degrees (NMME grid)"
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding={name: {"zlib": True, "complevel": 4}})
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
    name = OBS_VARIABLE[variable]
    if integer_centred:
        months = _cached_integer_grid_archive(cfg, variable, ctx)[name]
        dekads = months.isel(time=slice(0, 0))        # no dekads on this grid
    elif variable == "precip":
        dekads, months = obs_chirps.load_archives(cfg, "1p0")
    else:
        dek_ds, mon_ds = obs_era5.load_archives(cfg, "1p0")
        dekads, months = dek_ds[name], mon_ds[name]
    return obs_period_totals(dekads, months, periods, years, how=how).load()


def _load_hindcast(cfg, system: str, model: str, variable: str) -> xr.DataArray:
    """Hindcast period values of one model (C3S keeps its members, NMME is an ensemble mean)."""
    if system == "c3s":
        return c3s_totals.load_totals(cfg, model, "hindcast", variable)
    return nmme_totals.load_totals(cfg, model, variable, "hindcast")


def prepare_pairs(cfg, system: str, model: str, variable: str, scales, ctx=None):
    """
    Forecast/observation pairs of one model and variable, and their periods.

    Shared by the scores (this module) and the diagrams
    (:mod:`eccas_s2s.operations.skill_diagrams`) so both read exactly the same
    hindcast, the same observation and the same leave-one-year-out categories.
    """
    scales = [s for s in scales if s in SYSTEM_SCALES[system]]
    hind = _load_hindcast(cfg, system, model, variable).load()
    keep = [p for p in hind["period"].values
            if str(hind["scale"].sel(period=p).values) in scales]
    hind = hind.sel(period=keep)
    periods = build_periods(
        cfg.init_date,
        max((cfg.c3s_models[model].max_lead_days if system == "c3s" else 400), 1),
        scales=tuple(scales))
    periods = [p for p in periods if p.key in set(keep)]
    years = [int(y) for y in hind["year"].values]
    obs = _observation_for(cfg, variable, periods, years, hind.isel(period=0), ctx)
    obs = obs.sel(period=[p.key for p in periods])
    hind = match_model_grid(hind.sel(period=[p.key for p in periods]),
                            obs.isel(year=0, period=0, drop=True))
    return build_pairs(hind, obs), periods


def summary_rows(maps: xr.Dataset, masks: dict, system: str, model: str, variable: str,
                 lead_of: dict, min_fraction: float = MIN_FRACTION) -> list[dict]:
    """
    One summary row per period: domain medians, positive-skill fractions, eligibility.

    ``lead_of`` maps a period key to its lead in months. A system without
    members carries no probabilistic column at all, and its eligibility
    (Draft §3.3) rests on the deterministic criterion alone.
    """
    has_prob = PROBABILISTIC_CRITERION in maps
    rows = []
    for key in [str(k) for k in maps["period"].values]:
        row = {"system": system, "model": model, "variable": variable,
               "scale": str(maps["scale"].sel(period=key).values),
               "period": key, "label": str(maps["label"].sel(period=key).values),
               "lead_month": lead_of.get(key), "n_years": maps.attrs.get("n_years"),
               "n_members": maps.attrs.get("n_members"), "probabilistic": has_prob}
        scores = ["pearson", "spearman", "acc", "bias", "rmse", "msess"]
        scores += ["rpss"] if has_prob else []
        for score in scores:
            row[f"{score}_domain_median"] = float(
                maps[score].sel(period=key).where(masks[DOMAIN]).median())
        row["frac_pearson_positive"] = fraction_positive(
            maps[DETERMINISTIC_CRITERION].sel(period=key), masks[DOMAIN])
        if has_prob:
            row["frac_rpss_positive"] = fraction_positive(
                maps[PROBABILISTIC_CRITERION].sel(period=key), masks[DOMAIN])
            for cat in maps["category"].values:
                row[f"roc_{cat}_domain_median"] = float(
                    maps["roc_area"].sel(period=key, category=cat)
                    .where(masks[DOMAIN]).median())
        row["eligible"] = bool(row["frac_pearson_positive"] >= min_fraction
                               and (not has_prob or row["frac_rpss_positive"] >= min_fraction))
        rows.append(row)
    return rows


def _write_summary(cfg, ctx, out: Path, rows: list[dict]) -> list[dict]:
    """Merge the new rows into ``skill_raw_summary.csv`` and refresh the eligibility register."""
    table = pd.DataFrame(rows)
    if table.empty:
        return []
    f = out / "skill_raw_summary.csv"
    keys = ["system", "model", "variable", "period"]
    if f.exists():              # keep the rows of the runs not repeated here
        old = pd.read_csv(f)
        old = old[~old.set_index(keys).index.isin(table.set_index(keys).index)]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["system", "model", "variable", "scale", "period"])
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
    return elig.to_dict("records")


def rebuild_summary(config: str, min_fraction: float = MIN_FRACTION) -> RunContext:
    """
    Rebuild the summary table and the eligibility register from the archived maps.

    The scores live in ``maps/*_skill.nc``; the table is only their domain
    summary. Rebuilding it costs seconds and avoids recomputing everything when
    a run is interrupted after the maps are written.
    """
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    with RunContext(cfg, step="skill_raw_summary") as ctx:
        rows = []
        for f in sorted((out / "maps").glob("*_skill.nc")):
            system, model, variable = split_skill_name(f.stem)
            with xr.open_dataset(f) as maps:
                maps = maps.load()
            ctx.record_input(f, role="skill_maps")
            valid = maps[DETERMINISTIC_CRITERION].notnull().any("period")
            masks = zone_masks(maps[DETERMINISTIC_CRITERION], cfg.domains, valid=valid)
            lead_of = {str(k): int(str(k).split("_")[-1].lstrip("msd") or 0)
                       for k in maps["period"].values}
            rows += summary_rows(maps, masks, system, model, variable, lead_of, min_fraction)
            ctx.log.info("%s %s %s : %d période(s)", system, model, variable,
                         maps.sizes["period"])
        ctx.record_parameter("eligibility", _write_summary(cfg, ctx, out, rows))
    return ctx


def rescore_zones(config: str) -> RunContext:
    """
    Recompute the zone scores in R from the ``pairs.csv`` already written.

    The couples are the expensive part; they are archived next to the scores, so
    a change in the R script (a corrected Brier decomposition, a new score) is
    replayed in seconds instead of rereading every hindcast.
    """
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    with RunContext(cfg, step="skill_raw_zones") as ctx:
        ctx.record_parameter("r_version", check_packages()["r_version"])
        for pairs_csv in sorted((out / "zones").glob("*/*/pairs.csv")):
            zdir = pairs_csv.parent
            zone = zdir.name
            system, model, variable = split_skill_name(zdir.parent.name)
            frame = pd.read_csv(pairs_csv)
            ctx.record_input(pairs_csv, role="pairs")
            try:
                run_zone_scores(frame, zdir, f"{system}|{model}|{variable}|{zone}")
            except RNotAvailable as exc:
                ctx.warn(f"scores R {zdir} : {exc}")
                continue
            ctx.log.info("%s %s %s %s : %d périodes", system, model, variable, zone,
                         frame["period"].nunique())
            for name in ("deterministic_scores.csv", "tercile_scores.csv",
                         "category_scores.csv", "reliability_bins.csv"):
                if (zdir / name).exists():
                    ctx.record_output(zdir / name, role="zone_scores", system=system,
                                      model=model, variable=variable, zone=zone)
    return ctx


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

            # NMME is distributed as monthly ensemble means: no daily field, hence
            # no dekad and no daily extremes (tmax/tmin), and no member either.
            sys_scales = [s for s in scales if s in SYSTEM_SCALES[system]]
            if not sys_scales:
                ctx.warn(f"{system} : aucune échelle demandée n'est disponible "
                         f"(disponibles : {', '.join(SYSTEM_SCALES[system])})")
                continue

            for model in selected:
                for variable in variables:
                    if system == "nmme" and variable in ("tmax", "tmin"):
                        continue                       # NMME is monthly: no daily extremes
                    try:
                        pairs, periods = prepare_pairs(cfg, system, model, variable,
                                                       sys_scales, ctx)
                    except (FileNotFoundError, KeyError) as exc:
                        ctx.warn(f"{system} {model} {variable} : données absentes ({exc})")
                        continue
                    ctx.log.info("%s %s %s : %d périodes, %d années", system, model, variable,
                                 len(periods), pairs.attrs["n_years"])

                    det = deterministic_scores(pairs["ensmean"], pairs["obs"])
                    has_prob = "prob" in pairs
                    if has_prob:
                        prob = tercile_skill(pairs["prob"], pairs["obs_cat"])
                        # both carry their own sample size; keep them apart
                        maps = xr.merge([det, prob.rename({"n_years": "n_years_prob"})])
                    else:
                        maps = det          # ensemble mean only: deterministic scores only
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
                    summary += summary_rows(maps, masks, system, model, variable,
                                            {p.key: p.month_offset for p in periods},
                                            min_fraction)

                    if not skip_r:
                        for zone, mask in masks.items():
                            idx = zone_index(pairs, mask)
                            frame = pairs_to_frame(idx)
                            zdir = out / "zones" / f"{system}_{model}_{variable}" / zone
                            try:
                                run_zone_scores(frame, zdir, f"{system}|{model}|{variable}|{zone}")
                            except RNotAvailable as exc:
                                ctx.warn(f"scores R {system} {model} {variable} {zone} : {exc}")
                    del pairs, maps
                    gc.collect()

        eligibility = _write_summary(cfg, ctx, out, summary) or eligibility
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
    ap.add_argument("--from-maps", action="store_true",
                    help="reconstruire seulement le tableau de synthèse à partir des cartes déjà écrites")
    ap.add_argument("--from-pairs", action="store_true",
                    help="recalculer seulement les scores de zone en R depuis les pairs.csv archivés")
    args = ap.parse_args(argv)
    if args.from_maps:
        rebuild_summary(args.config, args.min_fraction)
        return
    if args.from_pairs:
        rescore_zones(args.config)
        return
    run(args.config, args.systems, args.variables, args.models, args.scales,
        args.min_fraction, args.skip_r)


if __name__ == "__main__":
    main()
