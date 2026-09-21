"""
Raw hindcast skill of every model (workflow step E4, Draft §3.3).

For each system (C3S, NMME), model, variable and period, the raw forecast is
compared with the observation over the whole hindcast, leave-one-year-out:

* **maps** (Python, :mod:`eccas_s2s.validate.scores`): bias, MAE, RMSE, MSESS,
  Pearson, Spearman, ACC, RPS/RPSS, and per category the Brier skill score and
  the ROC area;
* **eligibility** (§3.3): a model is eligible for a variable, scale and period
  when its skill is positive on at least ``min_fraction`` of the domain for at
  least one deterministic score **and** one probabilistic score; a system
  delivered as an ensemble mean (NMME) has no raw probability and is judged on
  the deterministic criterion alone.

Everything is raw: no bias correction, no calibration. These numbers are the
reference that the calibration of phase P3 must beat.

Everything is restricted to the **CEEAC land mask** built from the shapefile
(:mod:`eccas_s2s.core.geo`), exactly as the product maps of the reference chain:
a score is computed, mapped and summarised on the same cells, and the ocean
never enters a median or a positive-skill fraction. Scores being computed grid
point by grid point, there is no zone averaging any more — a map says more than
a zone index, and the pooled diagrams (:mod:`eccas_s2s.operations.skill_diagrams`)
cover what a zone score used to give.

Outputs in ``<output_root>/skill/<YYYYMM>/raw/``, three trees with the same
branches ``<system>_<model>/<scale>/<variable>/``::

    netcdf/…/<metric>.nc              scores on the grid, one file per metric
    figures/…/<metric>/<period>.png   one map per period (skill_diagrams: diagrams/)
    skill_raw_summary.csv             domain medians and eligibility

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

from eccas_s2s.core.geo import fraction_above, mask_like
from eccas_s2s.core.periods import build_periods
from eccas_s2s.obs.climatology import obs_period_totals
from eccas_s2s.obs.regrid import conservative_to_degree, match_model_grid
from eccas_s2s.operations import c3s_totals, nmme_totals, obs_chirps, obs_era5
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.pairs import build_pairs
from eccas_s2s.validate.scores import deterministic_scores, tercile_skill

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
#: scores published as maps and as netCDF (decision: one file per metric).
METRICS = ("pearson", "spearman", "acc", "bias", "rmse", "msess",
           "rpss", "bss", "roc_area", "groc")
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

    The observation is restricted to the CEEAC mask **before** the pairs are
    built, so every score, map and diagram covers the same cells.
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
    mask = mask_like(cfg.raw["paths"]["shapefile"], obs.isel(year=0, period=0, drop=True))
    if ctx:
        ctx.log.info("    masque CEEAC : %d mailles sur %d", int(mask.sum()), int(mask.size))
    obs = obs.where(mask)
    pairs = build_pairs(hind, obs)
    pairs.attrs["mask_cells"] = int(mask.sum())
    return pairs, periods


def score_paths(out: Path, system: str, model: str, variable: str, scale: str) -> Path:
    """``<tree>/<system>_<model>/<scale>/<variable>/`` — the branch shared by every output."""
    return Path(out) / f"{system}_{model}" / scale / variable


def write_score_netcdf(maps: xr.Dataset, out: Path, system: str, model: str,
                       variable: str, scale: str, ctx=None) -> list[Path]:
    """
    One netCDF per metric, with explicit coordinates and units.

    Splitting the metrics makes each file self-describing (a reader opens
    ``rpss.nc`` and gets the RPSS, its periods and its no-skill value) and lets a
    later phase add a metric without rewriting the others. Dimensions are
    ``(period, latitude, longitude)`` plus ``category`` for the per-category
    scores; ``period`` carries its key, its English label, its French label and
    its scale.
    """
    folder = score_paths(out, system, model, variable, scale)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name in maps.data_vars:
        ds = maps[name].to_dataset(name=name)
        ds["latitude"].attrs.update(units="degrees_north", standard_name="latitude")
        ds["longitude"].attrs.update(units="degrees_east", standard_name="longitude")
        ds.attrs.update({**maps.attrs, "metric": str(name), "scale": scale})
        path = folder / f"{name}.nc"
        tmp = path.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp, encoding={name: {"zlib": True, "complevel": 4}})
        tmp.replace(path)
        written.append(path)
        if ctx:
            ctx.record_output(path, role="skill_netcdf", system=system, model=model,
                              variable=variable, scale=scale, metric=str(name))
    return written


def summary_rows(maps: xr.Dataset, mask: xr.DataArray, system: str, model: str, variable: str,
                 lead_of: dict, min_fraction: float = MIN_FRACTION) -> list[dict]:
    """
    One summary row per period: medians over the CEEAC mask and eligibility.

    A system without members carries no probabilistic column at all, and its
    eligibility (Draft §3.3) rests on the deterministic criterion alone.
    """
    has_prob = PROBABILISTIC_CRITERION in maps
    rows = []
    for key in [str(k) for k in maps["period"].values]:
        row = {"system": system, "model": model, "variable": variable,
               "scale": str(maps["scale"].sel(period=key).values),
               "period": key, "label": str(maps["label"].sel(period=key).values),
               "label_fr": (str(maps["label_fr"].sel(period=key).values)
                            if "label_fr" in maps.coords else ""),
               "lead_month": lead_of.get(key), "n_years": maps.attrs.get("n_years"),
               "n_members": maps.attrs.get("n_members"), "probabilistic": has_prob}
        for score in [m for m in METRICS if m in maps and "category" not in maps[m].dims]:
            row[f"{score}_median"] = float(maps[score].sel(period=key).where(mask).median())
        row["frac_pearson_positive"] = fraction_above(
            maps[DETERMINISTIC_CRITERION].sel(period=key), mask)
        if has_prob:
            row["frac_rpss_positive"] = fraction_above(
                maps[PROBABILISTIC_CRITERION].sel(period=key), mask)
            row["frac_groc_useful"] = fraction_above(maps["groc"].sel(period=key), mask, 0.5)
            for cat in maps["category"].values:
                row[f"roc_{cat}_median"] = float(
                    maps["roc_area"].sel(period=key, category=cat).where(mask).median())
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
    Rebuild the summary table and the eligibility register from the netCDF scores.

    The scores live in ``netcdf/<system>_<model>/<scale>/<variable>/<metric>.nc``;
    the table is only their summary over the mask, so rebuilding it costs seconds
    when a run is interrupted after the scores are written.
    """
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    shapefile = cfg.raw["paths"]["shapefile"]
    with RunContext(cfg, step="skill_raw_summary") as ctx:
        rows = []
        for folder in sorted((out / "netcdf").glob("*/*/*")):
            files = sorted(folder.glob("*.nc"))
            if not files:
                continue
            system, model = folder.parent.parent.name.split("_", 1)
            scale, variable = folder.parent.name, folder.name
            maps = xr.merge([xr.open_dataset(f) for f in files],
                            combine_attrs="override").load()
            for f in files:
                ctx.record_input(f, role="skill_netcdf")
            mask = mask_like(shapefile, maps[DETERMINISTIC_CRITERION].isel(period=0, drop=True))
            lead_of = {str(k): int(str(k).split("_")[1].lstrip("m") or 0)
                       for k in maps["period"].values}
            rows += summary_rows(maps, mask, system, model, variable, lead_of, min_fraction)
            ctx.log.info("%s %s %s %s : %d période(s)", system, model, variable, scale,
                         maps.sizes["period"])
        ctx.record_parameter("eligibility", _write_summary(cfg, ctx, out, rows))
    return ctx


def run(config: str, systems=("c3s", "nmme"), variables=("precip",), models=None,
        scales=None, min_fraction: float = MIN_FRACTION) -> RunContext:
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    scales = list(scales) if scales else cfg.scales

    with RunContext(cfg, step="skill_raw") as ctx:
        ctx.record_parameter("systems", list(systems))
        ctx.record_parameter("variables", list(variables))
        ctx.record_parameter("scales", scales)
        ctx.record_parameter("min_fraction", min_fraction)
        ctx.record_parameter("cross_validation", cfg.cv_scheme)
        ctx.record_parameter("mask", str(cfg.raw["paths"]["shapefile"]))

        summary, eligibility = [], []
        for system in systems:
            candidates = (list(cfg.c3s_models) if system == "c3s"
                          else list(cfg.raw["systems"]["nmme"]["models"]))
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
                        label_fr=("period", [p.label_fr(cfg.init_date.year, with_dates=True)
                                             for p in periods]),
                        scale=("period", [p.scale for p in periods]))
                    maps.attrs.update({**ctx.netcdf_attrs(), "system": system, "model": model,
                                       "variable": variable, "kind": "raw hindcast skill",
                                       "n_years": pairs.attrs["n_years"],
                                       "n_members": pairs.attrs["n_members"],
                                       "mask": "CEEAC (shapefile)",
                                       "mask_cells": pairs.attrs.get("mask_cells"),
                                       "init_date": str(cfg.init_date.date()),
                                       "cross_validation": cfg.cv_scheme})

                    mask = pairs["obs"].notnull().any(["year", "period"])
                    for scale in sorted({p.scale for p in periods}):
                        keys = [p.key for p in periods if p.scale == scale]
                        write_score_netcdf(maps.sel(period=keys), out / "netcdf", system,
                                           model, variable, scale, ctx)
                    summary += summary_rows(maps, mask, system, model, variable,
                                            {p.key: p.month_offset for p in periods},
                                            min_fraction)
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
    ap.add_argument("--from-maps", action="store_true",
                    help="reconstruire seulement le tableau de synthèse à partir des cartes déjà écrites")
    args = ap.parse_args(argv)
    if args.from_maps:
        rebuild_summary(args.config, args.min_fraction)
        return
    run(args.config, args.systems, args.variables, args.models, args.scales,
        args.min_fraction)


if __name__ == "__main__":
    main()
