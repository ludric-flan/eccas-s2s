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
from eccas_s2s.io.netcdf import open_cf, save as save_cf
from eccas_s2s.core.periods import (add_period_arguments, announce_subset,
                                    selected_periods)
from eccas_s2s.obs.climatology import obs_period_totals
from eccas_s2s.obs.regrid import conservative_to_degree, match_model_grid
from eccas_s2s.operations import c3s_totals, nmme_totals, obs_chirps, obs_era5
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.calibrate.base import RawEnsemble
from eccas_s2s.products.catalogue import VALUE
from eccas_s2s.validate.pairs import build_pairs
from eccas_s2s.validate.product_scores import ObservationContext, product_maps

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


def horizon_days(cfg, system: str, model: str) -> int:
    """
    Forecast horizon a system is processed over.

    A C3S model is processed over the horizon it actually delivers. NMME is
    capped at the **longest C3S horizon** (decision of 23/09/2026): it publishes
    twice as far ahead, but a period no C3S model reaches cannot enter the
    multi-model, and calibrating it would double the computation for a product
    that would stand alone.
    """
    if system == "c3s":
        return max(int(cfg.c3s_models[model].max_lead_days), 1)
    return max(int(m.max_lead_days) for m in cfg.c3s_models.values())


def skill_dir(cfg, kind: str = "raw") -> Path:
    """
    Where the scores of a family of forecasts live.

    ``kind`` is ``raw`` for the hindcasts as delivered (phase P2) and
    ``calibrated`` for the output of phase P3: the two trees have the same shape,
    which is what lets the same maps and diagrams describe both.
    """
    return cfg.output_root / "skill" / cfg.cycle_id / kind


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
        _, months = obs_chirps.load_archives(cfg)
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


def hindcast_streams(cfg, system: str, model: str, variable: str, scales) -> dict[str, list[str]]:
    """
    Which scales are read from which totals file.

    Only the C3S rainfall is split: UKMO, BoM and NCEP take their months and
    seasons from the monthly archive, where their lagged members are, and their
    dekads from the daily one (decision of 23/09/2026). The two files hold
    **different ensembles** — 28 members against 7 for UKMO — so they are scored
    separately, each with its own member count in the netCDF attributes and in
    the summary. Everything else has a single file per model and variable.
    """
    if system != "c3s" or variable != "precip":
        return {"daily": [s for s in scales if s in SYSTEM_SCALES[system]]}
    return c3s_totals.streams_for(cfg.c3s_models[model], scales)


def _load_hindcast(cfg, system: str, model: str, variable: str,
                   stream: str = "daily") -> xr.DataArray:
    """Hindcast period values of one model (C3S keeps its members, NMME is an ensemble mean)."""
    if system == "c3s":
        return c3s_totals.load_totals(cfg, model, "hindcast", variable, stream)
    return nmme_totals.load_totals(cfg, model, variable, "hindcast")


def prepare_pairs(cfg, system: str, model: str, variable: str, scales, ctx=None,
                  selection=None, stream: str = "daily"):
    """
    Forecast/observation pairs of one model and variable, and their periods.

    Shared by the scores (this module) and the diagrams
    (:mod:`eccas_s2s.operations.skill_diagrams`) so both read exactly the same
    hindcast, the same observation and the same leave-one-year-out categories.

    The observation is restricted to the CEEAC mask **before** the pairs are
    built, so every score, map and diagram covers the same cells.
    """
    scales = [s for s in scales if s in SYSTEM_SCALES[system]]
    hind = _load_hindcast(cfg, system, model, variable, stream).load()
    keep = [p for p in hind["period"].values
            if str(hind["scale"].sel(period=p).values) in scales]
    hind = hind.sel(period=keep)
    periods = cfg.periods_for(horizon_days(cfg, system, model), tuple(scales),
                              selection=selection)
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
    pairs.attrs["stream"] = stream
    return pairs, periods


def score_paths(out: Path, system: str, model: str, variable: str, scale: str,
                product: str | None = None) -> Path:
    """
    ``<tree>/<system>_<model>/<scale>/<variable>[/<product>]/`` — the shared branch.

    The product level is what makes the tree readable: a forecaster looking for
    "can I trust the 200 mm map for OND?" opens
    ``…/season/precip/depassement_200mm/bss/`` and finds exactly that, instead of
    a folder of metrics that says nothing about which product they judge.
    """
    base = Path(out) / f"{system}_{model}" / scale / variable
    return base / product if product else base


def write_product_netcdf(maps: xr.Dataset, out: Path, system: str, model: str, variable: str,
                         scale: str, product: str, ctx=None) -> list[Path]:
    """
    One netCDF per metric inside the product's folder, with explicit coordinates.

    Splitting the metrics keeps each file self-describing (a reader opens
    ``rpss.nc`` and gets the RPSS of that product, its periods and its no-skill
    value). Dimensions are ``(period, latitude, longitude)``, plus ``category``
    for the per-category ROC area; ``period`` carries its key, its English label,
    its French label and its scale.
    """
    folder = score_paths(out, system, model, variable, scale, product)
    folder.mkdir(parents=True, exist_ok=True)
    init = pd.Timestamp(maps.attrs.get("init_date", "1970-01-01"))
    written = []
    for name in maps.data_vars:
        ds = maps[name].to_dataset(name=name)
        ds.attrs.update({**maps.attrs, "metric": str(name), "scale": scale, "product": product})
        # written CF-1.8: a real time axis, numeric category flags and character
        # strings, so the file opens in CDO, ncview or QGIS as well as in xarray
        path = save_cf(ds, folder / f"{name}.nc", init_year=init.year, init_month=init.month)
        written.append(path)
        if ctx:
            ctx.record_output(path, role="skill_netcdf", system=system, model=model,
                              variable=variable, scale=scale, product=product, metric=str(name))
    return written


def product_summary_rows(maps: xr.Dataset, mask: xr.DataArray, system: str, model: str,
                         variable: str, scale: str, product: str, lead_of: dict,
                         stream: str = "daily") -> list[dict]:
    """
    One row per period and metric: median over the CEEAC mask and positive fraction.

    Two flags carry the meaning of each row. ``deciding`` marks the metric that
    answers "is this product usable as delivered?"; ``discrimination`` marks the
    one that answers "is there signal a calibration could recover?". The
    eligibility register is built from those two alone.
    """
    from eccas_s2s.validate.product_scores import (DISCRIMINATION_OF, LOWER_IS_BETTER,
                                                   NO_SKILL_VALUE)

    deciding = maps.attrs.get("deciding", "")
    discrimination = DISCRIMINATION_OF.get(maps.attrs.get("kind", ""), "")
    rows = []
    for key in [str(k) for k in maps["period"].values]:
        common = {"system": system, "model": model, "variable": variable, "stream": stream,
                  "scale": scale, "period": key,
                  "label": str(maps["label"].sel(period=key).values),
                  "label_fr": (str(maps["label_fr"].sel(period=key).values)
                               if "label_fr" in maps.coords else ""),
                  "lead_month": lead_of.get(key),
                  "product": product, "product_label": maps.attrs.get("label", product),
                  "kind": maps.attrs.get("kind", ""),
                  "n_years": int(maps["n_years"].sel(period=key).max()) if "n_years" in maps else None,
                  "n_members": maps.attrs.get("n_members"),
                  "own_climatology": maps.attrs.get("own_climatology")}
        for name, field in maps.data_vars.items():
            if name in ("n_years", "base_rate"):
                continue
            slices = ([(f"{name}_{c}", field.sel(period=key, category=c))
                       for c in field["category"].values] if "category" in field.dims
                      else [(name, field.sel(period=key))])
            for metric, values in slices:
                base = metric.split("_")[0]
                good_when_positive = base not in LOWER_IS_BETTER
                # an area under the ROC curve has no skill at 0.5, not at 0
                no_skill = NO_SKILL_VALUE.get(name, NO_SKILL_VALUE.get(base, 0.0))
                rows.append({**common, "metric": metric,
                             "deciding": metric == deciding,
                             "discrimination": name == discrimination,
                             "no_skill_value": no_skill,
                             "median": float(values.where(mask).median()),
                             "fraction_positive": (fraction_above(values, mask, no_skill)
                                                   if good_when_positive else np.nan)})
    return rows


def eligibility_table(table: pd.DataFrame, min_fraction: float = MIN_FRACTION) -> pd.DataFrame:
    """
    Which models the workflow can work with, product by product (Draft §3.3).

    The question this register answers is **admission**, not quality: can this
    model be used at all? The criterion is therefore the **discrimination** — the
    GROC of a categorical product, the ROC skill of an event, the correlation of
    a value. A model that cannot separate the years it should from the years it
    should not carries no information, and no calibration can create any; a model
    that discriminates but is overconfident or biased is *kept*, because that is
    precisely what the calibration of phase P3 repairs.

    It is perfectly normal for every model to be admitted. Admission does not say
    the product will be used **raw**: the column ``usable_raw`` reports, for
    information, whether the deciding metric is positive on ``min_fraction`` of
    the mask — and the register of phase P3 will say which calibration to apply.

    Three states:

    ``exploitable — utilisable brut``
        discrimination present, and the deciding metric positive too;
    ``exploitable — à calibrer``
        discrimination present, deciding metric not: the signal is there, the
        probabilities or the values are not usable as they stand;
    ``non exploitable``
        no discrimination on any period — the model is dropped for this product.
    """
    keys = ["system", "model", "variable", "stream", "scale", "product", "product_label"]
    deciding = (table[table["deciding"]].groupby(keys + ["period"])["fraction_positive"]
                .max().reset_index().rename(columns={"fraction_positive": "frac_deciding"}))
    signal = (table[table["discrimination"]].groupby(keys + ["period"])["fraction_positive"]
              .max().reset_index().rename(columns={"fraction_positive": "frac_discrimination"}))
    per_period = signal.merge(deciding, on=keys + ["period"], how="outer")
    per_period["has_signal"] = per_period["frac_discrimination"] >= min_fraction
    per_period["usable_raw"] = per_period["frac_deciding"] >= min_fraction

    agg = (per_period.groupby(keys)
           .agg(periods=("period", "count"), periods_signal=("has_signal", "sum"),
                periods_usable_raw=("usable_raw", "sum"),
                median_frac_discrimination=("frac_discrimination", "median"),
                median_frac_deciding=("frac_deciding", "median"))
           .reset_index())
    agg["eligible"] = agg["periods_signal"] > 0
    agg["usable_raw"] = agg["periods_usable_raw"] > 0
    agg["status"] = np.where(~agg["eligible"], "non exploitable",
                             np.where(agg["usable_raw"], "exploitable — utilisable brut",
                                      "exploitable — à calibrer"))
    return agg.sort_values(keys)


def models_eligibility(register: pd.DataFrame) -> pd.DataFrame:
    """
    One line per model, variable and scale: is it admitted to the workflow?

    A model enters the workflow as soon as **one** of its products discriminates;
    the count of admitted products says how broadly it is usable, and the count
    of products usable raw says how much of it needs no calibration.
    """
    keys = ["system", "model", "variable", "stream", "scale"]
    out = (register.groupby(keys)
           .agg(products=("product", "count"), products_eligible=("eligible", "sum"),
                products_usable_raw=("usable_raw", "sum"),
                median_frac_discrimination=("median_frac_discrimination", "median"))
           .reset_index())
    out["eligible"] = out["products_eligible"] > 0
    return out.sort_values(keys)


def _write_summary(cfg, ctx, out: Path, rows: list[dict],
                   min_fraction: float = MIN_FRACTION) -> list[dict]:
    """Merge the new rows into ``skill_raw_summary.csv`` and refresh the register."""
    table = pd.DataFrame(rows)
    if table.empty:
        return []
    f = out / "skill_raw_summary.csv"
    keys = ["system", "model", "variable", "scale", "period", "product", "metric"]
    if f.exists():              # keep the rows of the runs not repeated here
        old = pd.read_csv(f)
        if set(keys) <= set(old.columns):
            old = old[~old.set_index(keys).index.isin(table.set_index(keys).index)]
            table = pd.concat([old, table], ignore_index=True)
        else:
            ctx.warn("ancienne synthèse dans un format antérieur (sans produit) : remplacée")
    table = table.sort_values(keys)
    table.to_csv(f, index=False)
    ctx.record_output(f, role="summary")

    register = eligibility_table(table, min_fraction)
    reg_dir = cfg.path_of("output_root") / "registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    f = reg_dir / "products_eligibility.csv"
    register.to_csv(f, index=False)
    ctx.record_output(f, role="eligibility")
    models = models_eligibility(register)
    f = reg_dir / "models_eligibility.csv"
    models.to_csv(f, index=False)
    ctx.record_output(f, role="models_eligibility")
    counts = register["status"].value_counts().to_dict()
    ctx.log.info("registre : %s", ", ".join(f"{v} {k}" for k, v in counts.items()))
    ctx.log.info("modèles admis : %d sur %d (système × modèle × variable × échelle)",
                 int(models["eligible"].sum()), len(models))
    ctx.record_parameter("n_rows", len(table))
    ctx.record_parameter("register_status", counts)
    return register.to_dict("records")


def rebuild_summary(config: str, min_fraction: float = MIN_FRACTION) -> RunContext:
    """
    Rebuild the summary table and the register from the netCDF scores.

    The scores live in ``netcdf/<system>_<model>/<scale>/<variable>/<product>/<metric>.nc``;
    the table is only their summary over the mask, so rebuilding it costs seconds
    when a run is interrupted after the scores are written.
    """
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    shapefile = cfg.raw["paths"]["shapefile"]
    with RunContext(cfg, step="skill_raw_summary") as ctx:
        rows = []
        for folder in sorted((out / "netcdf").glob("*/*/*/*")):
            files = sorted(folder.glob("*.nc"))
            if not files:
                continue
            system, model = folder.parent.parent.parent.name.split("_", 1)
            scale, variable, product = (folder.parent.parent.name, folder.parent.name,
                                        folder.name)
            maps = xr.merge([open_cf(f) for f in files],
                            combine_attrs="override").load()
            for f in files:
                ctx.record_input(f, role="skill_netcdf")
            first = maps[[v for v in maps.data_vars if v != "n_years"][0]]
            mask = mask_like(shapefile, first.isel(period=0, drop=True, missing_dims="ignore")
                             .isel(category=0, drop=True) if "category" in first.dims
                             else first.isel(period=0, drop=True))
            lead_of = {str(k): int(str(k).split("_")[1].lstrip("m") or 0)
                       for k in maps["period"].values}
            rows += product_summary_rows(maps, mask, system, model, variable, scale, product,
                                         lead_of, stream=maps.attrs.get("stream", "daily"))
            ctx.log.info("%s %s %s %s %s : %d période(s)", system, model, variable, scale,
                         product, maps.sizes["period"])
        ctx.record_parameter("eligibility", _write_summary(cfg, ctx, out, rows, min_fraction))
    return ctx


def run(config: str, systems=("c3s", "nmme"), variables=("precip",), models=None,
        scales=None, min_fraction: float = MIN_FRACTION,
        selection=None, metrics: str = "deciding") -> RunContext:
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    scales = list(scales) if scales else cfg.scales

    # a system switched off in the configuration is skipped, not attempted
    systems = [s for s in systems if s in cfg.enabled_systems()]
    if not systems:
        raise ValueError("aucun système actif : voir `enabled` dans la configuration du cycle")

    with RunContext(cfg, step="skill_raw") as ctx:
        ctx.record_parameter("systems", list(systems))
        ctx.record_parameter("variables", list(variables))
        ctx.record_parameter("scales", scales)
        ctx.record_parameter("min_fraction", min_fraction)
        ctx.record_parameter("metrics", metrics)
        announce_subset(cfg, ctx, selection)
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
                    groups = hindcast_streams(cfg, system, model, variable, sys_scales)
                    for stream, stream_scales in groups.items():
                        if not stream_scales:
                            continue
                        try:
                            pairs, periods = prepare_pairs(cfg, system, model, variable,
                                                           stream_scales, ctx, selection,
                                                           stream)
                        except (FileNotFoundError, KeyError) as exc:
                            ctx.warn(f"{system} {model} {variable} ({stream}) : "
                                     f"données absentes ({exc})")
                            continue
                        ctx.log.info("%s %s %s [%s] : %d périodes, %d années, %d membres",
                                     system, model, variable, stream, len(periods),
                                     pairs.attrs["n_years"], pairs.attrs["n_members"])

                        mask = pairs["obs"].notnull().any(["year", "period"])
                        n_members = int(pairs.attrs["n_members"])
                        members = pairs["members"] if "members" in pairs else None
                        base_attrs = {**ctx.netcdf_attrs(), "system": system, "model": model,
                                      "variable": variable, "kind_of_run": "raw hindcast skill",
                                      "n_years": pairs.attrs["n_years"],
                                      "hindcast_period": pairs.attrs.get("years", ""),
                                      "n_members": n_members, "stream": stream,
                                      "mask": "CEEAC (shapefile)",
                                      "mask_cells": pairs.attrs.get("mask_cells"),
                                      "init_date": str(cfg.init_date.date()),
                                      "cross_validation": cfg.cv_scheme,
                                      "metrics_mode": str(metrics)}
                        var_thresholds = cfg.thresholds.get(variable, {})

                        for scale in sorted({p.scale for p in periods}):
                            scale_periods = [p for p in periods if p.scale == scale]
                            by_product: dict[str, list] = {}
                            for period in scale_periods:
                                key = period.key
                                ref = pairs["obs"].sel(period=key, drop=True)
                                octx = ObservationContext(ref, variable, scale, var_thresholds,
                                                          mask, "year")
                                if members is not None:
                                    dist = RawEnsemble(members.sel(period=key, drop=True))
                                    kinds = None
                                else:
                                    # a system delivered as an ensemble mean carries no
                                    # probability (D22): only the value products are scored
                                    dist = RawEnsemble(
                                        pairs["ensmean"].sel(period=key, drop=True)
                                        .expand_dims({"number": [0]}))
                                    kinds = (VALUE,)
                                maps = product_maps(dist, octx, mode=metrics,
                                                    n_members=n_members, kinds=kinds)
                                for name, ds in maps.items():
                                    by_product.setdefault(name, []).append(
                                        ds.expand_dims({"period": [key]}))
                                del maps
                            labels = {p.key: p.label(cfg.init_date.year) for p in scale_periods}
                            labels_fr = {p.key: p.label_fr(cfg.init_date.year, with_dates=True)
                                         for p in scale_periods}
                            for name, per_period in by_product.items():
                                merged = xr.concat(per_period, dim="period",
                                                   combine_attrs="override")
                                keys = [str(k) for k in merged["period"].values]
                                merged = merged.assign_coords(
                                    label=("period", [labels[k] for k in keys]),
                                    label_fr=("period", [labels_fr[k] for k in keys]),
                                    scale=("period", [scale] * len(keys)))
                                merged.attrs = {**base_attrs, **per_period[0].attrs}
                                write_product_netcdf(merged, out / "netcdf", system, model,
                                                     variable, scale, name, ctx)
                                summary += product_summary_rows(
                                    merged, mask, system, model, variable, scale, name,
                                    {p.key: p.month_offset for p in scale_periods}, stream)
                            ctx.log.info("  %s %s : %d produit(s) noté(s) sur %d période(s)",
                                         scale, variable, len(by_product), len(scale_periods))
                            del by_product
                            gc.collect()

                        del pairs
                        gc.collect()

        eligibility = _write_summary(cfg, ctx, out, summary, min_fraction) or eligibility
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
    ap.add_argument("--metrics", nargs="+", default=["deciding"],
                    help="métriques par produit : deciding (défaut : la métrique décisive, "
                         "sa version débiaisée et la discrimination), reported, all, "
                         "ou une liste explicite (rpss groc roc_area)")
    add_period_arguments(ap)
    ap.add_argument("--from-maps", action="store_true",
                    help="reconstruire seulement le tableau de synthèse à partir des cartes déjà écrites")
    args = ap.parse_args(argv)
    if args.from_maps:
        rebuild_summary(args.config, args.min_fraction)
        return
    run(args.config, args.systems, args.variables, args.models, args.scales,
        args.min_fraction, selection=selected_periods(args),
        metrics=args.metrics[0] if len(args.metrics) == 1 else tuple(args.metrics))


if __name__ == "__main__":
    main()
