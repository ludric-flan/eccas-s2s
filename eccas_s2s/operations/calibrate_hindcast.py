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
import xarray as xr

from eccas_s2s.calibrate.base import CATEGORIES
from eccas_s2s.calibrate.bias import bias_calibrators
from eccas_s2s.calibrate.logistic import logistic_calibrators
from eccas_s2s.calibrate.ngr import ngr_calibrators
from eccas_s2s.calibrate.qmap import qmap_calibrators
from eccas_s2s.core.geo import mask_like
from eccas_s2s.core.periods import build_periods
from eccas_s2s.operations import obs_chirps, obs_era5
from eccas_s2s.operations.skill_raw import (AGGREGATION, OBS_VARIABLE, SYSTEM_SCALES,
                                            _load_hindcast, split_skill_name)
from eccas_s2s.obs.climatology import obs_period_totals
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.cv import loyo_quantile

TERCILES = (1 / 3, 2 / 3)
#: the method families of the CAPC-AC note, in the order the report lists them.
FAMILIES = ("bias", "qmap", "logistic", "ngr")


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
        families=FAMILIES) -> RunContext:
    cfg = load_cycle(config)
    out_root = calibrated_dir(cfg)

    with RunContext(cfg, step="calibrate_hindcast") as ctx:
        ctx.record_parameter("systems", list(systems))
        ctx.record_parameter("variables", list(variables))
        ctx.record_parameter("scales", list(scales))
        ctx.record_parameter("families", list(families))
        ctx.record_parameter("cross_validation", cfg.cv_scheme)

        for system in systems:
            candidates = (list(cfg.c3s_models) if system == "c3s"
                          else list(cfg.raw["systems"]["nmme"]["models"]))
            sys_scales = [s for s in scales if s in SYSTEM_SCALES[system]]
            for model in [m for m in candidates if models is None or m in models]:
                for variable in variables:
                    if system == "nmme" and variable in ("tmax", "tmin"):
                        continue
                    try:
                        hind = _load_hindcast(cfg, system, model, variable).load()
                    except (FileNotFoundError, KeyError) as exc:
                        ctx.warn(f"{system} {model} {variable} : données absentes ({exc})")
                        continue
                    keep = [p for p in hind["period"].values
                            if str(hind["scale"].sel(period=p).values) in sys_scales]
                    hind = hind.sel(period=keep)
                    periods = build_periods(
                        cfg.init_date,
                        max((cfg.c3s_models[model].max_lead_days if system == "c3s" else 400), 1),
                        scales=tuple(sys_scales))
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
                        timings: dict[str, float] = {}
                        for key in keys:                      # period by period: memory
                            ens = model_on_grid(hind.sel(period=key, drop=True), obs)
                            ref = obs.sel(period=key, drop=True)
                            for cal in cals:
                                t0 = time.perf_counter()
                                dist = cal.fit_predict_loyo(ens, ref)
                                summary = _summary(dist, ref, variable).load()
                                timings[cal.name] = timings.get(cal.name, 0.0) + (
                                    time.perf_counter() - t0)
                                results[cal.name].append(summary.expand_dims(period=[key]))
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
                            path = folder / f"{name}.nc"
                            tmp = path.with_suffix(".tmp.nc")
                            ds.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4}
                                                        for v in ds.data_vars})
                            tmp.replace(path)
                            ctx.record_output(path, role="calibrated_hindcast", system=system,
                                              model=model, variable=variable, scale=scale,
                                              method=name)
                        ctx.log.info("  %s %s : %d méthode(s) écrite(s) dans %s",
                                     scale, variable, len(results), folder)
                        ctx.log.info("  temps par méthode (s) : %s",
                                     ", ".join(f"{k} {v:.0f}" for k, v in
                                               sorted(timings.items(), key=lambda kv: -kv[1])))
                        ctx.record_parameter(f"timings_{scale}_{variable}",
                                             {k: round(v, 1) for k, v in timings.items()})
                    del hind, obs
                    gc.collect()
    return ctx


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
    args = ap.parse_args(argv)
    run(args.config, args.systems, args.variables, args.models, args.scales, args.families)


if __name__ == "__main__":
    main()
