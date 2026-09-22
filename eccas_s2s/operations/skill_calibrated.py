"""
Skill of the calibrated hindcasts (workflow E5, phase P3).

Scores the files written by :mod:`eccas_s2s.operations.calibrate_hindcast` with
**exactly** the metrics of phase P2 — same scores, same leave-one-year-out
categories, same CEEAC mask — because the only question worth asking of a
calibration is: *does it beat the raw forecast, and does it beat climatology?*
Anything computed differently on the two sides would answer a different question.

The tree mirrors the raw one, with the method as an extra branch::

    skill/<cycle>/calibrated/netcdf/<system>_<model>/<method>/<scale>/<variable>/<metric>.nc
    skill/<cycle>/calibrated/figures/…            (same branches, one map per period)
    skill/<cycle>/calibrated/diagrams/…
    skill/<cycle>/calibrated/skill_calibrated_summary.csv

The summary table carries one row per model, method, variable and period, next to
the raw scores of the same period: that table is what the method selection of
E5 and the multi-model weighting of P6 read.

Example::

    python scripts/run_skill_calibrated.py --config config/cycle_202609.yaml
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import xarray as xr

from eccas_s2s.core.geo import mask_like
from eccas_s2s.operations.calibrate_hindcast import calibrated_dir, observation_on_grid
from eccas_s2s.operations.skill_raw import (METRICS, _write_summary, skill_dir, summary_rows,
                                            write_score_netcdf)
from eccas_s2s.core.periods import build_periods
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.pairs import observed_categories
from eccas_s2s.validate.scores import deterministic_scores, tercile_skill


def _periods_of(cfg, scale: str, keys) -> list:
    """The Period objects matching the keys stored in a calibrated file."""
    periods = build_periods(cfg.init_date, 400, scales=(scale,))
    by_key = {p.key: p for p in periods}
    return [by_key[k] for k in keys if k in by_key]


def score_file(cfg, path: Path, ctx=None) -> tuple[xr.Dataset, dict]:
    """Scores of one calibrated file, on the grid it was written on."""
    with xr.open_dataset(path) as ds:
        cal = ds.load()
    system, model = cal.attrs["system"], cal.attrs["model"]
    variable, scale, method = cal.attrs["variable"], cal.attrs["scale"], cal.attrs["method"]
    keys = [str(k) for k in cal["period"].values]
    periods = _periods_of(cfg, scale, keys)
    years = [int(y) for y in cal["year"].values]
    obs = observation_on_grid(cfg, variable, periods, years).sel(period=keys)
    obs_cat, _ = observed_categories(obs)

    det = deterministic_scores(cal["forecast"], obs)
    prob = tercile_skill(cal["prob"], obs_cat).rename({"n_years": "n_years_prob"})
    maps = xr.merge([det, prob])
    maps = maps.assign_coords(label=cal["label"], label_fr=cal["label_fr"], scale=cal["scale"])
    maps.attrs.update({**(ctx.netcdf_attrs() if ctx else {}), **cal.attrs,
                       "kind": "calibrated hindcast skill"})
    meta = {"system": system, "model": model, "variable": variable, "scale": scale,
            "method": method, "periods": keys, "obs": obs}
    return maps, meta


def run(config: str, systems=("c3s",), variables=None, models=None, methods=None,
        scales=None) -> RunContext:
    cfg = load_cycle(config)
    src = calibrated_dir(cfg)
    out = skill_dir(cfg, "calibrated")
    shapefile = cfg.raw["paths"]["shapefile"]

    with RunContext(cfg, step="skill_calibrated") as ctx:
        files = sorted(src.glob("*/*/*/*.nc"))
        if not files:
            raise FileNotFoundError(f"aucun hindcast calibré dans {src} "
                                    "(lancer d'abord run_calibrate_hindcast.py)")
        ctx.record_parameter("n_files", len(files))
        summary = []
        for f in files:
            system, model = f.parent.parent.parent.name.split("_", 1)
            scale, variable, method = f.parent.parent.name, f.parent.name, f.stem
            if system not in systems or (models and model not in models):
                continue
            if (variables and variable not in variables) or (methods and method not in methods):
                continue
            if scales and scale not in scales:
                continue
            maps, meta = score_file(cfg, f, ctx)
            ctx.record_input(f, role="calibrated_hindcast")
            mask = mask_like(shapefile, maps["pearson"].isel(period=0, drop=True))
            branch = out / "netcdf" / f"{system}_{model}" / method
            write_score_netcdf(maps, branch, system, model, variable, scale, ctx)
            lead_of = {k: int(k.split("_")[1].lstrip("m") or 0) for k in meta["periods"]}
            rows = summary_rows(maps, mask, system, model, variable, lead_of)
            for row in rows:
                row["method"] = method
            summary += rows
            ctx.log.info("%s %s %s %s %s : %d période(s)", system, model, method, variable,
                         scale, maps.sizes["period"])
            del maps
            gc.collect()

        table = pd.DataFrame(summary)
        if not table.empty:
            path = out / "skill_calibrated_summary.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            keys = ["system", "model", "method", "variable", "period"]
            if path.exists():
                old = pd.read_csv(path)
                old = old[~old.set_index(keys).index.isin(table.set_index(keys).index)]
                table = pd.concat([old, table], ignore_index=True)
            table = table.sort_values(["system", "model", "variable", "scale", "method", "period"])
            table.to_csv(path, index=False)
            ctx.record_output(path, role="summary")
            ctx.record_parameter("n_rows", len(table))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--systems", nargs="+", default=["c3s", "nmme"], choices=["c3s", "nmme"])
    ap.add_argument("--variables", nargs="+", choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    args = ap.parse_args(argv)
    run(args.config, args.systems, args.variables, args.models, args.methods, args.scales)


if __name__ == "__main__":
    main()
